# -*- coding: utf-8 -*-
"""
Construire des tuiles mesh au pas de 1 km à partir des “chunks” PLY colorisés produits par WaSuRe,
en utilisant les dalles MNS LiDAR (GeoTIFF) comme grille de référence.

- Entrée : un dossier contenant beaucoup de PLY (typiquement des carrés ~125 m), en EPSG:2154 (Lambert-93)
- Référence : des dalles MNS LiDAR (GeoTIFF) qui définissent exactement la grille 1 km (emprises exactes)
- Sortie : un PLY par dalle MNS, construit en :
    1) sélectionnant les chunks dont la bbox intersecte la bbox de la dalle MNS (+ buffer)
    2) concaténant les meshes sélectionnés
    3) conservant les faces de manière conservative pour éviter de créer des trous :
        garder un triangle si au moins un sommet est dans la bbox de la dalle (+ eps)
        OU si au moins une arête du triangle intersecte la bbox (+ eps)
   (On ne “découpe” pas géométriquement les triangles.)

Notes :
- L’objectif est d’éviter les trous aux limites de dalles (effet “watertight”), au prix d’un recouvrement.
- Le recouvrement peut provoquer du z-fighting dans certains viewers ; si besoin, on pourra ajouter une règle
  déterministe “d’appartenance” pour réduire/éliminer les doublons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Optional
from plyfile import PlyData, PlyElement

import numpy as np

# rasterio est utilisé uniquement pour lire l’emprise (bounds) des dalles MNS GeoTIFF.
try:
    import rasterio
except ImportError as e:
    raise ImportError("rasterio is required to read DSM tile bounds.") from e

# meshio sert à lire/écrire des meshes tout en conservant des attributs de sommets (notamment couleurs).
try:
    import meshio
except ImportError as e:
    raise ImportError("meshio is required to read/write PLY while preserving vertex colors.") from e


# -----------------------------
# Configuration
# -----------------------------

@dataclass(frozen=True)
class KmTilesConfig:
    # Dossier contenant les dalles MNS (GeoTIFF) qui définissent la grille 1 km.
    dsm_dir: Path
    # Filtre des fichiers MNS (par défaut : tous les .tif).
    dsm_glob: str = "*.tif"
    # Nom du dossier de sortie, créé à côté du dossier d’entrée.
    out_dirname: str = "ply_km_tiles"
    # Suffixe de nommage des PLY de sortie (ex: "ortho" ou "origin").
    suffix: str = "ortho"  # or "origin"

    # Paramètres de sélection / robustesse
    preselect_buffer_m: float = 2.0   # buffer (en m) pour élargir la bbox MNS lors de la présélection des chunks
    eps_m: float = 0.05               # epsilon numérique (en m) pour les tests “dans la bbox” / “segment-box” (5 cm)
    overwrite: bool = True            # écraser si les PLY de sortie existent déjà

    # Format de sortie
    binary_out: bool = True           # True -> PLY binaire, False -> PLY ASCII


# -----------------------------
# Outils géométriques (tests AABB)
# -----------------------------

def _bbox_expand(b: Tuple[float, float, float, float, float, float], e: float) -> Tuple[float, float, float, float, float, float]:
    """
    Élargit une AABB (Axis-Aligned Bounding Box) d’une marge e sur chaque dimension.

    b = (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    xmin, xmax, ymin, ymax, zmin, zmax = b
    return (xmin - e, xmax + e, ymin - e, ymax + e, zmin - e, zmax + e)


def _bbox_intersects(a: Tuple[float, float, float, float, float, float],
                     b: Tuple[float, float, float, float, float, float]) -> bool:
    """
    Test d’intersection entre deux AABB.
    Retourne True si les deux boîtes se recouvrent (intersection non vide).
    """
    ax0, ax1, ay0, ay1, az0, az1 = a
    bx0, bx1, by0, by1, bz0, bz1 = b
    return (ax0 <= bx1 and ax1 >= bx0 and
            ay0 <= by1 and ay1 >= by0 and
            az0 <= bz1 and az1 >= bz0)


def _points_in_aabb(pts: np.ndarray, aabb: Tuple[float, float, float, float, float, float]) -> np.ndarray:
    """
    Test vectorisé : indique quels points 3D appartiennent à l’AABB.

    pts : tableau (N, 3)
    Retour : masque booléen (N,)
    """
    xmin, xmax, ymin, ymax, zmin, zmax = aabb
    return ((pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
            (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax) &
            (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax))


def _segment_intersects_aabb(p0: np.ndarray, p1: np.ndarray,
                            aabb: Tuple[float, float, float, float, float, float]) -> bool:
    """
    Test d’intersection d’un segment avec une AABB par méthode des “slabs”.

    Retourne True si le segment [p0, p1] intersecte la boîte.

    Principe :
    - On paramètre le segment p(t) = p0 + t*(p1-p0), t ∈ [0,1]
    - On intersecte l’intervalle [0,1] avec les intervalles de t imposés par chaque “slab”
      (xmin<=x<=xmax, ymin<=y<=ymax, zmin<=z<=zmax).
    """
    xmin, xmax, ymin, ymax, zmin, zmax = aabb

    d = p1 - p0
    tmin = 0.0
    tmax = 1.0

    for i, (bmin, bmax) in enumerate(((xmin, xmax), (ymin, ymax), (zmin, zmax))):
        if abs(d[i]) < 1e-15:
            # Segment quasi parallèle au slab : il faut que p0 soit déjà dans l’intervalle du slab.
            if p0[i] < bmin or p0[i] > bmax:
                return False
        else:
            # Calcul des paramètres t où le segment coupe les plans bmin et bmax.
            ood = 1.0 / d[i]
            t1 = (bmin - p0[i]) * ood
            t2 = (bmax - p0[i]) * ood
            if t1 > t2:
                t1, t2 = t2, t1
            # On restreint l’intervalle de t admissible.
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False

    return True


def _tri_keep_mask(vertices: np.ndarray,
                   faces: np.ndarray,
                   aabb: Tuple[float, float, float, float, float, float]) -> np.ndarray:
    """
    Sélection conservative des triangles :
    - On garde un triangle si au moins un sommet est dans l’AABB
      OU si au moins une arête intersecte l’AABB.

    But : éviter de “couper” des triangles (ce qui créerait des trous) aux limites de dalles.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Test rapide : un triangle est gardé si au moins un sommet est à l’intérieur.
    inside0 = _points_in_aabb(v0, aabb)
    inside1 = _points_in_aabb(v1, aabb)
    inside2 = _points_in_aabb(v2, aabb)
    keep = inside0 | inside1 | inside2

    # Pour les triangles dont aucun sommet n’est dedans, on teste l’intersection des arêtes.
    idx = np.where(~keep)[0]
    if idx.size == 0:
        return keep

    for k in idx:
        p0 = v0[k]
        p1 = v1[k]
        p2 = v2[k]
        if (_segment_intersects_aabb(p0, p1, aabb) or
            _segment_intersects_aabb(p1, p2, aabb) or
            _segment_intersects_aabb(p2, p0, aabb)):
            keep[k] = True

    return keep


# -----------------------------
# Outils PLY
# -----------------------------

def _write_ply_with_plyfile(path: Path, mesh: meshio.Mesh, binary: bool = True) -> None:
    """
    Écrit un mesh triangulé au format PLY via plyfile, en conservant les attributs de sommets
    (ex : red/green/blue) pour que les outils aval (3D Tiles) gardent les couleurs.

    Notes :
    - Les couleurs PLY doivent être des champs scalaires 1D (red, green, blue), pas un champ (N,3).
    - Les faces sont écrites en propriété 'vertex_indices' sous forme de listes d'indices.
    """
    points = np.asarray(mesh.points)
    n = points.shape[0]

    # --- Construction du type de sommet (dtype structuré)
    # On force en float32 pour limiter la taille des fichiers (sauf besoin explicite de float64).
    vx = points[:, 0].astype(np.float32, copy=False)
    vy = points[:, 1].astype(np.float32, copy=False)
    vz = points[:, 2].astype(np.float32, copy=False)

    vertex_fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    vertex_arrays = {"x": vx, "y": vy, "z": vz}

    pd = mesh.point_data or {}

    # Si meshio fournit un champ 'rgb' (N,3), on le convertit en red/green/blue (uint8).
    if "rgb" in pd and getattr(pd["rgb"], "ndim", 1) == 2 and pd["rgb"].shape[1] == 3:
        rgb = pd["rgb"].astype(np.uint8, copy=False)
        pd = dict(pd)  # copie pour ne pas modifier l’original
        pd["red"] = rgb[:, 0]
        pd["green"] = rgb[:, 1]
        pd["blue"] = rgb[:, 2]
        pd.pop("rgb", None)

    # Préservation des champs couleurs “standards” si présents
    for cname in ("red", "green", "blue", "r", "g", "b"):
        if cname in pd:
            arr = np.asarray(pd[cname])
            if arr.shape != (n,):
                # On ignore les champs non scalaires (sécurité).
                continue
            # On impose uint8 (format attendu pour les couleurs PLY).
            arr = arr.astype(np.uint8, copy=False)
            vertex_fields.append((cname, "u1"))
            vertex_arrays[cname] = arr

    # Préservation des normales si disponibles (nx, ny, nz)
    for nname in ("nx", "ny", "nz"):
        if nname in pd:
            arr = np.asarray(pd[nname])
            if arr.shape != (n,):
                continue
            arr = arr.astype(np.float32, copy=False)
            vertex_fields.append((nname, "f4"))
            vertex_arrays[nname] = arr

    # Optionnel : préserver d’autres champs 1D scalaires (int/float)
    # (On évite les tableaux multi-dim car l’écriture PLY peut varier selon les conventions.)
    for k, v in pd.items():
        if k in vertex_arrays:
            continue
        arr = np.asarray(v)
        if arr.ndim != 1 or arr.shape[0] != n:
            continue
        if arr.dtype.kind in ("u", "i"):
            # Entiers -> int32
            arr2 = arr.astype(np.int32, copy=False)
            vertex_fields.append((k, "i4"))
            vertex_arrays[k] = arr2
        elif arr.dtype.kind == "f":
            # Flottants -> float32
            arr2 = arr.astype(np.float32, copy=False)
            vertex_fields.append((k, "f4"))
            vertex_arrays[k] = arr2
        # Autres types (string/object) ignorés.

    # --- Création du tableau structuré des sommets
    vdata = np.empty(n, dtype=vertex_fields)
    for name in vdata.dtype.names:
        vdata[name] = vertex_arrays[name]

    vertex_el = PlyElement.describe(vdata, "vertex")

    # --- Récupération des triangles
    tri = None
    for cb in mesh.cells:
        if cb.type == "triangle":
            tri = cb.data
            break

    if tri is None:
        # Cas rare : pas de faces triangulées.
        tri = np.zeros((0, 3), dtype=np.int32)

    tri = np.asarray(tri, dtype=np.int32)

    # plyfile attend une propriété “liste d’indices” par face.
    fdata = np.empty(tri.shape[0], dtype=[("vertex_indices", "O")])
    fdata["vertex_indices"] = [t.tolist() for t in tri]
    face_el = PlyElement.describe(fdata, "face")

    # text=False => binaire, text=True => ASCII
    ply = PlyData([vertex_el, face_el], text=not binary)
    ply.write(str(path))


def _read_ply_points_only(path: Path) -> np.ndarray:
    """Lit uniquement les coordonnées XYZ des sommets d’un PLY (voie rapide)."""
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    pts = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64, copy=False)
    return pts


def _read_ply_with_plyfile_as_meshio(path: Path) -> meshio.Mesh:
    """
    Lecteur robuste pour les PLY WaSuRe :
    - On lit via plyfile (plus tolérant sur certains formats)
    - On reconstruit ensuite un meshio.Mesh.

    Objectif : contourner certains “quirks” de meshio sur des variantes PLY
    (notamment les propriétés de faces).
    """
    ply = PlyData.read(str(path))

    # --- sommets
    v = ply["vertex"].data
    points = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64, copy=False)

    point_data = {}

    # --- couleurs (cas fréquents : red/green/blue ou r/g/b)
    if ("red" in v.dtype.names) and ("green" in v.dtype.names) and ("blue" in v.dtype.names):
        rgb = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.uint8, copy=False)
        point_data["rgb"] = rgb
    elif ("r" in v.dtype.names) and ("g" in v.dtype.names) and ("b" in v.dtype.names):
        rgb = np.column_stack([v["r"], v["g"], v["b"]]).astype(np.uint8, copy=False)
        point_data["rgb"] = rgb

    # --- normales (optionnel)
    if ("nx" in v.dtype.names) and ("ny" in v.dtype.names) and ("nz" in v.dtype.names):
        nrm = np.column_stack([v["nx"], v["ny"], v["nz"]]).astype(np.float32, copy=False)
        point_data["normals"] = nrm

    # --- faces
    if "face" not in ply:
        # Pas de faces : on renvoie un mesh “nuage” avec un bloc triangles vide.
        return meshio.Mesh(points=points, cells=[("triangle", np.zeros((0, 3), dtype=np.int64))], point_data=point_data)

    f = ply["face"].data

    # Le champ d’indices peut s’appeler vertex_indices ou vertex_index selon les fichiers.
    if "vertex_indices" in f.dtype.names:
        inds = f["vertex_indices"]
    elif "vertex_index" in f.dtype.names:
        inds = f["vertex_index"]
    else:
        raise ValueError(f"Unsupported PLY face format in {path.name}: {f.dtype.names}")

    # Si des faces ne sont pas triangulées, on les triangule en “éventail” (fan triangulation).
    tris = []
    for poly in inds:
        poly = np.asarray(poly, dtype=np.int64)
        if poly.size < 3:
            continue
        if poly.size == 3:
            tris.append(poly)
        else:
            # Triangulation en éventail : (0,i,i+1)
            v0 = poly[0]
            for i in range(1, poly.size - 1):
                tris.append(np.array([v0, poly[i], poly[i + 1]], dtype=np.int64))

    tri = np.vstack(tris) if tris else np.zeros((0, 3), dtype=np.int64)

    return meshio.Mesh(points=points, cells=[("triangle", tri)], point_data=point_data)


def _meshio_read_ply(path: Path) -> meshio.Mesh:
    """
    Lecteur “safe” :
    - On tente meshio.read() (souvent plus rapide quand ça marche)
    - En cas d’exception, on bascule sur le lecteur plyfile (plus robuste pour WaSuRe)
    """
    try:
        return meshio.read(str(path))
    except Exception:
        return _read_ply_with_plyfile_as_meshio(path)


# Bloc d’écriture meshio laissé commenté volontairement (ancienne approche / compat versions).
# def _meshio_write_ply(path: Path, mesh: meshio.Mesh, binary: bool) -> None:
#     ...


def _mesh_bbox(mesh: meshio.Mesh) -> Tuple[float, float, float, float, float, float]:
    """
    Calcule la bbox 3D d’un meshio.Mesh sous forme d’AABB :
    (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    pts = mesh.points
    xmin, ymin, zmin = np.min(pts, axis=0)
    xmax, ymax, zmax = np.max(pts, axis=0)
    return (float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax))


def _subset_mesh_faces(mesh: meshio.Mesh, keep_faces: np.ndarray) -> meshio.Mesh:
    """
    Conserve uniquement les faces (triangles) sélectionnées par keep_faces,
    et remappe les indices de sommets pour compacter le mesh.

    On préserve aussi les attributs de sommets (point_data) pour les sommets conservés
    (ex : couleurs).
    """
    # Recherche du bloc de triangles
    tri_cells = None
    other_cells = []
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            tri_cells = cell_block.data
        else:
            other_cells.append(cell_block)

    if tri_cells is None:
        raise ValueError("No triangle cells found in mesh.")

    tri_kept = tri_cells[keep_faces]
    if tri_kept.size == 0:
        # Cas : aucun triangle conservé -> on renvoie un mesh vide, mais avec les mêmes clés de point_data.
        empty_points = np.zeros((0, 3), dtype=np.float64)
        empty_cells = [("triangle", np.zeros((0, 3), dtype=np.int64))]
        empty_point_data = {
            k: np.zeros((0,), dtype=v.dtype) if v.ndim == 1 else np.zeros((0, v.shape[1]), dtype=v.dtype)
            for k, v in (mesh.point_data or {}).items()
        }
        return meshio.Mesh(points=empty_points, cells=empty_cells, point_data=empty_point_data)

    # Sommets effectivement utilisés par les triangles conservés
    used = np.unique(tri_kept.reshape(-1))

    # Construction d’un mapping ancien_index -> nouveau_index
    new_index = -np.ones(mesh.points.shape[0], dtype=np.int64)
    new_index[used] = np.arange(used.size, dtype=np.int64)
    tri_remap = new_index[tri_kept]

    # Sous-ensemble de sommets
    new_points = mesh.points[used]

    # Sous-ensemble des attributs de sommets
    new_point_data = {}
    if mesh.point_data:
        for k, v in mesh.point_data.items():
            new_point_data[k] = v[used]

    # On ne conserve que les triangles (si besoin d’autres primitives, il faudra étendre).
    return meshio.Mesh(points=new_points, cells=[("triangle", tri_remap)], point_data=new_point_data)


def _concat_meshes(meshes: Iterable[meshio.Mesh]) -> meshio.Mesh:
    """
    Concatène une liste de meshes triangulés sans dédoublonner les sommets.
    Les indices de triangles sont décalés au fur et à mesure (offset cumulatif).

    Les attributs de sommets (point_data) sont conservés uniquement pour les clés communes à tous les meshes
    (ex : 'rgb' si présent partout).
    """
    meshes = list(meshes)
    if not meshes:
        return meshio.Mesh(
            points=np.zeros((0, 3), dtype=np.float64),
            cells=[("triangle", np.zeros((0, 3), dtype=np.int64))]
        )

    # Calcul des clés de point_data communes à tous les meshes
    common_keys = None
    for m in meshes:
        keys = set((m.point_data or {}).keys())
        common_keys = keys if common_keys is None else (common_keys & keys)
    common_keys = common_keys or set()

    points_all = []
    tri_all = []
    pdata_all = {k: [] for k in common_keys}

    offset = 0
    for m in meshes:
        # Extraction des triangles
        tri = None
        for cb in m.cells:
            if cb.type == "triangle":
                tri = cb.data
                break
        if tri is None:
            continue

        pts = m.points
        points_all.append(pts)
        tri_all.append(tri + offset)

        # Accumulation des attributs communs
        for k in common_keys:
            pdata_all[k].append(m.point_data[k])

        offset += pts.shape[0]

    if not points_all:
        return meshio.Mesh(
            points=np.zeros((0, 3), dtype=np.float64),
            cells=[("triangle", np.zeros((0, 3), dtype=np.int64))]
        )

    points_cat = np.vstack(points_all)
    tri_cat = np.vstack(tri_all)

    point_data_cat = {}
    for k, chunks in pdata_all.items():
        point_data_cat[k] = np.concatenate(chunks, axis=0)

    return meshio.Mesh(points=points_cat, cells=[("triangle", tri_cat)], point_data=point_data_cat)


def _extract_km_index_from_mns_name(stem: str) -> str:
    """
    Extrait l’identifiant de dalle km depuis le nom d’un fichier MNS LiDAR IGN.

    Exemple :
        LHD_FXX_0606_6933_MNS_O_0M50_LAMB93_IGN69
        -> 0606_6933
    """
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected MNS filename format: {stem}")

    # Pattern IGN typique : LHD_FXX_XXXX_YYYY_...
    x_km = parts[2]
    y_km = parts[3]

    if not (x_km.isdigit() and y_km.isdigit()):
        raise ValueError(f"Cannot extract km index from: {stem}")

    return f"{x_km}_{y_km}"


# -----------------------------
# Fonction principale
# -----------------------------

def run_post_wasure_make_km_tiles(
    ply_color_dir: Path,
    logger: logging.Logger,
    cfg: KmTilesConfig,
) -> Path:
    """
    Construit des tuiles PLY 1 km à partir des chunks PLY colorisés (WaSuRe) en EPSG:2154.

    Entrées
    -------
    ply_color_dir
        Dossier contenant de nombreux PLY colorisés (EPSG:2154).
    logger
        Logger à utiliser.
    cfg
        Configuration KmTilesConfig.

    Sortie
    ------
    Path
        Dossier de sortie contenant un PLY par dalle MNS (grille 1 km).
    """
    ply_color_dir = Path(ply_color_dir)
    dsm_dir = Path(cfg.dsm_dir)

    # Dossier de sortie : à côté du dossier d’entrée (run_*/<out_dirname>)
    out_dir = ply_color_dir.parent / cfg.out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    # Liste des dalles MNS (référence grille 1 km)
    dsm_paths = sorted(dsm_dir.glob(cfg.dsm_glob))
    if not dsm_paths:
        raise FileNotFoundError(f"No DSM tiles found in {dsm_dir} with glob={cfg.dsm_glob}")

    # Liste des chunks PLY colorisés (petites tuiles)
    ply_paths = sorted(ply_color_dir.glob("*.ply"))
    if not ply_paths:
        raise FileNotFoundError(f"No PLY files found in {ply_color_dir}")

    # Pré-calcul de la bbox de chaque chunk PLY (gain de perf : on ne relit pas les PLY à chaque dalle km)
    logger.info("Precomputing PLY chunk bboxes (%d files)...", len(ply_paths))
    chunk_info = []
    for p in ply_paths:
        pts = _read_ply_points_only(p)
        xmin, ymin, zmin = np.min(pts, axis=0)
        xmax, ymax, zmax = np.max(pts, axis=0)
        bb = (float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax))
        chunk_info.append((p, bb))

    logger.info("Building 1 km tiles from %d DSM tiles...", len(dsm_paths))

    # Boucle : une dalle MNS = une sortie PLY km
    for dsm_path in dsm_paths:
        stem = dsm_path.stem
        km_index = _extract_km_index_from_mns_name(stem)
        out_ply = out_dir / f"{km_index}_mesh_km_{cfg.suffix}.ply"
        tile_id = km_index  # identifiant utilisé dans les logs

        # Politique d’écrasement
        if out_ply.exists() and not cfg.overwrite:
            logger.info("Skip (exists): %s", out_ply)
            continue

        # Lecture de l’emprise XY de la dalle MNS (GeoTIFF)
        with rasterio.open(dsm_path) as ds:
            b = ds.bounds  # left, bottom, right, top
            # Le MNS définit seulement l’emprise XY. Pour Z, on met une plage énorme pour ne rien couper.
            tile_aabb = (float(b.left), float(b.right), float(b.bottom), float(b.top), -1e9, 1e9)

        # Deux AABB :
        # - select_aabb : bbox élargie (buffer) pour sélectionner les chunks candidats
        # - test_aabb   : bbox élargie (epsilon) pour tester triangles/segments de manière robuste
        select_aabb = _bbox_expand(tile_aabb, cfg.preselect_buffer_m)
        test_aabb = _bbox_expand(tile_aabb, cfg.eps_m)

        # Sélection des chunks candidats via intersection de bbox
        candidates = [p for (p, bb) in chunk_info if _bbox_intersects(bb, select_aabb)]
        if not candidates:
            logger.warning("No candidate chunks for tile %s", tile_id)
            # Écriture d’un mesh vide pour garder un pipeline aval “prévisible”.
            empty = meshio.Mesh(
                points=np.zeros((0, 3), dtype=np.float64),
                cells=[("triangle", np.zeros((0, 3), dtype=np.int64))]
            )
            _write_ply_with_plyfile(out_ply, out_mesh, binary=cfg.binary_out)
            continue

        logger.info("Tile %s: %d candidate chunks", tile_id, len(candidates))

        # Lecture de tous les chunks candidats + concaténation
        meshes = []
        for p in candidates:
            meshes.append(_meshio_read_ply(p))

        big = _concat_meshes(meshes)
        if big.points.shape[0] == 0:
            logger.warning("Tile %s: empty concatenated mesh.", tile_id)
            _write_ply_with_plyfile(out_ply, out_mesh, binary=cfg.binary_out)
            continue

        # Récupération du bloc triangles dans le mesh concaténé
        tri = None
        for cb in big.cells:
            if cb.type == "triangle":
                tri = cb.data
                break
        if tri is None or tri.size == 0:
            logger.warning("Tile %s: no triangles.", tile_id)
            _write_ply_with_plyfile(out_ply, out_mesh, binary=cfg.binary_out)
            continue

        # Filtrage conservative des triangles par rapport à la bbox de la dalle km
        keep = _tri_keep_mask(big.points, tri, test_aabb)
        kept_count = int(np.count_nonzero(keep))
        logger.info("Tile %s: keep %d / %d triangles", tile_id, kept_count, tri.shape[0])

        # Extraction du sous-mesh (triangles conservés) + écriture PLY
        out_mesh = _subset_mesh_faces(big, keep)
        _write_ply_with_plyfile(out_ply, out_mesh, binary=cfg.binary_out)

    logger.info("Done. Output dir: %s", out_dir)
    return out_dir