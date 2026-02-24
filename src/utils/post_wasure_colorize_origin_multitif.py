# -*- coding: utf-8 -*-
"""
Étape 9 - Colorisation des tuiles PLY shiftées (L93) par origine (LiDAR vs MNS) à partir de multiples GeoTIFF de masque.

Objectif
--------
Coloriser chaque sommet d'une tuile PLY selon l'origine des données :
- zones inchangées (masque == 0) : LiDAR,
- zones changées (masque == 1) : MNS (corrélation recalée),
- sommets hors couverture masque : couleur par défaut.

Robustesse multi-masques
------------------------
Une tuile PLY peut intersecter plusieurs tuiles de masque GeoTIFF. La stratégie est :
- initialiser tous les sommets à une couleur par défaut,
- pour chaque masque intersectant : lire seulement une fenêtre (window) correspondant à la bbox de la tuile PLY,
  puis mettre à jour uniquement les sommets couverts,
- en cas de recouvrement multiple d'un sommet par plusieurs masques, résoudre le conflit selon `resolve_conflict` :
  - "first" : premier masque gagnant,
  - "last" : dernier masque gagnant,
  - "prefer_change" : on préfère "changé" (1) à "inchangé" (0).

Notes
-----
- Les champs ajoutés/écrasés sur les sommets sont : red, green, blue (uint8).
- Le mode d'échantillonnage est "nearest" (arrondi des indices raster).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from plyfile import PlyData, PlyElement


@dataclass(frozen=True)
class OriginColorConfig:
    """
    Configuration de la colorisation par origine à partir de masques GeoTIFF.

    Attributs
    ---------
    swap_meaning : bool
        Si True, inverse l'interprétation des valeurs de masque (0/1).
        Utile si, selon la convention d'entrée, 1 représente "LiDAR" et 0 représente "MNS".
    lidar_rgb : tuple[int,int,int]
        Couleur RGB (uint8) attribuée aux sommets classés "LiDAR".
    mns_rgb : tuple[int,int,int]
        Couleur RGB (uint8) attribuée aux sommets classés "MNS".
    default_rgb : tuple[int,int,int]
        Couleur par défaut pour les sommets hors couverture de masque.
    overwrite : bool
        Si False, ne réécrit pas les fichiers PLY de sortie déjà existants.
    sampling : Literal["nearest"]
        Mode d'échantillonnage (ici uniquement "nearest").
    bbox_buffer_m : float
        Marge (mètres) ajoutée à la bbox XY de la tuile PLY pour l'intersection avec les masques.
    resolve_conflict : {"first","last","prefer_change"}
        Politique de résolution quand plusieurs masques couvrent le même sommet :
        - "first" : ne met à jour que les sommets non encore assignés,
        - "last" : écrase systématiquement l'assignation précédente,
        - "prefer_change" : écrase seulement si la nouvelle valeur est 1 et l'ancienne est 0
          (ou si non assigné).
    """
    swap_meaning: bool = False
    lidar_rgb: tuple[int, int, int] = (0, 140, 255)
    mns_rgb: tuple[int, int, int] = (255, 120, 0)
    default_rgb: tuple[int, int, int] = (200, 200, 200)
    overwrite: bool = True
    sampling: Literal["nearest"] = "nearest"
    bbox_buffer_m: float = 2.0
    # Résolution de conflit si plusieurs masques couvrent un même sommet :
    # "prefer_change" => masque==1 gagne sur masque==0
    resolve_conflict: Literal["first", "last", "prefer_change"] = "prefer_change"


def _ensure_vertex_rgb_fields(vertex_arr: np.ndarray) -> np.ndarray:
    """
    Garantit que le tableau structuré des sommets contient les champs couleur uint8 : red/green/blue.

    - Si ces champs existent déjà, on renvoie le tableau tel quel (ils seront écrasés ensuite).
    - Sinon, on crée un nouveau dtype en ajoutant ces champs et on copie les données.

    Parameters
    ----------
    vertex_arr : np.ndarray
        Tableau structuré issu de l'élément "vertex" d'un PLY.

    Returns
    -------
    np.ndarray
        Tableau structuré avec champs red/green/blue présents.
    """
    names = vertex_arr.dtype.names or ()
    need = [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if all(n in names for n, _ in need):
        return vertex_arr

    new_descr = list(vertex_arr.dtype.descr)
    for n, dt in need:
        if n not in names:
            new_descr.append((n, dt))

    out = np.zeros(vertex_arr.shape[0], dtype=new_descr)
    for n in vertex_arr.dtype.names:
        out[n] = vertex_arr[n]
    return out


def _compute_bbox_xy(x: np.ndarray, y: np.ndarray, buffer_m: float) -> tuple[float, float, float, float]:
    """
    Calcule la bbox XY (xmin, ymin, xmax, ymax) d'une tuile PLY à partir des sommets.

    Parameters
    ----------
    x, y : np.ndarray
        Coordonnées des sommets.
    buffer_m : float
        Marge ajoutée (mètres).

    Returns
    -------
    tuple[float,float,float,float]
        (xmin, ymin, xmax, ymax)
    """
    xmin = float(np.nanmin(x)) - buffer_m
    xmax = float(np.nanmax(x)) + buffer_m
    ymin = float(np.nanmin(y)) - buffer_m
    ymax = float(np.nanmax(y)) + buffer_m
    return xmin, ymin, xmax, ymax


def _bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """
    Test d'intersection entre deux bbox axis-aligned en XY.

    Parameters
    ----------
    a, b : tuple
        (xmin, ymin, xmax, ymax)

    Returns
    -------
    bool
        True si les bbox se recouvrent (intersection non vide), False sinon.
    """
    axmin, aymin, axmax, aymax = a
    bxmin, bymin, bxmax, bymax = b
    return not (axmax < bxmin or axmin > bxmax or aymax < bymin or aymin > bymax)


def run_post_wasure_colorize_origin_multitif(
    *,
    ply_l93_dir: str | Path,
    mask_dir: str | Path,
    logger: logging.Logger,
    cfg: OriginColorConfig,
) -> Path:
    """
    Étape 9 : colorise les tuiles PLY (L93) selon l'origine (LiDAR vs MNS) à partir de plusieurs masques GeoTIFF.

    Entrées
    -------
    ply_l93_dir :
        Répertoire contenant les tuiles PLY shiftées en EPSG:2154 (typiquement `run_*/ply_L93`).
    mask_dir :
        Répertoire contenant les masques GeoTIFF (une ou plusieurs tuiles).
        Chaque masque est supposé coder les zones changées (1) vs inchangées (0).

    Sortie
    ------
    Répertoire `run_*/ply_L93_origin` contenant des PLY avec champs red/green/blue.

    Stratégie
    ---------
    - Construire un index léger des bbox des masques (sans charger les rasters en mémoire).
    - Pour chaque tuile PLY :
        * trouver les masques intersectants via les bbox,
        * initialiser tous les sommets à la couleur par défaut,
        * lire pour chaque masque une fenêtre correspondant à la bbox PLY,
        * affecter/mettre à jour la valeur masque (0/1) par sommet avec une règle de conflit,
        * convertir la valeur 0/1 en couleur (LiDAR/MNS), puis écrire le PLY.

    Returns
    -------
    Path
        Répertoire de sortie contenant les PLY colorisés par origine.
    """
    ply_l93_dir = Path(ply_l93_dir)
    mask_dir = Path(mask_dir)

    if not ply_l93_dir.is_dir():
        raise RuntimeError(f"Input directory not found: {ply_l93_dir}")
    if not mask_dir.is_dir():
        raise RuntimeError(f"Mask directory not found: {mask_dir}")

    # On suppose ply_l93_dir = <run_dir>/ply_L93 => run_dir = parent
    run_dir = ply_l93_dir.parent

    # Sortie : PLY colorisés par origine
    out_dir = run_dir / "ply_L93_origin"
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(ply_l93_dir.glob("*.ply"))
    if not ply_files:
        raise RuntimeError(f"No PLY found in: {ply_l93_dir}")

    mask_files = sorted(mask_dir.glob("*.tif"))
    if not mask_files:
        raise RuntimeError(f"No mask .tif found in: {mask_dir}")

    # Index léger : on stocke (chemin, bbox) pour éviter de garder des rasters ouverts ou en RAM
    mask_index: list[tuple[Path, tuple[float, float, float, float]]] = []
    for p in mask_files:
        with rasterio.open(p) as ds:
            mask_index.append((p, (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)))

    logger.info(
        "Step 9 - Origin colorization (multi-mask) | tiles=%d | masks=%d | resolve=%s",
        len(ply_files), len(mask_index), cfg.resolve_conflict
    )

    # Couleurs en uint8 (vecteurs 3)
    lidar_rgb = np.array(cfg.lidar_rgb, dtype=np.uint8)
    mns_rgb = np.array(cfg.mns_rgb, dtype=np.uint8)
    default_rgb = np.array(cfg.default_rgb, dtype=np.uint8)

    # Statistiques globales (sur toutes les tuiles)
    n_written = 0
    n_skipped = 0
    n_total_vertices = 0
    n_covered_vertices = 0
    n_outside_vertices = 0
    n_lidar = 0
    n_mns = 0
    n_tiles_multi_masks = 0

    for i, in_ply in enumerate(ply_files, start=1):
        out_ply = out_dir / in_ply.name
        if out_ply.exists() and not cfg.overwrite:
            n_skipped += 1
            continue

        # Lecture PLY
        ply = PlyData.read(str(in_ply))
        if "vertex" not in ply:
            logger.warning("No vertex element, skipping: %s", in_ply)
            continue

        v = np.array(ply["vertex"].data)
        if ("x" not in v.dtype.names) or ("y" not in v.dtype.names):
            raise RuntimeError(f"Missing x/y fields in vertex data: {in_ply}")

        # Coordonnées sommets
        x = v["x"].astype(np.float64)
        y = v["y"].astype(np.float64)

        # Bbox XY de la tuile PLY (avec buffer)
        bbox = _compute_bbox_xy(x, y, cfg.bbox_buffer_m)

        # Recherche de tous les masques dont la bbox intersecte la bbox de la tuile PLY
        intersecting = [p for (p, b) in mask_index if _bbox_intersects(bbox, b)]
        if len(intersecting) > 1:
            n_tiles_multi_masks += 1

        # Initialiser la couleur par défaut pour tous les sommets
        colors = np.empty((v.shape[0], 3), dtype=np.uint8)
        colors[:] = default_rgb

        # Suivi des sommets couverts par au moins un masque
        covered = np.zeros(v.shape[0], dtype=bool)

        # Valeur masque assignée par sommet (0/1). 255 = non assigné (hors couverture / jamais touché)
        assigned_val = np.full(v.shape[0], fill_value=255, dtype=np.uint8)

        # Pour chaque masque intersectant, lire uniquement une fenêtre et mettre à jour les sommets couverts
        for mask_path in intersecting:
            with rasterio.open(mask_path) as ds:
                xmin, ymin, xmax, ymax = bbox

                # Fenêtre raster correspondant à la bbox PLY
                win = from_bounds(xmin, ymin, xmax, ymax, transform=ds.transform)
                win = win.round_offsets().round_lengths()

                # Lecture de la fenêtre seulement.
                # boundless=True + fill_value=0 : ce qui sort du raster est rempli en 0
                arr = ds.read(1, window=win, boundless=True, fill_value=0)

                # Normalisation en 0/1
                nodata = ds.nodata
                if nodata is not None:
                    arr = np.where(arr == nodata, 0, arr)
                if np.issubdtype(arr.dtype, np.floating):
                    arr = np.where(np.isfinite(arr), arr, 0)
                m01 = (arr > 0).astype(np.uint8)

                # Transform de fenêtre : permet de projeter (x,y) vers (row,col) dans le tableau `m01`
                w_transform = ds.window_transform(win)
                inv = ~w_transform
                colf = inv.a * x + inv.b * y + inv.c
                rowf = inv.d * x + inv.e * y + inv.f

                # Échantillonnage "nearest" : arrondi des indices pixel
                row = np.rint(rowf).astype(np.int64)
                col = np.rint(colf).astype(np.int64)

                H, W = m01.shape
                valid = (row >= 0) & (row < H) & (col >= 0) & (col < W)
                if not np.any(valid):
                    continue

                # Valeurs masque 0/1 pour les sommets couverts par ce masque
                mvals = m01[row[valid], col[valid]]
                idx = np.where(valid)[0]

                # Résolution de conflit : quels sommets doit-on mettre à jour ?
                if cfg.resolve_conflict == "first":
                    # On ne modifie que les sommets non encore assignés
                    upd = (assigned_val[idx] == 255)
                elif cfg.resolve_conflict == "last":
                    # On écrase systématiquement
                    upd = np.ones(idx.size, dtype=bool)
                else:  # prefer_change
                    # Mettre à jour si non assigné OU si le nouveau vaut 1 et l'ancien vaut 0
                    prev = assigned_val[idx]
                    upd = (prev == 255) | ((mvals == 1) & (prev == 0))

                idx_upd = idx[upd]
                mvals_upd = mvals[upd]

                # Écriture de la valeur assignée et marquage "covered"
                assigned_val[idx_upd] = mvals_upd
                covered[idx_upd] = True

        # Conversion des valeurs assignées (0/1) en couleurs
        # 255 => par défaut (hors couverture)
        is_assigned = (assigned_val != 255)
        if np.any(is_assigned):
            vals = assigned_val[is_assigned]
            idxa = np.where(is_assigned)[0]

            # Convention : par défaut 0=LiDAR, 1=MNS, sauf inversion explicite
            if cfg.swap_meaning:
                is_lidar = (vals == 1)
                is_mns = (vals == 0)
            else:
                is_lidar = (vals == 0)
                is_mns = (vals == 1)

            idx_lidar = idxa[is_lidar]
            idx_mns = idxa[is_mns]

            if idx_lidar.size:
                colors[idx_lidar] = lidar_rgb
            if idx_mns.size:
                colors[idx_mns] = mns_rgb

            # Stats globales (cumulées)
            n_lidar += int(idx_lidar.size)
            n_mns += int(idx_mns.size)

        # Stats globales de couverture
        n_total_vertices += int(v.shape[0])
        n_covered_vertices += int(covered.sum())
        n_outside_vertices += int((~covered).sum())

        # Écriture PLY avec champs RGB
        v2 = _ensure_vertex_rgb_fields(v)
        v2["red"] = colors[:, 0]
        v2["green"] = colors[:, 1]
        v2["blue"] = colors[:, 2]

        # Remplacer l'élément vertex, conserver les autres (faces, etc.)
        ply.elements = tuple([PlyElement.describe(v2, "vertex")] + [e for e in ply.elements if e.name != "vertex"])
        ply.write(str(out_ply))
        n_written += 1

        # Progress léger dans le log principal
        if (i % 50) == 0:
            logger.info("Step 9 progress: %d/%d", i, len(ply_files))

    logger.info(
        "Step 9 done | written=%d | skipped=%d | vertices=%d | covered=%d | outside=%d | lidar=%d | mns=%d | tiles_multi_masks=%d | out=%s",
        n_written, n_skipped, n_total_vertices, n_covered_vertices, n_outside_vertices, n_lidar, n_mns, n_tiles_multi_masks, out_dir
    )
    return out_dir