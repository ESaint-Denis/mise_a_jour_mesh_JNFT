# -*- coding: utf-8 -*-
"""
Étape 8 - Colorisation des tuiles PLY WaSuRe (déjà shiftées en L93) avec les orthophotos IGN via WMS-R GetMap.

Principe
--------
Pour chaque tuile PLY (coordonnées L93 / EPSG:2154) :
- on calcule sa bbox XY (avec un buffer optionnel),
- on requête une image WMS GetMap correspondant à cette bbox, à une résolution sol (GSD) cible,
- on échantillonne la couleur RGB au niveau de chaque sommet (nearest ou bilinear),
- on écrit un nouveau PLY avec des champs couleur par sommet (red/green/blue en uint8),
  en conservant les autres éléments (faces, etc.) inchangés.

Choix de conception
------------------
- Le mode par défaut est "nearest" : rapide, et cohérent avec les tests initiaux.
- Un cache disque évite de re-télécharger les mêmes images WMS lors de relances.

Notes
-----
- Les champs ajoutés/écrasés sur les sommets sont : red, green, blue (uint8).
- Les images WMS sont lues en mémoire (rasterio MemoryFile) pour éviter de gérer un format géoréférencé sur disque.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import requests
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from plyfile import PlyData, PlyElement

import warnings
from rasterio.errors import NotGeoreferencedWarning

# On ignore cet avertissement : les images WMS (JPEG/PNG) ne sont pas forcément géoréférencées
# au sens strict de rasterio, mais on reconstruit nous-même un transform à partir de la bbox demandée.
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# ----------------------------- Configuration -----------------------------


@dataclass(frozen=True)
class OrthoWmsConfig:
    """
    Paramètres de colorisation par orthophoto via WMS.

    Attributs
    ---------
    wms_url : str
        URL du service WMS-R (GéoPlateforme IGN).
    layer : str
        Nom de couche WMS (par défaut : orthophotos).
    crs : str
        CRS de la requête WMS. Après l'étape 7, les PLY sont en EPSG:2154.
    img_format : str
        Format de sortie image demandé au WMS (ex. image/jpeg).
    gsd_m : float
        Résolution sol cible (Ground Sampling Distance) en mètres.
    bbox_buffer_m : float
        Marge ajoutée à la bbox XY (en mètres) pour éviter des artefacts de bord.
    sampling : {"nearest","bilinear"}
        Mode d'échantillonnage RGB au niveau des sommets.
    default_rgb : tuple[int,int,int]
        Couleur par défaut utilisée si un sommet tombe hors image (ou hors zone valide).
    overwrite : bool
        Si False, ne réécrit pas les PLY de sortie existants.
    cache_dirname : str
        Nom du dossier de cache (créé dans le répertoire du run WaSuRe).
    sleep_s : float
        Pause entre requêtes pour limiter la charge côté service.
    timeout_s : float
        Timeout réseau (requests).
    retries : int
        Nombre de tentatives en cas d'erreurs transitoires.
    """
    # URL WMS-R (GeoPlateforme IGN)
    wms_url: str = "https://data.geopf.fr/wms-r/wms"
    # Couche : BD ORTHO / orthophotos
    layer: str = "HR.ORTHOIMAGERY.ORTHOPHOTOS"
    # CRS de la requête (les tuiles mesh sont en EPSG:2154 après step 7)
    crs: str = "EPSG:2154"
    # Format image de sortie
    img_format: str = "image/jpeg"
    # GSD cible en mètres (0.20 m par défaut)
    gsd_m: float = 0.20
    # Marge autour de la bbox (mètres) pour éviter les artefacts en bordure
    bbox_buffer_m: float = 2.0
    # Mode d'échantillonnage
    sampling: Literal["nearest", "bilinear"] = "nearest"
    # Couleur par défaut si hors image
    default_rgb: tuple[int, int, int] = (200, 200, 200)
    # Réécrire les PLY de sortie s'ils existent déjà
    overwrite: bool = True
    # Nom du dossier cache (créé dans le dossier du run)
    cache_dirname: str = "ortho_cache_wms"
    # Limitation de débit : pause entre requêtes
    sleep_s: float = 0.05
    # Timeout réseau
    timeout_s: float = 60.0
    # Nombre de retries en cas d'erreurs transitoires
    retries: int = 3


# ----------------------------- Petites fonctions utilitaires -----------------------------


def _compute_bbox_xy(x: np.ndarray, y: np.ndarray, buffer_m: float) -> tuple[float, float, float, float]:
    """
    Calcule la bbox XY (xmin, ymin, xmax, ymax) à partir des coordonnées des sommets.

    Parameters
    ----------
    x, y : np.ndarray
        Coordonnées des sommets.
    buffer_m : float
        Marge ajoutée (en m).

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


def _wms_image_size_for_bbox(bbox: tuple[float, float, float, float], gsd_m: float) -> tuple[int, int]:
    """
    Convertit une bbox métrique en dimensions pixel (width, height) à une GSD donnée.

    On impose au minimum 2x2 pixels pour éviter des comportements indésirables
    côté serveur ou côté lecture.

    Parameters
    ----------
    bbox : tuple
        (xmin, ymin, xmax, ymax)
    gsd_m : float
        Résolution sol (m/pixel).

    Returns
    -------
    tuple[int,int]
        (width_px, height_px)
    """
    xmin, ymin, xmax, ymax = bbox
    w_m = max(0.0, xmax - xmin)
    h_m = max(0.0, ymax - ymin)
    width = max(2, int(math.ceil(w_m / gsd_m)))
    height = max(2, int(math.ceil(h_m / gsd_m)))
    return width, height


def _cache_key(cfg: OrthoWmsConfig, bbox: tuple[float, float, float, float], width: int, height: int) -> str:
    """
    Produit une clé de cache stable à partir des paramètres impactant les valeurs pixels.

    On inclut notamment : URL, layer, CRS, format, bbox, width/height.

    Returns
    -------
    str
        Hash court (24 caractères) utilisable comme nom de fichier.
    """
    s = f"{cfg.wms_url}|{cfg.layer}|{cfg.crs}|{cfg.img_format}|{bbox}|{width}|{height}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]


def _download_wms_getmap(
    *,
    cfg: OrthoWmsConfig,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    out_path: Path,
    logger: logging.Logger,
) -> Path:
    """
    Télécharge une image WMS GetMap et l'écrit dans `out_path` (avec retries).

    Parameters
    ----------
    cfg : OrthoWmsConfig
        Configuration WMS.
    bbox : tuple
        (xmin, ymin, xmax, ymax) dans le CRS de la requête.
    width, height : int
        Dimensions pixel demandées.
    out_path : Path
        Chemin de sortie (fichier image).
    logger : logging.Logger
        Logger (non utilisé ici mais conservé pour cohérence d'interface).

    Returns
    -------
    Path
        Chemin vers l'image téléchargée.

    Raises
    ------
    RuntimeError
        Si la requête échoue après `cfg.retries` tentatives.
    """
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": cfg.layer,
        "STYLES": "",
        "CRS": cfg.crs,
        # Pour EPSG:2154, l'ordre d'axes attendu ici est x,y => xmin,ymin,xmax,ymax
        "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "FORMAT": cfg.img_format,
        "TRANSPARENT": "FALSE",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for k in range(cfg.retries):
        try:
            r = requests.get(cfg.wms_url, params=params, timeout=cfg.timeout_s)
            r.raise_for_status()
            out_path.write_bytes(r.content)
            return out_path
        except Exception as e:
            last_err = e
            # Petit backoff (progressif)
            time.sleep(0.5 * (k + 1))

    raise RuntimeError(f"WMS GetMap failed after {cfg.retries} retries: {last_err}") from last_err


def _read_rgb_from_image_bytes(img_path: Path) -> np.ndarray:
    """
    Lit une image RGB (JPEG/PNG) et renvoie un tableau HxWx3 uint8 via rasterio.

    Notes
    -----
    - Beaucoup de JPEG sont en 3 bandes (RGB).
    - Certains flux peuvent avoir 4 bandes (RGBA) : on lit uniquement les 3 premières.

    Parameters
    ----------
    img_path : Path
        Chemin vers l'image téléchargée.

    Returns
    -------
    np.ndarray
        Tableau (H, W, 3) en uint8.
    """
    data = img_path.read_bytes()
    with MemoryFile(data) as mem:
        with mem.open() as ds:
            count = ds.count
            if count < 3:
                raise RuntimeError(f"Expected at least 3 bands in WMS image, got {count} for {img_path}")
            r = ds.read(1)
            g = ds.read(2)
            b = ds.read(3)

            # Conversion robuste en uint8
            if r.dtype != np.uint8:
                r = np.clip(r, 0, 255).astype(np.uint8)
                g = np.clip(g, 0, 255).astype(np.uint8)
                b = np.clip(b, 0, 255).astype(np.uint8)

            rgb = np.dstack([r, g, b])
            return rgb


def _xy_to_rowcol(transform: rasterio.Affine, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Conversion (x,y) -> (row,col) en utilisant l'inverse de l'affine raster.

    Ici, on arrondit au pixel le plus proche (rint) pour un échantillonnage "nearest".

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (rows, cols) en int64.
    """
    inv = ~transform
    colf = inv.a * x + inv.b * y + inv.c
    rowf = inv.d * x + inv.e * y + inv.f
    row = np.rint(rowf).astype(np.int64)
    col = np.rint(colf).astype(np.int64)
    return row, col


def _sample_rgb_nearest(rgb: np.ndarray, rows: np.ndarray, cols: np.ndarray, default_rgb: tuple[int, int, int]) -> np.ndarray:
    """
    Échantillonne RGB au plus proche voisin.

    Parameters
    ----------
    rgb : np.ndarray
        Image (H,W,3) uint8.
    rows, cols : np.ndarray
        Indices pixel.
    default_rgb : tuple
        Couleur par défaut si hors image.

    Returns
    -------
    np.ndarray
        Tableau (N,3) uint8.
    """
    H, W, _ = rgb.shape
    out = np.empty((rows.size, 3), dtype=np.uint8)

    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    out[~valid] = np.array(default_rgb, dtype=np.uint8)

    rv = rows[valid]
    cv = cols[valid]
    out[valid] = rgb[rv, cv, :]
    return out


def _sample_rgb_bilinear(rgb: np.ndarray, transform: rasterio.Affine, x: np.ndarray, y: np.ndarray, default_rgb: tuple[int, int, int]) -> np.ndarray:
    """
    Échantillonnage bilinéaire en espace raster.

    - On calcule les coordonnées flottantes (rowf, colf),
    - on interpole entre les 4 pixels voisins,
    - si un point sort des bornes, on laisse la couleur par défaut.

    Returns
    -------
    np.ndarray
        Tableau (N,3) uint8.
    """
    # Bilinear in raster space
    inv = ~transform
    colf = inv.a * x + inv.b * y + inv.c
    rowf = inv.d * x + inv.e * y + inv.f

    H, W, _ = rgb.shape
    out = np.empty((rowf.size, 3), dtype=np.uint8)
    out[:] = np.array(default_rgb, dtype=np.uint8)

    r0 = np.floor(rowf).astype(np.int64)
    c0 = np.floor(colf).astype(np.int64)
    r1 = r0 + 1
    c1 = c0 + 1

    wr = (rowf - r0).astype(np.float64)
    wc = (colf - c0).astype(np.float64)

    valid = (r0 >= 0) & (r1 < H) & (c0 >= 0) & (c1 < W)
    if not np.any(valid):
        return out

    vr0 = r0[valid]; vr1 = r1[valid]
    vc0 = c0[valid]; vc1 = c1[valid]
    vwr = wr[valid][:, None]
    vwc = wc[valid][:, None]

    q00 = rgb[vr0, vc0, :].astype(np.float64)
    q01 = rgb[vr0, vc1, :].astype(np.float64)
    q10 = rgb[vr1, vc0, :].astype(np.float64)
    q11 = rgb[vr1, vc1, :].astype(np.float64)

    top = (1.0 - vwc) * q00 + vwc * q01
    bot = (1.0 - vwc) * q10 + vwc * q11
    val = (1.0 - vwr) * top + vwr * bot
    out[valid] = np.clip(np.rint(val), 0, 255).astype(np.uint8)
    return out


def _ensure_vertex_rgb_fields(vertex_arr: np.ndarray) -> np.ndarray:
    """
    Garantit que le tableau structuré des sommets contient les champs uint8 : red/green/blue.

    - Si les champs existent déjà : on retourne le tableau tel quel (ils seront écrasés ensuite).
    - Sinon : on crée un nouveau dtype en ajoutant ces champs, et on copie les données.

    Parameters
    ----------
    vertex_arr : np.ndarray
        Tableau structuré (vertex) issu du PLY.

    Returns
    -------
    np.ndarray
        Tableau structuré avec champs couleur présents.
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


# ----------------------------- Fonction principale de l'étape -----------------------------


def run_post_wasure_colorize_ortho_wms(
    *,
    ply_l93_dir: str | Path,
    logger: logging.Logger,
    cfg: OrthoWmsConfig,
) -> Path:
    """
    Étape 8 : colorise chaque tuile PLY à partir des orthophotos IGN via WMS GetMap.

    Entrée
    ------
    ply_l93_dir :
        Répertoire contenant les tuiles PLY déjà shiftées en L93 (typiquement : `run_*/ply_L93`).

    Sortie
    ------
    Un répertoire `run_*/ply_L93_ortho` contenant les tuiles PLY colorisées.

    Détails
    -------
    - Pour chaque tuile : calcul bbox -> GetMap -> lecture RGB -> échantillonnage -> écriture PLY.
    - Un cache d'images WMS est créé dans `run_*/<cfg.cache_dirname>`.

    Parameters
    ----------
    ply_l93_dir : str | Path
        Répertoire d'entrée des PLY (L93).
    logger : logging.Logger
        Logger du pipeline.
    cfg : OrthoWmsConfig
        Configuration WMS et paramètres d'échantillonnage.

    Returns
    -------
    Path
        Répertoire de sortie contenant les PLY colorisés.
    """
    ply_l93_dir = Path(ply_l93_dir)
    if not ply_l93_dir.is_dir():
        raise RuntimeError(f"Input directory not found: {ply_l93_dir}")

    # On suppose que ply_l93_dir = <run_dir>/ply_L93 => run_dir = parent
    run_dir = ply_l93_dir.parent

    # Sortie : PLY colorisés
    out_dir = run_dir / "ply_L93_ortho"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cache des images GetMap
    cache_dir = run_dir / cfg.cache_dirname
    cache_dir.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(ply_l93_dir.glob("*.ply"))
    if not ply_files:
        raise RuntimeError(f"No PLY found in: {ply_l93_dir}")

    logger.info(
        "Step 8 - Ortho colorization (WMS) | tiles=%d | layer=%s | gsd=%.2fm | fmt=%s",
        len(ply_files), cfg.layer, cfg.gsd_m, cfg.img_format
    )

    n_written = 0
    n_skipped = 0
    n_cached = 0

    for i, in_ply in enumerate(ply_files, start=1):
        out_ply = out_dir / in_ply.name
        if out_ply.exists() and not cfg.overwrite:
            n_skipped += 1
            continue

        # Lecture du PLY
        ply = PlyData.read(str(in_ply))
        if "vertex" not in ply:
            logger.warning("No vertex element, skipping: %s", in_ply)
            continue

        v = np.array(ply["vertex"].data)

        # Vérification des champs XY nécessaires au calcul bbox + sampling
        if "x" not in v.dtype.names or "y" not in v.dtype.names:
            raise RuntimeError(f"Missing x/y fields in vertex data: {in_ply}")

        x = v["x"].astype(np.float64)
        y = v["y"].astype(np.float64)

        # Bbox de tuile + dimensions image à la GSD cible
        bbox = _compute_bbox_xy(x, y, cfg.bbox_buffer_m)
        width, height = _wms_image_size_for_bbox(bbox, cfg.gsd_m)

        # Cache key déterministe
        key = _cache_key(cfg, bbox, width, height)
        img_path = cache_dir / f"{key}.jpg"

        # Récupération image WMS (cache hit si déjà présent)
        if img_path.is_file():
            n_cached += 1
        else:
            _download_wms_getmap(cfg=cfg, bbox=bbox, width=width, height=height, out_path=img_path, logger=logger)
            if cfg.sleep_s > 0:
                time.sleep(cfg.sleep_s)

        # Lecture image RGB
        rgb = _read_rgb_from_image_bytes(img_path)

        # Construction d'un transform affine cohérent avec la bbox demandée et la taille image renvoyée
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width=rgb.shape[1], height=rgb.shape[0])

        # Échantillonnage des couleurs
        if cfg.sampling == "nearest":
            rows, cols = _xy_to_rowcol(transform, x, y)
            sampled = _sample_rgb_nearest(rgb, rows, cols, cfg.default_rgb)
        else:
            sampled = _sample_rgb_bilinear(rgb, transform, x, y, cfg.default_rgb)

        # Ajout/garantie des champs couleurs et écriture dans v2
        v2 = _ensure_vertex_rgb_fields(v)
        v2["red"] = sampled[:, 0]
        v2["green"] = sampled[:, 1]
        v2["blue"] = sampled[:, 2]

        # Remplacement de l'élément vertex, conservation des autres éléments (faces, etc.)
        new_elems = [PlyElement.describe(v2, "vertex")] + [e for e in ply.elements if e.name != "vertex"]
        ply.elements = tuple(new_elems)

        # Écriture du PLY colorisé
        ply.write(str(out_ply))
        n_written += 1

        # On garde le log principal léger : message de progression toutes les 50 tuiles
        if (i % 50) == 0:
            logger.info("Step 8 progress: %d/%d", i, len(ply_files))

    logger.info(
        "Step 8 done | written=%d | skipped=%d | cache_hits=%d | out=%s",
        n_written, n_skipped, n_cached, out_dir
    )
    return out_dir