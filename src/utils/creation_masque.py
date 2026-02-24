# -*- coding: utf-8 -*-
"""
Création du masque de changement à partir d'un différentiel de MNS.

Ce module produit un masque binaire de changement (par dalle) en comparant :
- un MNS de corrélation recalé (newer),
- un MNS LiDAR (older), reprojeté/ré-échantillonné localement sur la grille du MNS corrélation.

Le masque est calculé en plusieurs étapes :
1) Détection brute par comparaison d'intervalles (min/max locaux) entre les deux MNS.
2) Nettoyage morphologique (ouverture : érosion puis dilatation).
3) Filtrage des petites composantes connexes (seuil en m²).
4) Buffer métrique (dilatation elliptique approximant un buffer en mètres).

Codage du masque (uint8) :
- 0   : pas de changement
- 1   : changement
- 255 : nodata (valeur configurable)

Auteur : ESaint-Denis
"""

# utils/creation_masque.py
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage import label as cc_label

from utils.creation_arborescence import ProjectPaths
from utils.recuperation_donnees import setup_logger


class MissingMatchingTileError(RuntimeError):
    """Erreur levée si aucune dalle MNS LiDAR ne correspond à une dalle de corrélation recalée."""


# --- regex: MNS corrélation contient ..._0605000_6933000.tif ---
_TILE_RE_M = re.compile(r"_(\d{7})_(\d{7})")


@dataclass(frozen=True)
class MaskConfig:
    """
    Paramètres de génération du masque de changement.

    Étapes
    ------
    1) Masque par comparaison d'intervalles (min/max locaux)
       - z_tolerance_m : tolérance verticale (m) appliquée aux tests "abaissé" / "relevé"
       - window_radius : rayon (pixels) de la fenêtre locale (R=2 -> 5x5)
       - block_size    : taille de bloc (pixels) pour traiter par tuiles mémoire
       - resampling    : méthode de ré-échantillonnage pour reprojecter le MNS LiDAR

    2) Ouverture morphologique
       - radius_open   : rayon (pixels) de l'élément structurant carré (R=4 -> 9x9)

    3) Filtrage des petites composantes
       - min_area_m2   : aire minimale (m²) des composantes conservées
       - connectivity  : connectivité des composantes (4 ou 8)

    4) Buffer métrique
       - buffer_m      : distance de buffer (m) appliquée au masque binaire
       - buffer_closing: si True, fait une fermeture (dilatation puis érosion),
                         sinon dilatation seule (comportement du script actuel)

    I/O
    ---
    - mask_nodata : valeur nodata du masque (uint8)
    - compress    : compression GeoTIFF
    - overwrite   : réécrit la sortie si déjà présente
    """
    # Étape 1: masque intervalle
    z_tolerance_m: float | tuple[float, float] = 1.0
    window_radius: int = 2          # 2 -> 5x5
    block_size: int = 1024
    resampling: Resampling = Resampling.bilinear

    # Étape 2: morpho open
    radius_open: int = 4            # 4 -> 9x9

    # Étape 3: filtrage petites composantes
    min_area_m2: float = 16.0
    connectivity: int = 8

    # Étape 4: buffer
    buffer_m: float = 2.0
    buffer_closing: bool = False    # False = dilatation seule

    # I/O
    mask_nodata: int = 255
    compress: str = "LZW"
    overwrite: bool = True


def _normalize_tolerance(z_tolerance_m: float | tuple[float, float]) -> tuple[float, float]:
    """
    Normalise la tolérance verticale en un couple (tol_abaissement, tol_rehaussement).

    Parameters
    ----------
    z_tolerance_m : float | tuple[float, float]
        - si float : tolérance symétrique (t, t)
        - si tuple/list (t_low, t_high) : tolérance asymétrique

    Returns
    -------
    tuple[float, float]
        (tol_low, tol_high) en mètres.
    """
    if isinstance(z_tolerance_m, (tuple, list)) and len(z_tolerance_m) == 2:
        return float(z_tolerance_m[0]), float(z_tolerance_m[1])
    t = float(z_tolerance_m or 0.0)
    return t, t


def _parse_tile_xy_km_from_corr_name(name: str) -> tuple[int, int]:
    """
    Extrait l'identifiant de dalle (x_km, y_km) depuis un nom de fichier de corrélation.

    Le nom contient typiquement des coordonnées en mètres :
    - ..._0605000_6933000...

    Les valeurs extraites sont divisées par 1000 pour obtenir les indices km.

    Parameters
    ----------
    name : str
        Nom du fichier MNS corrélation (recalé).

    Returns
    -------
    tuple[int, int]
        (x_km, y_km)

    Raises
    ------
    ValueError
        Si le motif n'est pas détecté ou si les coordonnées ne sont pas multiples de 1000.
    """
    m = _TILE_RE_M.search(name)
    if not m:
        raise ValueError(f"Impossible d'extraire l'identifiant dalle depuis: {name}")
    x_m = int(m.group(1))
    y_m = int(m.group(2))
    if (x_m % 1000) != 0 or (y_m % 1000) != 0:
        raise ValueError(f"Coordonnées mètres non multiples de 1000 dans: {name}")
    return x_m // 1000, y_m // 1000


def _find_mns_lidar_file_for_tile(mns_lidar_dir: Path, x_km: int, y_km: int) -> Path:
    """
    Retrouve la dalle MNS LiDAR correspondant à une dalle km.

    Convention recherchée dans le nom :
    - présence de `_{XXXX}_{YYYY}_`
    - et de `_MNS_` (pour limiter les faux positifs)

    Si plusieurs versions sont trouvées, la plus récente (mtime) est sélectionnée.

    Parameters
    ----------
    mns_lidar_dir : Path
        Répertoire des MNS LiDAR.
    x_km, y_km : int
        Identifiant km de la dalle.

    Returns
    -------
    Path
        Chemin du GeoTIFF MNS LiDAR retenu.

    Raises
    ------
    MissingMatchingTileError
        Si aucune dalle correspondante n'est trouvée.
    """
    pattern = f"_{x_km:04d}_{y_km:04d}_"
    candidates = [p for p in mns_lidar_dir.glob("*.tif") if pattern in p.name and "_MNS_" in p.name]

    if not candidates:
        raise MissingMatchingTileError(
            f"Aucun MNS LiDAR trouvé pour la dalle {x_km:04d}_{y_km:04d} dans {mns_lidar_dir}"
        )

    # Si plusieurs versions, on prend la plus récente.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _read_as_float32_with_nan(ds: rasterio.io.DatasetReader, window: Window, nodata_override=None) -> np.ndarray:
    """
    Lit une fenêtre raster et remplace le nodata par NaN (tableau float32).

    Parameters
    ----------
    ds : rasterio.io.DatasetReader
        Dataset rasterio ouvert.
    window : rasterio.windows.Window
        Fenêtre à lire.
    nodata_override : optional
        Valeur nodata à utiliser à la place de ds.nodata (si fournie).

    Returns
    -------
    np.ndarray
        Tableau float32 avec nodata remplacé par NaN.
    """
    arr = ds.read(1, window=window, masked=False).astype(np.float32)
    nodata = nodata_override if nodata_override is not None else ds.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.float32(np.nan), arr)
    else:
        arr = np.where(np.isfinite(arr), arr, np.float32(np.nan))
    return arr


def _expand_window_with_halo(core: Window, ds: rasterio.io.DatasetReader, halo: int) -> tuple[Window, int, int]:
    """
    Étend une fenêtre "core" avec un halo, borné aux limites du raster.

    Objectif : permettre le calcul de min/max glissants sur le core, tout en disposant
    des pixels voisins nécessaires au voisinage (rayon = halo).

    Parameters
    ----------
    core : Window
        Fenêtre cœur à traiter.
    ds : rasterio.io.DatasetReader
        Dataset rasterio (pour les limites width/height).
    halo : int
        Rayon (pixels) de l'extension.

    Returns
    -------
    tuple[Window, int, int]
        (big_window, y_off, x_off) où (y_off, x_off) est l'origine de core dans big_window.
    """
    halo = max(0, int(halo))
    x0 = int(core.col_off); y0 = int(core.row_off)
    w  = int(core.width);   h  = int(core.height)

    x0b = max(0, x0 - halo); y0b = max(0, y0 - halo)
    x1b = min(ds.width,  x0 + w + halo)
    y1b = min(ds.height, y0 + h + halo)

    big = Window(col_off=x0b, row_off=y0b, width=x1b - x0b, height=y1b - y0b)
    return big, (y0 - y0b), (x0 - x0b)


def _rolling_nan_minmax(arr: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule un minimum et un maximum glissants, en ignorant les NaN, sur une fenêtre carrée.

    Fenêtre utilisée : (2R+1) x (2R+1), avec R = radius.

    Implementation :
    - construction d'une vue 4D via `as_strided` sur un tableau paddé,
    - réduction par `nanmin` / `nanmax`.

    Parameters
    ----------
    arr : np.ndarray
        Tableau 2D (float32) contenant éventuellement des NaN.
    radius : int
        Rayon du voisinage en pixels (R<=0 : renvoie arr, arr).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (min_glissant, max_glissant) float32.
    """
    if radius <= 0:
        a = arr.astype(np.float32, copy=False)
        return a, a

    H, W = arr.shape
    k = 2 * radius + 1

    pad = np.pad(arr, ((radius, radius), (radius, radius)), mode="constant", constant_values=np.nan)
    s0, s1 = pad.strides[:2]

    win_view = as_strided(
        pad,
        shape=(H, W, k, k),
        strides=(s0, s1, s0, s1),
        writeable=False
    )

    win_flat = win_view.reshape(H, W, k * k)
    mn = np.nanmin(win_flat, axis=2).astype(np.float32, copy=False)
    mx = np.nanmax(win_flat, axis=2).astype(np.float32, copy=False)

    all_nan = np.all(~np.isfinite(win_flat), axis=2)
    mn[all_nan] = np.nan
    mx[all_nan] = np.nan
    return mn, mx


def _pixel_area_from_affine(transform) -> float:
    """
    Calcule l'aire d'un pixel dans les unités du CRS (ex. m² en EPSG:2154).

    Formule générale (incluant rotation/skew) : |a*e - b*d|.

    Parameters
    ----------
    transform :
        Transform affine rasterio (src.transform).

    Returns
    -------
    float
        Aire d'un pixel.
    """
    a = float(getattr(transform, "a"))
    b = float(getattr(transform, "b"))
    d = float(getattr(transform, "d"))
    e = float(getattr(transform, "e"))
    return abs(a * e - b * d)


def _pixel_scales_from_affine(transform) -> tuple[float, float]:
    """
    Estime l'échelle pixel en X et Y (m/pixel en EPSG:2154), robuste à la rotation.

    sx = sqrt(a² + b²)
    sy = sqrt(d² + e²)

    Parameters
    ----------
    transform :
        Transform affine rasterio.

    Returns
    -------
    tuple[float, float]
        (sx, sy) en unités du CRS / pixel.
    """
    a = float(getattr(transform, "a"))
    b = float(getattr(transform, "b"))
    d = float(getattr(transform, "d"))
    e = float(getattr(transform, "e"))
    sx = (a*a + b*b) ** 0.5
    sy = (d*d + e*e) ** 0.5
    return sx, sy


def _elliptical_structure(rx_px: int, ry_px: int) -> np.ndarray:
    """
    Construit un élément structurant elliptique (approximation d'un cercle métrique).

    L'ellipse est définie par ses rayons en pixels (rx_px, ry_px).

    Parameters
    ----------
    rx_px, ry_px : int
        Rayons en pixels selon X et Y.

    Returns
    -------
    np.ndarray
        Matrice booléenne (élément structurant).
    """
    rx_px = max(0, int(rx_px))
    ry_px = max(0, int(ry_px))
    if rx_px == 0 and ry_px == 0:
        return np.ones((1, 1), dtype=bool)

    h = 2 * ry_px + 1
    w = 2 * rx_px + 1
    Y, X = np.ogrid[-ry_px:ry_px + 1, -rx_px:rx_px + 1]

    if rx_px == 0:
        S = (np.abs(Y) <= ry_px)
    elif ry_px == 0:
        S = (np.abs(X) <= rx_px)
    else:
        S = (X * X) / (rx_px * rx_px) + (Y * Y) / (ry_px * ry_px) <= 1.0

    return S.astype(bool)


def _compute_change_mask_interval_array(
    newer_path: Path,
    older_path: Path,
    cfg: MaskConfig,
    logger: logging.Logger,
) -> tuple[np.ndarray, dict]:
    """
    Calcule le masque brut (0/1/nodata) sur la grille du raster "newer".

    Ici :
    - newer = MNS corrélation recalé (grille de référence)
    - older = MNS LiDAR (reprojeté localement sur la fenêtre courante)

    Méthode "intervalle disjoint" (robuste à du bruit) :
    - on calcule pour chaque pixel le min/max locaux sur une fenêtre (rayon cfg.window_radius),
      séparément pour newer et older,
    - changement si l'intervalle [new_min, new_max] est disjoint de [old_min, old_max],
      en tenant compte de tolérances verticales.

    Paramètres de tolérance :
    - tol_low  : seuil pour détecter un abaissement (new < old - tol_low)
    - tol_high : seuil pour détecter un rehaussement (new > old + tol_high)

    Parameters
    ----------
    newer_path : Path
        Raster "newer" (corrélation recalée).
    older_path : Path
        Raster "older" (LiDAR).
    cfg : MaskConfig
        Paramètres de calcul.
    logger : logging.Logger
        Logger (non utilisé directement ici mais conservé pour cohérence).

    Returns
    -------
    tuple[np.ndarray, dict]
        - mask : uint8 (0/1/nodata)
        - profile : profil rasterio du masque (dtype/nodata/compression mis à jour)
    """
    tol_low, tol_high = _normalize_tolerance(cfg.z_tolerance_m)

    with rasterio.open(newer_path) as ds_new, rasterio.open(older_path) as ds_old:
        prof = ds_new.profile.copy()
        nodata_out = cfg.mask_nodata

        mask = np.full((ds_new.height, ds_new.width), nodata_out, dtype=np.uint8)

        for y0 in range(0, ds_new.height, cfg.block_size):
            for x0 in range(0, ds_new.width, cfg.block_size):
                h = min(cfg.block_size, ds_new.height - y0)
                w = min(cfg.block_size, ds_new.width - x0)
                core = Window(x0, y0, w, h)

                big, y_off, x_off = _expand_window_with_halo(core, ds_new, halo=cfg.window_radius)

                # newer (corrélation recalée) -> NaN
                new_big = _read_as_float32_with_nan(ds_new, big)

                # older (LiDAR) reprojecté sur la fenêtre big -> NaN
                old_big = np.empty_like(new_big, dtype=np.float32)
                old_big[:] = np.nan

                reproject(
                    source=rasterio.band(ds_old, 1),
                    destination=old_big,
                    src_transform=ds_old.transform,
                    src_crs=ds_old.crs,
                    src_nodata=ds_old.nodata,
                    dst_transform=ds_new.window_transform(big),
                    dst_crs=ds_new.crs,
                    dst_nodata=np.nan,
                    resampling=cfg.resampling,
                )

                # enveloppes min/max
                new_min_big, new_max_big = _rolling_nan_minmax(new_big, cfg.window_radius)
                old_min_big, old_max_big = _rolling_nan_minmax(old_big, cfg.window_radius)

                # crop sur core
                new_min = new_min_big[y_off:y_off + h, x_off:x_off + w]
                new_max = new_max_big[y_off:y_off + h, x_off:x_off + w]
                old_min = old_min_big[y_off:y_off + h, x_off:x_off + w]
                old_max = old_max_big[y_off:y_off + h, x_off:x_off + w]

                valid = np.isfinite(new_min) & np.isfinite(new_max) & np.isfinite(old_min) & np.isfinite(old_max)
                if not np.any(valid):
                    mask[y0:y0 + h, x0:x0 + w] = nodata_out
                    continue

                # logique intervalle disjoint avec tolérance
                lowered = new_max < (old_min - np.float32(tol_low))
                raised  = new_min > (old_max + np.float32(tol_high))
                change  = lowered | raised

                out = np.zeros((h, w), dtype=np.uint8)
                out[valid & change] = 1
                out[~valid] = nodata_out
                mask[y0:y0 + h, x0:x0 + w] = out

        prof.update(dtype=np.uint8, nodata=nodata_out, compress=cfg.compress)
        return mask, prof


def _morpho_open(mask: np.ndarray, radius: int, nodata: int) -> np.ndarray:
    """
    Applique une ouverture morphologique (érosion puis dilatation) sur un masque binaire.

    Le nodata est préservé : les pixels nodata ne participent pas au calcul et restent nodata.

    Parameters
    ----------
    mask : np.ndarray
        Masque uint8 (0/1/nodata).
    radius : int
        Rayon (pixels) de l'élément structurant carré (taille = 2*radius+1).
    nodata : int
        Valeur nodata du masque.

    Returns
    -------
    np.ndarray
        Masque uint8 après ouverture morphologique.
    """
    nod = (mask == nodata)
    change = (mask == 1)
    change[nod] = False

    k = 2 * int(radius) + 1
    structure = np.ones((k, k), dtype=bool)

    opened = binary_dilation(binary_erosion(change, structure=structure), structure=structure)

    out = np.full(mask.shape, nodata, dtype=np.uint8)
    out[~nod] = np.where(opened[~nod], 1, 0).astype(np.uint8)
    return out


def _remove_small_components(mask: np.ndarray, transform, min_area_m2: float, connectivity: int, nodata: int, logger: logging.Logger) -> np.ndarray:
    """
    Supprime les composantes connexes dont l'aire est inférieure à un seuil.

    Le seuil est exprimé en m² et converti en nombre minimal de pixels via l'aire pixel
    déduite de la transform affine.

    Parameters
    ----------
    mask : np.ndarray
        Masque uint8 (0/1/nodata).
    transform :
        Transform affine rasterio (pour estimer aire pixel).
    min_area_m2 : float
        Aire minimale des composantes à conserver (m²).
    connectivity : int
        Connectivité des composantes (4 ou 8).
    nodata : int
        Valeur nodata.
    logger : logging.Logger
        Logger (journalise les statistiques).

    Returns
    -------
    np.ndarray
        Masque uint8 filtré.
    """
    nod = (mask == nodata)
    change = (mask == 1)
    change[nod] = False

    px_area = _pixel_area_from_affine(transform)
    min_pixels = int(np.ceil(float(min_area_m2) / max(px_area, np.finfo(float).eps)))
    min_pixels = max(1, min_pixels)

    if connectivity == 8:
        structure = np.ones((3, 3), dtype=bool)
    elif connectivity == 4:
        structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)
    else:
        raise ValueError("connectivity doit être 4 ou 8")

    labels, nlab = cc_label(change, structure=structure)
    if nlab == 0:
        return mask

    sizes = np.bincount(labels.ravel())
    keep_label = np.flatnonzero(sizes >= min_pixels)
    keep_label = keep_label[keep_label != 0]
    keep = np.isin(labels, keep_label)

    out = np.zeros(mask.shape, dtype=np.uint8)
    out[keep] = 1
    out[nod] = nodata

    removed = int((sizes[1:] < min_pixels).sum())
    kept = int((sizes[1:] >= min_pixels).sum())
    logger.info("Aire pixel: %.4f m² | seuil: %.1f m² -> min_pixels=%d | composantes: kept=%d removed=%d total=%d",
                px_area, min_area_m2, min_pixels, kept, removed, nlab)

    return out


def _buffer_mask_metric(mask: np.ndarray, transform, buffer_m: float, closing: bool, nodata: int) -> np.ndarray:
    """
    Applique un buffer métrique sur le masque binaire par opérations morphologiques.

    Implementation :
    - conversion du buffer (mètres) en rayons (pixels) selon l'échelle pixel (sx, sy),
    - dilatation avec un élément structurant elliptique,
    - optionnellement fermeture : dilatation suivie d'érosion.

    Le nodata est préservé.

    Parameters
    ----------
    mask : np.ndarray
        Masque uint8 (0/1/nodata).
    transform :
        Transform affine rasterio (pour estimer sx, sy).
    buffer_m : float
        Rayon de buffer (mètres).
    closing : bool
        Si True : fermeture (dilatation puis érosion), sinon dilatation seule.
    nodata : int
        Valeur nodata.

    Returns
    -------
    np.ndarray
        Masque uint8 après buffer.
    """
    nod = (mask == nodata)
    change = (mask == 1)
    change[nod] = False

    sx, sy = _pixel_scales_from_affine(transform)
    rx_px = int(np.ceil(float(buffer_m) / max(sx, np.finfo(float).eps)))
    ry_px = int(np.ceil(float(buffer_m) / max(sy, np.finfo(float).eps)))
    se = _elliptical_structure(rx_px, ry_px)

    dil = binary_dilation(change, structure=se)
    if closing:
        proc = binary_erosion(dil, structure=se)
    else:
        proc = dil

    out = np.full(mask.shape, nodata, dtype=np.uint8)
    out[~nod] = np.where(proc[~nod], 1, 0).astype(np.uint8)
    return out


def _write_mask(mask: np.ndarray, profile: dict, out_path: Path) -> None:
    """
    Écrit un masque raster (bande 1) sur disque.

    Parameters
    ----------
    mask : np.ndarray
        Masque uint8.
    profile : dict
        Profil rasterio (dtype, nodata, transform, crs, compression, etc.).
    out_path : Path
        Chemin de sortie.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)


def run_creation_masque(paths: ProjectPaths, cfg: MaskConfig) -> None:
    """
    Crée les masques de changement (un par dalle) en comparant :

    - newer = MNS corrélation recalé : `paths.mns_recale`
    - older = MNS LiDAR             : `paths.mns_lidar`

    Sortie :
    - masque final (0/1/nodata) écrit dans `paths.masque`,
      nommé : MASK_CHANGE_XXXX_YYYY.tif

    Parameters
    ----------
    paths : ProjectPaths
        Arborescence du projet.
    cfg : MaskConfig
        Paramètres de création du masque.

    Raises
    ------
    RuntimeError
        Si aucune dalle n'est trouvée dans `paths.mns_recale`.
    MissingMatchingTileError
        Si une dalle LiDAR correspondante ne peut pas être trouvée.
    """
    logger = setup_logger(paths.logs, name="creation_masque")

    corr_files = sorted(paths.mns_recale.glob("*.tif"))
    if not corr_files:
        raise RuntimeError(f"Aucune dalle trouvée dans {paths.mns_recale}")

    logger.info("Nombre de dalles à traiter: %d", len(corr_files))
    logger.info("Paramètres: tol=%s, window_radius=%d, open_r=%d, min_area=%.1f m², buffer=%.1f m",
                str(cfg.z_tolerance_m), cfg.window_radius, cfg.radius_open, cfg.min_area_m2, cfg.buffer_m)

    for i, corr_path in enumerate(corr_files, start=1):
        t0 = time.time()

        x_km, y_km = _parse_tile_xy_km_from_corr_name(corr_path.name)
        lidar_path = _find_mns_lidar_file_for_tile(paths.mns_lidar, x_km, y_km)

        out_path = paths.masque / f"MASK_CHANGE_{x_km:04d}_{y_km:04d}.tif"
        if out_path.exists() and not cfg.overwrite:
            logger.info("Skip (déjà présent): %s", out_path.name)
            continue

        logger.info("---- Dalle %d / %d : %s ----", i, len(corr_files), f"{x_km:04d}_{y_km:04d}")
        logger.info("Newer (corr recalé): %s", corr_path.name)
        logger.info("Older (LiDAR): %s", lidar_path.name)

        # 1) Masque brut intervalle
        mask, prof_mask = _compute_change_mask_interval_array(
            newer_path=corr_path,
            older_path=lidar_path,
            cfg=cfg,
            logger=logger,
        )

        nodata = cfg.mask_nodata
        n_valid = int((mask != nodata).sum())
        n_change = int((mask == 1).sum())
        logger.info("Masque brut: valid=%d | change=%d (%.2f%%)",
                    n_valid, n_change, 100.0 * n_change / max(1, n_valid))

        # On a besoin du transform pour aire/buffer
        with rasterio.open(corr_path) as ds_new:
            transform = ds_new.transform

        # 2) Open morphologique
        mask2 = _morpho_open(mask, radius=cfg.radius_open, nodata=nodata)

        # 3) Filtre petites composantes
        mask3 = _remove_small_components(
            mask2, transform=transform, min_area_m2=cfg.min_area_m2,
            connectivity=cfg.connectivity, nodata=nodata, logger=logger
        )

        # 4) Buffer métrique
        mask4 = _buffer_mask_metric(
            mask3, transform=transform, buffer_m=cfg.buffer_m,
            closing=cfg.buffer_closing, nodata=nodata
        )

        n_change_final = int((mask4 == 1).sum())
        logger.info("Masque final: change=%d (%.2f%% des pixels valides initiaux)",
                    n_change_final, 100.0 * n_change_final / max(1, n_valid))

        # 5) Écriture finale uniquement
        _write_mask(mask4.astype(np.uint8, copy=False), prof_mask, out_path)

        dt = time.time() - t0
        logger.info("Écrit: %s | Temps: %.1f s", out_path.name, dt)

    logger.info("Création des masques terminée.")