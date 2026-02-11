# -*- coding: utf-8 -*-
"""
Création du masque de changement par différentiel de MNS entre MNS LiDAr et MNS de corrélation.
Opérations géométriques et de filtrage sur le masque

@author: ESaint-Denis
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
    """Erreur levée si on ne trouve pas la dalle MNS LiDAR correspondant à la dalle corrélation recalée."""


# --- regex: MNS corrélation contient souvent ..._0605000_6933000.tif ---
_TILE_RE_M = re.compile(r"_(\d{7})_(\d{7})")


@dataclass(frozen=True)
class MaskConfig:
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
    buffer_closing: bool = False    # False = dilatation seule (comme ton script)

    # I/O
    mask_nodata: int = 255
    compress: str = "LZW"
    overwrite: bool = True


def _normalize_tolerance(z_tolerance_m: float | tuple[float, float]) -> tuple[float, float]:
    """Retourne (tol_lowering, tol_raising) en mètres."""
    if isinstance(z_tolerance_m, (tuple, list)) and len(z_tolerance_m) == 2:
        return float(z_tolerance_m[0]), float(z_tolerance_m[1])
    t = float(z_tolerance_m or 0.0)
    return t, t


def _parse_tile_xy_km_from_corr_name(name: str) -> tuple[int, int]:
    """
    Extrait (x_km, y_km) depuis un nom MNS corrélation contenant _0605000_6933000.
    (on divise par 1000)
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
    Retrouve la dalle MNS LiDAR correspondante dans MNS_lidar/.
    Exemple attendu: ..._0605_6933_..._MNS_....tif
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
    Lit une fenêtre et remplace nodata par NaN.
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
    Étend une fenêtre core avec un halo (pour calculer un min/max glissant),
    clampé aux limites du raster.
    Retourne (big_window, y_off, x_off) où (y_off,x_off) est l'origine de core dans big_window.
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
    Min/max glissant NaN-aware sur fenêtre carrée (2R+1)^2, sans SciPy.
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
    Aire d'un pixel en unités CRS² : |a*e - b*d|.
    (OK aussi si rotation/skew)
    """
    a = float(getattr(transform, "a"))
    b = float(getattr(transform, "b"))
    d = float(getattr(transform, "d"))
    e = float(getattr(transform, "e"))
    return abs(a * e - b * d)


def _pixel_scales_from_affine(transform) -> tuple[float, float]:
    """
    Échelle pixel en X et Y (m/pixel en LAMB93), robuste à la rotation.
    sx = sqrt(a² + b²), sy = sqrt(d² + e²)
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
    Élément structurant elliptique (approx cercle métrique).
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
    Calcule le masque brut (0/1/255) sur la grille du raster newer (corrélation recalée).
    Retourne (mask_uint8, profile_newer).
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
    Open morphologique (érosion puis dilatation) sur mask binaire, en préservant nodata.
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
    Supprime les composantes connexes dont l'aire < min_area_m2.
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
    Dilatation métrique (ellipse) sur le masque binaire, nodata préservé.
    Option closing: dilatation puis érosion.
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)


def run_creation_masque(paths: ProjectPaths, cfg: MaskConfig) -> None:
    """
    Crée un masque final par dalle, en comparant:
      - newer = MNS corrélation recalé (paths.mns_recale)
      - older = MNS LiDAR (paths.mns_lidar)

    Sortie:
      - masque final (0/1/255) dans paths.masque, même identifiant de dalle que corr.
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
