# -*- coding: utf-8 -*-
"""
Recalage altimétrique des dalles MNS de corrélation sur les dalles MNS LiDAR

@author: ESaint-Denis
"""

# utils/recalage_altimetrique.py
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from utils.creation_arborescence import ProjectPaths
from utils.recuperation_donnees import setup_logger  # on réutilise le logger déjà en place


_TILE_RE_KM = re.compile(r"_(\d{4})_(\d{4})_")
_TILE_RE_M  = re.compile(r"_(\d{7})_(\d{7})")  # ex: _0605000_6933000


class MissingMatchingTileError(RuntimeError):
    """Erreur levée quand on ne trouve pas de correspondance MNS LiDAR / MNS corrélation."""


@dataclass(frozen=True)
class RecalageConfig:
    k_mad: float = 3.0
    n_iter: int = 3
    resampling: Resampling = Resampling.bilinear
    compress: str = "deflate"   # compression GeoTIFF
    overwrite: bool = True      # réécrire si déjà présent


def _parse_tile_xy_from_any_filename(name: str) -> tuple[int, int]:
    """
    Extrait (x_km, y_km) depuis un nom de fichier.

    Supporte 2 formats:
    - km: ..._0605_6933_...
    - m : ..._0605000_6933000...
    """
    m_km = _TILE_RE_KM.search(name)
    if m_km:
        return int(m_km.group(1)), int(m_km.group(2))

    m_m = _TILE_RE_M.search(name)
    if m_m:
        x_m = int(m_m.group(1))
        y_m = int(m_m.group(2))

        if x_m % 1000 != 0 or y_m % 1000 != 0:
            raise ValueError(f"Coordonnées mètres non multiples de 1000 dans: {name}")

        return x_m // 1000, y_m // 1000

    raise ValueError(f"Impossible d'extraire l'identifiant de dalle (xxxx_yyyy) depuis: {name}")


def _find_mns_lidar_file_for_tile(mns_lidar_dir: Path, x_km: int, y_km: int) -> Path:
    """
    Retrouve le fichier MNS LiDAR correspondant à la dalle (x_km, y_km) dans MNS_lidar/.
    On cherche un .tif qui contient _XXXX_YYYY_ et aussi '_MNS_' pour éviter les faux positifs.
    """
    pattern = f"_{x_km:04d}_{y_km:04d}_"
    candidates = []
    for p in mns_lidar_dir.glob("*.tif"):
        n = p.name
        if pattern in n and "_MNS_" in n:
            candidates.append(p)

    if len(candidates) == 0:
        raise MissingMatchingTileError(
            f"Aucun MNS LiDAR trouvé pour la dalle {x_km:04d}_{y_km:04d} dans {mns_lidar_dir}"
        )
    if len(candidates) > 1:
        # Cas rare : plusieurs versions. On prend la plus récente (mtime), et on loguera.
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return candidates[0]


def read_raster(path: Path) -> tuple[np.ndarray, float | int | None, dict]:
    """
    Lit la bande 1 d'un raster et renvoie (array float32, nodata, profile).
    """
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        profile = src.profile.copy()
    return arr, nodata, profile


def resample_to_reference_grid(
    src_path: Path,
    ref_profile: dict,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """
    Reprojette / ré-échantillonne src_path sur la grille définie par ref_profile.
    """
    ref_h = ref_profile["height"]
    ref_w = ref_profile["width"]
    dst_nodata = ref_profile.get("nodata", None)

    dst = np.full((ref_h, ref_w), dst_nodata if dst_nodata is not None else np.nan, dtype=np.float32)

    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            dst_nodata=dst_nodata,
            resampling=resampling,
        )

    return dst


def robust_offset_from_dz(
    dz: np.ndarray,
    valid_mask: np.ndarray,
    k_mad: float = 3.0,
    n_iter: int = 3,
) -> tuple[float, np.ndarray]:
    """
    Estime un décalage vertical robuste (médiane) via filtrage MAD itératif.
    Renvoie (delta_z, stable_mask).

    Convention:
      dz = new - old
      delta_z = median(dz sur pixels stables)
    """
    mask = valid_mask.copy()
    delta = 0.0

    for _ in range(max(1, n_iter)):
        v = dz[mask]
        if v.size == 0:
            raise RuntimeError("Plus aucun pixel valide pour estimer le décalage vertical.")

        delta = float(np.median(v))
        r = dz - delta

        rv = r[mask]
        med = float(np.median(rv))
        mad = float(np.median(np.abs(rv - med)))

        if mad < 1e-12:
            # quasi constant : on s'arrête
            break

        robust_std = 1.4826 * mad
        mask = valid_mask & (np.abs(r - med) <= k_mad * robust_std)

    return delta, mask


def _build_valid_mask(arr: np.ndarray, nodata: float | int | None) -> np.ndarray:
    """
    Construit un masque de validité pour un raster : finite + != nodata.
    """
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= (arr != nodata)
    return valid


def write_corrected_dem(
    ref_profile: dict,
    dem_new: np.ndarray,
    nodata_new: float | int | None,
    delta_z: float,
    out_path: Path,
    compress: str = "deflate",
    overwrite: bool = True,
) -> None:
    """
    Écrit le MNS corrélation recalé : dem_new_corr = dem_new - delta_z
    en préservant le nodata.
    """
    if out_path.exists() and not overwrite:
        return

    dem_corr = dem_new - delta_z

    if nodata_new is not None:
        dem_corr[dem_new == nodata_new] = nodata_new

    profile = ref_profile.copy()
    profile.update(dtype="float32", compress=compress)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dem_corr.astype(np.float32), 1)


def run_recalage_altimetrique(
    paths: ProjectPaths,
    cfg: RecalageConfig,
) -> None:
    """
    Recale altimétriquement les dalles MNS de corrélation sur les dalles MNS LiDAR.

    - Entrées :
        * paths.mns_correlation : dalles MNS corrélation (grille de référence)
        * paths.mns_lidar       : dalles MNS LiDAR (à resampler sur la grille corr)
    - Sorties :
        * paths.mns_recale      : dalles MNS corrélation corrigées (mêmes noms)

    Convention du décalage :
        dz = corr - lidar_resampled
        delta_z = median(dz) sur pixels stables
        corr_recale = corr - delta_z
    """
    logger = setup_logger(paths.logs, name="recalage_altimetrique")

    corr_files = sorted(paths.mns_correlation.glob("*.tif"))
    if not corr_files:
        raise RuntimeError(f"Aucune dalle trouvée dans {paths.mns_correlation}")

    logger.info("Nombre de dalles MNS corrélation à traiter: %d", len(corr_files))
    logger.info("Paramètres: k_mad=%.2f, n_iter=%d, resampling=%s", cfg.k_mad, cfg.n_iter, cfg.resampling)

    for i, corr_path in enumerate(corr_files, start=1):
        t0 = time.time()
        x_km, y_km = _parse_tile_xy_from_any_filename(corr_path.name)

        logger.info("---- Dalle %d / %d : %s (id %04d_%04d) ----", i, len(corr_files), corr_path.name, x_km, y_km)

        # 1) Trouver le MNS LiDAR correspondant
        lidar_path = _find_mns_lidar_file_for_tile(paths.mns_lidar, x_km, y_km)
        logger.info("MNS LiDAR correspondant: %s", lidar_path.name)

        # 2) Lire le MNS corrélation (référence)
        corr, nodata_corr, prof_corr = read_raster(corr_path)

        ref_profile = {
            "height": prof_corr["height"],
            "width": prof_corr["width"],
            "transform": prof_corr["transform"],
            "crs": prof_corr["crs"],
            "nodata": prof_corr.get("nodata", None),
            **prof_corr,  # on garde le profil complet pour l'écriture
        }

        # 3) Resampler LiDAR sur la grille corr
        lidar_on_corr = resample_to_reference_grid(
            src_path=lidar_path,
            ref_profile=ref_profile,
            resampling=cfg.resampling,
        )

        # 4) Masque de validité + dz
        valid = _build_valid_mask(corr, nodata_corr)
        # nodata de la grille destination lors du reproject = ref_profile["nodata"]
        nodata_dst = ref_profile.get("nodata", None)
        valid &= _build_valid_mask(lidar_on_corr, nodata_dst)

        dz = corr - lidar_on_corr

        # 5) Estimation robuste
        delta_z, stable_mask = robust_offset_from_dz(
            dz=dz,
            valid_mask=valid,
            k_mad=cfg.k_mad,
            n_iter=cfg.n_iter,
        )

        n_stable = int(stable_mask.sum())
        n_valid = int(valid.sum())
        dt = time.time() - t0

        logger.info("Δz estimé (corr - lidar): %+0.3f m", delta_z)
        logger.info("Pixels valides: %d | Pixels stables: %d (%.1f%%)", n_valid, n_stable, 100.0 * n_stable / max(1, n_valid))

        # 6) Écriture du MNS corrélation recalé (dans MNS_recale) avec le même nom
        out_path = paths.mns_recale / corr_path.name
        write_corrected_dem(
            ref_profile=prof_corr,
            dem_new=corr,
            nodata_new=nodata_corr,
            delta_z=delta_z,
            out_path=out_path,
            compress=cfg.compress,
            overwrite=cfg.overwrite,
        )

        logger.info("Écrit: %s", out_path.name)
        logger.info("Temps: %.1f s", dt)

    logger.info("Recalage altimétrique terminé.")
