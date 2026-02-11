# -*- coding: utf-8 -*-
"""
Découpage des nuages LiDAR et MNS de corrélation recalé selon le masque de changement.
Fusion des nuages découpé en un .laz

@author: ESaint-Denis
"""

# utils/fusion_nuages.py
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window



import laspy

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

from utils.creation_arborescence import ProjectPaths
from utils.recuperation_donnees import setup_logger


class MissingInputsError(RuntimeError):
    """Erreur levée si on ne trouve pas un des fichiers nécessaires pour une dalle."""


# Regex
_TILE_RE_KM = re.compile(r"_(\d{4})_(\d{4})_")        # ex LiDAR: _0605_6933_
_TILE_RE_M  = re.compile(r"_(\d{7})_(\d{7})")         # ex corr: _0605000_6933000


@dataclass(frozen=True)
class FusionConfig:
    # Lecture/écriture
    chunk_size_lidar: int = 5_000_000        # points par itération LiDAR
    block_size_dsm: int = 1024               # pixels (pour itérer DSM/mask par blocs)
    overwrite: bool = True

    # LAS/LAZ
    las_version: str = "1.4"
    point_format: int = 6                    # LAS 1.4 PF6 pour WKT CRS
    scales: Tuple[float, float, float] = (0.01, 0.01, 0.01)  # précision cm


def _parse_tile_xy_km_from_name(name: str) -> tuple[int, int]:
    """
    Renvoie (x_km, y_km) à partir de:
    - ..._0605_6933_...   (km)
    - ..._0605000_6933000... (m -> /1000)
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

    raise ValueError(f"Impossible d'extraire l'id dalle depuis: {name}")


def _find_lidar_laz_for_tile(lidar_dir: Path, x_km: int, y_km: int) -> Path:
    """
    Trouve la dalle LiDAR (copc.laz ou laz) correspondant à x_km,y_km.
    """
    pattern = f"_{x_km:04d}_{y_km:04d}_"
    candidates = []
    for ext in ("*.laz", "*.LAZ"):
        for p in lidar_dir.glob(ext):
            if pattern in p.name:
                candidates.append(p)

    if not candidates:
        raise MissingInputsError(f"Aucun fichier LiDAR (.laz) trouvé pour {x_km:04d}_{y_km:04d} dans {lidar_dir}")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_dsm_recale_for_tile(mns_recale_dir: Path, x_km: int, y_km: int) -> Path:
    """
    Trouve le MNS corrélation recalé (tif) correspondant.
    On cherche la version en mètres: _0605000_6933000
    """
    x_m = f"{x_km*1000:07d}"
    y_m = f"{y_km*1000:07d}"
    pattern = f"_{x_m}_{y_m}"
    candidates = [p for p in mns_recale_dir.glob("*.tif") if pattern in p.name]

    if not candidates:
        raise MissingInputsError(f"Aucun MNS recalé trouvé pour {x_km:04d}_{y_km:04d} dans {mns_recale_dir}")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_mask_for_tile(mask_dir: Path, x_km: int, y_km: int) -> Path:
    """
    Trouve le masque final.
    Par défaut, le script masque proposait MASK_CHANGE_0605_6933.tif.
    On accepte aussi d'autres variantes tant que _0605_6933 apparaît.
    """
    pattern = f"_{x_km:04d}_{y_km:04d}"
    candidates = [p for p in mask_dir.glob("*.tif") if pattern in p.stem]

    if not candidates:
        raise MissingInputsError(f"Aucun masque trouvé pour {x_km:04d}_{y_km:04d} dans {mask_dir}")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _assert_same_grid_except_extent(ds1: rasterio.io.DatasetReader,
                                   ds2: rasterio.io.DatasetReader,
                                   name1="DSM", name2="mask") -> None:
    """
    Vérifie CRS + résolution/orientation identiques (a,e,b,d).
    Autorise origines et emprises différentes.
    """
    if ds1.crs is None or ds2.crs is None:
        raise RuntimeError(f"{name1} et/ou {name2} n'ont pas de CRS.")

    if ds1.crs != ds2.crs:
        raise RuntimeError(f"{name1} et {name2} CRS différents: {ds1.crs} vs {ds2.crs}")

    t1, t2 = ds1.transform, ds2.transform
    ok = (
        np.isclose(t1.a, t2.a, atol=1e-9) and
        np.isclose(t1.e, t2.e, atol=1e-9) and
        np.isclose(t1.b, t2.b, atol=1e-9) and
        np.isclose(t1.d, t2.d, atol=1e-9)
    )
    if not ok:
        raise RuntimeError(f"{name1} et {name2} n'ont pas la même grille (résolution/orientation diffèrent).")


def _load_common_mask_and_meta(dsm_path: Path, mask_path: Path) -> tuple[np.ndarray, dict, tuple[float, float, float, float]]:
    """
    Charge le masque sur l'intersection DSM/mask, et renvoie:
    - mask_arr uint8 {0,1}
    - meta dict avec transform/crs/width/height/nodata
    - bounds communs (left,bottom,right,top)
    """
    with rasterio.open(dsm_path) as ds_dsm, rasterio.open(mask_path) as ds_mask:
        _assert_same_grid_except_extent(ds_dsm, ds_mask, name1="DSM", name2="mask")

        b1, b2 = ds_dsm.bounds, ds_mask.bounds
        left = max(b1.left, b2.left)
        right = min(b1.right, b2.right)
        bottom = max(b1.bottom, b2.bottom)
        top = min(b1.top, b2.top)

        if left >= right or bottom >= top:
            raise RuntimeError("DSM et masque ne se recouvrent pas (intersection vide).")

        win_dsm = from_bounds(left, bottom, right, top, transform=ds_dsm.transform).round_offsets().round_lengths()
        win_mask = from_bounds(left, bottom, right, top, transform=ds_mask.transform).round_offsets().round_lengths()

        if (win_dsm.width != win_mask.width) or (win_dsm.height != win_mask.height):
            raise RuntimeError("Fenêtres DSM et masque de tailles différentes: grilles pas parfaitement alignées.")

        # masque binaire 0/1
        m = ds_mask.read(1, window=win_mask)
        nodata_mask = ds_mask.nodata
        if nodata_mask is not None:
            valid = (m != nodata_mask)
            m = np.where(valid & (m > 0), 1, 0).astype(np.uint8)
        else:
            m = np.where(np.isfinite(m) & (m > 0), 1, 0).astype(np.uint8)

        # transform commun = transform DSM sur la fenêtre
        t_common = ds_dsm.window_transform(win_dsm)

        meta = {
            "transform": t_common,
            "crs": ds_dsm.crs,
            "width": int(win_dsm.width),
            "height": int(win_dsm.height),
            "bounds": (left, bottom, right, top),
            "win_dsm": win_dsm,   # utile pour lire DSM par blocs
        }

    return m, meta, (left, bottom, right, top)


def _lidar_points_in_extent(x: np.ndarray, y: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    left, bottom, right, top = bounds
    return (x >= left) & (x <= right) & (y >= bottom) & (y <= top)


def _index_from_xy(x: np.ndarray, y: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray]:
    """
    (x,y) -> (row,col) vectorisé via inverse affine (indices floor).
    """
    inv = ~transform
    col = inv.a * x + inv.b * y + inv.c
    row = inv.d * x + inv.e * y + inv.f
    return np.floor(row).astype(np.int64), np.floor(col).astype(np.int64)


def _iter_kept_lidar_xyz(laz_path: Path,
                         mask_arr: np.ndarray,
                         meta: dict,
                         chunk_size: int):
    """
    Itère sur les points LiDAR à conserver (mask==0), renvoie des chunks (x,y,z) float64.
    """
    transform = meta["transform"]
    H, W = meta["height"], meta["width"]
    bounds = meta["bounds"]

    with laspy.open(laz_path) as reader:
        for pts in reader.chunk_iterator(chunk_size):
            x = pts.x.copy()
            y = pts.y.copy()
            z = pts.z.copy()

            inside = _lidar_points_in_extent(x, y, bounds)
            if not np.any(inside):
                continue

            x = x[inside]; y = y[inside]; z = z[inside]
            rows, cols = _index_from_xy(x, y, transform)

            valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
            if not np.any(valid):
                continue

            rows = rows[valid]; cols = cols[valid]
            x = x[valid]; y = y[valid]; z = z[valid]

            keep = (mask_arr[rows, cols] == 0)
            if np.any(keep):
                yield x[keep], y[keep], z[keep]


def _iter_dsm_points_where_changed(dsm_path: Path,
                                  mask_arr: np.ndarray,
                                  meta: dict,
                                  block_size: int):
    """
    Itère les points DSM (centre pixel) là où mask==1, en lisant DSM par blocs.
    Renvoie des chunks (x,y,z) float64.
    """
    transform = meta["transform"]
    win_dsm_full = meta["win_dsm"]

    with rasterio.open(dsm_path) as ds:
        nodata = ds.nodata

        H = meta["height"]
        W = meta["width"]

        # on parcourt en blocs sur la grille commune
        for row0 in range(0, H, block_size):
            h = min(block_size, H - row0)
            for col0 in range(0, W, block_size):
                w = min(block_size, W - col0)

                # masque bloc
                mblk = mask_arr[row0:row0+h, col0:col0+w]
                if not np.any(mblk == 1):
                    continue

                # fenêtre correspondante dans le DSM d'origine (win_dsm_full + offset)
                win = Window(
                    col_off=win_dsm_full.col_off + col0,
                    row_off=win_dsm_full.row_off + row0,
                    width=w,
                    height=h,
                )

                dsm = ds.read(1, window=win, masked=False).astype(np.float32)
                if nodata is not None:
                    valid_dsm = (dsm != nodata) & np.isfinite(dsm)
                else:
                    valid_dsm = np.isfinite(dsm)

                changed = (mblk == 1) & valid_dsm
                if not np.any(changed):
                    continue

                rr, cc = np.nonzero(changed)
                # indices en coordonnées "grille commune"
                rows = rr.astype(np.int64) + row0
                cols = cc.astype(np.int64) + col0

                # centres pixel: (col+0.5,row+0.5)
                colc = cols.astype(np.float64) + 0.5
                rowc = rows.astype(np.float64) + 0.5

                a, b, c0, d, e, f0 = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
                x = c0 + a * colc + b * rowc
                y = f0 + d * colc + e * rowc
                z = dsm[rr, cc].astype(np.float64)

                yield x, y, z


def _update_minmax(mins: Optional[np.ndarray], maxs: Optional[np.ndarray],
                   x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cur_min = np.array([x.min(), y.min(), z.min()], dtype=np.float64)
    cur_max = np.array([x.max(), y.max(), z.max()], dtype=np.float64)

    if mins is None:
        mins = cur_min
        maxs = cur_max
    else:
        mins = np.minimum(mins, cur_min)
        maxs = np.maximum(maxs, cur_max)
    return mins, maxs


def _write_laz_streaming(out_path: Path,
                         iter_lidar,
                         iter_dsm,
                         crs_wkt: Optional[str],
                         cfg: FusionConfig,
                         mins_xyz: np.ndarray,
                         maxs_xyz: np.ndarray,
                         logger: logging.Logger) -> None:
    """
    Écrit le LAZ en streaming (pas de concat RAM).
    """
    header = laspy.LasHeader(point_format=cfg.point_format, version=cfg.las_version)
    header.scales = np.array(cfg.scales, dtype=np.float64)
    header.offsets = np.array([mins_xyz[0], mins_xyz[1], mins_xyz[2]], dtype=np.float64)

    if crs_wkt:
        try:
            header.parse_crs_wkt(crs_wkt)
            logger.info("CRS WKT embarqué dans le LAZ.")
        except Exception as e:
            logger.warning("Impossible d'embarquer le CRS WKT: %s", e)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sur Windows, si le fichier existe et est ouvert ailleurs => Permission denied
    if out_path.exists() and cfg.overwrite:
        out_path.unlink()

    with laspy.open(out_path, mode="w", header=header) as writer:
        # Écrire LiDAR (mask==0)
        for x, y, z in iter_lidar:
            rec = laspy.ScaleAwarePointRecord.zeros(len(x), header=header)
            rec.x = x
            rec.y = y
            rec.z = z
            writer.write_points(rec)

        # Écrire DSM (mask==1)
        for x, y, z in iter_dsm:
            rec = laspy.ScaleAwarePointRecord.zeros(len(x), header=header)
            rec.x = x
            rec.y = y
            rec.z = z
            writer.write_points(rec)

    logger.info("LAZ écrit: %s", out_path)


def fuse_one_tile(paths: ProjectPaths, x_km: int, y_km: int, cfg: FusionConfig, logger: logging.Logger) -> None:
    """
    Fusionne une dalle:
      - LiDAR: garder mask==0
      - DSM recalé: générer points mask==1
      - Masque: binaire 0/1

    Sortie: nuage_combine/combined_XXXX_YYYY.laz
    """
    lidar_path = _find_lidar_laz_for_tile(paths.nuage_points_lidar, x_km, y_km)
    dsm_path = _find_dsm_recale_for_tile(paths.mns_recale, x_km, y_km)
    mask_path = _find_mask_for_tile(paths.masque, x_km, y_km)

    out_path = paths.nuage_combine / f"combined_{x_km:04d}_{y_km:04d}.laz"
    if out_path.exists() and not cfg.overwrite:
        logger.info("Skip (déjà présent): %s", out_path.name)
        return

    logger.info("LiDAR: %s", lidar_path.name)
    logger.info("DSM recalé: %s", dsm_path.name)
    logger.info("Masque: %s", mask_path.name)
    logger.info("Sortie: %s", out_path.name)

    # Charger masque sur intersection DSM/mask + meta commun
    mask_arr, meta, _ = _load_common_mask_and_meta(dsm_path, mask_path)

    # CRS WKT depuis rasters
    crs_wkt = None
    if meta["crs"] is not None:
        try:
            crs_wkt = meta["crs"].to_wkt()
        except Exception:
            crs_wkt = None

    # ---- PASS 1: stats min/max + comptages (pour header offsets) ----
    mins = None
    maxs = None
    count_lidar = 0
    count_dsm = 0

    logger.info("Pass 1/2: calcul des bornes XYZ + comptage...")
    for x, y, z in _iter_kept_lidar_xyz(lidar_path, mask_arr, meta, cfg.chunk_size_lidar):
        mins, maxs = _update_minmax(mins, maxs, x, y, z)
        count_lidar += int(x.size)

    for x, y, z in _iter_dsm_points_where_changed(dsm_path, mask_arr, meta, cfg.block_size_dsm):
        mins, maxs = _update_minmax(mins, maxs, x, y, z)
        count_dsm += int(x.size)

    if mins is None:
        raise RuntimeError("Aucun point à écrire (LiDAR filtré vide + DSM change vide).")

    logger.info("LiDAR conservés: %d points", count_lidar)
    logger.info("DSM créés: %d points", count_dsm)
    logger.info("Total: %d points", count_lidar + count_dsm)

    # ---- PASS 2: écriture streaming ----
    logger.info("Pass 2/2: écriture LAZ en streaming...")
    iter_lidar = _iter_kept_lidar_xyz(lidar_path, mask_arr, meta, cfg.chunk_size_lidar)
    iter_dsm = _iter_dsm_points_where_changed(dsm_path, mask_arr, meta, cfg.block_size_dsm)

    _write_laz_streaming(
        out_path=out_path,
        iter_lidar=iter_lidar,
        iter_dsm=iter_dsm,
        crs_wkt=crs_wkt,
        cfg=cfg,
        mins_xyz=mins,
        maxs_xyz=maxs,
        logger=logger,
    )


def run_fusion_nuages(paths: ProjectPaths, cfg: FusionConfig) -> None:
    """
    Lance la fusion dalle par dalle.
    On se base sur les dalles présentes dans MNS_recale (corr recalé), et on déduit x_km,y_km.
    """
    logger = setup_logger(paths.logs, name="fusion_nuages")

    dsm_files = sorted(paths.mns_recale.glob("*.tif"))
    if not dsm_files:
        raise RuntimeError(f"Aucune dalle dans {paths.mns_recale}")

    logger.info("Nombre de dalles DSM recalées à traiter: %d", len(dsm_files))
    logger.info("Paramètres: chunk_lidar=%d | block_dsm=%d | overwrite=%s",
                cfg.chunk_size_lidar, cfg.block_size_dsm, cfg.overwrite)

    for i, p in enumerate(dsm_files, start=1):
        t0 = time.time()
        x_km, y_km = _parse_tile_xy_km_from_name(p.name)

        logger.info("==== Dalle %d/%d : %04d_%04d ====", i, len(dsm_files), x_km, y_km)
        fuse_one_tile(paths, x_km, y_km, cfg, logger)
        logger.info("Temps dalle: %.1f s", time.time() - t0)

    logger.info("Fusion des nuages terminée.")
