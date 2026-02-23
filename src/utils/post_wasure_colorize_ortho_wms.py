# -*- coding: utf-8 -*-
"""
Step 8 - Colorize shifted WaSuRe PLY tiles with IGN orthophotos via WMS-R GetMap.

Design:
- For each PLY tile (in L93), compute its XY bbox (+ optional buffer).
- Request a WMS GetMap image for that bbox at a target ground sampling distance (GSD).
- Sample RGB at each vertex (nearest or bilinear).
- Write a new PLY with per-vertex colors (red/green/blue uint8), keeping faces intact.

All comments in English.

Notes:
- Default mode uses nearest neighbor sampling (fast, matches your experimental script).
- A disk cache avoids re-downloading WMS images on re-runs.
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

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# ----------------------------- Configuration -----------------------------


@dataclass(frozen=True)
class OrthoWmsConfig:
    # WMS-R endpoint (GeoPlateforme IGN)
    wms_url: str = "https://data.geopf.fr/wms-r/wms"
    # Layer name for BD ORTHO 20 cm
    layer: str = "HR.ORTHOIMAGERY.ORTHOPHOTOS"
    # CRS of the request (mesh is in EPSG:2154 after step 7)
    crs: str = "EPSG:2154"
    # Output image format
    img_format: str = "image/jpeg"
    # Target ground sampling distance in meters (0.20 m by default)
    gsd_m: float = 0.20
    # Extra margin around bbox (meters) to avoid border artifacts
    bbox_buffer_m: float = 2.0
    # Sampling mode
    sampling: Literal["nearest", "bilinear"] = "nearest"
    # Default color when outside image
    default_rgb: tuple[int, int, int] = (200, 200, 200)
    # Overwrite output PLY if it already exists
    overwrite: bool = True
    # Cache directory name (created inside run dir)
    cache_dirname: str = "ortho_cache_wms"
    # Request throttling (seconds) to be gentle with the service
    sleep_s: float = 0.05
    # Network timeouts
    timeout_s: float = 60.0
    # Retry count for transient errors
    retries: int = 3


# ----------------------------- Small helpers -----------------------------


def _compute_bbox_xy(x: np.ndarray, y: np.ndarray, buffer_m: float) -> tuple[float, float, float, float]:
    xmin = float(np.nanmin(x)) - buffer_m
    xmax = float(np.nanmax(x)) + buffer_m
    ymin = float(np.nanmin(y)) - buffer_m
    ymax = float(np.nanmax(y)) + buffer_m
    return xmin, ymin, xmax, ymax


def _wms_image_size_for_bbox(bbox: tuple[float, float, float, float], gsd_m: float) -> tuple[int, int]:
    xmin, ymin, xmax, ymax = bbox
    w_m = max(0.0, xmax - xmin)
    h_m = max(0.0, ymax - ymin)
    # Ensure at least 2x2 pixels to avoid server/client issues
    width = max(2, int(math.ceil(w_m / gsd_m)))
    height = max(2, int(math.ceil(h_m / gsd_m)))
    return width, height


def _cache_key(cfg: OrthoWmsConfig, bbox: tuple[float, float, float, float], width: int, height: int) -> str:
    # Stable key: parameters that affect pixel values
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
    Download a WMS GetMap image to out_path (with retries).
    """
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": cfg.layer,
        "STYLES": "",
        "CRS": cfg.crs,
        # For EPSG:2154, axis order is x,y (bbox = xmin,ymin,xmax,ymax)
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
            # Small backoff
            time.sleep(0.5 * (k + 1))

    raise RuntimeError(f"WMS GetMap failed after {cfg.retries} retries: {last_err}") from last_err


def _read_rgb_from_image_bytes(img_path: Path) -> np.ndarray:
    """
    Read an RGB image (JPEG/PNG) into an HxWx3 uint8 array using rasterio.
    """
    data = img_path.read_bytes()
    with MemoryFile(data) as mem:
        with mem.open() as ds:
            # Many JPEG are 3-band (RGB). Some may be 4-band (RGBA).
            count = ds.count
            if count < 3:
                raise RuntimeError(f"Expected at least 3 bands in WMS image, got {count} for {img_path}")
            r = ds.read(1)
            g = ds.read(2)
            b = ds.read(3)

            # Ensure uint8 output
            if r.dtype != np.uint8:
                r = np.clip(r, 0, 255).astype(np.uint8)
                g = np.clip(g, 0, 255).astype(np.uint8)
                b = np.clip(b, 0, 255).astype(np.uint8)

            rgb = np.dstack([r, g, b])
            return rgb


def _xy_to_rowcol(transform: rasterio.Affine, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv = ~transform
    colf = inv.a * x + inv.b * y + inv.c
    rowf = inv.d * x + inv.e * y + inv.f
    row = np.rint(rowf).astype(np.int64)
    col = np.rint(colf).astype(np.int64)
    return row, col


def _sample_rgb_nearest(rgb: np.ndarray, rows: np.ndarray, cols: np.ndarray, default_rgb: tuple[int, int, int]) -> np.ndarray:
    H, W, _ = rgb.shape
    out = np.empty((rows.size, 3), dtype=np.uint8)

    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    out[~valid] = np.array(default_rgb, dtype=np.uint8)

    rv = rows[valid]
    cv = cols[valid]
    out[valid] = rgb[rv, cv, :]
    return out


def _sample_rgb_bilinear(rgb: np.ndarray, transform: rasterio.Affine, x: np.ndarray, y: np.ndarray, default_rgb: tuple[int, int, int]) -> np.ndarray:
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
    Ensure the structured vertex array contains uint8 fields red/green/blue.
    If already present, keep them (will be overwritten).
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


# ----------------------------- Main step function -----------------------------


def run_post_wasure_colorize_ortho_wms(
    *,
    ply_l93_dir: str | Path,
    logger: logging.Logger,
    cfg: OrthoWmsConfig,
) -> Path:
    """
    Step 8: colorize each PLY tile from IGN ortho via WMS GetMap.

    Input
    -----
    ply_l93_dir : directory containing shifted PLY tiles (L93), typically run_*/ply_L93.

    Output
    ------
    Directory run_*/ply_L93_ortho with colorized PLY tiles.
    """
    ply_l93_dir = Path(ply_l93_dir)
    if not ply_l93_dir.is_dir():
        raise RuntimeError(f"Input directory not found: {ply_l93_dir}")

    run_dir = ply_l93_dir.parent
    out_dir = run_dir / "ply_L93_ortho"
    out_dir.mkdir(parents=True, exist_ok=True)

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

        ply = PlyData.read(str(in_ply))
        if "vertex" not in ply:
            logger.warning("No vertex element, skipping: %s", in_ply)
            continue

        v = np.array(ply["vertex"].data)
        if "x" not in v.dtype.names or "y" not in v.dtype.names:
            raise RuntimeError(f"Missing x/y fields in vertex data: {in_ply}")

        x = v["x"].astype(np.float64)
        y = v["y"].astype(np.float64)

        bbox = _compute_bbox_xy(x, y, cfg.bbox_buffer_m)
        width, height = _wms_image_size_for_bbox(bbox, cfg.gsd_m)
        key = _cache_key(cfg, bbox, width, height)
        img_path = cache_dir / f"{key}.jpg"

        if img_path.is_file():
            n_cached += 1
        else:
            _download_wms_getmap(cfg=cfg, bbox=bbox, width=width, height=height, out_path=img_path, logger=logger)
            if cfg.sleep_s > 0:
                time.sleep(cfg.sleep_s)

        rgb = _read_rgb_from_image_bytes(img_path)

        # Build an affine transform consistent with the requested bbox and image size
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width=rgb.shape[1], height=rgb.shape[0])

        if cfg.sampling == "nearest":
            rows, cols = _xy_to_rowcol(transform, x, y)
            sampled = _sample_rgb_nearest(rgb, rows, cols, cfg.default_rgb)
        else:
            sampled = _sample_rgb_bilinear(rgb, transform, x, y, cfg.default_rgb)

        v2 = _ensure_vertex_rgb_fields(v)
        v2["red"] = sampled[:, 0]
        v2["green"] = sampled[:, 1]
        v2["blue"] = sampled[:, 2]

        # Replace vertex element, keep all other elements (faces, etc.)
        new_elems = [PlyElement.describe(v2, "vertex")] + [e for e in ply.elements if e.name != "vertex"]
        ply.elements = tuple(new_elems)

        ply.write(str(out_ply))
        n_written += 1

        # Keep main log light: progress every 50 tiles
        if (i % 50) == 0:
            logger.info("Step 8 progress: %d/%d", i, len(ply_files))

    logger.info(
        "Step 8 done | written=%d | skipped=%d | cache_hits=%d | out=%s",
        n_written, n_skipped, n_cached, out_dir
    )
    return out_dir
