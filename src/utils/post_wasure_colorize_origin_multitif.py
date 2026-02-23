# -*- coding: utf-8 -*-
"""
Step 9 - Colorize shifted PLY tiles by origin (LiDAR vs MNS) using many mask GeoTIFF tiles.

Robust handling when a PLY tile intersects multiple mask tiles:
- Start with default color for all vertices.
- For each intersecting mask tile, read a small window and update only vertices covered by it.
- If a vertex is covered multiple times, we prefer "changed" (mask==1) over "unchanged" (mask==0).

All comments in English.
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
    swap_meaning: bool = False
    lidar_rgb: tuple[int, int, int] = (0, 140, 255)
    mns_rgb: tuple[int, int, int] = (255, 120, 0)
    default_rgb: tuple[int, int, int] = (200, 200, 200)
    overwrite: bool = True
    sampling: Literal["nearest"] = "nearest"
    bbox_buffer_m: float = 2.0
    # Conflict resolution when multiple masks cover the same vertex:
    # "prefer_change": mask==1 wins over mask==0
    resolve_conflict: Literal["first", "last", "prefer_change"] = "prefer_change"


def _ensure_vertex_rgb_fields(vertex_arr: np.ndarray) -> np.ndarray:
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
    xmin = float(np.nanmin(x)) - buffer_m
    xmax = float(np.nanmax(x)) + buffer_m
    ymin = float(np.nanmin(y)) - buffer_m
    ymax = float(np.nanmax(y)) + buffer_m
    return xmin, ymin, xmax, ymax


def _bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
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
    ply_l93_dir = Path(ply_l93_dir)
    mask_dir = Path(mask_dir)

    if not ply_l93_dir.is_dir():
        raise RuntimeError(f"Input directory not found: {ply_l93_dir}")
    if not mask_dir.is_dir():
        raise RuntimeError(f"Mask directory not found: {mask_dir}")

    run_dir = ply_l93_dir.parent
    out_dir = run_dir / "ply_L93_origin"
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(ply_l93_dir.glob("*.ply"))
    if not ply_files:
        raise RuntimeError(f"No PLY found in: {ply_l93_dir}")

    mask_files = sorted(mask_dir.glob("*.tif"))
    if not mask_files:
        raise RuntimeError(f"No mask .tif found in: {mask_dir}")

    # Build a lightweight bbox index (no raster data in memory)
    mask_index: list[tuple[Path, tuple[float, float, float, float]]] = []
    for p in mask_files:
        with rasterio.open(p) as ds:
            mask_index.append((p, (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)))

    logger.info(
        "Step 9 - Origin colorization (multi-mask) | tiles=%d | masks=%d | resolve=%s",
        len(ply_files), len(mask_index), cfg.resolve_conflict
    )

    lidar_rgb = np.array(cfg.lidar_rgb, dtype=np.uint8)
    mns_rgb = np.array(cfg.mns_rgb, dtype=np.uint8)
    default_rgb = np.array(cfg.default_rgb, dtype=np.uint8)

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

        ply = PlyData.read(str(in_ply))
        if "vertex" not in ply:
            logger.warning("No vertex element, skipping: %s", in_ply)
            continue

        v = np.array(ply["vertex"].data)
        if ("x" not in v.dtype.names) or ("y" not in v.dtype.names):
            raise RuntimeError(f"Missing x/y fields in vertex data: {in_ply}")

        x = v["x"].astype(np.float64)
        y = v["y"].astype(np.float64)
        bbox = _compute_bbox_xy(x, y, cfg.bbox_buffer_m)

        # Find all intersecting mask tiles
        intersecting = [p for (p, b) in mask_index if _bbox_intersects(bbox, b)]
        if len(intersecting) > 1:
            n_tiles_multi_masks += 1

        # Start with default for all vertices
        colors = np.empty((v.shape[0], 3), dtype=np.uint8)
        colors[:] = default_rgb

        # Track which vertices have been covered by at least one mask
        covered = np.zeros(v.shape[0], dtype=bool)
        # Track mask value assigned (0/1) to solve conflicts consistently
        assigned_val = np.full(v.shape[0], fill_value=255, dtype=np.uint8)  # 255 = unassigned

        for mask_path in intersecting:
            with rasterio.open(mask_path) as ds:
                xmin, ymin, xmax, ymax = bbox
                win = from_bounds(xmin, ymin, xmax, ymax, transform=ds.transform)
                win = win.round_offsets().round_lengths()

                # Read only the window; boundless fill_value=0 means outside -> 0
                arr = ds.read(1, window=win, boundless=True, fill_value=0)

                # Normalize to 0/1
                nodata = ds.nodata
                if nodata is not None:
                    arr = np.where(arr == nodata, 0, arr)
                if np.issubdtype(arr.dtype, np.floating):
                    arr = np.where(np.isfinite(arr), arr, 0)
                m01 = (arr > 0).astype(np.uint8)

                # Window transform for mapping XY -> row/col in this window
                w_transform = ds.window_transform(win)
                inv = ~w_transform
                colf = inv.a * x + inv.b * y + inv.c
                rowf = inv.d * x + inv.e * y + inv.f
                row = np.rint(rowf).astype(np.int64)
                col = np.rint(colf).astype(np.int64)

                H, W = m01.shape
                valid = (row >= 0) & (row < H) & (col >= 0) & (col < W)
                if not np.any(valid):
                    continue

                mvals = m01[row[valid], col[valid]]  # 0/1 for covered vertices
                idx = np.where(valid)[0]

                # Decide which vertices we should update in case of conflict
                if cfg.resolve_conflict == "first":
                    upd = (assigned_val[idx] == 255)
                elif cfg.resolve_conflict == "last":
                    upd = np.ones(idx.size, dtype=bool)
                else:  # prefer_change
                    # Update if unassigned OR new is 1 and previous is 0
                    prev = assigned_val[idx]
                    upd = (prev == 255) | ((mvals == 1) & (prev == 0))

                idx_upd = idx[upd]
                mvals_upd = mvals[upd]

                assigned_val[idx_upd] = mvals_upd
                covered[idx_upd] = True

        # Convert assigned mask values to colors
        # 255 => default (outside)
        is_assigned = (assigned_val != 255)
        if np.any(is_assigned):
            vals = assigned_val[is_assigned]
            idxa = np.where(is_assigned)[0]

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

            n_lidar += int(idx_lidar.size)
            n_mns += int(idx_mns.size)

        # Stats
        n_total_vertices += int(v.shape[0])
        n_covered_vertices += int(covered.sum())
        n_outside_vertices += int((~covered).sum())

        # Write PLY with RGB fields
        v2 = _ensure_vertex_rgb_fields(v)
        v2["red"] = colors[:, 0]
        v2["green"] = colors[:, 1]
        v2["blue"] = colors[:, 2]

        ply.elements = tuple([PlyElement.describe(v2, "vertex")] + [e for e in ply.elements if e.name != "vertex"])
        ply.write(str(out_ply))
        n_written += 1

        if (i % 50) == 0:
            logger.info("Step 9 progress: %d/%d", i, len(ply_files))

    logger.info(
        "Step 9 done | written=%d | skipped=%d | vertices=%d | covered=%d | outside=%d | lidar=%d | mns=%d | tiles_multi_masks=%d | out=%s",
        n_written, n_skipped, n_total_vertices, n_covered_vertices, n_outside_vertices, n_lidar, n_mns, n_tiles_multi_masks, out_dir
    )
    return out_dir
