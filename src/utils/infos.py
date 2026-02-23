#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build per-tile summary from two log files:
- recalage_altimetrique_*.log  -> tile id, correlation DSM filename, dz
- creation_masque_*.log        -> tile id, correlation DSM filename, final change percent

Outputs:
- tile_summary.csv
- tile_grid_pct_changed.csv (pivot y_km rows, x_km columns)
- tile_grid_dz.csv
- tile_grid_corr_year.csv
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd


@dataclass(frozen=True)
class TileInfo:
    x_km: int
    y_km: int
    corr_name: Optional[str] = None
    corr_year: Optional[int] = None
    dz_m: Optional[float] = None
    pct_changed: Optional[float] = None


# -----------------------------
# Helpers
# -----------------------------

_RE_ID = re.compile(r"\(id\s+(?P<id>\d{4}_\d{4})\)")
_RE_TILE_RECAL = re.compile(
    r"----\s+Dalle\s+\d+\s+/\s+\d+\s+:\s+(?P<corr_name>.+?\.tif)\s+\(id\s+(?P<id>\d{4}_\d{4})\)\s+----"
)
_RE_DZ = re.compile(r"z\s+estim(?:é|e)\s+\(corr\s*-\s*lidar\)\s*:\s*(?P<dz>[-+]?\d+(?:\.\d+)?)\s*m", re.IGNORECASE)

_RE_TILE_MASK = re.compile(r"----\s+Dalle\s+\d+\s+/\s+\d+\s+:\s+(?P<id>\d{4}_\d{4})\s+----")
_RE_NEWER = re.compile(r"Newer\s+\(corr recalé\)\s*:\s*(?P<corr_name>.+?\.tif)")
_RE_MASK_FINAL = re.compile(
    r"Masque final:\s+change=\d+\s+\((?P<pct>\d+(?:\.\d+)?)% des pixels valides initiaux\)"
)

_RE_YEAR_FROM_PATH = re.compile(r"(?:^|/|\\)LAMB93_(?P<yyyy>\d{4})(?:/|\\)")


def _parse_xy_from_id(tile_id: str) -> Tuple[int, int]:
    x_str, y_str = tile_id.split("_")
    return int(x_str), int(y_str)


def _try_parse_year_from_corr_name(corr_name: str) -> Optional[int]:
    """
    Try to infer year from correlation DSM filename.
    Example: MNS_CORREL_1-0_LAMB93_24FD7320_...
    -> year 2024 (assumes 2000+YY)
    """
    m = re.search(r"_LAMB93_(?P<yy>\d{2})FD", corr_name)
    if not m:
        return None
    yy = int(m.group("yy"))
    return 2000 + yy


# -----------------------------
# Parsers
# -----------------------------

def parse_recalage_log(path: Path) -> Dict[str, TileInfo]:
    """
    Parse recalage_altimetrique log.
    Returns dict: tile_id -> TileInfo (with dz and corr tile name).
    """
    tiles: Dict[str, TileInfo] = {}
    current_id: Optional[str] = None
    current_corr: Optional[str] = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _RE_TILE_RECAL.search(line)
        if m:
            current_id = m.group("id")
            current_corr = m.group("corr_name").strip()
            x_km, y_km = _parse_xy_from_id(current_id)
            tiles[current_id] = TileInfo(
                x_km=x_km,
                y_km=y_km,
                corr_name=current_corr,
                corr_year=_try_parse_year_from_corr_name(current_corr),
                dz_m=None,
                pct_changed=None,
            )
            continue

        if current_id is not None:
            m2 = _RE_DZ.search(line)
            if m2:
                dz = float(m2.group("dz"))
                ti = tiles[current_id]
                tiles[current_id] = TileInfo(
                    x_km=ti.x_km,
                    y_km=ti.y_km,
                    corr_name=ti.corr_name,
                    corr_year=ti.corr_year,
                    dz_m=dz,
                    pct_changed=ti.pct_changed,
                )

    return tiles


def parse_mask_log(path: Path) -> Dict[str, TileInfo]:
    """
    Parse creation_masque log.
    Returns dict: tile_id -> TileInfo (with corr tile name and final percent).
    """
    tiles: Dict[str, TileInfo] = {}
    current_id: Optional[str] = None
    current_corr: Optional[str] = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _RE_TILE_MASK.search(line)
        if m:
            current_id = m.group("id")
            current_corr = None
            x_km, y_km = _parse_xy_from_id(current_id)
            tiles.setdefault(current_id, TileInfo(x_km=x_km, y_km=y_km))
            continue

        if current_id is None:
            continue

        m2 = _RE_NEWER.search(line)
        if m2:
            current_corr = m2.group("corr_name").strip()
            ti = tiles[current_id]
            tiles[current_id] = TileInfo(
                x_km=ti.x_km,
                y_km=ti.y_km,
                corr_name=current_corr,
                corr_year=_try_parse_year_from_corr_name(current_corr),
                dz_m=ti.dz_m,
                pct_changed=ti.pct_changed,
            )
            continue

        m3 = _RE_MASK_FINAL.search(line)
        if m3:
            pct = float(m3.group("pct"))
            ti = tiles[current_id]
            # Keep corr_name if already found
            corr_name = ti.corr_name if ti.corr_name else current_corr
            corr_year = ti.corr_year if ti.corr_year else (_try_parse_year_from_corr_name(corr_name) if corr_name else None)
            tiles[current_id] = TileInfo(
                x_km=ti.x_km,
                y_km=ti.y_km,
                corr_name=corr_name,
                corr_year=corr_year,
                dz_m=ti.dz_m,
                pct_changed=pct,
            )

    return tiles


# -----------------------------
# Merge + Outputs
# -----------------------------

def merge_tile_infos(a: Dict[str, TileInfo], b: Dict[str, TileInfo]) -> pd.DataFrame:
    """
    Merge two dicts of TileInfo (same tile_id key).
    Preference: keep corr_name from mask log if present (it is the 'corr recal�' actually used).
    """
    keys = sorted(set(a.keys()) | set(b.keys()))

    rows: List[dict] = []
    for k in keys:
        ta = a.get(k)
        tb = b.get(k)

        x_km = (tb.x_km if tb else ta.x_km) if (ta or tb) else None
        y_km = (tb.y_km if tb else ta.y_km) if (ta or tb) else None

        corr_name = None
        corr_year = None
        dz_m = None
        pct = None

        if ta:
            corr_name = ta.corr_name or corr_name
            corr_year = ta.corr_year or corr_year
            dz_m = ta.dz_m if ta.dz_m is not None else dz_m
            pct = ta.pct_changed if ta.pct_changed is not None else pct

        if tb:
            # Prefer mask corr_name if available
            corr_name = tb.corr_name or corr_name
            corr_year = tb.corr_year or corr_year
            dz_m = tb.dz_m if tb.dz_m is not None else dz_m
            pct = tb.pct_changed if tb.pct_changed is not None else pct

        # If year missing but corr_name present, try parse
        if corr_year is None and corr_name:
            corr_year = _try_parse_year_from_corr_name(corr_name)

        rows.append(
            {
                "tile_id": k,
                "x_km": x_km,
                "y_km": y_km,
                "corr_name": corr_name,
                "corr_year": corr_year,
                "dz_m": dz_m,
                "pct_changed": pct,
            }
        )

    df = pd.DataFrame(rows).sort_values(["y_km", "x_km"]).reset_index(drop=True)
    return df


def write_grids(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pivot grids: y as rows, x as columns
    # Use descending y for "map-like" view (north at top)
    if "pct_changed" in df.columns:
        grid = df.pivot(index="y_km", columns="x_km", values="pct_changed").sort_index(ascending=False)
        grid.to_csv(out_dir / "tile_grid_pct_changed.csv")

    if "dz_m" in df.columns:
        grid = df.pivot(index="y_km", columns="x_km", values="dz_m").sort_index(ascending=False)
        grid.to_csv(out_dir / "tile_grid_dz.csv")

    if "corr_year" in df.columns:
        grid = df.pivot(index="y_km", columns="x_km", values="corr_year").sort_index(ascending=False)
        grid.to_csv(out_dir / "tile_grid_corr_year.csv")

def build_summary_from_logs_dir(logs_dir: Path, out_dir: Path | None = None) -> Path:
    """
    Build summary CSV + grids from the latest recalage_altimetrique_*.log
    and creation_masque_*.log found in logs_dir.

    Returns the path to tile_summary.csv.
    """
    out_dir = out_dir or logs_dir

    recal_logs = sorted(logs_dir.glob("recalage_altimetrique_*.log"))
    if not recal_logs:
        raise FileNotFoundError(f"No recalage_altimetrique_*.log found in {logs_dir}")
    recal_log = recal_logs[-1]

    mask_logs = sorted(logs_dir.glob("creation_masque_*.log"))
    if not mask_logs:
        raise FileNotFoundError(f"No creation_masque_*.log found in {logs_dir}")
    mask_log = mask_logs[-1]

    rec = parse_recalage_log(recal_log)
    msk = parse_mask_log(mask_log)
    df = merge_tile_infos(rec, msk)

    out_csv = out_dir / "tile_summary.csv"
    df.to_csv(out_csv, index=False)
    write_grids(df, out_dir)

    return out_csv

def build_summary_from_logs_dir(logs_dir: Path, out_dir: Path | None = None) -> Path:
    """
    Build summary CSV + grids from the latest recalage_altimetrique_*.log
    and creation_masque_*.log found in logs_dir.

    Returns the path to tile_summary.csv.
    """
    out_dir = out_dir or logs_dir

    recal_logs = sorted(logs_dir.glob("recalage_altimetrique_*.log"))
    if not recal_logs:
        raise FileNotFoundError(f"No recalage_altimetrique_*.log found in {logs_dir}")
    recal_log = recal_logs[-1]

    mask_logs = sorted(logs_dir.glob("creation_masque_*.log"))
    if not mask_logs:
        raise FileNotFoundError(f"No creation_masque_*.log found in {logs_dir}")
    mask_log = mask_logs[-1]

    rec = parse_recalage_log(recal_log)
    msk = parse_mask_log(mask_log)
    df = merge_tile_infos(rec, msk)

    out_csv = out_dir / "tile_summary.csv"
    df.to_csv(out_csv, index=False)
    write_grids(df, out_dir)

    return out_csv

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build per-tile summary from pipeline logs.")
    parser.add_argument("--logs-dir", type=Path, required=True, help="Path to logs directory")
    parser.add_argument("--recalage-log", type=Path, default=None, help="Path to recalage_altimetrique_*.log")
    parser.add_argument("--mask-log", type=Path, default=None, help="Path to creation_masque_*.log")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: logs-dir)")
    args = parser.parse_args()

    logs_dir: Path = args.logs_dir
    out_dir: Path = args.out_dir or logs_dir

    # Auto-pick latest logs if not provided
    if args.recalage_log is None:
        candidates = sorted(logs_dir.glob("recalage_altimetrique_*.log"))
        if not candidates:
            raise FileNotFoundError(f"No recalage_altimetrique_*.log found in {logs_dir}")
        recal_log = candidates[-1]
    else:
        recal_log = args.recalage_log

    if args.mask_log is None:
        candidates = sorted(logs_dir.glob("creation_masque_*.log"))
        if not candidates:
            raise FileNotFoundError(f"No creation_masque_*.log found in {logs_dir}")
        mask_log = candidates[-1]
    else:
        mask_log = args.mask_log

    rec = parse_recalage_log(recal_log)
    msk = parse_mask_log(mask_log)

    df = merge_tile_infos(rec, msk)

    out_csv = out_dir / "tile_summary.csv"
    df.to_csv(out_csv, index=False)

    write_grids(df, out_dir)

    print(f"Wrote: {out_csv}")
    print(f"Wrote grids in: {out_dir}")


if __name__ == "__main__":
    main()