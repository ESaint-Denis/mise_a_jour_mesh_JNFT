# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import requests
from shapely.geometry import box, shape


@dataclass(frozen=True)
class DepartementWfsConfig:
    # WFS endpoint for BD TOPO V3
    wfs_endpoint: str = "https://data.geopf.fr/wfs/ows"
    type_name: str = "BDTOPO_V3:departement"
    srs: str = "EPSG:2154"
    timeout_s: int = 60


def get_departements_for_tile_bbox(
    x_km: int,
    y_km: int,
    cfg: DepartementWfsConfig,
    tile_size_m: int = 1000,
) -> List[Tuple[str, float]]:
    """
    Return list of (code_insee, intersection_area_m2) for departments intersecting the 1 km tile.

    Notes:
    - Uses WFS GetFeature with BBOX filter.
    - Computes exact intersection areas from returned geometries (shapely).
    - Sorted by decreasing intersection area (stable, deterministic).
    """
    xmin = x_km * 1000
    ymin = y_km * 1000
    xmax = xmin + tile_size_m
    ymax = ymin + tile_size_m

    tile_geom = box(xmin, ymin, xmax, ymax)

# --- Build WFS GetFeature params (robust)
    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": cfg.type_name,
        "SRSNAME": cfg.srs,
        "OUTPUTFORMAT": "application/json",
        # WFS 2.0: prefer 4-values BBOX; CRS is given by SRSNAME
        # WFS 2.0 on this service requires CRS suffix in BBOX
        "BBOX": f"{xmin},{ymin},{xmax},{ymax},{cfg.srs}",
        # Optional: limit number of features (departements are few anyway)
        # "COUNT": 50,
    }

    r = requests.get(cfg.wfs_endpoint, params=params, timeout=cfg.timeout_s)

    # If server still returns 400, try a fallback variant with CRS in BBOX (some servers expect it)
    if r.status_code == 400:
        params2 = dict(params)
        params2["BBOX"] = f"{xmin},{ymin},{xmax},{ymax},{cfg.srs}"
        r = requests.get(cfg.wfs_endpoint, params=params2, timeout=cfg.timeout_s)

    r.raise_for_status()
    gj = r.json()

    out: dict[str, float] = {}
    for feat in gj.get("features", []):
        props = feat.get("properties", {}) or {}
        code = props.get("code_insee")

        # Optional fallback if field name differs
        if not code:
            for k in ("CODE_INSEE", "code_insee_dep", "insee_dep", "code_dep", "code_dept"):
                if k in props:
                    code = props[k]
                    break

        geom = feat.get("geometry")

        if not code or geom is None:
            continue

        dep_geom = shape(geom)
        inter_area = dep_geom.intersection(tile_geom).area
        if inter_area > 0:
            out[code] = out.get(code, 0.0) + float(inter_area)

    # Sort by intersection area (desc)
    return sorted(out.items(), key=lambda kv: kv[1], reverse=True)