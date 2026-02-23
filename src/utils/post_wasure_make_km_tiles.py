# -*- coding: utf-8 -*-
"""
Build 1 km mesh tiles from WaSuRe colored PLY chunks, using LiDAR DSM (MNS) tiles as reference grid.

- Input: directory with many PLY chunks (typically ~125 m squares), in EPSG:2154 (Lambert-93)
- Reference: LiDAR DSM (GeoTIFF) tiles defining the 1 km grid (exact bounds)
- Output: one PLY per DSM tile, built by:
    1) selecting chunks whose bbox intersects the DSM tile bbox (+ buffer)
    2) concatenating meshes
    3) keeping faces conservatively so we don't create holes:
        keep triangle if any vertex is inside tile bbox (+ eps) OR any edge intersects bbox (+ eps)
   (No geometric cutting of triangles.)

Notes:
- This is designed to avoid holes at tile boundaries (watertight feel), at the cost of overlap.
- Overlap can cause z-fighting in some viewers; if that becomes an issue, we can add a deterministic "ownership" rule.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Optional
from plyfile import PlyData, PlyElement


import numpy as np

try:
    import rasterio
except ImportError as e:
    raise ImportError("rasterio is required to read DSM tile bounds.") from e

try:
    import meshio
except ImportError as e:
    raise ImportError("meshio is required to read/write PLY while preserving vertex colors.") from e


# -----------------------------
# Configuration
# -----------------------------

@dataclass(frozen=True)
class KmTilesConfig:
    dsm_dir: Path
    dsm_glob: str = "*.tif"
    out_dirname: str = "ply_km_tiles"
    suffix: str = "ortho"  # or "origin"

    # Selection / robustness parameters
    preselect_buffer_m: float = 2.0   # expand DSM bbox to select input chunks
    eps_m: float = 0.05               # numeric epsilon for in-box / segment-box tests (5 cm)
    overwrite: bool = True

    # Output format
    binary_out: bool = True           # True -> binary PLY, False -> ascii


# -----------------------------
# Geometry helpers (AABB tests)
# -----------------------------

def _bbox_expand(b: Tuple[float, float, float, float, float, float], e: float) -> Tuple[float, float, float, float, float, float]:
    xmin, xmax, ymin, ymax, zmin, zmax = b
    return (xmin - e, xmax + e, ymin - e, ymax + e, zmin - e, zmax + e)


def _bbox_intersects(a: Tuple[float, float, float, float, float, float],
                     b: Tuple[float, float, float, float, float, float]) -> bool:
    ax0, ax1, ay0, ay1, az0, az1 = a
    bx0, bx1, by0, by1, bz0, bz1 = b
    return (ax0 <= bx1 and ax1 >= bx0 and
            ay0 <= by1 and ay1 >= by0 and
            az0 <= bz1 and az1 >= bz0)


def _points_in_aabb(pts: np.ndarray, aabb: Tuple[float, float, float, float, float, float]) -> np.ndarray:
    # pts: (N, 3)
    xmin, xmax, ymin, ymax, zmin, zmax = aabb
    return ((pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
            (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax) &
            (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax))


def _segment_intersects_aabb(p0: np.ndarray, p1: np.ndarray,
                            aabb: Tuple[float, float, float, float, float, float]) -> bool:
    """
    Slab intersection test for segment vs AABB.
    Returns True if the segment [p0, p1] intersects the box.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = aabb

    d = p1 - p0
    tmin = 0.0
    tmax = 1.0

    for i, (bmin, bmax) in enumerate(((xmin, xmax), (ymin, ymax), (zmin, zmax))):
        if abs(d[i]) < 1e-15:
            # Segment parallel to slab; must be within slab
            if p0[i] < bmin or p0[i] > bmax:
                return False
        else:
            ood = 1.0 / d[i]
            t1 = (bmin - p0[i]) * ood
            t2 = (bmax - p0[i]) * ood
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False

    return True


def _tri_keep_mask(vertices: np.ndarray,
                   faces: np.ndarray,
                   aabb: Tuple[float, float, float, float, float, float]) -> np.ndarray:
    """
    Conservative triangle selection:
    keep triangle if any vertex is inside AABB OR any edge intersects AABB.
    This avoids cutting triangles and reduces risk of holes on boundaries.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    inside0 = _points_in_aabb(v0, aabb)
    inside1 = _points_in_aabb(v1, aabb)
    inside2 = _points_in_aabb(v2, aabb)
    keep = inside0 | inside1 | inside2

    # For triangles without inside vertices, test edge intersection
    idx = np.where(~keep)[0]
    if idx.size == 0:
        return keep

    for k in idx:
        p0 = v0[k]
        p1 = v1[k]
        p2 = v2[k]
        if (_segment_intersects_aabb(p0, p1, aabb) or
            _segment_intersects_aabb(p1, p2, aabb) or
            _segment_intersects_aabb(p2, p0, aabb)):
            keep[k] = True

    return keep


# -----------------------------
# PLY helpers
# -----------------------------
def _write_ply_with_plyfile(path: Path, mesh: meshio.Mesh, binary: bool = True) -> None:
    """
    Write a triangle mesh to PLY using plyfile, preserving vertex attributes as scalar fields
    (e.g., red/green/blue) so downstream tools (3D Tiles) keep colors.

    Notes:
    - PLY colors must be stored as separate 1D arrays (red, green, blue), not (N,3) rgb.
    - Faces are written as 'vertex_indices' list of int32.
    """

    points = np.asarray(mesh.points)
    n = points.shape[0]

    # --- Build vertex dtype
    # Use float32 for size unless you explicitly need float64
    vx = points[:, 0].astype(np.float32, copy=False)
    vy = points[:, 1].astype(np.float32, copy=False)
    vz = points[:, 2].astype(np.float32, copy=False)

    vertex_fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    vertex_arrays = {"x": vx, "y": vy, "z": vz}

    pd = mesh.point_data or {}

    # If meshio-style 'rgb' exists (N,3), split it
    if "rgb" in pd and getattr(pd["rgb"], "ndim", 1) == 2 and pd["rgb"].shape[1] == 3:
        rgb = pd["rgb"].astype(np.uint8, copy=False)
        pd = dict(pd)  # copy
        pd["red"] = rgb[:, 0]
        pd["green"] = rgb[:, 1]
        pd["blue"] = rgb[:, 2]
        pd.pop("rgb", None)

    # Preserve standard color fields if present
    for cname in ("red", "green", "blue", "r", "g", "b"):
        if cname in pd:
            arr = np.asarray(pd[cname])
            if arr.shape != (n,):
                # Skip non-scalar fields safely
                continue
            # Force uint8 for PLY colors
            arr = arr.astype(np.uint8, copy=False)
            vertex_fields.append((cname, "u1"))
            vertex_arrays[cname] = arr

    # Preserve normals if present (nx, ny, nz)
    for nname in ("nx", "ny", "nz"):
        if nname in pd:
            arr = np.asarray(pd[nname])
            if arr.shape != (n,):
                continue
            arr = arr.astype(np.float32, copy=False)
            vertex_fields.append((nname, "f4"))
            vertex_arrays[nname] = arr

    # Optionally preserve any other scalar 1D fields (safe)
    # (Avoid multidimensional arrays because PLY writer conventions vary.)
    for k, v in pd.items():
        if k in vertex_arrays:
            continue
        arr = np.asarray(v)
        if arr.ndim != 1 or arr.shape[0] != n:
            continue
        # Map dtype to something plyfile can write safely
        if arr.dtype.kind in ("u", "i"):
            # int -> int32
            arr2 = arr.astype(np.int32, copy=False)
            vertex_fields.append((k, "i4"))
            vertex_arrays[k] = arr2
        elif arr.dtype.kind == "f":
            arr2 = arr.astype(np.float32, copy=False)
            vertex_fields.append((k, "f4"))
            vertex_arrays[k] = arr2
        # else ignore (strings/objects)

    # --- Build structured vertex array
    vdata = np.empty(n, dtype=vertex_fields)
    for name in vdata.dtype.names:
        vdata[name] = vertex_arrays[name]

    vertex_el = PlyElement.describe(vdata, "vertex")

    # --- Faces (triangles)
    tri = None
    for cb in mesh.cells:
        if cb.type == "triangle":
            tri = cb.data
            break

    if tri is None:
        tri = np.zeros((0, 3), dtype=np.int32)

    tri = np.asarray(tri, dtype=np.int32)
    # plyfile expects a list-like property for vertex_indices
    fdata = np.empty(tri.shape[0], dtype=[("vertex_indices", "O")])
    fdata["vertex_indices"] = [t.tolist() for t in tri]
    face_el = PlyElement.describe(fdata, "face")

    ply = PlyData([vertex_el, face_el], text=not binary)
    ply.write(str(path))


def _read_ply_points_only(path: Path) -> np.ndarray:
    """Read only vertex XYZ from a PLY file (fast path)."""
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    pts = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64, copy=False)
    return pts


def _read_ply_with_plyfile_as_meshio(path: Path) -> meshio.Mesh:
    """
    Read WaSuRe PLY robustly using plyfile, then rebuild a meshio.Mesh.
    This bypasses meshio's PLY quirks with face properties like 'vertex_index'.
    """
    ply = PlyData.read(str(path))

    # --- vertices
    v = ply["vertex"].data
    points = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64, copy=False)

    point_data = {}

    # --- colors (common: red/green/blue or r/g/b)
    if ("red" in v.dtype.names) and ("green" in v.dtype.names) and ("blue" in v.dtype.names):
        rgb = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.uint8, copy=False)
        point_data["rgb"] = rgb
    elif ("r" in v.dtype.names) and ("g" in v.dtype.names) and ("b" in v.dtype.names):
        rgb = np.column_stack([v["r"], v["g"], v["b"]]).astype(np.uint8, copy=False)
        point_data["rgb"] = rgb

    # --- normals (optional)
    if ("nx" in v.dtype.names) and ("ny" in v.dtype.names) and ("nz" in v.dtype.names):
        nrm = np.column_stack([v["nx"], v["ny"], v["nz"]]).astype(np.float32, copy=False)
        point_data["normals"] = nrm

    # --- faces
    if "face" not in ply:
        # No faces: return point cloud mesh
        return meshio.Mesh(points=points, cells=[("triangle", np.zeros((0, 3), dtype=np.int64))], point_data=point_data)

    f = ply["face"].data

    # Face indices field name can be 'vertex_indices' or 'vertex_index'
    if "vertex_indices" in f.dtype.names:
        inds = f["vertex_indices"]
    elif "vertex_index" in f.dtype.names:
        inds = f["vertex_index"]
    else:
        raise ValueError(f"Unsupported PLY face format in {path.name}: {f.dtype.names}")

    # Triangulate if needed (fan triangulation for polygons)
    tris = []
    for poly in inds:
        poly = np.asarray(poly, dtype=np.int64)
        if poly.size < 3:
            continue
        if poly.size == 3:
            tris.append(poly)
        else:
            # Fan triangulation: (0,i,i+1)
            v0 = poly[0]
            for i in range(1, poly.size - 1):
                tris.append(np.array([v0, poly[i], poly[i + 1]], dtype=np.int64))

    tri = np.vstack(tris) if tris else np.zeros((0, 3), dtype=np.int64)

    return meshio.Mesh(points=points, cells=[("triangle", tri)], point_data=point_data)


def _meshio_read_ply(path: Path) -> meshio.Mesh:
    """
    Safe reader:
    - Try meshio first (fast when it works)
    - Fallback to plyfile-based reader for WaSuRe PLY variants
    """
    try:
        return meshio.read(str(path))
    except Exception:
        return _read_ply_with_plyfile_as_meshio(path)


# def _meshio_write_ply(path: Path, mesh: meshio.Mesh, binary: bool) -> None:
#     file_format = "ply"
#     # meshio chooses ascii/binary via "binary" argument for some formats,
#     # but for PLY it uses "binary" inside "file_format" options depending on version.
#     # We handle both patterns.
#     try:
#         meshio.write(str(path), mesh, file_format=file_format, binary=binary)
#     except TypeError:
#         # Older meshio versions
#         meshio.write(str(path), mesh, file_format=file_format)


def _mesh_bbox(mesh: meshio.Mesh) -> Tuple[float, float, float, float, float, float]:
    pts = mesh.points
    xmin, ymin, zmin = np.min(pts, axis=0)
    xmax, ymax, zmax = np.max(pts, axis=0)
    return (float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax))


def _subset_mesh_faces(mesh: meshio.Mesh, keep_faces: np.ndarray) -> meshio.Mesh:
    """
    Keep only selected faces and remap vertices to a compact indexing.
    Preserves point_data (e.g., colors) for remaining vertices.
    """
    # Find triangles cell block
    tri_cells = None
    other_cells = []
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            tri_cells = cell_block.data
        else:
            other_cells.append(cell_block)

    if tri_cells is None:
        raise ValueError("No triangle cells found in mesh.")

    tri_kept = tri_cells[keep_faces]
    if tri_kept.size == 0:
        # Return an empty mesh with same point_data keys
        empty_points = np.zeros((0, 3), dtype=np.float64)
        empty_cells = [("triangle", np.zeros((0, 3), dtype=np.int64))]
        empty_point_data = {k: np.zeros((0,), dtype=v.dtype) if v.ndim == 1 else np.zeros((0, v.shape[1]), dtype=v.dtype)
                            for k, v in (mesh.point_data or {}).items()}
        return meshio.Mesh(points=empty_points, cells=empty_cells, point_data=empty_point_data)

    used = np.unique(tri_kept.reshape(-1))
    new_index = -np.ones(mesh.points.shape[0], dtype=np.int64)
    new_index[used] = np.arange(used.size, dtype=np.int64)
    tri_remap = new_index[tri_kept]

    new_points = mesh.points[used]

    new_point_data = {}
    if mesh.point_data:
        for k, v in mesh.point_data.items():
            new_point_data[k] = v[used]

    # Only triangles are kept; if you need other primitives, we can extend later.
    return meshio.Mesh(points=new_points, cells=[("triangle", tri_remap)], point_data=new_point_data)


def _concat_meshes(meshes: Iterable[meshio.Mesh]) -> meshio.Mesh:
    """
    Concatenate triangle meshes without deduplicating vertices.
    Preserves point_data keys common to all meshes.
    """
    meshes = list(meshes)
    if not meshes:
        return meshio.Mesh(points=np.zeros((0, 3), dtype=np.float64),
                           cells=[("triangle", np.zeros((0, 3), dtype=np.int64))])

    # Determine common point_data keys (e.g., colors)
    common_keys = None
    for m in meshes:
        keys = set((m.point_data or {}).keys())
        common_keys = keys if common_keys is None else (common_keys & keys)
    common_keys = common_keys or set()

    points_all = []
    tri_all = []
    pdata_all = {k: [] for k in common_keys}

    offset = 0
    for m in meshes:
        # Extract triangles
        tri = None
        for cb in m.cells:
            if cb.type == "triangle":
                tri = cb.data
                break
        if tri is None:
            continue

        pts = m.points
        points_all.append(pts)
        tri_all.append(tri + offset)

        for k in common_keys:
            pdata_all[k].append(m.point_data[k])

        offset += pts.shape[0]

    if not points_all:
        return meshio.Mesh(points=np.zeros((0, 3), dtype=np.float64),
                           cells=[("triangle", np.zeros((0, 3), dtype=np.int64))])

    points_cat = np.vstack(points_all)
    tri_cat = np.vstack(tri_all)

    point_data_cat = {}
    for k, chunks in pdata_all.items():
        point_data_cat[k] = np.concatenate(chunks, axis=0)

    return meshio.Mesh(points=points_cat, cells=[("triangle", tri_cat)], point_data=point_data_cat)

def _extract_km_index_from_mns_name(stem: str) -> str:
    """
    Extract km tile index from IGN LiDAR MNS filename.

    Example:
        LHD_FXX_0606_6933_MNS_O_0M50_LAMB93_IGN69
        -> 0606_6933
    """
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected MNS filename format: {stem}")

    # IGN pattern: LHD_FXX_XXXX_YYYY_...
    x_km = parts[2]
    y_km = parts[3]

    if not (x_km.isdigit() and y_km.isdigit()):
        raise ValueError(f"Cannot extract km index from: {stem}")

    return f"{x_km}_{y_km}"


# -----------------------------
# Main function
# -----------------------------

def run_post_wasure_make_km_tiles(
    ply_color_dir: Path,
    logger: logging.Logger,
    cfg: KmTilesConfig,
) -> Path:
    """
    Build 1 km PLY tiles from colored WaSuRe PLY chunks.

    Parameters
    ----------
    ply_color_dir
        Directory containing many colored PLY chunks (EPSG:2154).
    logger
        Logger instance.
    cfg
        KmTilesConfig.

    Returns
    -------
    Path
        Output directory containing one PLY per DSM tile.
    """
    ply_color_dir = Path(ply_color_dir)
    dsm_dir = Path(cfg.dsm_dir)

    out_dir = ply_color_dir.parent / cfg.out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    dsm_paths = sorted(dsm_dir.glob(cfg.dsm_glob))
    if not dsm_paths:
        raise FileNotFoundError(f"No DSM tiles found in {dsm_dir} with glob={cfg.dsm_glob}")

    ply_paths = sorted(ply_color_dir.glob("*.ply"))
    if not ply_paths:
        raise FileNotFoundError(f"No PLY files found in {ply_color_dir}")

    # Precompute bbox of each chunk PLY (fast pass)
    logger.info("Precomputing PLY chunk bboxes (%d files)...", len(ply_paths))
    chunk_info = []
    for p in ply_paths:
        pts = _read_ply_points_only(p)
        xmin, ymin, zmin = np.min(pts, axis=0)
        xmax, ymax, zmax = np.max(pts, axis=0)
        bb = (float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax))
        chunk_info.append((p, bb))

    logger.info("Building 1 km tiles from %d DSM tiles...", len(dsm_paths))

    for dsm_path in dsm_paths:
        stem = dsm_path.stem
        km_index = _extract_km_index_from_mns_name(stem)
        out_ply = out_dir / f"{km_index}_mesh_km_{cfg.suffix}.ply"
        tile_id = km_index  # for logging


        if out_ply.exists() and not cfg.overwrite:
            logger.info("Skip (exists): %s", out_ply)
            continue

        with rasterio.open(dsm_path) as ds:
            b = ds.bounds  # left, bottom, right, top
            # DSM tiles define XY bbox; Z range is unknown, so use a wide Z box.
            # We use +/- 1e9 to ensure Z doesn't clip anything.
            tile_aabb = (float(b.left), float(b.right), float(b.bottom), float(b.top), -1e9, 1e9)

        # Expand bbox for selection + numeric epsilon
        select_aabb = _bbox_expand(tile_aabb, cfg.preselect_buffer_m)
        test_aabb = _bbox_expand(tile_aabb, cfg.eps_m)

        # Select candidate chunks
        candidates = [p for (p, bb) in chunk_info if _bbox_intersects(bb, select_aabb)]
        if not candidates:
            logger.warning("No candidate chunks for tile %s", tile_id)
            # Write an empty mesh to keep downstream predictable
            empty = meshio.Mesh(points=np.zeros((0, 3), dtype=np.float64),
                                cells=[("triangle", np.zeros((0, 3), dtype=np.int64))])
            _write_ply_with_plyfile(out_ply, out_mesh, binary=cfg.binary_out)
            continue

        logger.info("Tile %s: %d candidate chunks", tile_id, len(candidates))

        # Read and concatenate candidates
        meshes = []
        for p in candidates:
            meshes.append(_meshio_read_ply(p))

        big = _concat_meshes(meshes)
        if big.points.shape[0] == 0:
            logger.warning("Tile %s: empty concatenated mesh.", tile_id)
            _write_ply_with_plyfile(out_ply, out_mesh, binary=cfg.binary_out)

            continue

        # Filter faces conservatively
        tri = None
        for cb in big.cells:
            if cb.type == "triangle":
                tri = cb.data
                break
        if tri is None or tri.size == 0:
            logger.warning("Tile %s: no triangles.", tile_id)
            _write_ply_with_plyfile(out_ply, out_mesh, binary=cfg.binary_out)
            continue

        keep = _tri_keep_mask(big.points, tri, test_aabb)
        kept_count = int(np.count_nonzero(keep))
        logger.info("Tile %s: keep %d / %d triangles", tile_id, kept_count, tri.shape[0])

        out_mesh = _subset_mesh_faces(big, keep)
        _write_ply_with_plyfile(out_ply, out_mesh, binary=cfg.binary_out)

    logger.info("Done. Output dir: %s", out_dir)
    return out_dir
