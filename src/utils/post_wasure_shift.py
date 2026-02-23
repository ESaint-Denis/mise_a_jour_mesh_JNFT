# -*- coding: utf-8 -*-
"""
Post-processing step after WaSuRe: translate PLY tiles (shift or unshift).

This module:
- reads the shift from wasure_metadata_3d_gen.xml (<bbox_ori>),
- applies it (+shift or -shift) to all PLY tiles (ASCII or binary),
- writes translated PLY files into <wasure_run_dir>/<out_subdir>.

Notes:
- This only updates vertex coordinates (and optional extra coordinate fields).
- It does NOT update any custom header comments such as "comment bbox ...".
  This is intentional: for WaSuRe mesh23dtile, keeping local "comment bbox" untouched
  is usually desirable when working with local-coordinate PLY tiles.

All code comments are in English.
"""

from __future__ import annotations

import glob
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from plyfile import PlyData, PlyElement, PlyElementParseError


# ----------------------------- XML shift extraction -----------------------------


def get_shift_from_xml(xml_file: str | Path) -> list[float]:
    """
    Extract the shift (Xmin, Ymin, Zmin) from the <bbox_ori> tag of an XML file.

    Notes
    -----
    The <bbox_ori> tag is expected to contain a string of the form:
    'xminxXmax:yminxYmax:zminxZmax'
    Example:
    '916000.0x917000.0:6457000.0x6458000.0:118.93x10000.0'
    """
    xml_file = str(xml_file)
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bbox_ori = root.find(".//bbox_ori")
    if bbox_ori is None or not bbox_ori.text:
        raise RuntimeError("Missing or empty <bbox_ori> tag in XML.")

    fields = bbox_ori.text.strip().split(":")
    xmin = float(fields[0].split("x")[0])
    ymin = float(fields[1].split("x")[0])
    zmin = float(fields[2].split("x")[0])
    return [xmin, ymin, zmin]


# ----------------------------- PLY helpers -----------------------------
def fix_bbox_comment_trailing_space(ply_path: str | Path) -> bool:
    """
    Ensure the header line 'comment bbox ...' ends with a trailing space before newline.

    This is required for WaSuRe mesh23dtile.py which parses the bbox line with
    a split logic that expects an extra trailing separator token.

    Returns
    -------
    bool
        True if the file was modified, False otherwise.
    """
    ply_path = Path(ply_path)

    data = ply_path.read_bytes()

    # Support both LF and CRLF
    end_token_lf = b"end_header\n"
    end_token_crlf = b"end_header\r\n"

    if end_token_lf in data:
        end_pos = data.index(end_token_lf) + len(end_token_lf)
        nl = b"\n"
    elif end_token_crlf in data:
        end_pos = data.index(end_token_crlf) + len(end_token_crlf)
        nl = b"\r\n"
    else:
        # Not a valid PLY header
        return False

    header = data[:end_pos]
    body = data[end_pos:]

    lines = header.splitlines(keepends=True)
    modified = False

    for i, line in enumerate(lines):
        # line includes newline
        if line.startswith(b"comment bbox "):
            # Remove newline temporarily
            raw = line[:-len(nl)] if line.endswith(nl) else line
            if not raw.endswith(b" "):
                lines[i] = raw + b" " + nl
                modified = True
            break

    if not modified:
        return False

    ply_path.write_bytes(b"".join(lines) + body)
    return True


def reformat_ply_ascii_flat(in_path: str | Path, out_path: str | Path) -> None:
    """
    Convert a flattened ASCII PLY into a standard one (one vertex/face per line).
    """
    in_path = str(in_path)
    out_path = str(out_path)

    with open(in_path, "r", encoding="utf-8") as fin:
        lines = []
        vertex_count = 0
        face_count = 0
        vertex_props = 0
        parsing_vertex = False

        # Read header
        for line in fin:
            lines.append(line)
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                parsing_vertex = True
                continue
            if line.startswith("element face"):
                face_count = int(line.split()[-1])
                parsing_vertex = False
                continue
            if line.startswith("property") and parsing_vertex:
                vertex_props += 1
            if line.startswith("end_header"):
                break

        # Read remaining tokens
        data = fin.read().split()
        idx = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        # Write header
        for l in lines:
            fout.write(l)

        # Write vertices
        for _ in range(vertex_count):
            vals = data[idx : idx + vertex_props]
            fout.write(" ".join(vals) + "\n")
            idx += vertex_props

        # Write faces
        for _ in range(face_count):
            n_idx = int(data[idx])
            vals = data[idx : idx + 1 + n_idx]
            fout.write(" ".join(vals) + "\n")
            idx += 1 + n_idx

        # Write remaining tokens if any
        if idx < len(data):
            fout.write(" ".join(data[idx:]) + "\n")


def upgrade_vertex_dtype_to_double(vertex_arr: np.ndarray) -> np.ndarray:
    """
    Convert all float32 fields of a structured vertex array to float64.
    """
    new_descr = []
    for name, dtype in vertex_arr.dtype.descr:
        if dtype.startswith("<f4") or dtype.startswith(">f4") or dtype == "float32":
            new_descr.append((name, "f8"))
        else:
            new_descr.append((name, dtype))

    vertex_arr_double = np.zeros(vertex_arr.shape[0], dtype=new_descr)
    for name in vertex_arr.dtype.names:
        vertex_arr_double[name] = vertex_arr[name]
    return vertex_arr_double


def apply_shift_to_ply(
    input_ply: str | Path,
    output_ply: str | Path,
    *,
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    t_coords: tuple[str, ...] = ("x", "y", "z"),
    ascii_out: bool = False,
    xml_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> bool:
    """
    Apply a translation shift to selected vertex properties in a PLY file.

    Parameters
    ----------
    shift
        Translation (dx, dy, dz) to apply.
    t_coords
        Vertex properties to translate (default: x,y,z). If extra fields exist,
        you may include them (e.g., x0,y0,z0 or x_origin,y_origin,z_origin).
    xml_path
        If provided, shift is read from the XML (<bbox_ori>) and overrides `shift`.
        Note: In the orchestrator we typically read XML once and pass shift directly.

    Returns
    -------
    bool
        True if the input was detected as "flattened ASCII" and had to be repaired.
    """
    input_ply = str(input_ply)
    output_ply = str(output_ply)

    repaired_flattened = False

    try:
        plydata = PlyData.read(input_ply)
    except PlyElementParseError as e:
        if "expected end-of-line" in str(e) and "element 'vertex'" in str(e):
            repaired_flattened = True
            if logger:
                # Use DEBUG to avoid spamming the main log
                logger.debug("Flattened ASCII PLY detected, repairing temporarily: %s", input_ply)
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".ply") as tmpf:
                tmp_path = tmpf.name
            reformat_ply_ascii_flat(input_ply, tmp_path)
            plydata = PlyData.read(tmp_path)
            os.remove(tmp_path)
        else:
            raise

    if xml_path is not None:
        sx, sy, sz = get_shift_from_xml(xml_path)
        shift = (sx, sy, sz)
        if logger:
            logger.debug("Shift read from XML: %s", shift)
    else:
        # Fallback to PLY comment
        for comment in getattr(plydata, "comments", []):
            if comment.startswith("IGN offset Pos"):
                parts = comment.split()
                shift = (float(parts[-3]), float(parts[-2]), float(parts[-1]))
                if logger:
                    logger.debug("Shift read from PLY comment: %s", shift)
                break

    if "vertex" not in plydata:
        if logger:
            logger.warning("No vertex element in PLY, skipping: %s", input_ply)
        return repaired_flattened

    vertex_dtype = plydata["vertex"].data.dtype
    vertex_arr = np.array(plydata["vertex"].data.copy())
    vertex_arr = upgrade_vertex_dtype_to_double(vertex_arr)

    for i, axis in enumerate(t_coords):
        if axis in vertex_dtype.names:
            vertex_arr[axis] = vertex_arr[axis] + shift[i % 3]

    plydata.elements = tuple(
        [PlyElement.describe(vertex_arr, "vertex")]
        + [e for e in plydata.elements if e.name != "vertex"]
    )

    plydata.text = ascii_out
    plydata.write(output_ply)
    
    # WaSuRe compatibility: bbox parser expects a trailing space on "comment bbox" line
    try:
        fix_bbox_comment_trailing_space(output_ply)
    except Exception:
        if logger:
            logger.debug("Failed to patch bbox trailing space for: %s", output_ply)


    return repaired_flattened


# ----------------------------- Orchestration -----------------------------


@dataclass(frozen=True)
class PostWasureShiftConfig:
    t_coords: tuple[str, ...] = ("x", "y", "z")
    ascii_out: bool = False
    overwrite: bool = True

    # +1 => shift (local -> global), -1 => unshift (global -> local)
    sign: int = 1

    # Output folder name under the WaSuRe run directory
    out_subdir: str = "ply_L93"


def _find_tiles_dir(wasure_out_dir: Path) -> Path:
    """
    Try common WaSuRe layouts and return the directory containing *.ply tiles.
    """
    candidates = [
        wasure_out_dir / "outputs" / "tiles",
        # wasure_out_dir / "outputs" / "outputs" / "tiles",
        # wasure_out_dir / "tiles",
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("*.ply")):
            return c
    raise RuntimeError(f"Unable to locate a tiles directory with .ply in: {wasure_out_dir}")


def run_post_wasure_shift(
    *,
    wasure_out_dir: str | Path,
    logger: logging.Logger,
    cfg: PostWasureShiftConfig,
    tiles_dir: str | Path | None = None,
) -> Path:
    """
    Translate all PLY tiles.

    Parameters
    ----------
    wasure_out_dir
        WaSuRe run directory (run_*).
    tiles_dir
        Optional input tiles directory. If None, defaults to <run_*/outputs/tiles>.
        Use this to "unshift" already processed tiles (e.g., ply_L93_ortho -> local).
    cfg.sign
        +1 shift, -1 unshift.
    cfg.out_subdir
        Output folder under run_*.

    Returns
    -------
    Path
        Output directory containing translated PLY tiles.
    """
    wasure_out_dir = Path(wasure_out_dir)

    xml_path = wasure_out_dir / "wasure_metadata_3d_gen.xml"
    if not xml_path.is_file():
        raise RuntimeError(f"Missing XML metadata file: {xml_path}")

    if tiles_dir is None:
        in_tiles_dir = _find_tiles_dir(wasure_out_dir)
    else:
        in_tiles_dir = Path(tiles_dir)
        if not in_tiles_dir.is_dir():
            raise RuntimeError(f"Input tiles_dir not found: {in_tiles_dir}")

    out_dir = wasure_out_dir / cfg.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(Path(p) for p in glob.glob(str(in_tiles_dir / "*.ply")))
    if not ply_files:
        raise RuntimeError(f"No .ply files found in: {in_tiles_dir}")

    # Read shift ONCE, then apply sign (+shift or -shift)
    base_shift = tuple(get_shift_from_xml(xml_path))
    shift = (cfg.sign * base_shift[0], cfg.sign * base_shift[1], cfg.sign * base_shift[2])

    action = "shift" if cfg.sign >= 0 else "unshift"
    logger.info(
        "Step 7 - WaSuRe %s | tiles=%d | shift=%s | in=%s | out=%s",
        action, len(ply_files), shift, in_tiles_dir, out_dir
    )

    n_written = 0
    n_skipped = 0
    n_flattened = 0

    for in_ply in ply_files:
        out_ply = out_dir / in_ply.name
        if out_ply.exists() and not cfg.overwrite:
            n_skipped += 1
            continue

        try:
            repaired = apply_shift_to_ply(
                input_ply=in_ply,
                output_ply=out_ply,
                shift=shift,
                t_coords=cfg.t_coords,
                ascii_out=cfg.ascii_out,
                xml_path=None,  # IMPORTANT: do not re-read XML per tile
                logger=logger,
            )
            n_flattened += int(repaired)
            n_written += 1
        except Exception:
            # Keep traceback in main log (useful for debugging one bad tile)
            logger.exception("%s failed for: %s", action, in_ply)
            raise

    logger.info(
        "Step 7 done (%s) | written=%d | skipped=%d | repaired_flattened=%d | out=%s",
        action, n_written, n_skipped, n_flattened, out_dir
    )
    return out_dir
