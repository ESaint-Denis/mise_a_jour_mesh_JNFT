# -*- coding: utf-8 -*-
"""
Read the last WaSuRe run information written by run_wasure().
"""

from __future__ import annotations

import json
from pathlib import Path


def read_last_wasure_output(out_wasure_root: str | Path) -> Path:
    """
    Read wasure_last_run.json and return the output_dir of the last run.

    Parameters
    ----------
    out_wasure_root : Path
        Root directory where WaSuRe runs are stored.

    Returns
    -------
    Path
        Path to the last WaSuRe output directory.
    """
    out_wasure_root = Path(out_wasure_root)
    last_path = out_wasure_root / "wasure_last_run.json"

    if not last_path.is_file():
        raise RuntimeError(f"Missing wasure_last_run.json: {last_path}")

    meta = json.loads(last_path.read_text(encoding="utf-8"))

    if "output_dir" not in meta:
        raise RuntimeError(f"Invalid JSON (missing output_dir): {last_path}")

    out_dir = Path(meta["output_dir"])
    if not out_dir.exists():
        raise RuntimeError(f"Stored output_dir does not exist: {out_dir}")

    return out_dir
