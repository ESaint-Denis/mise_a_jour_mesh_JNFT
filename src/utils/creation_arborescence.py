# -*- coding: utf-8 -*-
"""
Création de l'arborescence du projet mise à jour de mesh

@author: ESaint-Denis
"""

# utils/creation_arborescence.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    nuage_points_lidar: Path
    mns_lidar: Path
    mns_correlation: Path
    mns_recale: Path
    masque: Path
    nuage_combine: Path
    out_wasure: Path
    tmp: Path
    logs: Path


def create_project_tree(project_dir: str | Path) -> ProjectPaths:
    """
    Create (if missing) the standard directory tree for the mesh update pipeline.

    Parameters
    ----------
    project_dir
        Project root directory chosen by the user.

    Returns
    -------
    ProjectPaths
        Convenient object holding all key project paths.
    """
    root = Path(project_dir).expanduser().resolve()

    paths = ProjectPaths(
        root=root,
        nuage_points_lidar=root / "nuage_points_lidar",
        mns_lidar=root / "MNS_lidar",
        mns_correlation=root / "MNS_correlation",
        mns_recale=root / "MNS_recale",
        masque=root / "masque",
        nuage_combine=root / "nuage_combine",
        out_wasure=root / "out_WASURE",
        tmp=root / "tmp",
        logs=root / "logs",
    )

    # Create all directories (idempotent).
    for p in paths.__dict__.values():
        if isinstance(p, Path):
            p.mkdir(parents=True, exist_ok=True)

    return paths
