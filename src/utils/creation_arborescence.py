# -*- coding: utf-8 -*-
"""
Création de l'arborescence standard du projet de mise à jour de mesh.

Ce module définit la structure de répertoires utilisée par le pipeline
de mise à jour de mesh (fusion LiDAR / MNS corrélation / WaSuRe).

Il fournit :
    - une structure de données immuable regroupant tous les chemins utiles,
    - une fonction permettant de créer automatiquement l'arborescence
      du projet si elle n'existe pas.

Auteur : ESaint-Denis
"""

# utils/creation_arborescence.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """
    Structure immuable regroupant l'ensemble des chemins du projet.

    Attributs
    ---------
    root : Path
        Répertoire racine du projet.

    nuage_points_lidar : Path
        Dossier contenant les nuages de points LiDAR d'origine.

    mns_lidar : Path
        Dossier contenant les MNS issus du LiDAR.

    mns_correlation : Path
        Dossier contenant les MNS issus de la corrélation d'images.

    mns_recale : Path
        Dossier contenant les MNS après recalage altimétrique.

    masque : Path
        Dossier contenant les masques de changement (zones modifiées).

    nuage_combine : Path
        Dossier contenant les nuages fusionnés (LiDAR + MNS corrélation).

    out_wasure : Path
        Dossier de sortie des traitements WaSuRe (reconstruction mesh).

    tmp : Path
        Dossier temporaire pour les fichiers intermédiaires.

    logs : Path
        Dossier contenant les fichiers journaux du pipeline.
    """
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
    Crée (si nécessaire) l'arborescence standard du projet de mise à jour de mesh.

    Cette fonction :
        1. Résout le chemin racine du projet.
        2. Construit l'ensemble des sous-répertoires nécessaires au pipeline.
        3. Crée ces dossiers sur le disque de manière idempotente
           (aucune erreur si le dossier existe déjà).

    Parameters
    ----------
    project_dir : str | Path
        Chemin du répertoire racine choisi pour le projet.

    Returns
    -------
    ProjectPaths
        Objet regroupant tous les chemins structurés du projet.

    Notes
    -----
    - Tous les dossiers sont créés avec `parents=True` et `exist_ok=True`,
      ce qui garantit un comportement sûr en cas d'exécution répétée.
    - Aucun fichier existant n'est modifié.
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