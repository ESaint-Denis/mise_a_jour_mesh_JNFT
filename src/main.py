# -*- coding: utf-8 -*-
"""
Script principal de la msise à jour de mesh LiDAR par masque de changement, fusion des nuages de points et création d'un mesh WaSuRe'

@author: ESaint-Denis
"""

# main.py
from __future__ import annotations

from utils.creation_arborescence import create_project_tree
from utils.recuperation_donnees import RetrievalConfig, run_retrieval
from utils.recalage_altimetrique import RecalageConfig, run_recalage_altimetrique
from utils.creation_masque import MaskConfig, run_creation_masque
from utils.fusion_nuages import FusionConfig, run_fusion_nuages

PROJECT_DIR = r"/media/DATA/MESH_3D/out_mise_a_jour_mesh_JNFT/test_dep76"
LIDAR_URLS_TXT = r"/media/DATA/MESH_3D/out_mise_a_jour_mesh_JNFT/test_dep76/dalles_test_76.txt"

MNS_YEAR_YYYY = 2025
DEPARTEMENT = "76"

def main() -> None:
    paths = create_project_tree(PROJECT_DIR)

    cfg_retrieval = RetrievalConfig(
        mns_year_yyyy=MNS_YEAR_YYYY,
        departement=DEPARTEMENT,
    )

    run_retrieval(
        paths=paths,
        lidar_urls_txt=LIDAR_URLS_TXT,
        cfg=cfg_retrieval,
        strict_missing_mns_correlation=True,
        fetch_mns_lidar=True,
        strict_missing_mns_lidar=True,
    )

    cfg_recal = RecalageConfig(
        k_mad=3.0,
        n_iter=3,
        overwrite=True,
    )

    run_recalage_altimetrique(paths=paths, cfg=cfg_recal)
    
    cfg_mask = MaskConfig(
        z_tolerance_m=1.0,
        window_radius=2,
        radius_open=4,
        min_area_m2=16.0,
        buffer_m=2.0,
        buffer_closing=False,
        overwrite=True,
    )
    run_creation_masque(paths=paths, cfg=cfg_mask)
    
    cfg_fusion = FusionConfig(
        chunk_size_lidar=5_000_000,
        block_size_dsm=1024,
        overwrite=True,
    )
    
    run_fusion_nuages(paths=paths, cfg=cfg_fusion)
    

if __name__ == "__main__":
    main()