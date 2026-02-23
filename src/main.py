# -*- coding: utf-8 -*-
"""
Script principal de la mise à jour de mesh LiDAR :
- récupération données
- recalage altimétrique
- création masque
- fusion nuages
- lancement WaSuRe
- création de divers produits (mesh au format ply kilométrique, 3Dtiles ...) colorisé avec l'ortho ou selon l'origine
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from utils.creation_arborescence import create_project_tree
from utils.recuperation_donnees import RetrievalConfig, run_retrieval
from utils.recalage_altimetrique import RecalageConfig, run_recalage_altimetrique
from utils.creation_masque import MaskConfig, run_creation_masque
from utils.fusion_nuages import FusionConfig, run_fusion_nuages
from utils.run_wasure import WaSuReConfig, run_wasure
from utils.read_last_wasure import read_last_wasure_output
from utils.post_wasure_shift import PostWasureShiftConfig, run_post_wasure_shift
from utils.post_wasure_colorize_ortho_wms import OrthoWmsConfig, run_post_wasure_colorize_ortho_wms
from utils.post_wasure_colorize_origin_multitif import OriginColorConfig, run_post_wasure_colorize_origin_multitif
from utils.run_mesh23dtile import Mesh23DTileConfig, run_mesh23dtile
from utils.post_wasure_make_km_tiles import KmTilesConfig, run_post_wasure_make_km_tiles
from utils.infos import build_summary_from_logs_dir


RUN_WASURE = True  # True = relance WaSuRe, False = utilise le dernier run
PROJECT_DIR = r"/media/DATA/MESH_3D/out_mise_a_jour_mesh_JNFT/toulouse"
LIDAR_URLS_TXT = r"/media/DATA/MESH_3D/out_mise_a_jour_mesh_JNFT/toulouse/dalles_toulouse.txt"


def setup_main_logger(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = logs_dir / f"main_{ts}.log"

    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Log principal: %s", log_file)
    return logger


def main() -> None:

    # 1️⃣ Création arborescence
    paths = create_project_tree(PROJECT_DIR)
    logger = setup_main_logger(paths.logs)

    # 2️⃣ Récupération données
    cfg_retrieval = RetrievalConfig(
        store_root=Path("/media/stores/cifs/store-REF/produits/modeles-numeriques-3D/MNS/MNS_CORREL_1-0_TIFF"),
        year_start=time.localtime().tm_year, 
        year_stop=2020,
    )

    run_retrieval(
        paths=paths,
        lidar_urls_txt=LIDAR_URLS_TXT,
        cfg=cfg_retrieval,
        strict_missing_mns_correlation=True,
        fetch_mns_lidar=True,
        strict_missing_mns_lidar=True,
    )

    # 3️⃣ Recalage altimétrique
    cfg_recal = RecalageConfig(
        k_mad=3.0,
        n_iter=3,
        overwrite=True,
    )

    run_recalage_altimetrique(paths=paths, cfg=cfg_recal)

    # 4️⃣ Création masque
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
    
    # 4bis) Build per-tile summary table (from logs) after mask creation
    try:
        summary_csv = build_summary_from_logs_dir(paths.logs, out_dir=paths.logs)
        logger.info("Résumé par tuile écrit: %s", summary_csv)
    except Exception as e:
        logger.error("Impossible de générer le résumé par tuile: %s", e)

    # 5️⃣ Fusion nuages
    cfg_fusion = FusionConfig(
        chunk_size_lidar=5_000_000,
        block_size_dsm=1024,
        overwrite=True,
    )

    run_fusion_nuages(paths=paths, cfg=cfg_fusion)

    # 6️⃣ WaSuRe
    cfg_wasure = WaSuReConfig(
        wasure_repo_dir=Path("/home/data_ssd/esaint-denis/sparkling-wasure"),
        wasure_script="./run_lidarhd.sh",
        extra_args=None,
        fail_if_input_empty=True,
    )

    if RUN_WASURE:
        wasure_out_dir = run_wasure(
            input_dir=paths.nuage_combine,
            out_wasure_root=paths.out_wasure,
            logs_dir=paths.logs,
            logger=logger,
            cfg=cfg_wasure,
        )
    else:
        wasure_out_dir = read_last_wasure_output(paths.out_wasure)
        logger.info("Mode sans WaSuRe - utilisation du dernier run : %s", wasure_out_dir)

    # 7️⃣ Shift des tuiles
    cfg_shift = PostWasureShiftConfig(
        t_coords=("x", "y", "z"),
        ascii_out=False,
        overwrite=True,
        sign=+1,
        out_subdir="ply_L93",
    )

    ply_l93_dir = run_post_wasure_shift(
        wasure_out_dir=wasure_out_dir,
        logger=logger,
        cfg=cfg_shift,
    )


    logger.info("Dossier PLY shiftés (L93) : %s", ply_l93_dir)

    # 8️⃣ Colorisation ortho (WMS)
    cfg_ortho = OrthoWmsConfig(
        # Defaults validated:
        img_format="image/jpeg",
        gsd_m=0.20,
        sampling="nearest",
        bbox_buffer_m=2.0,
        overwrite=True,
    )

    ply_ortho_dir = run_post_wasure_colorize_ortho_wms(
        ply_l93_dir=ply_l93_dir,
        logger=logger,
        cfg=cfg_ortho,
    )

    logger.info("Dossier PLY colorisés ortho : %s", ply_ortho_dir)
    
    # 9️⃣ Colorisation par origine (LiDAR vs MNS) via masque
    cfg_origin = OriginColorConfig(
        swap_meaning=False,
        lidar_rgb=(0, 140, 255),
        mns_rgb=(255, 120, 0),
        default_rgb=(200, 200, 200),
        overwrite=True,
        bbox_buffer_m=2.0,
    )

    ply_origin_dir = run_post_wasure_colorize_origin_multitif(
        ply_l93_dir=ply_l93_dir,
        mask_dir=paths.masque,
        logger=logger,
        cfg=cfg_origin,
    )

    logger.info("Dossier PLY colorisés origine (L93) : %s", ply_origin_dir)
    
    # 9bis: Build 1 km PLY tiles in L93
    cfg_km_ortho = KmTilesConfig(
        dsm_dir=paths.mns_lidar,
        overwrite=True,
        out_dirname="ply_km_tiles_ortho_L93",
        suffix = "ortho" 
    )

    ply_km_ortho_dir = run_post_wasure_make_km_tiles(
        ply_color_dir=ply_ortho_dir,
        logger=logger,
        cfg=cfg_km_ortho,
    )

    cfg_km_origin = KmTilesConfig(
        dsm_dir=paths.mns_lidar,
        overwrite=True,
        out_dirname="ply_km_tiles_origin_L93",
        suffix = "origin"
    )

    ply_km_origin_dir = run_post_wasure_make_km_tiles(
        ply_color_dir=ply_origin_dir,
        logger=logger,
        cfg=cfg_km_origin,
    )

    logger.info("PLY km ortho (L93): %s", ply_km_ortho_dir)
    logger.info("PLY km origin (L93): %s", ply_km_origin_dir)

    # 🔟 Déshift des PLY colorisés (L93 -> local WaSuRe)

    # 🔟 Déshift ortho
    cfg_unshift_ortho = PostWasureShiftConfig(
        t_coords=("x", "y", "z"),
        ascii_out=False,
        overwrite=True,
        sign=-1,
        out_subdir="colorized_tiles_ortho_local",
    )

    ortho_local_dir = run_post_wasure_shift(
        wasure_out_dir=wasure_out_dir,
        tiles_dir=ply_ortho_dir,
        logger=logger,
        cfg=cfg_unshift_ortho,
    )

    # 🔟 Déshift origin
    cfg_unshift_origin = PostWasureShiftConfig(
        t_coords=("x", "y", "z"),
        ascii_out=False,
        overwrite=True,
        sign=-1,
        out_subdir="colorized_tiles_origin_local",
    )

    origin_local_dir = run_post_wasure_shift(
        wasure_out_dir=wasure_out_dir,
        tiles_dir=ply_origin_dir,
        logger=logger,
        cfg=cfg_unshift_origin,
    )

    logger.info("PLY locaux ortho : %s", ortho_local_dir)
    logger.info("PLY locaux origine : %s", origin_local_dir)

    # 1️⃣1️⃣ Génération des 3D Tiles (pipeline mesh23dtile WaSuRe)

    cfg_3dtiles = Mesh23DTileConfig(
        wasure_repo_dir=Path("/home/data_ssd/esaint-denis/sparkling-wasure"),
        num_process=30,
        overwrite=True,
        exec_mode="docker",
        docker_image="ddt_img_base_devel_proxy",
        docker_user=True,
    )


    tileset_ortho_dir = run_mesh23dtile(
        input_dir=ortho_local_dir,
        wasure_run_dir=wasure_out_dir,
        output_dir=wasure_out_dir / "3dtiles_ortho",
        logs_dir=paths.logs,
        logger=logger,
        cfg=cfg_3dtiles,
        tag="ortho",
    )

    tileset_origin_dir = run_mesh23dtile(
        input_dir=origin_local_dir, 
        wasure_run_dir=wasure_out_dir,
        output_dir=wasure_out_dir / "3dtiles_origin",
        logs_dir=paths.logs,
        logger=logger,
        cfg=cfg_3dtiles,
        tag="origin",
    )

    logger.info("Tileset ortho : %s", tileset_ortho_dir)
    logger.info("Tileset origine : %s", tileset_origin_dir)

if __name__ == "__main__":
    main()
