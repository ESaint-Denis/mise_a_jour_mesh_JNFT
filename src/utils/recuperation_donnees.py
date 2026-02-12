# -*- coding: utf-8 -*-
"""
Script de récupération des données nécessaires aux traitements

@author: ESaint-Denis
"""

# utils/recuperation_donnees.py
from __future__ import annotations

import json
import logging
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
import os
import subprocess
import urllib.error


from utils.creation_arborescence import ProjectPaths


_TILE_RE = re.compile(r"_(\d{4})_(\d{4})_")


class MissingCorrelationDSMError(RuntimeError):
    """Erreur levée quand au moins une dalle MNS corrélation manque sur le store (bloquant)."""


class MissingLidarDSMError(RuntimeError):
    """Erreur levée quand au moins une dalle MNS LiDAR HD ne peut pas être récupérée (optionnellement bloquant)."""


@dataclass(frozen=True)
class RetrievalConfig:
    # MNS corrélation (store interne)
    mns_year_yyyy: int           # ex: 2025
    departement: str             # ex: "76"
    store_root: Path = Path(r"\\store.ign.fr\store-REF\produits\modeles-numeriques-3D\MNS\MNS_CORREL_1-0_TIFF")

    # MNS LiDAR HD via WFS (annuaire) + WMS-R (téléchargement)
    wfs_endpoint: str = "https://data.geopf.fr/wfs/ows"
    wfs_type_name: str = "IGNF_MNS-LIDAR-HD:dalle"
    wfs_srs: str = "EPSG:2154"


def setup_logger(logs_dir: Path, name: str = "recuperation_donnees") -> logging.Logger:
    """Configure un logger avec sortie fichier horodatée + console."""
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = logs_dir / f"{name}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Log file: %s", log_file)
    return logger

def _log_proxy_env(logger) -> None:
    """Log proxy-related environment variables (useful on servers)."""
    keys = ["http_proxy", "https_proxy", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "all_proxy", "ALL_PROXY"]
    vals = {k: os.environ.get(k) for k in keys if os.environ.get(k)}
    if vals:
        logger.info("Proxy env détecté: %s", vals)
    else:
        logger.info("Aucun proxy env détecté (http_proxy/https_proxy/no_proxy).")


def _download_with_curl(url: str, dst_path: Path, logger, timeout_s: int = 300, retries: int = 5) -> None:
    """
    Download using curl (often more robust behind corporate proxies).
    Uses resume (-C -) and follows redirects (-L).
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "curl",
        "-L",                 # follow redirects
        "-f",                 # fail on HTTP errors
        "--retry", str(retries),
        "--retry-delay", "3",
        "--connect-timeout", "20",
        "--max-time", str(timeout_s),
        "-A", "mise_a_jour_mesh/1.0",
        "-C", "-",            # resume if partial exists
        "-o", str(dst_path),
        url,
    ]

    logger.info("Téléchargement (curl): %s", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.returncode != 0:
        stderr = (p.stderr or "").strip()
        if stderr:
            logger.error("curl stderr: %s", stderr)
        raise RuntimeError(f"Echec curl (code={p.returncode}) pour {url}")


def read_lidar_urls(txt_path: str | Path) -> list[str]:
    """Lit un fichier texte contenant 1 URL LiDAR par ligne (ignore lignes vides et commentaires '#')."""
    p = Path(txt_path).expanduser().resolve()
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()

    urls: list[str] = []
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        urls.append(s)
    return urls


def _filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name:
        raise ValueError(f"Impossible d'extraire le nom de fichier depuis l'URL: {url}")
    return name


def _parse_tile_xy_from_lidar_filename(lidar_filename: str) -> tuple[int, int]:
    """
    Extrait (x_km, y_km) depuis un nom du style:
    LHD_FXX_0605_6933_PTS_LAMB93_IGN69.copc.laz
    """
    m = _TILE_RE.search(lidar_filename)
    if not m:
        raise ValueError(f"Impossible d'extraire (x_km, y_km) depuis: {lidar_filename}")
    x_km = int(m.group(1))
    y_km = int(m.group(2))
    return x_km, y_km


def _km_to_m_str(km: int) -> str:
    """Convertit un index km sur 4 chiffres en coordonnée mètres sur 7 chiffres. Ex: 0605 -> 0605000"""
    m = km * 1000
    return str(m).zfill(7)


# -----------------------------
# MNS corrélation (store interne)
# -----------------------------
def build_mns_correlation_folder(cfg: RetrievalConfig) -> Path:
    """
    Construit:
    \\store...\MNS_CORREL_1-0_TIFF\LAMB93_{YYYY}\MNS_TIFF_RGF93LAMB93_{YY}FD{DEP}20\data
    """
    yyyy = int(cfg.mns_year_yyyy)
    yy = yyyy % 100
    dep = str(cfg.departement)
    lot = f"{yy:02d}FD{dep}20"
    return cfg.store_root / f"LAMB93_{yyyy}" / f"MNS_TIFF_RGF93LAMB93_{lot}" / "data"


def build_mns_correlation_filename(cfg: RetrievalConfig, x_km: int, y_km: int) -> str:
    """Construit le nom de dalle MNS corrélation attendu à partir de (x_km, y_km)."""
    yyyy = int(cfg.mns_year_yyyy)
    yy = yyyy % 100
    dep = str(cfg.departement)
    lot = f"{yy:02d}FD{dep}20"

    x_m = _km_to_m_str(x_km)
    y_m = _km_to_m_str(y_km)
    return f"MNS_CORREL_1-0_LAMB93_{lot}_{x_m}_{y_m}.tif"


# -----------------------------
# MNS LiDAR HD (WFS -> url WMS-R)
# -----------------------------
def _http_get_json(url: str, timeout_s: int = 60) -> dict[str, Any]:
    """GET HTTP et parse JSON."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8", errors="replace"))


def _find_url_property(properties: dict[str, Any]) -> str | None:
    """Récupère la propriété 'url' quelle que soit la casse éventuelle."""
    for k, v in properties.items():
        if isinstance(k, str) and k.lower() == "url":
            return str(v) if v is not None else None
    return None


def get_mns_lidar_download_url_from_wfs(cfg: RetrievalConfig, x_km: int, y_km: int, logger: logging.Logger) -> str:
    """
    Interroge le WFS sur une petite BBOX au centre de la dalle km, et récupère l'attribut 'url'.

    Hypothèse de repère:
    - x_km -> x0 = x_km * 1000 est le bord Ouest (mètres).
    - y_km -> y_top = y_km * 1000 est le bord Nord (mètres),
      et la dalle fait 1000 m vers le Sud.
    """
    x0 = x_km * 1000
    y_top = y_km * 1000

    # Point au centre de la dalle
    xc = x0 + 500.0
    yc = y_top - 500.0

    # Petite BBOX de 2m x 2m autour du centre
    eps = 1.0
    bbox = f"{xc-eps},{yc-eps},{xc+eps},{yc+eps},{cfg.wfs_srs}"

    # Requête WFS GetFeature (GeoJSON)
    # Important: outputFormat en JSON, et restriction par BBOX.
    wfs_url = (
        f"{cfg.wfs_endpoint}"
        f"?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
        f"&TYPENAMES={cfg.wfs_type_name}"
        f"&SRSNAME={cfg.wfs_srs}"
        f"&BBOX={bbox}"
        f"&OUTPUTFORMAT=application%2Fjson"
    )

    logger.info("WFS query for MNS LiDAR HD: %s", wfs_url)

    js = _http_get_json(wfs_url, timeout_s=60)

    feats = js.get("features", [])
    if not feats:
        raise RuntimeError(f"Aucune feature WFS trouvée pour la dalle {x_km}_{y_km} (bbox centre).")

    # Si plusieurs, on prend la première (la bbox est minuscule, ça devrait être unique)
    props = feats[0].get("properties", {})
    if not isinstance(props, dict):
        raise RuntimeError("Réponse WFS inattendue: 'properties' invalide.")

    url = _find_url_property(props)
    if not url:
        raise RuntimeError("La feature WFS ne contient pas d'attribut 'url' exploitable.")

    return url


def _filename_from_wmsr_url(wms_url: str, fallback: str) -> str:
    """
    Extrait le paramètre FILENAME=... depuis l'URL WMS-R si présent.
    Sinon retourne fallback.
    """
    parsed = urlparse(wms_url)
    qs = parse_qs(parsed.query)
    fn = qs.get("FILENAME") or qs.get("filename")
    if fn and len(fn) > 0 and fn[0]:
        return Path(fn[0]).name
    return fallback


# -----------------------------
# IO utilitaires (download/copy)
# -----------------------------
def download_file(
    url: str,
    dst_path: Path,
    tmp_dir: Path,
    logger,
    timeout_s: int = 300,
    overwrite: bool = False,
    use_curl_fallback: bool = True,
) -> None:
    """
    Télécharge un fichier URL -> dst_path, avec:
    - logs détaillés
    - écriture via fichier .part dans tmp_dir (puis rename atomique)
    - retries en cas d'erreurs transitoires (502/503/504, timeouts)
    - fallback curl (optionnel) si urllib échoue (utile en environnement proxy)

    Parameters
    ----------
    url : str
        URL du fichier à télécharger.
    dst_path : Path
        Chemin final.
    tmp_dir : Path
        Dossier temporaire pour le fichier .part.
    logger :
        Logger (logging.Logger).
    timeout_s : int
        Timeout réseau pour urllib/curl.
    overwrite : bool
        Si False et dst existe => on skip.
    use_curl_fallback : bool
        Si True => fallback curl si urllib échoue.
    """
    dst_path = Path(dst_path)
    tmp_dir = Path(tmp_dir)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if dst_path.exists() and not overwrite and dst_path.stat().st_size > 0:
        logger.info("Déjà présent, skip: %s", dst_path)
        return

    _log_proxy_env(logger)

    part_path = tmp_dir / (dst_path.name + ".part")

    # Nettoyage éventuel d'un vieux .part (si taille 0, ou si overwrite)
    try:
        if part_path.exists() and (overwrite or part_path.stat().st_size == 0):
            part_path.unlink()
    except Exception as e:
        logger.warning("Impossible de supprimer l'ancien .part (%s): %s", part_path, e)

    last_err = None

    # --- tentative urllib (3 essais) ---
    for attempt in range(1, 4):
        try:
            logger.info("Téléchargement (urllib) tentative %d/3: %s", attempt, url)

            req = Request(url, headers={"User-Agent": "mise_a_jour_mesh/1.0"})

            with urlopen(req, timeout=timeout_s) as resp:
                status = getattr(resp, "status", None)
                if status is not None:
                    logger.info("HTTP status: %s", status)

                # Écriture stream -> .part
                with open(part_path, "wb") as f:
                    shutil.copyfileobj(resp, f)

            # Vérif basique
            if not part_path.exists() or part_path.stat().st_size == 0:
                raise RuntimeError(f"Téléchargement vide (0 octet) pour {url}")

            # Move atomique vers destination
            part_path.replace(dst_path)

            logger.info("OK téléchargé: %s (%.1f MB)", dst_path, dst_path.stat().st_size / (1024 * 1024))
            return

        except urllib.error.HTTPError as e:
            last_err = e
            logger.warning("HTTPError tentative %d/3: %s", attempt, e)

            # Erreurs souvent transitoires côté proxy/gateway
            if e.code in (502, 503, 504):
                time.sleep(2 * attempt)
                continue

            # autres codes => pas de retry agressif
            break

        except Exception as e:
            last_err = e
            logger.warning("Erreur tentative %d/3: %s", attempt, e)
            time.sleep(2 * attempt)

    # --- fallback curl ---
    if use_curl_fallback:
        logger.warning("urllib en échec (%s). Fallback curl pour: %s", last_err, url)
        _download_with_curl(url, dst_path, logger, timeout_s=timeout_s, retries=5)

        if not dst_path.exists() or dst_path.stat().st_size == 0:
            raise RuntimeError(f"Fallback curl a produit un fichier vide pour {url}")

        logger.info("OK téléchargé (curl): %s (%.1f MB)", dst_path, dst_path.stat().st_size / (1024 * 1024))
        return

    raise RuntimeError(f"Echec téléchargement: {url}: {last_err}") from last_err

def copy_if_needed(src: Path, dst: Path, logger: logging.Logger) -> None:
    """Copie src -> dst si dst n'existe pas."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        logger.info("Déjà présent, skip: %s", dst.name)
        return

    shutil.copy2(src, dst)
    logger.info("Copie OK: %s", dst.name)


# -----------------------------
# Orchestration
# -----------------------------
def run_retrieval(
    paths: ProjectPaths,
    lidar_urls_txt: str | Path,
    cfg: RetrievalConfig,
    *,
    strict_missing_mns_correlation: bool = True,
    fetch_mns_lidar: bool = True,
    strict_missing_mns_lidar: bool = False,
) -> None:
    """
    Récupère:
    - Nuage points LiDAR (download depuis URLs)
    - MNS corrélation (copy depuis store interne) -> bloquant si manquant (selon strict_missing_mns_correlation)
    - MNS LiDAR HD (optionnel) via WFS->url (WMS-R) -> téléchargement GeoTIFF

    Paramètres
    ----------
    strict_missing_mns_correlation
        Si True: manque d'une dalle MNS corrélation = arrêt immédiat (bloquant).
    fetch_mns_lidar
        Si True: tente de récupérer les dalles MNS LiDAR HD via WFS.
    strict_missing_mns_lidar
        Si True: manque MNS LiDAR HD = arrêt (sinon warning).
    """
    logger = setup_logger(paths.logs)

    logger.info("Racine projet: %s", paths.root)
    logger.info("Fichier URLs LiDAR: %s", Path(lidar_urls_txt).resolve())
    logger.info("Année MNS corrélation: %s", cfg.mns_year_yyyy)
    logger.info("Département: %s", cfg.departement)
    logger.info("Fetch MNS LiDAR HD via WFS: %s", fetch_mns_lidar)

    urls = read_lidar_urls(lidar_urls_txt)
    logger.info("URLs lues: %d", len(urls))

    mns_corr_folder = build_mns_correlation_folder(cfg)
    logger.info("Dossier MNS corrélation (source): %s", mns_corr_folder)

    for i, url in enumerate(urls, start=1):
        logger.info("---- Dalle %d / %d ----", i, len(urls))

        lidar_name = _filename_from_url(url)
        lidar_dst = paths.nuage_points_lidar / lidar_name

        # 1) Téléchargement LiDAR
        try:
            download_file(url, lidar_dst, paths.tmp, logger, timeout_s=300)
        except Exception as e:
            logger.error("Erreur téléchargement LiDAR: %s", e)
            raise

        # 2) Dalle MNS corrélation correspondante
        x_km, y_km = _parse_tile_xy_from_lidar_filename(lidar_name)
        corr_name = build_mns_correlation_filename(cfg, x_km, y_km)
        corr_src = mns_corr_folder / corr_name
        corr_dst = paths.mns_correlation / corr_name

        if not corr_src.exists():
            msg = f"MNS corrélation manquant (store): {corr_src}"
            logger.error(msg)
            if strict_missing_mns_correlation:
                raise MissingCorrelationDSMError(msg)
            else:
                continue

        try:
            copy_if_needed(corr_src, corr_dst, logger)
        except Exception as e:
            logger.error("Erreur copie MNS corrélation %s: %s", corr_name, e)
            raise

        # 3) MNS LiDAR HD via WFS (optionnel)
        if fetch_mns_lidar:
            try:
                wms_url = get_mns_lidar_download_url_from_wfs(cfg, x_km, y_km, logger)
                fallback_name = f"LHD_FXX_{x_km:04d}_{y_km:04d}_MNS_LIDARHD.tif"
                mns_lidar_name = _filename_from_wmsr_url(wms_url, fallback=fallback_name)
                mns_lidar_dst = paths.mns_lidar / mns_lidar_name

                download_file(wms_url, mns_lidar_dst, paths.tmp, logger, timeout_s=300)

            except Exception as e:
                msg = f"Echec récupération MNS LiDAR HD pour dalle {x_km:04d}_{y_km:04d}: {e}"
                if strict_missing_mns_lidar:
                    logger.error(msg)
                    raise MissingLidarDSMError(msg)
                else:
                    logger.warning(msg)

    logger.info("Récupération terminée.")
