# -*- coding: utf-8 -*-
"""
Récupération des données nécessaires au pipeline de mise à jour de mesh.

Ce module orchestre l'acquisition des données d'entrée du pipeline :
- Nuages de points LiDAR : téléchargés depuis une liste d'URL (un fichier texte).
- MNS de corrélation : copiés depuis un store interne (arborescence normalisée),
  avec recherche automatique du département via un WFS "départements", puis
  sélection du millésime le plus récent disponible.
- MNS LiDAR HD : optionnellement récupérés via une requête WFS (GeoJSON) permettant
  d'obtenir une URL WMS-R/HTTP de téléchargement du GeoTIFF.

Le module inclut :
- une configuration (RetrievalConfig),
- des utilitaires robustes de téléchargement (urllib + fallback curl),
- des utilitaires de parsing (indices de dalle km depuis les noms de fichiers),
- une fonction principale d'orchestration (run_retrieval).

Auteur : ESaint-Denis
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
import platform
from datetime import datetime


from utils.creation_arborescence import ProjectPaths
from utils.departement_wfs import DepartementWfsConfig, get_departements_for_tile_bbox


_TILE_RE = re.compile(r"_(\d{4})_(\d{4})_")


class MissingCorrelationDSMError(RuntimeError):
    """Erreur levée lorsqu'au moins une dalle MNS de corrélation est manquante sur le store (cas bloquant)."""


class MissingLidarDSMError(RuntimeError):
    """Erreur levée lorsqu'une dalle MNS LiDAR HD ne peut pas être récupérée (cas éventuellement bloquant)."""


@dataclass(frozen=True)
class RetrievalConfig:
    """
    Configuration de la récupération des données.

    Attributs
    ---------
    store_root : Path
        Racine du store interne contenant les dalles MNS de corrélation.

    year_start : int
        Année de départ pour la recherche des MNS de corrélation (par défaut : année courante),
        la recherche se fait ensuite à rebours (year_start, year_start-1, ...).

    year_stop : int
        Année minimale incluse pour la recherche des MNS de corrélation.

    dep_wfs : DepartementWfsConfig
        Configuration du WFS "départements" utilisé pour déterminer les départements
        intersectant une dalle km (utile pour retrouver le bon répertoire de store).

    wfs_endpoint : str
        URL du endpoint WFS utilisé pour interroger la couche des dalles MNS LiDAR HD.

    wfs_type_name : str
        Nom de la couche WFS des dalles MNS LiDAR HD (typeName / typeNames).

    wfs_srs : str
        Système de référence spatiale utilisé dans les requêtes WFS (ex. EPSG:2154).

    Notes
    -----
    - La recherche des MNS de corrélation combine (département via WFS) + (année décroissante)
      pour trouver le millésime le plus récent disponible.
    - La récupération des MNS LiDAR HD repose sur un attribut 'url' renvoyé par le WFS.
    """
    # Correlation DSM store
    store_root: Path

    # Correlation DSM search policy
    year_start: int = datetime.now().year   # search from current year downwards
    year_stop: int = 2020                  # inclusive lower bound (adjust if needed)

    # Department WFS
    dep_wfs: DepartementWfsConfig = DepartementWfsConfig()

    # LiDAR HD DSM via WFS/WMS-R (unchanged)
    wfs_endpoint: str = "https://data.geopf.fr/wfs/ows"
    wfs_type_name: str = "IGNF_MNS-LIDAR-HD:dalle"
    wfs_srs: str = "EPSG:2154"


def setup_logger(logs_dir: Path, name: str = "recuperation_donnees") -> logging.Logger:
    """
    Configure un logger avec :
    - sortie fichier horodatée dans le répertoire de logs,
    - sortie console.

    Parameters
    ----------
    logs_dir : Path
        Répertoire où écrire les logs.
    name : str
        Nom du logger (et préfixe du fichier de log).

    Returns
    -------
    logging.Logger
        Logger prêt à l'emploi.
    """
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
    """
    Journalise les variables d'environnement liées au proxy.

    Utile pour diagnostiquer des problèmes de téléchargement sur serveurs
    ou environnements d'entreprise (proxy HTTP/HTTPS).
    """
    keys = ["http_proxy", "https_proxy", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "all_proxy", "ALL_PROXY"]
    vals = {k: os.environ.get(k) for k in keys if os.environ.get(k)}
    if vals:
        logger.info("Proxy env détecté: %s", vals)
    else:
        logger.info("Aucun proxy env détecté (http_proxy/https_proxy/no_proxy).")


def resolve_store_root(cfg: RetrievalConfig) -> Path:
    """
    Détermine la racine du store des dalles MNS de corrélation.

    Ordre de priorité :
    1) cfg.store_root si fourni
    2) variable d'environnement MNS_CORREL_STORE_ROOT
    3) valeurs par défaut dépendantes de la plateforme (Linux puis UNC Windows)

    Parameters
    ----------
    cfg : RetrievalConfig
        Configuration de récupération.

    Returns
    -------
    Path
        Chemin de la racine du store à utiliser.

    Notes
    -----
    - Si aucun candidat n'existe réellement, la fonction renvoie quand même
      un chemin "le plus probable" selon l'OS, afin d'améliorer les messages
      d'erreur en aval.
    """
    # 1) Explicit config
    if cfg.store_root is not None:
        return Path(cfg.store_root)

    # 2) Environment override
    env = os.environ.get("MNS_CORREL_STORE_ROOT")
    if env:
        return Path(env)

    # 3) Defaults
    candidates: list[Path] = []

    # Linux mount (what your admin fixed)
    candidates.append(Path("/media/stores/cifs/store-REF/produits/modeles-numeriques-3D/MNS/MNS_CORREL_1-0_TIFF"))

    # Windows UNC (works when run on Windows)
    candidates.append(Path(r"\\store.ign.fr\store-REF\produits\modeles-numeriques-3D\MNS\MNS_CORREL_1-0_TIFF"))

    for p in candidates:
        if p.exists():
            return p

    # If nothing exists, return the most likely default for the current OS (helps error messages)
    if platform.system().lower().startswith("win"):
        return candidates[1]
    return candidates[0]


def _download_with_curl(url: str, dst_path: Path, logger, timeout_s: int = 300, retries: int = 5) -> None:
    """
    Télécharge un fichier en utilisant `curl`.

    Cette méthode est souvent plus robuste derrière des proxys (ou en présence
    d'erreurs intermittentes). Elle :
    - suit les redirections (-L),
    - échoue explicitement sur erreurs HTTP (-f),
    - reprend un téléchargement partiel (-C -),
    - réessaie automatiquement (--retry).

    Parameters
    ----------
    url : str
        URL de téléchargement.
    dst_path : Path
        Chemin du fichier de sortie.
    logger :
        Logger.
    timeout_s : int
        Durée maximale de téléchargement.
    retries : int
        Nombre de tentatives côté curl.
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
    """
    Lit un fichier texte contenant une URL LiDAR par ligne.

    Les lignes vides et les lignes commençant par `#` (commentaires) sont ignorées.

    Parameters
    ----------
    txt_path : str | Path
        Chemin du fichier texte.

    Returns
    -------
    list[str]
        Liste des URL lues.
    """
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
    """
    Extrait le nom de fichier depuis une URL.

    Parameters
    ----------
    url : str
        URL d'un fichier (ex. .../mon_fichier.laz).

    Returns
    -------
    str
        Nom du fichier (sans répertoires).

    Raises
    ------
    ValueError
        Si aucun nom n'est extractible depuis l'URL.
    """
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name:
        raise ValueError(f"Impossible d'extraire le nom de fichier depuis l'URL: {url}")
    return name


def _parse_tile_xy_from_lidar_filename(lidar_filename: str) -> tuple[int, int]:
    """
    Extrait les indices de dalle kilométrique (x_km, y_km) depuis un nom LiDAR.

    Exemple de motif attendu (présence de _XXXX_YYYY_) :
    LHD_FXX_0605_6933_PTS_LAMB93_IGN69.copc.laz

    Parameters
    ----------
    lidar_filename : str
        Nom de fichier LiDAR.

    Returns
    -------
    tuple[int, int]
        (x_km, y_km) indices kilométriques extraits.

    Raises
    ------
    ValueError
        Si le motif n'est pas détecté.
    """
    m = _TILE_RE.search(lidar_filename)
    if not m:
        raise ValueError(f"Impossible d'extraire (x_km, y_km) depuis: {lidar_filename}")
    x_km = int(m.group(1))
    y_km = int(m.group(2))
    return x_km, y_km


def _km_to_m_str(km: int) -> str:
    """
    Convertit un index kilométrique (4 chiffres) en coordonnée mètres (7 chiffres).

    Exemple :
    - 0605 -> 0605000

    Parameters
    ----------
    km : int
        Indice kilométrique.

    Returns
    -------
    str
        Coordonnée en mètres formatée sur 7 chiffres.
    """
    m = km * 1000
    return str(m).zfill(7)


# -----------------------------
# MNS corrélation (store interne)
# -----------------------------
def build_mns_correlation_folder(store_root: Path, year_yyyy: int, dep_code_insee: str) -> Path:
    """
    Construit le chemin de dossier du store interne pour les MNS de corrélation.

    Convention attendue :
    <store_root>/LAMB93_<YYYY>/MNS_TIFF_RGF93LAMB93_<YY>FD<DEP>20/data

    Parameters
    ----------
    store_root : Path
        Racine du store.
    year_yyyy : int
        Millésime (année sur 4 chiffres).
    dep_code_insee : str
        Code département (convention store, voir `normalize_dep_code_for_store`).

    Returns
    -------
    Path
        Chemin du dossier `data` contenant les tuiles.
    """
    yyyy = int(year_yyyy)
    yy = yyyy % 100
    dep = str(dep_code_insee)
    lot = f"{yy:02d}FD{dep}20"
    return Path(store_root) / f"LAMB93_{yyyy}" / f"MNS_TIFF_RGF93LAMB93_{lot}" / "data"


def build_mns_correlation_filename(year_yyyy: int, dep_code_insee: str, x_km: int, y_km: int) -> str:
    """
    Construit le nom attendu d'une dalle MNS de corrélation à partir des paramètres.

    Parameters
    ----------
    year_yyyy : int
        Millésime (année sur 4 chiffres).
    dep_code_insee : str
        Code département selon la convention du store.
    x_km, y_km : int
        Indices kilométriques de la tuile.

    Returns
    -------
    str
        Nom de fichier GeoTIFF attendu.
    """
    yyyy = int(year_yyyy)
    yy = yyyy % 100
    dep = str(dep_code_insee)
    lot = f"{yy:02d}FD{dep}20"

    x_m = _km_to_m_str(x_km)
    y_m = _km_to_m_str(y_km)
    return f"MNS_CORREL_1-0_LAMB93_{lot}_{x_m}_{y_m}.tif"


def normalize_dep_code_for_store(dep: str) -> str:
    """
    Normalise un code département pour correspondre à la convention du store MNS corrélation.

    Conventions du store :
    - La Corse est stockée sous la forme '20' (couvre 2A et 2B).
    - Départements métropolitains : 2 chiffres (01..95).
    - Départements d'outre-mer : 3 chiffres (971..976).
    - Certaines sources (WFS) peuvent renvoyer un code à 3 chiffres avec zéro initial (ex. '076').

    Exemples :
    - '076' -> '76'
    - '6'   -> '06'
    - '06'  -> '06'
    - '2A'  -> '20'
    - '2B'  -> '20'
    - '971' -> '971'

    Parameters
    ----------
    dep : str
        Code département en entrée.

    Returns
    -------
    str
        Code normalisé pour le store.
    """
    dep = str(dep).strip().upper()

    # Corsica special case (store uses 20)
    if dep in ("2A", "2B"):
        return "20"

    # Numeric codes
    if dep.isdigit():
        dep2 = dep.lstrip("0") or "0"
        # Overseas (3 digits)
        if len(dep2) >= 3:
            return dep2
        # Metropolitan (2 digits)
        return dep2.zfill(2)

    return dep


def find_latest_mns_correlation_tile(
    cfg: RetrievalConfig,
    x_km: int,
    y_km: int,
) -> Path | None:
    """
    Recherche la dalle MNS de corrélation la plus récente disponible pour une tuile donnée.

    Stratégie :
    1) détermination des départements intersectant la tuile via WFS,
    2) normalisation des codes département selon la convention du store,
    3) parcours des années de cfg.year_start à cfg.year_stop (décroissant),
    4) pour chaque année, test des départements (ordre décroissant d'aire d'intersection),
    5) renvoi du premier chemin existant.

    Parameters
    ----------
    cfg : RetrievalConfig
        Configuration (store + politiques de recherche + WFS départements).
    x_km, y_km : int
        Indices kilométriques de la tuile.

    Returns
    -------
    Path | None
        Chemin du fichier existant si trouvé, sinon None.

    Notes
    -----
    - La fonction tente de déduire le "bon" département si une tuile intersecte plusieurs départements.
    - Les messages [DEBUG] actuels sont laissés tels quels.
    """
    deps = get_departements_for_tile_bbox(x_km, y_km, cfg.dep_wfs)
    dep_codes = [normalize_dep_code_for_store(d) for (d, _area) in deps]
    seen = set()
    dep_codes = [d for d in dep_codes if not (d in seen or seen.add(d))]
    print(f"[DEBUG] tile {x_km:04d}_{y_km:04d} dep_codes={dep_codes}")
    
    for year in range(int(cfg.year_start), int(cfg.year_stop) - 1, -1):
        for dep in dep_codes:
            folder = build_mns_correlation_folder(cfg.store_root, year, dep)
            name = build_mns_correlation_filename(year, dep, x_km, y_km)
            p = folder / name
            print(f"[DEBUG] try year={year} dep={dep} path={p}")
            if p.exists():
                return p

    return None


# -----------------------------
# MNS LiDAR HD (WFS -> url WMS-R)
# -----------------------------
def _http_get_json(url: str, timeout_s: int = 60) -> dict[str, Any]:
    """
    Effectue une requête HTTP GET et interprète la réponse comme du JSON.

    Parameters
    ----------
    url : str
        URL à interroger.
    timeout_s : int
        Timeout réseau.

    Returns
    -------
    dict[str, Any]
        Objet JSON décodé.
    """
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8", errors="replace"))


def _find_url_property(properties: dict[str, Any]) -> str | None:
    """
    Recherche une propriété nommée 'url' indépendamment de la casse.

    Parameters
    ----------
    properties : dict[str, Any]
        Dictionnaire de propriétés (ex. properties d'une feature GeoJSON).

    Returns
    -------
    str | None
        Valeur de l'URL si trouvée, sinon None.
    """
    for k, v in properties.items():
        if isinstance(k, str) and k.lower() == "url":
            return str(v) if v is not None else None
    return None


def get_mns_lidar_download_url_from_wfs(cfg: RetrievalConfig, x_km: int, y_km: int, logger: logging.Logger) -> str:
    """
    Interroge le WFS des dalles MNS LiDAR HD et récupère l'URL de téléchargement.

    La requête WFS est faite sur une très petite BBOX centrée sur la dalle km afin
    d'obtenir la feature correspondante, puis l'attribut `url`.

    Hypothèses de repère (EPSG:2154) :
    - x_km -> x0 = x_km * 1000 : bord Ouest (mètres).
    - y_km -> y_top = y_km * 1000 : bord Nord (mètres).
    - la dalle fait 1000 m vers le Sud.

    Parameters
    ----------
    cfg : RetrievalConfig
        Configuration WFS (endpoint, typename, SRS).
    x_km, y_km : int
        Indices kilométriques de la dalle.
    logger : logging.Logger
        Logger.

    Returns
    -------
    str
        URL de téléchargement (souvent un lien WMS-R/HTTP).

    Raises
    ------
    RuntimeError
        Si aucune feature n'est trouvée ou si l'attribut 'url' est absent/inexploitable.
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

    props = feats[0].get("properties", {})
    if not isinstance(props, dict):
        raise RuntimeError("Réponse WFS inattendue: 'properties' invalide.")

    url = _find_url_property(props)
    if not url:
        raise RuntimeError("La feature WFS ne contient pas d'attribut 'url' exploitable.")

    return url


def _filename_from_wmsr_url(wms_url: str, fallback: str) -> str:
    """
    Extrait un nom de fichier depuis une URL WMS-R.

    Si le paramètre FILENAME=... (ou filename=...) est présent, il est utilisé.
    Sinon, la valeur `fallback` est renvoyée.

    Parameters
    ----------
    wms_url : str
        URL WMS-R/HTTP.
    fallback : str
        Nom utilisé si aucun paramètre n'est détecté.

    Returns
    -------
    str
        Nom de fichier retenu.
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
    Télécharge un fichier depuis une URL vers un chemin local.

    Fonctionnalités :
    - journalisation détaillée,
    - écriture via un fichier temporaire `.part` dans tmp_dir (puis renommage atomique),
    - tentatives multiples en cas d'erreurs transitoires (timeouts, 502/503/504),
    - fallback `curl` (optionnel) si `urllib` échoue (utile derrière proxy).

    Parameters
    ----------
    url : str
        URL du fichier à télécharger.
    dst_path : Path
        Chemin final du fichier.
    tmp_dir : Path
        Dossier temporaire pour le fichier `.part`.
    logger :
        Logger (logging.Logger).
    timeout_s : int
        Timeout réseau pour urllib/curl.
    overwrite : bool
        Si False et dst_path existe déjà (taille > 0), le téléchargement est ignoré.
    use_curl_fallback : bool
        Si True, déclenche un téléchargement via curl en cas d'échec urllib.

    Raises
    ------
    RuntimeError
        Si le téléchargement échoue (urllib + éventuellement curl) ou produit un fichier vide.
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
    """
    Copie un fichier source vers une destination si la destination n'existe pas déjà.

    Parameters
    ----------
    src : Path
        Fichier source.
    dst : Path
        Fichier destination.
    logger : logging.Logger
        Logger.
    """
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
    Exécute la récupération des données d'entrée du pipeline.

    Étapes :
    1) Téléchargement des nuages de points LiDAR à partir d'un fichier d'URL.
    2) Pour chaque tuile (x_km, y_km) déduite du nom LiDAR :
       - recherche de la dalle MNS de corrélation la plus récente dans le store interne,
       - copie de la dalle trouvée dans le dossier projet correspondant.
    3) Optionnel : récupération du MNS LiDAR HD via WFS (URL WMS-R), puis téléchargement du GeoTIFF.

    Parameters
    ----------
    paths : ProjectPaths
        Arborescence du projet (répertoires de sortie).
    lidar_urls_txt : str | Path
        Fichier texte contenant une URL LiDAR par ligne.
    cfg : RetrievalConfig
        Configuration (store, politiques de recherche, accès WFS).
    strict_missing_mns_correlation : bool, default=True
        Si True : l'absence d'une dalle MNS de corrélation est bloquante (exception).
        Si False : l'absence est journalisée, et la tuile est ignorée.
    fetch_mns_lidar : bool, default=True
        Si True : tente de récupérer les dalles MNS LiDAR HD via WFS.
    strict_missing_mns_lidar : bool, default=False
        Si True : l'échec de récupération MNS LiDAR HD est bloquant (exception).
        Si False : l'échec produit un warning et le traitement continue.

    Raises
    ------
    MissingCorrelationDSMError
        Si une dalle MNS corrélation est manquante et `strict_missing_mns_correlation=True`.
    MissingLidarDSMError
        Si une dalle MNS LiDAR HD est manquante et `strict_missing_mns_lidar=True`.
    RuntimeError
        En cas d'erreur de téléchargement ou de requête WFS.
    """
    logger = setup_logger(paths.logs)

    logger.info("Racine projet: %s", paths.root)
    logger.info("Fichier URLs LiDAR: %s", Path(lidar_urls_txt).resolve())
    logger.info(
        "MNS corrélation: auto-détection département via WFS (%s)",
        cfg.dep_wfs.type_name,
    )
    logger.info(
        "MNS corrélation: recherche années %d → %d",
        cfg.year_start,
        cfg.year_stop,
    )
    logger.info("MNS corrélation store root: %s", cfg.store_root)
    logger.info("Fetch MNS LiDAR HD via WFS: %s", fetch_mns_lidar)

    urls = read_lidar_urls(lidar_urls_txt)
    logger.info("URLs lues: %d", len(urls))


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

        corr_src = find_latest_mns_correlation_tile(cfg, x_km, y_km)
        logger.info("MNS corrélation retenu: %s", corr_src)
        if corr_src is None:
            msg = (
                f"MNS corrélation introuvable (store) pour tuile {x_km:04d}_{y_km:04d} "
                f"en testant années {cfg.year_start}->{cfg.year_stop}"
            )
            logger.error(msg)
            if strict_missing_mns_correlation:
                raise MissingCorrelationDSMError(msg)
            else:
                continue

        corr_dst = paths.mns_correlation / corr_src.name
        copy_if_needed(corr_src, corr_dst, logger)

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