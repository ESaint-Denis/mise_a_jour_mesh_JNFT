# -*- coding: utf-8 -*-
"""
Exécution du générateur LOD / 3D Tiles "mesh23dtile" de WaSuRe.

Ce module encapsule l'appel :
  <wasure_repo>/services/mesh23dtile/run.sh

Deux modes d'exécution
---------------------
- Mode Docker (recommandé) :
    Exécute run.sh dans l'image Docker WaSuRe contenant l'environnement conda + obj-tiler.
- Mode Host :
    Exécute run.sh directement sur la machine hôte (nécessite conda + dépendances présentes sur l'hôte).

Notes importantes
-----------------
- Le service WaSuRe mesh23dtile attend des tuiles PLY en coordonnées LOCALES (type outputs/tiles).
  Il utilise le fichier wasure_metadata_3d_gen.xml (dans le run_*) pour appliquer les offsets
  et convertir vers EPSG:4978 lors de la génération des 3D Tiles.
- Les logs de cette étape sont séparés du log principal du pipeline (fichier dédié).
- La sortie attendue est typiquement un tileset.json dans le répertoire de sortie.

Tous les commentaires de code sont en français.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


# ----------------------------- Configuration -----------------------------


@dataclass(frozen=True)
class Mesh23DTileConfig:
    """
    Configuration pour l'exécution de WaSuRe mesh23dtile.

    Attributs
    ---------
    wasure_repo_dir : Path
        Chemin vers le dépôt sparkling-wasure sur le système de fichiers HÔTE.
        Il doit contenir services/mesh23dtile/run.sh.
    num_process : int
        Nombre maximal de jobs parallèles utilisés par run.sh (variable d'environnement NUM_PROCESS).
    overwrite : bool
        Si False et qu'un tileset.json existe déjà dans output_dir, l'exécution est ignorée.
    exec_mode : str
        Backend d'exécution : "docker" (recommandé) ou "host".
    docker_image : str
        Nom de l'image Docker contenant l'environnement WaSuRe (conda env mesh23Dtile, obj-tiler, etc.).
    docker_user : bool
        Si True, exécute le conteneur avec l'UID/GID courant pour éviter de produire des fichiers root.
    docker_extra_args : tuple[str, ...]
        Arguments supplémentaires passés à `docker run` (ex: "--network=host").
    extra_env : tuple[tuple[str, str], ...]
        Variables d'environnement additionnelles à passer au conteneur (en plus de NUM_PROCESS).
    """
    # Chemin vers le dépôt sparkling-wasure sur l'HÔTE.
    wasure_repo_dir: Path

    # Nombre de processus parallèles (variable NUM_PROCESS consommée par run.sh).
    num_process: int = 30

    # Si False et la sortie contient déjà tileset.json, on skip.
    overwrite: bool = True

    # Mode d'exécution : "docker" (recommandé) ou "host".
    exec_mode: str = "docker"

    # Image Docker contenant l'environnement WaSuRe (conda env mesh23Dtile, obj-tiler, etc.).
    docker_image: str = "ddt_img_base_devel_proxy"

    # Si True, exécute le conteneur en tant qu'utilisateur courant (évite les sorties appartenant à root).
    docker_user: bool = True

    # Arguments docker supplémentaires (ex: ["--network=host"]).
    docker_extra_args: tuple[str, ...] = ()

    # Variables d'environnement supplémentaires (en plus de NUM_PROCESS).
    extra_env: tuple[tuple[str, str], ...] = ()


def _timestamp() -> str:
    """Horodatage lisible (utilisé pour nommer logs et métadonnées)."""
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _ensure_dir(p: Path) -> None:
    """Crée un répertoire s'il n'existe pas (parents inclus)."""
    p.mkdir(parents=True, exist_ok=True)


def _has_tileset(output_dir: Path) -> bool:
    """
    Détecte la présence d'une sortie mesh23dtile déjà produite.

    WaSuRe finalize.py génère généralement tileset.json à la racine du dossier de sortie.
    """
    return (output_dir / "tileset.json").is_file()


def _check_prereqs_host(logger: logging.Logger) -> None:
    """
    Vérifie les prérequis minimaux en mode 'host'.

    Le mode host est en général fragile car run.sh peut sourcer /opt/conda/etc/profile.d/conda.sh
    et s'attend à trouver l'environnement conda mesh23Dtile + obj-tiler. On conserve ce mode,
    mais on prévient clairement dans les logs.
    """
    if shutil.which("bash") is None:
        raise RuntimeError("[mesh23dtile] 'bash' introuvable dans PATH (requis pour exec_mode='host').")
    logger.warning(
        "[mesh23dtile] exec_mode='host' sélectionné. Ce mode est souvent fragile car WaSuRe run.sh "
        "source /opt/conda/etc/profile.d/conda.sh et s'attend à l'env mesh23Dtile + obj-tiler. "
        "Préférer exec_mode='docker'."
    )


def _docker_available() -> bool:
    """Retourne True si la commande `docker` est disponible dans PATH."""
    return shutil.which("docker") is not None


def _run_subprocess_to_log(
    *,
    cmd: list[str],
    cwd: Path | None,
    env: dict[str, str] | None,
    log_file: Path,
    header_lines: list[str],
) -> subprocess.CompletedProcess:
    """
    Exécute une commande et redirige stdout+stderr vers un fichier log.

    Parameters
    ----------
    cmd : list[str]
        Commande complète à exécuter.
    cwd : Path | None
        Répertoire de travail pour le subprocess (None => ne pas forcer).
    env : dict[str,str] | None
        Environnement à passer au subprocess (None => héritage).
    log_file : Path
        Fichier de log de sortie.
    header_lines : list[str]
        Lignes préfixées au début du log (traçabilité : run_id, cmd, chemins...).

    Returns
    -------
    subprocess.CompletedProcess
        Résultat de l'exécution (returncode, etc.).
    """
    with open(log_file, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line.rstrip("\n") + "\n")
        f.write("\n")
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )


def _build_docker_cmd(
    *,
    cfg: Mesh23DTileConfig,
    host_wasure_repo_dir: Path,
    input_dir: Path,
    xml_file: Path,
    output_dir: Path,
) -> list[str]:
    """
    Construit une commande `docker run` qui exécute mesh23dtile/run.sh DANS le conteneur.

    Stratégie de montage (volumes)
    ------------------------------
    On monte :
    - /media -> /media, si input/output/xml sont sous /media/... (cas typique /media/DATA/...)
    - le dépôt WaSuRe (host_wasure_repo_dir) pour accéder à services/mesh23dtile/run.sh
    - /tmp -> /tmp pour les fichiers temporaires (certains outils le supposent)

    Note
    ----
    Monter '/' en entier serait robuste mais généralement interdit / déconseillé.
    Ici on se limite à des préfixes usuels (/media, repo, /tmp).
    """
    if not _docker_available():
        raise RuntimeError("[mesh23dtile] docker introuvable dans PATH alors que exec_mode='docker'.")

    mounts: list[str] = []

    # Monter /media si les chemins de données sont sous /media
    if str(input_dir).startswith("/media/") or str(output_dir).startswith("/media/") or str(xml_file).startswith("/media/"):
        mounts += ["-v", "/media:/media"]

    # Monter le repo WaSuRe (monté à l'identique du chemin hôte)
    mounts += ["-v", f"{host_wasure_repo_dir}:{host_wasure_repo_dir}"]

    # Monter /tmp (souvent utile)
    mounts += ["-v", "/tmp:/tmp"]

    # Mapper l'utilisateur courant pour éviter de créer des sorties appartenant à root
    user_args: list[str] = []
    if cfg.docker_user:
        user_args = ["--user", f"{os.getuid()}:{os.getgid()}"]

    # Variables d'environnement passées au conteneur
    env_args: list[str] = ["-e", f"NUM_PROCESS={int(cfg.num_process)}"]
    for k, v in cfg.extra_env:
        env_args += ["-e", f"{k}={v}"]

    # Répertoire de travail dans le conteneur : racine du repo (même chemin que sur l'hôte)
    workdir_args = ["-w", str(host_wasure_repo_dir)]

    # Script à exécuter dans le conteneur
    script_in_container = host_wasure_repo_dir / "services" / "mesh23dtile" / "run.sh"
    inner_cmd = [
        "bash",
        str(script_in_container),
        "--input_dir", str(input_dir),
        "--xml_file", str(xml_file),
        "--output_dir", str(output_dir),
    ]

    cmd = [
        "docker", "run", "--rm",
        *user_args,
        *cfg.docker_extra_args,
        *env_args,
        *mounts,
        *workdir_args,
        cfg.docker_image,
        *inner_cmd,
    ]
    return cmd


# ----------------------------- API publique -----------------------------


def run_mesh23dtile(
    *,
    input_dir: str | Path,
    wasure_run_dir: str | Path,
    output_dir: str | Path,
    logs_dir: str | Path,
    logger: logging.Logger,
    cfg: Mesh23DTileConfig,
    tag: str,
) -> Path:
    """
    Lance WaSuRe mesh23dtile sur un dossier de tuiles PLY.

    Parameters
    ----------
    input_dir
        Dossier contenant des tuiles PLY (coordonnées locales, style WaSuRe outputs/tiles).
    wasure_run_dir
        Dossier run_* contenant le fichier wasure_metadata_3d_gen.xml (métadonnées + offset).
    output_dir
        Dossier de sortie du tileset (créé si besoin).
    logs_dir
        Dossier pour les logs de cette étape (fichier dédié).
    logger
        Logger du pipeline (log principal). Cette fonction écrit un log séparé pour mesh23dtile.
    cfg
        Configuration Mesh23DTileConfig.
    tag
        Chaîne utilisée pour nommer les logs/JSON (ex : "ortho" ou "origin").

    Returns
    -------
    Path
        Le dossier output_dir (doit contenir tileset.json en cas de succès).
    """
    input_dir = Path(input_dir)
    wasure_run_dir = Path(wasure_run_dir)
    output_dir = Path(output_dir)
    logs_dir = Path(logs_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"[mesh23dtile] input_dir introuvable: {input_dir}")

    # Vérifier qu'il y a au moins une tuile .ply
    n_ply = len(list(input_dir.glob("*.ply")))
    if n_ply == 0:
        raise RuntimeError(f"[mesh23dtile] input_dir ne contient aucune tuile .ply: {input_dir}")

    # Métadonnées WaSuRe nécessaires (offset/CRS, etc.)
    xml_file = wasure_run_dir / "wasure_metadata_3d_gen.xml"
    if not xml_file.is_file():
        raise FileNotFoundError(f"[mesh23dtile] XML manquant: {xml_file}")

    # Script mesh23dtile
    script = cfg.wasure_repo_dir / "services" / "mesh23dtile" / "run.sh"
    if not script.is_file():
        raise FileNotFoundError(f"[mesh23dtile] script manquant: {script}")

    _ensure_dir(output_dir)
    _ensure_dir(logs_dir)

    # Politique overwrite : si tileset.json existe déjà et overwrite=False, on skip
    if _has_tileset(output_dir) and not cfg.overwrite:
        logger.info("==== Step 11 - mesh23dtile (%s) ====", tag)
        logger.info("La sortie existe déjà et overwrite=False -> skip: %s", output_dir / "tileset.json")
        return output_dir

    run_id = _timestamp()
    log_file = logs_dir / f"mesh23dtile_{tag}_{run_id}.log"
    meta_file = output_dir / f"mesh23dtile_{tag}_{run_id}.json"
    last_file = output_dir / f"mesh23dtile_{tag}_last.json"

    # Construction de la commande selon le mode d'exécution
    if cfg.exec_mode.lower() == "docker":
        cmd = _build_docker_cmd(
            cfg=cfg,
            host_wasure_repo_dir=cfg.wasure_repo_dir,
            input_dir=input_dir,
            xml_file=xml_file,
            output_dir=output_dir,
        )
        cwd = None  # docker gère -w
        env = None  # docker gère -e
    elif cfg.exec_mode.lower() == "host":
        _check_prereqs_host(logger)
        cmd = [
            str(script),
            "--input_dir", str(input_dir),
            "--xml_file", str(xml_file),
            "--output_dir", str(output_dir),
        ]
        cwd = cfg.wasure_repo_dir
        env = dict(os.environ)
        env["NUM_PROCESS"] = str(int(cfg.num_process))
        for k, v in cfg.extra_env:
            env[k] = v
    else:
        raise ValueError(f"[mesh23dtile] exec_mode inconnu: {cfg.exec_mode!r} (attendu: 'docker' ou 'host').")

    # Logs principaux (résumés)
    logger.info("==== Step 11 - mesh23dtile (%s) ====", tag)
    logger.info("Dossier PLY entrée: %s (ply=%d)", input_dir, n_ply)
    logger.info("XML: %s", xml_file)
    logger.info("Dossier sortie tileset: %s", output_dir)
    logger.info("Log (séparé): %s", log_file)
    logger.info("NUM_PROCESS=%d", int(cfg.num_process))
    logger.info("Mode exécution: %s", cfg.exec_mode)
    if cfg.exec_mode.lower() == "docker":
        logger.info("Image Docker: %s", cfg.docker_image)
    logger.info("Commande: %s", " ".join(cmd))

    start_t = time.time()

    # Entête écrit dans le log séparé (traçabilité reproductible)
    header = [
        f"[PIPELINE] mesh23dtile tag={tag} run_id={run_id}",
        f"[PIPELINE] exec_mode={cfg.exec_mode}",
        f"[PIPELINE] cmd={' '.join(cmd)}",
        f"[PIPELINE] NUM_PROCESS={int(cfg.num_process)}",
        f"[PIPELINE] input_dir={input_dir}",
        f"[PIPELINE] xml_file={xml_file}",
        f"[PIPELINE] output_dir={output_dir}",
    ]

    proc = _run_subprocess_to_log(cmd=cmd, cwd=cwd, env=env, log_file=log_file, header_lines=header)
    elapsed_s = time.time() - start_t

    # Métadonnées JSON (écrit même en cas d'échec)
    meta = {
        "tag": tag,
        "run_id": run_id,
        "timestamp": run_id,
        "exec_mode": cfg.exec_mode,
        "input_dir": str(input_dir),
        "xml_file": str(xml_file),
        "output_dir": str(output_dir),
        "wasure_repo_dir": str(cfg.wasure_repo_dir),
        "script": str(script),
        "command": cmd,
        "num_process": int(cfg.num_process),
        "docker_image": cfg.docker_image if cfg.exec_mode.lower() == "docker" else None,
        "log_file": str(log_file),
        "returncode": int(proc.returncode),
        "elapsed_seconds": float(elapsed_s),
    }
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    last_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Gestion des erreurs d'exécution
    if proc.returncode != 0:
        logger.error("mesh23dtile a échoué (returncode=%d). Voir: %s", proc.returncode, log_file)
        raise RuntimeError(f"mesh23dtile a échoué (returncode={proc.returncode}). Log: {log_file}")

    # Post-check : tileset.json doit exister
    tileset = output_dir / "tileset.json"
    if not tileset.is_file():
        # Symptôme courant si run.sh échoue sans propager correctement l'erreur
        logger.error("mesh23dtile a retourné 0 mais tileset.json est manquant: %s", tileset)
        logger.error("Voir log: %s", log_file)
        raise RuntimeError(f"mesh23dtile n'a pas produit tileset.json. Log: {log_file}")

    # Sanity check : selon versions, le dossier tiles/ peut ou non être présent
    has_tiles_dir = (output_dir / "tiles").is_dir()
    if not has_tiles_dir:
        logger.warning(
            "mesh23dtile a produit tileset.json mais aucun dossier 'tiles/' n'a été trouvé dans %s. "
            "Cela peut dépendre de la version WaSuRe/finalize. Vérifier le log: %s",
            output_dir, log_file
        )

    logger.info("mesh23dtile OK (%.1fs) | tileset: %s", elapsed_s, tileset)
    return output_dir