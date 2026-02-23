# -*- coding: utf-8 -*-
"""
Lancement de WaSuRe depuis le pipeline Python.

Objectifs :
- Ne JAMAIS réutiliser / effacer un dossier out_WASURE existant (problème de droits).
- Créer un sous-dossier horodaté à chaque run.
- Écrire les logs WaSuRe (très verbeux) dans un fichier dédié (pas dans le log principal).
- Écrire un JSON "wasure_last_run.json" pour que la suite du pipeline retrouve la sortie.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WaSuReConfig:
    # Dossier contenant run_lidarhd.sh
    wasure_repo_dir: Path = Path("/home/data_ssd/esaint-denis/sparkling-wasure")
    # Script à appeler (relatif à wasure_repo_dir)
    wasure_script: str = "./run_lidarhd.sh"
    # Arguments additionnels éventuels (si tu ajoutes des options WaSuRe plus tard)
    extra_args: list[str] | None = None
    # Si True, on échoue si le dossier input_dir est vide (sécurité)
    fail_if_input_empty: bool = True


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _dir_is_empty(p: Path) -> bool:
    try:
        return not any(p.iterdir())
    except Exception:
        return True


def run_wasure(
    *,
    input_dir: Path,
    out_wasure_root: Path,
    logs_dir: Path,
    logger: logging.Logger,
    cfg: WaSuReConfig,
) -> Path:
    """
    Lance WaSuRe sur input_dir (nuages LAZ combinés) et écrit dans out_wasure_root/run_YYYY-mm-dd_HH-MM-SS.

    Retour
    ------
    Path : dossier de sortie WaSuRe de ce run
    """
    input_dir = Path(input_dir)
    out_wasure_root = Path(out_wasure_root)
    logs_dir = Path(logs_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"[WaSuRe] input_dir introuvable: {input_dir}")

    if cfg.fail_if_input_empty and _dir_is_empty(input_dir):
        raise RuntimeError(f"[WaSuRe] input_dir existe mais semble vide: {input_dir}")

    out_wasure_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_id = _timestamp()
    out_dir = out_wasure_root / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    wasure_log = logs_dir / f"wasure_{run_id}.log"
    meta_path = out_wasure_root / f"wasure_run_{run_id}.json"
    last_path = out_wasure_root / "wasure_last_run.json"

    # Commande (appel du wrapper, docker ou pas: transparent)
    cmd = [
        cfg.wasure_script,
        "--input_dir", str(input_dir),
        "--output_dir", str(out_dir),
    ]
    if cfg.extra_args:
        cmd.extend(cfg.extra_args)

    # Log principal: court et utile
    logger.info("==== Étape WaSuRe ====")
    logger.info("Repo WaSuRe: %s", cfg.wasure_repo_dir)
    logger.info("Commande: %s", " ".join(cmd))
    logger.info("Sortie WaSuRe: %s", out_dir)
    logger.info("Log WaSuRe (séparé): %s", wasure_log)

    start_t = time.time()

    # Exécution : stdout+stderr -> fichier dédié
    with open(wasure_log, "w", encoding="utf-8") as f:
        f.write(f"[PIPELINE] WaSuRe run_id={run_id}\n")
        f.write(f"[PIPELINE] cwd={cfg.wasure_repo_dir}\n")
        f.write(f"[PIPELINE] cmd={' '.join(cmd)}\n")
        f.write(f"[PIPELINE] input_dir={input_dir}\n")
        f.write(f"[PIPELINE] output_dir={out_dir}\n")
        f.write("\n")

        proc = subprocess.run(
            cmd,
            cwd=str(cfg.wasure_repo_dir),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    elapsed_s = time.time() - start_t

    # JSON de traçabilité (écrit même si échec)
    meta = {
        "run_id": run_id,
        "timestamp": run_id,
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "wasure_repo_dir": str(cfg.wasure_repo_dir),
        "command": cmd,
        "log_file": str(wasure_log),
        "returncode": int(proc.returncode),
        "elapsed_seconds": float(elapsed_s),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    last_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if proc.returncode != 0:
        logger.error("WaSuRe a échoué (returncode=%d). Voir: %s", proc.returncode, wasure_log)
        raise RuntimeError(f"WaSuRe a échoué (returncode={proc.returncode}). Log: {wasure_log}")

    logger.info("WaSuRe terminé OK en %.1f s", elapsed_s)
    logger.info("JSON run: %s", meta_path)
    logger.info("JSON last: %s", last_path)

    return out_dir
