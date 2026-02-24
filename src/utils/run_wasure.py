# -*- coding: utf-8 -*-
"""
Lancement de WaSuRe depuis le pipeline Python.

Ce module encapsule l'exécution de WaSuRe (via le script wrapper du dépôt) en respectant
des contraintes pratiques rencontrées en environnement serveur (droits, traçabilité,
verbeux des logs, etc.).

Objectifs
---------
- Ne jamais réutiliser / effacer un dossier out_WASURE existant (risque de problèmes de droits).
- Créer un sous-dossier horodaté à chaque exécution (un run = une sortie isolée).
- Écrire les logs WaSuRe (très verbeux) dans un fichier dédié (séparé du log principal).
- Écrire une métadonnée JSON par run et un JSON "last run" afin que les étapes suivantes
  du pipeline puissent retrouver la dernière sortie produite.

Sorties produites
----------------
Dans `out_wasure_root` :
- `run_<timestamp>/` :
    Dossier de sortie WaSuRe pour cette exécution.
- `wasure_run_<timestamp>.json` :
    Métadonnées (entrée, sortie, commande, log, returncode, durée) pour cette exécution.
- `wasure_last_run.json` :
    Copie des métadonnées du dernier run réussi/terminé (écrite même si échec, avec returncode).

Dans `logs_dir` :
- `wasure_<timestamp>.log` :
    Log complet stdout+stderr de WaSuRe (très verbeux).

Remarques
---------
- La commande est exécutée via `subprocess.run` avec `cwd` positionné sur le dépôt WaSuRe.
- En cas d'échec (returncode != 0), une exception est levée et le log dédié doit être consulté.

Auteur : ESaint-Denis
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
    """
    Configuration d'exécution de WaSuRe.

    Attributs
    ---------
    wasure_repo_dir : Path
        Chemin du dépôt WaSuRe contenant le script wrapper (ex. run_lidarhd.sh).

    wasure_script : str
        Script à exécuter (chemin relatif à `wasure_repo_dir` ou commande accessible).
        Exemple : "./run_lidarhd.sh".

    extra_args : list[str] | None
        Arguments supplémentaires optionnels, ajoutés tels quels à la commande.
        Permet d'étendre les options WaSuRe sans modifier l'interface principale.

    fail_if_input_empty : bool
        Si True, le pipeline échoue si `input_dir` existe mais est vide.
        Sert de garde-fou pour éviter de lancer WaSuRe sur une entrée invalide.
    """
    # Dossier contenant run_lidarhd.sh
    wasure_repo_dir: Path = Path("/home/data_ssd/esaint-denis/sparkling-wasure")
    # Script à appeler (relatif à wasure_repo_dir)
    wasure_script: str = "./run_lidarhd.sh"
    # Arguments additionnels éventuels (si tu ajoutes des options WaSuRe plus tard)
    extra_args: list[str] | None = None
    # Si True, on échoue si le dossier input_dir est vide (sécurité)
    fail_if_input_empty: bool = True


def _timestamp() -> str:
    """
    Génère un identifiant horodaté lisible pour nommer un run.

    Returns
    -------
    str
        Timestamp au format "YYYY-mm-dd_HH-MM-SS".
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _dir_is_empty(p: Path) -> bool:
    """
    Indique si un répertoire est vide (ou illisible).

    Notes
    -----
    En cas d'erreur (droits, répertoire inaccessible, etc.), la fonction renvoie True
    par prudence (comportement "fail safe").

    Parameters
    ----------
    p : Path
        Répertoire à tester.

    Returns
    -------
    bool
        True si vide (ou si une exception survient), False sinon.
    """
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
    Lance WaSuRe sur un répertoire d'entrée contenant des nuages LAZ combinés.

    L'exécution crée systématiquement un sous-dossier de sortie unique :
    `out_wasure_root/run_<timestamp>/`.

    Les logs WaSuRe (stdout+stderr) sont redirigés vers un fichier dédié dans `logs_dir`
    pour éviter de polluer le log principal du pipeline.

    Parameters
    ----------
    input_dir : Path
        Répertoire d'entrée contenant les nuages LAZ combinés (produits par l'étape de fusion).
    out_wasure_root : Path
        Répertoire racine des sorties WaSuRe.
    logs_dir : Path
        Répertoire des logs du pipeline (on y place un log dédié WaSuRe).
    logger : logging.Logger
        Logger du pipeline (log court et informatif).
    cfg : WaSuReConfig
        Configuration WaSuRe (dépôt, script, options, garde-fous).

    Returns
    -------
    Path
        Chemin du dossier de sortie WaSuRe pour ce run : `out_wasure_root/run_<timestamp>`.

    Raises
    ------
    FileNotFoundError
        Si `input_dir` n'existe pas.
    RuntimeError
        Si `input_dir` est vide (si `fail_if_input_empty=True`) ou si WaSuRe échoue
        (returncode != 0).
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

    # Commande (appel du wrapper, docker ou pas : transparent)
    cmd = [
        cfg.wasure_script,
        "--input_dir", str(input_dir),
        "--output_dir", str(out_dir),
    ]
    if cfg.extra_args:
        cmd.extend(cfg.extra_args)

    # Log principal : court et utile
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