# -*- coding: utf-8 -*-
"""
Lecture des informations du dernier run WaSuRe.

Ce module fournit une fonction utilitaire qui lit le fichier
`wasure_last_run.json` (écrit par `run_wasure()`) afin de retrouver
le répertoire de sortie (`output_dir`) du dernier run.

Usage typique dans le pipeline :
- une étape lance WaSuRe et écrit `wasure_last_run.json`,
- les étapes suivantes appellent `read_last_wasure_output()` pour récupérer
  le chemin de sortie du dernier run (sans avoir à recalculer/chercher).
"""

from __future__ import annotations

import json
from pathlib import Path


def read_last_wasure_output(out_wasure_root: str | Path) -> Path:
    """
    Lit `wasure_last_run.json` et renvoie le champ `output_dir` (dossier de sortie du dernier run).

    Le fichier `wasure_last_run.json` est censé être généré par le module d'exécution WaSuRe
    (ex. `run_wasure.py`). Il contient au minimum la clé `output_dir`.

    Parameters
    ----------
    out_wasure_root : str | Path
        Répertoire racine contenant les runs WaSuRe et le fichier `wasure_last_run.json`.

    Returns
    -------
    Path
        Chemin vers le dossier de sortie WaSuRe du dernier run (`output_dir`).

    Raises
    ------
    RuntimeError
        - si `wasure_last_run.json` est absent,
        - si le JSON est invalide (clé `output_dir` manquante),
        - si le `output_dir` stocké n'existe plus sur disque.
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