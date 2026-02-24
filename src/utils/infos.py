#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construire un résumé “par dalle” à partir de deux fichiers de logs :
- recalage_altimetrique_*.log  -> id de dalle, nom du MNS de corrélation, dz
- creation_masque_*.log        -> id de dalle, nom du MNS de corrélation, pourcentage final de changement

Sorties :
- tile_summary.csv
- tile_grid_pct_changed.csv (pivot : y_km en lignes, x_km en colonnes)
- tile_grid_dz.csv
- tile_grid_corr_year.csv
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd


@dataclass(frozen=True)
class TileInfo:
    # Indices km (x_km, y_km) extraits du tile_id "xxxx_yyyy"
    x_km: int
    y_km: int
    # Nom du MNS de corrélation (fichier .tif), si détecté dans les logs
    corr_name: Optional[str] = None
    # Année inférée depuis le nom du MNS de corrélation (si possible)
    corr_year: Optional[int] = None
    # Décalage vertical estimé (corr - lidar) en mètres
    dz_m: Optional[float] = None
    # Pourcentage final de pixels “changés” dans le masque
    pct_changed: Optional[float] = None


# -----------------------------
# Helpers
# -----------------------------

# Regex utilitaire : récupère l’identifiant de dalle sous forme "dddd_dddd" dans "(id dddd_dddd)"
_RE_ID = re.compile(r"\(id\s+(?P<id>\d{4}_\d{4})\)")

# Dans recalage_altimetrique : ligne d’en-tête de dalle avec le nom du tif corr et l’id
_RE_TILE_RECAL = re.compile(
    r"----\s+Dalle\s+\d+\s+/\s+\d+\s+:\s+(?P<corr_name>.+?\.tif)\s+\(id\s+(?P<id>\d{4}_\d{4})\)\s+----"
)

# Dans recalage_altimetrique : ligne qui contient le dz estimé (gère estimé/estime et quelques variations d’espaces)
_RE_DZ = re.compile(r"z\s+estim(?:é|e)\s+\(corr\s*-\s*lidar\)\s*:\s*(?P<dz>[-+]?\d+(?:\.\d+)?)\s*m", re.IGNORECASE)

# Dans creation_masque : ligne d’en-tête de dalle (contient uniquement l’id)
_RE_TILE_MASK = re.compile(r"----\s+Dalle\s+\d+\s+/\s+\d+\s+:\s+(?P<id>\d{4}_\d{4})\s+----")

# Dans creation_masque : ligne “Newer (corr recalé): <...>.tif” -> nom du MNS corr recalé réellement utilisé
_RE_NEWER = re.compile(r"Newer\s+\(corr recalé\)\s*:\s*(?P<corr_name>.+?\.tif)")

# Dans creation_masque : ligne finale du masque indiquant le % de pixels changés
_RE_MASK_FINAL = re.compile(
    r"Masque final:\s+change=\d+\s+\((?P<pct>\d+(?:\.\d+)?)% des pixels valides initiaux\)"
)

# Regex potentielle (non utilisée plus bas) pour inférer une année si un chemin contient ".../LAMB93_YYYY/..."
_RE_YEAR_FROM_PATH = re.compile(r"(?:^|/|\\)LAMB93_(?P<yyyy>\d{4})(?:/|\\)")


def _parse_xy_from_id(tile_id: str) -> Tuple[int, int]:
    """
    Convertit un tile_id 'xxxx_yyyy' en deux entiers (x_km, y_km).
    """
    x_str, y_str = tile_id.split("_")
    return int(x_str), int(y_str)


def _try_parse_year_from_corr_name(corr_name: str) -> Optional[int]:
    """
    Essaie d’inférer l’année depuis le nom de fichier du MNS de corrélation.

    Exemple : MNS_CORREL_1-0_LAMB93_24FD7320_...
    -> année 2024 (hypothèse : 2000 + YY)

    Retourne None si le pattern n’est pas trouvé.
    """
    m = re.search(r"_LAMB93_(?P<yy>\d{2})FD", corr_name)
    if not m:
        return None
    yy = int(m.group("yy"))
    return 2000 + yy


# -----------------------------
# Parsers
# -----------------------------

def parse_recalage_log(path: Path) -> Dict[str, TileInfo]:
    """
    Parse un log recalage_altimetrique.

    Retour :
    dict : tile_id -> TileInfo (contient dz_m et le nom du tif de corrélation vu dans le log)
    """
    tiles: Dict[str, TileInfo] = {}
    current_id: Optional[str] = None
    current_corr: Optional[str] = None

    # Lecture ligne par ligne (avec errors="replace" pour éviter de planter sur caractères invalides)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        # Détection du début d’une nouvelle dalle
        m = _RE_TILE_RECAL.search(line)
        if m:
            current_id = m.group("id")
            current_corr = m.group("corr_name").strip()
            x_km, y_km = _parse_xy_from_id(current_id)

            # Création/écrasement de l’entrée TileInfo pour cette dalle
            tiles[current_id] = TileInfo(
                x_km=x_km,
                y_km=y_km,
                corr_name=current_corr,
                corr_year=_try_parse_year_from_corr_name(current_corr),
                dz_m=None,
                pct_changed=None,
            )
            continue

        # Si on est “dans” une dalle, on cherche la ligne dz associée
        if current_id is not None:
            m2 = _RE_DZ.search(line)
            if m2:
                dz = float(m2.group("dz"))
                ti = tiles[current_id]

                # Comme TileInfo est frozen=True, on reconstruit un nouvel objet en recopiant les champs
                tiles[current_id] = TileInfo(
                    x_km=ti.x_km,
                    y_km=ti.y_km,
                    corr_name=ti.corr_name,
                    corr_year=ti.corr_year,
                    dz_m=dz,
                    pct_changed=ti.pct_changed,
                )

    return tiles


def parse_mask_log(path: Path) -> Dict[str, TileInfo]:
    """
    Parse un log creation_masque.

    Retour :
    dict : tile_id -> TileInfo (contient le nom du tif corr recalé réellement utilisé et le % final de changement)
    """
    tiles: Dict[str, TileInfo] = {}
    current_id: Optional[str] = None
    current_corr: Optional[str] = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        # Début d’une nouvelle dalle (dans le log masque)
        m = _RE_TILE_MASK.search(line)
        if m:
            current_id = m.group("id")
            current_corr = None
            x_km, y_km = _parse_xy_from_id(current_id)

            # Crée l’entrée si elle n’existe pas déjà (setdefault)
            tiles.setdefault(current_id, TileInfo(x_km=x_km, y_km=y_km))
            continue

        # Si on n’a pas encore identifié de dalle, on ignore la ligne
        if current_id is None:
            continue

        # Récupération du nom du tif “Newer (corr recalé)” (si présent)
        m2 = _RE_NEWER.search(line)
        if m2:
            current_corr = m2.group("corr_name").strip()
            ti = tiles[current_id]

            # Mise à jour : on stocke corr_name/corr_year ; dz/pct restent inchangés
            tiles[current_id] = TileInfo(
                x_km=ti.x_km,
                y_km=ti.y_km,
                corr_name=current_corr,
                corr_year=_try_parse_year_from_corr_name(current_corr),
                dz_m=ti.dz_m,
                pct_changed=ti.pct_changed,
            )
            continue

        # Lecture du % final du masque (ligne “Masque final: ... (xx.xx% ...)”)
        m3 = _RE_MASK_FINAL.search(line)
        if m3:
            pct = float(m3.group("pct"))
            ti = tiles[current_id]

            # On conserve corr_name si déjà connu (priorité à ce qui a été trouvé avant)
            corr_name = ti.corr_name if ti.corr_name else current_corr

            # Idem pour l’année : on conserve si déjà connue, sinon on tente de parser depuis corr_name
            corr_year = ti.corr_year if ti.corr_year else (
                _try_parse_year_from_corr_name(corr_name) if corr_name else None
            )

            tiles[current_id] = TileInfo(
                x_km=ti.x_km,
                y_km=ti.y_km,
                corr_name=corr_name,
                corr_year=corr_year,
                dz_m=ti.dz_m,
                pct_changed=pct,
            )

    return tiles


# -----------------------------
# Merge + Outputs
# -----------------------------

def merge_tile_infos(a: Dict[str, TileInfo], b: Dict[str, TileInfo]) -> pd.DataFrame:
    """
    Fusionne deux dictionnaires TileInfo indexés par tile_id.

    Règle de préférence :
    - on privilégie corr_name issu du log masque s’il est disponible
      (c’est censé être le “corr recalé” réellement utilisé).

    Sortie :
    - DataFrame trié par (y_km, x_km).
    """
    keys = sorted(set(a.keys()) | set(b.keys()))

    rows: List[dict] = []
    for k in keys:
        ta = a.get(k)
        tb = b.get(k)

        # x_km/y_km : on prend ceux de tb s’il existe, sinon ceux de ta
        x_km = (tb.x_km if tb else ta.x_km) if (ta or tb) else None
        y_km = (tb.y_km if tb else ta.y_km) if (ta or tb) else None

        corr_name = None
        corr_year = None
        dz_m = None
        pct = None

        # On commence par les infos du recalage (si présentes)
        if ta:
            corr_name = ta.corr_name or corr_name
            corr_year = ta.corr_year or corr_year
            dz_m = ta.dz_m if ta.dz_m is not None else dz_m
            pct = ta.pct_changed if ta.pct_changed is not None else pct

        # Puis on applique les infos du masque (si présentes), en écrasant/complétant
        if tb:
            corr_name = tb.corr_name or corr_name
            corr_year = tb.corr_year or corr_year
            dz_m = tb.dz_m if tb.dz_m is not None else dz_m
            pct = tb.pct_changed if tb.pct_changed is not None else pct

        # Si l’année est manquante mais qu’on a un corr_name, on tente une dernière inférence
        if corr_year is None and corr_name:
            corr_year = _try_parse_year_from_corr_name(corr_name)

        # Construction d’une ligne “plate” pour le DataFrame
        rows.append(
            {
                "tile_id": k,
                "x_km": x_km,
                "y_km": y_km,
                "corr_name": corr_name,
                "corr_year": corr_year,
                "dz_m": dz_m,
                "pct_changed": pct,
            }
        )

    # Tri pour avoir un ordre stable et “carto-friendly” (d’abord y puis x)
    df = pd.DataFrame(rows).sort_values(["y_km", "x_km"]).reset_index(drop=True)
    return df


def write_grids(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Écrit des grilles (CSV pivotées) à partir du DataFrame :
    - pct_changed
    - dz_m
    - corr_year

    Chaque grille est une matrice y_km (lignes) x_km (colonnes), avec y décroissant
    pour une lecture “comme une carte” (nord en haut).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grille % changé
    if "pct_changed" in df.columns:
        grid = df.pivot(index="y_km", columns="x_km", values="pct_changed").sort_index(ascending=False)
        grid.to_csv(out_dir / "tile_grid_pct_changed.csv")

    # Grille dz
    if "dz_m" in df.columns:
        grid = df.pivot(index="y_km", columns="x_km", values="dz_m").sort_index(ascending=False)
        grid.to_csv(out_dir / "tile_grid_dz.csv")

    # Grille année corr
    if "corr_year" in df.columns:
        grid = df.pivot(index="y_km", columns="x_km", values="corr_year").sort_index(ascending=False)
        grid.to_csv(out_dir / "tile_grid_corr_year.csv")


def build_summary_from_logs_dir(logs_dir: Path, out_dir: Path | None = None) -> Path:
    """
    Construit tile_summary.csv + grilles à partir des derniers logs trouvés dans logs_dir :
    - recalage_altimetrique_*.log (dernier par ordre de tri)
    - creation_masque_*.log       (dernier par ordre de tri)

    Retourne :
    - le chemin vers tile_summary.csv
    """
    out_dir = out_dir or logs_dir

    # On sélectionne automatiquement le dernier log de recalage
    recal_logs = sorted(logs_dir.glob("recalage_altimetrique_*.log"))
    if not recal_logs:
        raise FileNotFoundError(f"No recalage_altimetrique_*.log found in {logs_dir}")
    recal_log = recal_logs[-1]

    # On sélectionne automatiquement le dernier log de masque
    mask_logs = sorted(logs_dir.glob("creation_masque_*.log"))
    if not mask_logs:
        raise FileNotFoundError(f"No creation_masque_*.log found in {logs_dir}")
    mask_log = mask_logs[-1]

    # Parsing + fusion
    rec = parse_recalage_log(recal_log)
    msk = parse_mask_log(mask_log)
    df = merge_tile_infos(rec, msk)

    # Écritures
    out_csv = out_dir / "tile_summary.csv"
    df.to_csv(out_csv, index=False)
    write_grids(df, out_dir)

    return out_csv


def build_summary_from_logs_dir(logs_dir: Path, out_dir: Path | None = None) -> Path:
    """
    Construit tile_summary.csv + grilles à partir des derniers logs trouvés dans logs_dir :
    - recalage_altimetrique_*.log (dernier par ordre de tri)
    - creation_masque_*.log       (dernier par ordre de tri)

    Retourne :
    - le chemin vers tile_summary.csv
    """
    out_dir = out_dir or logs_dir

    recal_logs = sorted(logs_dir.glob("recalage_altimetrique_*.log"))
    if not recal_logs:
        raise FileNotFoundError(f"No recalage_altimetrique_*.log found in {logs_dir}")
    recal_log = recal_logs[-1]

    mask_logs = sorted(logs_dir.glob("creation_masque_*.log"))
    if not mask_logs:
        raise FileNotFoundError(f"No creation_masque_*.log found in {logs_dir}")
    mask_log = mask_logs[-1]

    rec = parse_recalage_log(recal_log)
    msk = parse_mask_log(mask_log)
    df = merge_tile_infos(rec, msk)

    out_csv = out_dir / "tile_summary.csv"
    df.to_csv(out_csv, index=False)
    write_grids(df, out_dir)

    return out_csv


def main() -> None:
    import argparse

    # Interface CLI : on peut soit donner un dossier de logs, soit forcer les chemins des 2 logs
    parser = argparse.ArgumentParser(description="Build per-tile summary from pipeline logs.")
    parser.add_argument("--logs-dir", type=Path, required=True, help="Path to logs directory")
    parser.add_argument("--recalage-log", type=Path, default=None, help="Path to recalage_altimetrique_*.log")
    parser.add_argument("--mask-log", type=Path, default=None, help="Path to creation_masque_*.log")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: logs-dir)")
    args = parser.parse_args()

    logs_dir: Path = args.logs_dir
    out_dir: Path = args.out_dir or logs_dir

    # Auto-sélection du dernier log recalage si non fourni
    if args.recalage_log is None:
        candidates = sorted(logs_dir.glob("recalage_altimetrique_*.log"))
        if not candidates:
            raise FileNotFoundError(f"No recalage_altimetrique_*.log found in {logs_dir}")
        recal_log = candidates[-1]
    else:
        recal_log = args.recalage_log

    # Auto-sélection du dernier log masque si non fourni
    if args.mask_log is None:
        candidates = sorted(logs_dir.glob("creation_masque_*.log"))
        if not candidates:
            raise FileNotFoundError(f"No creation_masque_*.log found in {logs_dir}")
        mask_log = candidates[-1]
    else:
        mask_log = args.mask_log

    # Parsing + fusion
    rec = parse_recalage_log(recal_log)
    msk = parse_mask_log(mask_log)
    df = merge_tile_infos(rec, msk)

    # Écriture du résumé + grilles
    out_csv = out_dir / "tile_summary.csv"
    df.to_csv(out_csv, index=False)
    write_grids(df, out_dir)

    print(f"Wrote: {out_csv}")
    print(f"Wrote grids in: {out_dir}")


if __name__ == "__main__":
    main()