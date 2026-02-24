# -*- coding: utf-8 -*-
"""
Post-traitement après WaSuRe : translation des tuiles PLY (shift / unshift).

Ce module :
- lit le décalage (Xmin, Ymin, Zmin) dans `wasure_metadata_3d_gen.xml` (balise `<bbox_ori>`),
- applique une translation (+shift ou -shift) à toutes les tuiles PLY (ASCII ou binaire),
- écrit les PLY translatés dans `<wasure_run_dir>/<out_subdir>`.

Notes importantes
-----------------
- Seules les coordonnées des sommets (vertex) sont mises à jour (et éventuellement des champs
  supplémentaires si demandés via `t_coords`).
- Les commentaires personnalisés de l'en-tête PLY (ex. "comment bbox ...") ne sont pas recalculés.
  C'est intentionnel : pour WaSuRe / mesh23dtile, conserver certains commentaires "locaux"
  inchangés est généralement souhaitable lorsque l'on travaille avec des tuiles PLY en coordonnées locales.
- Une correction de compatibilité est appliquée après écriture : s'assurer que la ligne
  `comment bbox ...` se termine par un espace avant le retour à la ligne (requis par certains parseurs).

"""

from __future__ import annotations

import glob
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from plyfile import PlyData, PlyElement, PlyElementParseError


# ----------------------------- XML shift extraction -----------------------------


def get_shift_from_xml(xml_file: str | Path) -> list[float]:
    """
    Extrait le shift (Xmin, Ymin, Zmin) depuis la balise `<bbox_ori>` d'un fichier XML WaSuRe.

    Le contenu de `<bbox_ori>` est attendu sous la forme :
    'xminxXmax:yminxYmax:zminxZmax'

    Exemple
    -------
    '916000.0x917000.0:6457000.0x6458000.0:118.93x10000.0'

    Parameters
    ----------
    xml_file : str | Path
        Chemin vers le fichier XML (ex. `wasure_metadata_3d_gen.xml`).

    Returns
    -------
    list[float]
        [xmin, ymin, zmin] extraits de `<bbox_ori>`.

    Raises
    ------
    RuntimeError
        Si la balise `<bbox_ori>` est absente ou vide.
    """
    xml_file = str(xml_file)
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bbox_ori = root.find(".//bbox_ori")
    if bbox_ori is None or not bbox_ori.text:
        raise RuntimeError("Missing or empty <bbox_ori> tag in XML.")

    fields = bbox_ori.text.strip().split(":")
    xmin = float(fields[0].split("x")[0])
    ymin = float(fields[1].split("x")[0])
    zmin = float(fields[2].split("x")[0])
    return [xmin, ymin, zmin]


# ----------------------------- PLY helpers -----------------------------
def fix_bbox_comment_trailing_space(ply_path: str | Path) -> bool:
    """
    Assure que la ligne d'en-tête `comment bbox ...` se termine par un espace avant le retour à la ligne.

    Cette correction est requise pour certains outils WaSuRe (ex. mesh23dtile.py) dont le parseur
    de la ligne bbox attend un séparateur final.

    Parameters
    ----------
    ply_path : str | Path
        Chemin vers le fichier PLY à corriger.

    Returns
    -------
    bool
        True si le fichier a été modifié, False sinon.
    """
    ply_path = Path(ply_path)

    data = ply_path.read_bytes()

    # Support both LF and CRLF
    end_token_lf = b"end_header\n"
    end_token_crlf = b"end_header\r\n"

    if end_token_lf in data:
        end_pos = data.index(end_token_lf) + len(end_token_lf)
        nl = b"\n"
    elif end_token_crlf in data:
        end_pos = data.index(end_token_crlf) + len(end_token_crlf)
        nl = b"\r\n"
    else:
        # Not a valid PLY header
        return False

    header = data[:end_pos]
    body = data[end_pos:]

    lines = header.splitlines(keepends=True)
    modified = False

    for i, line in enumerate(lines):
        # line includes newline
        if line.startswith(b"comment bbox "):
            # Remove newline temporarily
            raw = line[:-len(nl)] if line.endswith(nl) else line
            if not raw.endswith(b" "):
                lines[i] = raw + b" " + nl
                modified = True
            break

    if not modified:
        return False

    ply_path.write_bytes(b"".join(lines) + body)
    return True


def reformat_ply_ascii_flat(in_path: str | Path, out_path: str | Path) -> None:
    """
    Convertit un PLY ASCII "aplati" (tokens sans retours ligne attendus) en PLY ASCII standard.

    Certains PLY ASCII peuvent être écrits en une forme où les sommets/faces ne sont pas
    correctement séparés par ligne, ce qui déclenche des erreurs de parsing.
    Cette fonction reconstruit un format standard :
    - un sommet par ligne,
    - une face par ligne.

    Parameters
    ----------
    in_path : str | Path
        PLY ASCII d'entrée (potentiellement aplati).
    out_path : str | Path
        PLY ASCII de sortie (reformaté).
    """
    in_path = str(in_path)
    out_path = str(out_path)

    with open(in_path, "r", encoding="utf-8") as fin:
        lines = []
        vertex_count = 0
        face_count = 0
        vertex_props = 0
        parsing_vertex = False

        # Read header
        for line in fin:
            lines.append(line)
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                parsing_vertex = True
                continue
            if line.startswith("element face"):
                face_count = int(line.split()[-1])
                parsing_vertex = False
                continue
            if line.startswith("property") and parsing_vertex:
                vertex_props += 1
            if line.startswith("end_header"):
                break

        # Read remaining tokens
        data = fin.read().split()
        idx = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        # Write header
        for l in lines:
            fout.write(l)

        # Write vertices
        for _ in range(vertex_count):
            vals = data[idx : idx + vertex_props]
            fout.write(" ".join(vals) + "\n")
            idx += vertex_props

        # Write faces
        for _ in range(face_count):
            n_idx = int(data[idx])
            vals = data[idx : idx + 1 + n_idx]
            fout.write(" ".join(vals) + "\n")
            idx += 1 + n_idx

        # Write remaining tokens if any
        if idx < len(data):
            fout.write(" ".join(data[idx:]) + "\n")


def upgrade_vertex_dtype_to_double(vertex_arr: np.ndarray) -> np.ndarray:
    """
    Convertit tous les champs float32 d'un tableau structuré de sommets en float64.

    Objectif : éviter les pertes de précision lors de l'application d'un shift
    sur des coordonnées (notamment quand les valeurs sont grandes).

    Parameters
    ----------
    vertex_arr : np.ndarray
        Tableau structuré (vertex) issu d'un PLY.

    Returns
    -------
    np.ndarray
        Nouveau tableau structuré avec les champs float32 convertis en float64.
    """
    new_descr = []
    for name, dtype in vertex_arr.dtype.descr:
        if dtype.startswith("<f4") or dtype.startswith(">f4") or dtype == "float32":
            new_descr.append((name, "f8"))
        else:
            new_descr.append((name, dtype))

    vertex_arr_double = np.zeros(vertex_arr.shape[0], dtype=new_descr)
    for name in vertex_arr.dtype.names:
        vertex_arr_double[name] = vertex_arr[name]
    return vertex_arr_double


def apply_shift_to_ply(
    input_ply: str | Path,
    output_ply: str | Path,
    *,
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    t_coords: tuple[str, ...] = ("x", "y", "z"),
    ascii_out: bool = False,
    xml_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> bool:
    """
    Applique une translation à certaines propriétés "vertex" d'un fichier PLY.

    Le shift peut être fourni :
    - directement via `shift=(dx,dy,dz)`,
    - ou lu depuis un XML WaSuRe via `xml_path` (balise `<bbox_ori>`), ce qui écrase `shift`.

    En l'absence de `xml_path`, une tentative de lecture est faite dans les commentaires PLY
    si un commentaire commence par "IGN offset Pos" (fallback).

    Le module supporte :
    - PLY binaire / ASCII standard,
    - et une réparation automatique de certains PLY ASCII "aplatis" (tokens sans fins de lignes),
      qui provoquent une erreur de parsing (`PlyElementParseError`).

    Parameters
    ----------
    input_ply : str | Path
        Chemin vers le PLY source.
    output_ply : str | Path
        Chemin du PLY de sortie.
    shift : tuple[float, float, float]
        Translation (dx, dy, dz) à appliquer si `xml_path` n'est pas fourni.
    t_coords : tuple[str, ...]
        Noms des propriétés vertex à translater (par défaut : ("x","y","z")).
        Peut inclure d'autres champs s'ils existent, par ex. ("x0","y0","z0").
        La translation est appliquée cycliquement sur (dx,dy,dz) via i % 3.
    ascii_out : bool
        Si True, force une écriture ASCII (plydata.text = True). Sinon conserve/écrit en binaire.
    xml_path : str | Path | None
        Si fourni, lit le shift depuis le XML (<bbox_ori>) et écrase `shift`.
        Dans l'orchestrateur, on préfère lire le XML une seule fois et passer `shift` directement.
    logger : logging.Logger | None
        Logger optionnel (messages debug/warning).

    Returns
    -------
    bool
        True si le PLY d'entrée a été détecté comme "ASCII aplati" et réparé temporairement,
        False sinon.

    Raises
    ------
    PlyElementParseError
        Si le PLY est invalide et ne correspond pas au cas "ASCII aplati" géré.
    """
    input_ply = str(input_ply)
    output_ply = str(output_ply)

    repaired_flattened = False

    try:
        plydata = PlyData.read(input_ply)
    except PlyElementParseError as e:
        if "expected end-of-line" in str(e) and "element 'vertex'" in str(e):
            repaired_flattened = True
            if logger:
                # Use DEBUG to avoid spamming the main log
                logger.debug("Flattened ASCII PLY detected, repairing temporarily: %s", input_ply)
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".ply") as tmpf:
                tmp_path = tmpf.name
            reformat_ply_ascii_flat(input_ply, tmp_path)
            plydata = PlyData.read(tmp_path)
            os.remove(tmp_path)
        else:
            raise

    if xml_path is not None:
        sx, sy, sz = get_shift_from_xml(xml_path)
        shift = (sx, sy, sz)
        if logger:
            logger.debug("Shift read from XML: %s", shift)
    else:
        # Fallback to PLY comment
        for comment in getattr(plydata, "comments", []):
            if comment.startswith("IGN offset Pos"):
                parts = comment.split()
                shift = (float(parts[-3]), float(parts[-2]), float(parts[-1]))
                if logger:
                    logger.debug("Shift read from PLY comment: %s", shift)
                break

    if "vertex" not in plydata:
        if logger:
            logger.warning("No vertex element in PLY, skipping: %s", input_ply)
        return repaired_flattened

    vertex_dtype = plydata["vertex"].data.dtype
    vertex_arr = np.array(plydata["vertex"].data.copy())
    vertex_arr = upgrade_vertex_dtype_to_double(vertex_arr)

    for i, axis in enumerate(t_coords):
        if axis in vertex_dtype.names:
            vertex_arr[axis] = vertex_arr[axis] + shift[i % 3]

    plydata.elements = tuple(
        [PlyElement.describe(vertex_arr, "vertex")]
        + [e for e in plydata.elements if e.name != "vertex"]
    )

    plydata.text = ascii_out
    plydata.write(output_ply)
    
    # WaSuRe compatibility: bbox parser expects a trailing space on "comment bbox" line
    try:
        fix_bbox_comment_trailing_space(output_ply)
    except Exception:
        if logger:
            logger.debug("Failed to patch bbox trailing space for: %s", output_ply)


    return repaired_flattened


# ----------------------------- Orchestration -----------------------------


@dataclass(frozen=True)
class PostWasureShiftConfig:
    """
    Configuration du post-traitement de translation des tuiles PLY.

    Attributs
    ---------
    t_coords : tuple[str, ...]
        Propriétés vertex à translater (par défaut : x,y,z).
    ascii_out : bool
        Si True, écrit les PLY en ASCII.
    overwrite : bool
        Si False, ne réécrit pas les PLY de sortie existants.
    sign : int
        Signe appliqué au shift lu dans le XML :
        +1 => shift (coordonnées locales -> coordonnées globales),
        -1 => unshift (coordonnées globales -> coordonnées locales).
    out_subdir : str
        Nom du sous-dossier de sortie (créé sous `wasure_out_dir`).
    """
    t_coords: tuple[str, ...] = ("x", "y", "z")
    ascii_out: bool = False
    overwrite: bool = True

    # +1 => shift (local -> global), -1 => unshift (global -> local)
    sign: int = 1

    # Output folder name under the WaSuRe run directory
    out_subdir: str = "ply_L93"


def _find_tiles_dir(wasure_out_dir: Path) -> Path:
    """
    Recherche un répertoire contenant des tuiles `.ply` selon des layouts WaSuRe courants.

    Candidates testés (ordre) :
    - <run>/outputs/tiles

    Parameters
    ----------
    wasure_out_dir : Path
        Répertoire d'un run WaSuRe (`run_*`).

    Returns
    -------
    Path
        Répertoire contenant les `.ply`.

    Raises
    ------
    RuntimeError
        Si aucun répertoire candidat ne contient de `.ply`.
    """
    candidates = [
        wasure_out_dir / "outputs" / "tiles",
        # wasure_out_dir / "outputs" / "outputs" / "tiles",
        # wasure_out_dir / "tiles",
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("*.ply")):
            return c
    raise RuntimeError(f"Unable to locate a tiles directory with .ply in: {wasure_out_dir}")


def run_post_wasure_shift(
    *,
    wasure_out_dir: str | Path,
    logger: logging.Logger,
    cfg: PostWasureShiftConfig,
    tiles_dir: str | Path | None = None,
) -> Path:
    """
    Translate toutes les tuiles PLY d'un run WaSuRe (shift ou unshift).

    Le shift est lu une seule fois depuis :
    - `<wasure_out_dir>/wasure_metadata_3d_gen.xml` (balise `<bbox_ori>`),
    puis multiplié par `cfg.sign` :
    - +1 : shift (local -> global)
    - -1 : unshift (global -> local)

    Paramètres
    ----------
    wasure_out_dir : str | Path
        Répertoire du run WaSuRe (`run_*`).
    logger : logging.Logger
        Logger du pipeline.
    cfg : PostWasureShiftConfig
        Configuration (signe, propriétés à translater, ASCII/binaire, overwrite, sous-dossier de sortie).
    tiles_dir : str | Path | None
        Répertoire d'entrée contenant les tuiles PLY.
        - Si None : auto-détection via `_find_tiles_dir(<run>)` (par défaut `<run>/outputs/tiles`).
        - Utile pour traiter un jeu de tuiles déjà post-traitées (ex. pour "unshift" une sortie).

    Returns
    -------
    Path
        Répertoire de sortie contenant les PLY translatés : `<wasure_out_dir>/<cfg.out_subdir>`.

    Raises
    ------
    RuntimeError
        - si le fichier XML metadata est absent,
        - si le répertoire de tuiles est introuvable,
        - si aucune tuile `.ply` n'est trouvée,
        - si une tuile provoque une erreur (exception relancée après log).
    """
    wasure_out_dir = Path(wasure_out_dir)

    xml_path = wasure_out_dir / "wasure_metadata_3d_gen.xml"
    if not xml_path.is_file():
        raise RuntimeError(f"Missing XML metadata file: {xml_path}")

    if tiles_dir is None:
        in_tiles_dir = _find_tiles_dir(wasure_out_dir)
    else:
        in_tiles_dir = Path(tiles_dir)
        if not in_tiles_dir.is_dir():
            raise RuntimeError(f"Input tiles_dir not found: {in_tiles_dir}")

    out_dir = wasure_out_dir / cfg.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(Path(p) for p in glob.glob(str(in_tiles_dir / "*.ply")))
    if not ply_files:
        raise RuntimeError(f"No .ply files found in: {in_tiles_dir}")

    # Read shift ONCE, then apply sign (+shift or -shift)
    base_shift = tuple(get_shift_from_xml(xml_path))
    shift = (cfg.sign * base_shift[0], cfg.sign * base_shift[1], cfg.sign * base_shift[2])

    action = "shift" if cfg.sign >= 0 else "unshift"
    logger.info(
        "Step 7 - WaSuRe %s | tiles=%d | shift=%s | in=%s | out=%s",
        action, len(ply_files), shift, in_tiles_dir, out_dir
    )

    n_written = 0
    n_skipped = 0
    n_flattened = 0

    for in_ply in ply_files:
        out_ply = out_dir / in_ply.name
        if out_ply.exists() and not cfg.overwrite:
            n_skipped += 1
            continue

        try:
            repaired = apply_shift_to_ply(
                input_ply=in_ply,
                output_ply=out_ply,
                shift=shift,
                t_coords=cfg.t_coords,
                ascii_out=cfg.ascii_out,
                xml_path=None,  # IMPORTANT: do not re-read XML per tile
                logger=logger,
            )
            n_flattened += int(repaired)
            n_written += 1
        except Exception:
            # Keep traceback in main log (useful for debugging one bad tile)
            logger.exception("%s failed for: %s", action, in_ply)
            raise

    logger.info(
        "Step 7 done (%s) | written=%d | skipped=%d | repaired_flattened=%d | out=%s",
        action, n_written, n_skipped, n_flattened, out_dir
    )
    return out_dir