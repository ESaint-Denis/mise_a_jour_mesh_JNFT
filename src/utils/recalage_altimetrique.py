# -*- coding: utf-8 -*-
"""
Recalage altimétrique des dalles MNS de corrélation sur les dalles MNS LiDAR.

Ce module estime un décalage vertical (offset) entre :
- un MNS issu de corrélation (référence de grille),
- un MNS issu du LiDAR (ré-échantillonné sur la grille du MNS corrélation),

puis applique ce décalage au MNS de corrélation afin de produire un MNS recalé.

Principe général :
1) lecture du MNS corrélation (grille de référence),
2) ré-échantillonnage du MNS LiDAR sur la même grille,
3) calcul dz = corr - lidar_resampled,
4) estimation robuste de Δz par médiane + filtrage itératif MAD,
5) écriture du MNS corrélation corrigé : corr_recale = corr - Δz.

Auteur : ESaint-Denis
"""

# utils/recalage_altimetrique.py
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from utils.creation_arborescence import ProjectPaths
from utils.recuperation_donnees import setup_logger  # on réutilise le logger déjà en place


_TILE_RE_KM = re.compile(r"_(\d{4})_(\d{4})_")
_TILE_RE_M  = re.compile(r"_(\d{7})_(\d{7})")  # ex: _0605000_6933000


class MissingMatchingTileError(RuntimeError):
    """Erreur levée lorsqu'aucune correspondance MNS LiDAR / MNS corrélation n'est trouvée pour une dalle."""


@dataclass(frozen=True)
class RecalageConfig:
    """
    Paramètres du recalage altimétrique.

    Attributs
    ---------
    k_mad : float
        Facteur multiplicatif appliqué à l'écart-type robuste (estimé via MAD)
        pour filtrer les résidus (pixels considérés "stables").

    n_iter : int
        Nombre d'itérations maximum du filtrage itératif MAD.

    resampling : rasterio.warp.Resampling
        Méthode de ré-échantillonnage utilisée pour projeter le MNS LiDAR sur la
        grille du MNS corrélation (ex. bilinear).

    compress : str
        Compression GeoTIFF appliquée à l'écriture (ex. "deflate").

    overwrite : bool
        Si True, réécrit les fichiers de sortie s'ils existent déjà.
    """
    k_mad: float = 3.0
    n_iter: int = 3
    resampling: Resampling = Resampling.bilinear
    compress: str = "deflate"   # compression GeoTIFF
    overwrite: bool = True      # réécrire si déjà présent


def _parse_tile_xy_from_any_filename(name: str) -> tuple[int, int]:
    """
    Extrait les indices de dalle kilométrique (x_km, y_km) depuis un nom de fichier.

    Deux formats sont supportés :
    - Format "km" : ..._0605_6933_...
    - Format "m"  : ..._0605000_6933000...

    Parameters
    ----------
    name : str
        Nom de fichier.

    Returns
    -------
    tuple[int, int]
        (x_km, y_km) indices kilométriques.

    Raises
    ------
    ValueError
        Si le motif n'est pas détecté ou si des coordonnées mètres ne sont pas multiples de 1000.
    """
    m_km = _TILE_RE_KM.search(name)
    if m_km:
        return int(m_km.group(1)), int(m_km.group(2))

    m_m = _TILE_RE_M.search(name)
    if m_m:
        x_m = int(m_m.group(1))
        y_m = int(m_m.group(2))

        if x_m % 1000 != 0 or y_m % 1000 != 0:
            raise ValueError(f"Coordonnées mètres non multiples de 1000 dans: {name}")

        return x_m // 1000, y_m // 1000

    raise ValueError(f"Impossible d'extraire l'identifiant de dalle (xxxx_yyyy) depuis: {name}")


def _find_mns_lidar_file_for_tile(mns_lidar_dir: Path, x_km: int, y_km: int) -> Path:
    """
    Recherche le fichier MNS LiDAR correspondant à une dalle (x_km, y_km).

    Stratégie :
    - on parcourt les GeoTIFF (*.tif) du répertoire `mns_lidar_dir`,
    - on retient ceux dont le nom contient `_{XXXX}_{YYYY}_`,
    - on impose en plus la présence de `_MNS_` pour limiter les faux positifs,
    - si plusieurs candidats existent, on prend le plus récent (mtime).

    Parameters
    ----------
    mns_lidar_dir : Path
        Répertoire contenant les dalles MNS LiDAR.
    x_km, y_km : int
        Indices kilométriques de la dalle.

    Returns
    -------
    Path
        Chemin du fichier MNS LiDAR retenu.

    Raises
    ------
    MissingMatchingTileError
        Si aucun fichier correspondant n'est trouvé.
    """
    pattern = f"_{x_km:04d}_{y_km:04d}_"
    candidates = []
    for p in mns_lidar_dir.glob("*.tif"):
        n = p.name
        if pattern in n and "_MNS_" in n:
            candidates.append(p)

    if len(candidates) == 0:
        raise MissingMatchingTileError(
            f"Aucun MNS LiDAR trouvé pour la dalle {x_km:04d}_{y_km:04d} dans {mns_lidar_dir}"
        )
    if len(candidates) > 1:
        # Cas rare : plusieurs versions. On prend la plus récente (mtime), et on loguera.
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return candidates[0]


def read_raster(path: Path) -> tuple[np.ndarray, float | int | None, dict]:
    """
    Lit un raster (bande 1) et renvoie l'image + nodata + profil rasterio.

    Parameters
    ----------
    path : Path
        Chemin du fichier raster.

    Returns
    -------
    tuple[np.ndarray, float | int | None, dict]
        - array : tableau float32 de la bande 1,
        - nodata : valeur nodata (ou None),
        - profile : profil rasterio (copie) pour l'écriture.
    """
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        profile = src.profile.copy()
    return arr, nodata, profile


def resample_to_reference_grid(
    src_path: Path,
    ref_profile: dict,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """
    Ré-échantillonne un raster sur la grille définie par un profil de référence.

    Cette fonction utilise `rasterio.warp.reproject` pour :
    - reprojeter si nécessaire,
    - et/ou ré-échantillonner sur la transformée et la taille de la grille de référence.

    Parameters
    ----------
    src_path : Path
        Raster source à reprojecter / ré-échantillonner.
    ref_profile : dict
        Profil rasterio définissant la grille destination (width, height, transform, crs, nodata...).
    resampling : rasterio.warp.Resampling
        Méthode de ré-échantillonnage.

    Returns
    -------
    np.ndarray
        Tableau float32 ré-échantillonné sur la grille de référence.
    """
    ref_h = ref_profile["height"]
    ref_w = ref_profile["width"]
    dst_nodata = ref_profile.get("nodata", None)

    dst = np.full((ref_h, ref_w), dst_nodata if dst_nodata is not None else np.nan, dtype=np.float32)

    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            dst_nodata=dst_nodata,
            resampling=resampling,
        )

    return dst


def robust_offset_from_dz(
    dz: np.ndarray,
    valid_mask: np.ndarray,
    k_mad: float = 3.0,
    n_iter: int = 3,
) -> tuple[float, np.ndarray]:
    """
    Estime un décalage vertical robuste (Δz) par médiane avec filtrage MAD itératif.

    Entrées :
    - dz : tableau des différences d'altitude pixel à pixel,
    - valid_mask : masque des pixels exploitables (finite, != nodata, etc.).

    Algorithme (itératif) :
    1) calcul de la médiane sur les pixels courants,
    2) calcul des résidus r = dz - médiane,
    3) estimation robuste de la dispersion via MAD,
    4) conservation des pixels dont |r - med(r)| <= k_mad * 1.4826 * MAD,
    5) répétition jusqu'à n_iter.

    Convention :
    - dz = new - old (ici : dz = corr - lidar_resampled)
    - Δz = médiane(dz sur pixels stables)

    Parameters
    ----------
    dz : np.ndarray
        Différences verticales pixel à pixel.
    valid_mask : np.ndarray
        Masque booléen des pixels valides.
    k_mad : float
        Seuil de rejet en nombre d'écarts-types robustes.
    n_iter : int
        Nombre d'itérations du filtrage.

    Returns
    -------
    tuple[float, np.ndarray]
        - delta_z : décalage vertical robuste estimé (mètres),
        - stable_mask : masque final des pixels conservés ("stables").

    Raises
    ------
    RuntimeError
        Si aucun pixel valide ne subsiste pour estimer Δz.
    """
    mask = valid_mask.copy()
    delta = 0.0

    for _ in range(max(1, n_iter)):
        v = dz[mask]
        if v.size == 0:
            raise RuntimeError("Plus aucun pixel valide pour estimer le décalage vertical.")

        delta = float(np.median(v))
        r = dz - delta

        rv = r[mask]
        med = float(np.median(rv))
        mad = float(np.median(np.abs(rv - med)))

        if mad < 1e-12:
            # quasi constant : on s'arrête
            break

        robust_std = 1.4826 * mad
        mask = valid_mask & (np.abs(r - med) <= k_mad * robust_std)

    return delta, mask


def _build_valid_mask(arr: np.ndarray, nodata: float | int | None) -> np.ndarray:
    """
    Construit un masque des pixels valides d'un raster.

    Un pixel est valide si :
    - la valeur est finie (pas NaN/Inf),
    - et (si nodata est défini) la valeur est différente de nodata.

    Parameters
    ----------
    arr : np.ndarray
        Tableau raster.
    nodata : float | int | None
        Valeur nodata, ou None.

    Returns
    -------
    np.ndarray
        Masque booléen des pixels valides.
    """
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= (arr != nodata)
    return valid


def write_corrected_dem(
    ref_profile: dict,
    dem_new: np.ndarray,
    nodata_new: float | int | None,
    delta_z: float,
    out_path: Path,
    compress: str = "deflate",
    overwrite: bool = True,
) -> None:
    """
    Écrit un MNS corrigé par décalage vertical.

    Correction appliquée :
    - dem_corr = dem_new - delta_z

    Le nodata est préservé : les pixels nodata dans `dem_new` restent nodata dans `dem_corr`.

    Parameters
    ----------
    ref_profile : dict
        Profil rasterio à utiliser pour l'écriture (transform, crs, dimensions...).
    dem_new : np.ndarray
        MNS à corriger (tableau).
    nodata_new : float | int | None
        Valeur nodata associée à dem_new.
    delta_z : float
        Décalage vertical estimé (mètres).
    out_path : Path
        Chemin du fichier GeoTIFF de sortie.
    compress : str
        Compression GeoTIFF.
    overwrite : bool
        Si False et le fichier existe, ne réécrit pas.

    Returns
    -------
    None
    """
    if out_path.exists() and not overwrite:
        return

    dem_corr = dem_new - delta_z

    if nodata_new is not None:
        dem_corr[dem_new == nodata_new] = nodata_new

    profile = ref_profile.copy()
    profile.update(dtype="float32", compress=compress)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dem_corr.astype(np.float32), 1)


def run_recalage_altimetrique(
    paths: ProjectPaths,
    cfg: RecalageConfig,
) -> None:
    """
    Recale altimétriquement les dalles MNS de corrélation sur les dalles MNS LiDAR.

    Entrées
    -------
    - paths.mns_correlation :
        Dalles MNS de corrélation (utilisées comme grille de référence).
    - paths.mns_lidar :
        Dalles MNS LiDAR (ré-échantillonnées sur la grille corrélation).

    Sorties
    -------
    - paths.mns_recale :
        Dalles MNS de corrélation corrigées (même nom que l'entrée).

    Convention du décalage
    ----------------------
    - dz = corr - lidar_resampled
    - delta_z = médiane(dz) estimée sur pixels stables (filtrage MAD)
    - corr_recale = corr - delta_z

    Parameters
    ----------
    paths : ProjectPaths
        Arborescence du projet (répertoires d'entrée/sortie).
    cfg : RecalageConfig
        Paramètres du recalage (k_mad, n_iter, resampling, compression, overwrite).

    Raises
    ------
    RuntimeError
        Si aucun fichier MNS corrélation n'est trouvé, ou si l'estimation échoue.
    MissingMatchingTileError
        Si aucun MNS LiDAR correspondant n'est trouvé pour une dalle.
    """
    logger = setup_logger(paths.logs, name="recalage_altimetrique")

    corr_files = sorted(paths.mns_correlation.glob("*.tif"))
    if not corr_files:
        raise RuntimeError(f"Aucune dalle trouvée dans {paths.mns_correlation}")

    logger.info("Nombre de dalles MNS corrélation à traiter: %d", len(corr_files))
    logger.info("Paramètres: k_mad=%.2f, n_iter=%d, resampling=%s", cfg.k_mad, cfg.n_iter, cfg.resampling)

    for i, corr_path in enumerate(corr_files, start=1):
        t0 = time.time()
        x_km, y_km = _parse_tile_xy_from_any_filename(corr_path.name)

        logger.info("---- Dalle %d / %d : %s (id %04d_%04d) ----", i, len(corr_files), corr_path.name, x_km, y_km)

        # 1) Trouver le MNS LiDAR correspondant
        lidar_path = _find_mns_lidar_file_for_tile(paths.mns_lidar, x_km, y_km)
        logger.info("MNS LiDAR correspondant: %s", lidar_path.name)

        # 2) Lire le MNS corrélation (référence)
        corr, nodata_corr, prof_corr = read_raster(corr_path)

        ref_profile = {
            "height": prof_corr["height"],
            "width": prof_corr["width"],
            "transform": prof_corr["transform"],
            "crs": prof_corr["crs"],
            "nodata": prof_corr.get("nodata", None),
            **prof_corr,  # on garde le profil complet pour l'écriture
        }

        # 3) Resampler LiDAR sur la grille corr
        lidar_on_corr = resample_to_reference_grid(
            src_path=lidar_path,
            ref_profile=ref_profile,
            resampling=cfg.resampling,
        )

        # 4) Masque de validité + dz
        valid = _build_valid_mask(corr, nodata_corr)
        # nodata de la grille destination lors du reproject = ref_profile["nodata"]
        nodata_dst = ref_profile.get("nodata", None)
        valid &= _build_valid_mask(lidar_on_corr, nodata_dst)

        dz = corr - lidar_on_corr

        # 5) Estimation robuste
        delta_z, stable_mask = robust_offset_from_dz(
            dz=dz,
            valid_mask=valid,
            k_mad=cfg.k_mad,
            n_iter=cfg.n_iter,
        )

        n_stable = int(stable_mask.sum())
        n_valid = int(valid.sum())
        dt = time.time() - t0

        logger.info("Δz estimé (corr - lidar): %+0.3f m", delta_z)
        logger.info("Pixels valides: %d | Pixels stables: %d (%.1f%%)", n_valid, n_stable, 100.0 * n_stable / max(1, n_valid))

        # 6) Écriture du MNS corrélation recalé (dans MNS_recale) avec le même nom
        out_path = paths.mns_recale / corr_path.name
        write_corrected_dem(
            ref_profile=prof_corr,
            dem_new=corr,
            nodata_new=nodata_corr,
            delta_z=delta_z,
            out_path=out_path,
            compress=cfg.compress,
            overwrite=cfg.overwrite,
        )

        logger.info("Écrit: %s", out_path.name)
        logger.info("Temps: %.1f s", dt)

    logger.info("Recalage altimétrique terminé.")