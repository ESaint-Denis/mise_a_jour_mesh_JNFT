# -*- coding: utf-8 -*-
"""
Interrogation WFS (BD TOPO V3) pour connaître le(s) département(s) intersectant une tuile 1 km.

But du module
-------------
Ce module sert à déterminer, pour une tuile kilométrique (x_km, y_km) en Lambert-93 (EPSG:2154),
quels départements (BD TOPO V3) intersectent l’emprise de cette tuile.

Principe
--------
1) On construit la bbox de la tuile 1 km en mètres (xmin, ymin, xmax, ymax).
2) On interroge le WFS de la Géoplateforme IGN (GetFeature) avec un filtre BBOX.
3) Pour chaque département retourné, on calcule l’aire d’intersection exacte (Shapely),
   puis on renvoie une liste triée par aire décroissante.

Sortie
------
La fonction renvoie une liste de tuples :
    (code_insee, aire_intersection_en_m2)
triée par aire décroissante (déterministe et pratique si une tuile chevauche plusieurs départements).

Dépendances
-----------
- requests : pour l’appel HTTP WFS
- shapely  : pour manipuler les géométries (bbox tuile, géométries WFS, intersections)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import requests
from shapely.geometry import box, shape


@dataclass(frozen=True)
class DepartementWfsConfig:
    # URL du service WFS (Géoplateforme IGN). Par défaut celui exposant la BD TOPO.
    wfs_endpoint: str = "https://data.geopf.fr/wfs/ows"

    # Nom de couche (typeName) WFS : ici la couche "departement" de la BD TOPO V3.
    type_name: str = "BDTOPO_V3:departement"

    # Système de coordonnées demandé au service.
    # Ici EPSG:2154 (Lambert-93), cohérent avec tes tuiles km (x_km, y_km) en mètres.
    srs: str = "EPSG:2154"

    # Timeout réseau (secondes) pour éviter de bloquer le pipeline si le service répond mal.
    timeout_s: int = 60


def get_departements_for_tile_bbox(
    x_km: int,
    y_km: int,
    cfg: DepartementWfsConfig,
    tile_size_m: int = 1000,
) -> List[Tuple[str, float]]:
    """
    Renvoie la liste des départements intersectant une tuile 1 km.

    Paramètres
    ----------
    x_km, y_km
        Indices kilométriques de la tuile (Lambert-93). On suppose que :
        xmin = x_km * 1000, ymin = y_km * 1000.
    cfg
        Configuration WFS (endpoint, couche, CRS, timeout).
    tile_size_m
        Taille de la tuile en mètres (par défaut 1000 m -> tuile 1 km).

    Retour
    ------
    List[Tuple[str, float]]
        Liste de (code_insee, aire_intersection_m2), triée par aire décroissante.

    Notes
    -----
    - La requête WFS est filtrée par BBOX : on récupère uniquement les départements
      susceptibles d’intersecter la tuile (peu d’objets, donc rapide).
    - L’aire retournée est une aire plane en unités du CRS (m² en EPSG:2154),
      puisqu’on travaille directement en Lambert-93.
    - Tri décroissant : si la tuile chevauche deux départements, le premier élément est
      celui qui couvre la plus grande part de la tuile (utile pour une affectation “majoritaire”).
    """
    # Conversion indices km -> coordonnées métriques de l'emprise de tuile
    xmin = x_km * 1000
    ymin = y_km * 1000
    xmax = xmin + tile_size_m
    ymax = ymin + tile_size_m

    # Géométrie de la tuile (rectangle) pour calculer ensuite les intersections exactes
    tile_geom = box(xmin, ymin, xmax, ymax)

    # --- Construction des paramètres WFS GetFeature
    # On utilise WFS 2.0.0 + sortie GeoJSON.
    # Le filtre BBOX restreint fortement le volume de données retourné.
    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": cfg.type_name,
        "SRSNAME": cfg.srs,
        "OUTPUTFORMAT": "application/json",
        # BBOX : xmin,ymin,xmax,ymax + CRS (certains serveurs exigent le CRS en 5e champ)
        "BBOX": f"{xmin},{ymin},{xmax},{ymax},{cfg.srs}",
        # Optionnel : on pourrait limiter le nombre d'objets, mais ici les départements sont très peu nombreux.
        # "COUNT": 50,
    }

    # Appel HTTP au service WFS
    r = requests.get(cfg.wfs_endpoint, params=params, timeout=cfg.timeout_s)

    # Fallback : si le serveur répond 400, on retente une variante (historique/robustesse).
    # (Dans ce code, la variante est identique, mais l’intention est de pouvoir adapter facilement
    #  si le service change d’exigence sur le format du BBOX.)
    if r.status_code == 400:
        params2 = dict(params)
        params2["BBOX"] = f"{xmin},{ymin},{xmax},{ymax},{cfg.srs}"
        r = requests.get(cfg.wfs_endpoint, params=params2, timeout=cfg.timeout_s)

    # Lève une exception si code HTTP d’erreur (4xx/5xx)
    r.raise_for_status()

    # Parsing GeoJSON
    gj = r.json()

    # Dictionnaire d’accumulation : code_insee -> aire (m²)
    # (Accumuler permet de rester robuste si plusieurs features partageaient le même code,
    #  même si en pratique ce n’est pas censé arriver.)
    out: dict[str, float] = {}

    for feat in gj.get("features", []):
        props = feat.get("properties", {}) or {}

        # Champ attendu dans la BD TOPO V3 : "code_insee"
        code = props.get("code_insee")

        # Fallback si le champ a un nom différent (robustesse entre variantes/diffusions)
        if not code:
            for k in ("CODE_INSEE", "code_insee_dep", "insee_dep", "code_dep", "code_dept"):
                if k in props:
                    code = props[k]
                    break

        # Géométrie GeoJSON
        geom = feat.get("geometry")

        # Si on n’a pas de code ou pas de géométrie, on ignore la feature
        if not code or geom is None:
            continue

        # Conversion GeoJSON -> géométrie Shapely
        dep_geom = shape(geom)

        # Aire d'intersection exacte entre le département et la tuile (m² en EPSG:2154)
        inter_area = dep_geom.intersection(tile_geom).area

        # On ne conserve que les départements qui recouvrent réellement la tuile
        if inter_area > 0:
            out[code] = out.get(code, 0.0) + float(inter_area)

    # Tri par aire décroissante : le “département dominant” est en premier
    return sorted(out.items(), key=lambda kv: kv[1], reverse=True)