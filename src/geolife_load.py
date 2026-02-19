import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils_geo import gps_to_zone, Zone

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    # distance géodésique simple (km)
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def build_episodes(points_parquet: str,
                   out_episodes_parquet: str,
                   n_lat: int, n_lon: int,
                   step_minutes: int = 1,
                   max_steps_per_episode: int = 180) -> dict:
    """
    Construit des épisodes par (user_id, traj_id) :
    - ré-échantillonne à 1 min (ou step_minutes)
    - calcule dist_km entre points successifs
    - convertit GPS -> zone (grille)
    """
    df = pd.read_parquet(points_parquet).copy()
    df = df.dropna(subset=["dt", "lat", "lon", "user_id", "traj_id"])
    df = df.sort_values(["user_id", "traj_id", "dt"]).reset_index(drop=True)

    # bounds sur l'ensemble (pour grille stable)
    lat_min, lat_max = float(df["lat"].min()), float(df["lat"].max())
    lon_min, lon_max = float(df["lon"].min()), float(df["lon"].max())

    episodes = []
    kept = 0

    # groupby traj
    for (uid, tid), g in df.groupby(["user_id", "traj_id"], sort=False):
        g = g.sort_values("dt")
        # ré-échantillonnage : on garde 1 point par minute (le premier de la minute)
        g["dt_floor"] = g["dt"].dt.floor(f"{step_minutes}min")
        g = g.drop_duplicates(subset=["dt_floor"], keep="first")
        g = g.sort_values("dt_floor")

        if len(g) < 5:
            continue

        # limiter longueur épisode
        if len(g) > max_steps_per_episode:
            g = g.iloc[:max_steps_per_episode]

        lats = g["lat"].astype(float).to_numpy()
        lons = g["lon"].astype(float).to_numpy()
        dts = g["dt_floor"].astype("datetime64[ns]").to_numpy()

        # distances step->step
        dist = np.zeros(len(g), dtype=np.float32)
        for i in range(1, len(g)):
            dist[i] = float(haversine_km(lats[i-1], lons[i-1], lats[i], lons[i]))

        # zones
        zi = np.empty(len(g), dtype=np.int16)
        zj = np.empty(len(g), dtype=np.int16)
        for i in range(len(g)):
            z = gps_to_zone(
                float(lats[i]), float(lons[i]),
                lat_min, lat_max, lon_min, lon_max,
                n_lat, n_lon
            )
            zi[i], zj[i] = z.i, z.j

        ep = pd.DataFrame({
            "user_id": uid,
            "traj_id": tid,
            "t": np.arange(len(g), dtype=np.int32),
            "dt": dts,
            "lat": lats.astype(np.float32),
            "lon": lons.astype(np.float32),
            "dist_km": dist,
            "zi": zi,
            "zj": zj
        })
        episodes.append(ep)
        kept += 1

    if not episodes:
        raise RuntimeError("No episodes built. Check parsing / step_minutes / max_steps.")

    out = pd.concat(episodes, ignore_index=True)
    os.makedirs(Path(out_episodes_parquet).parent, exist_ok=True)
    out.to_parquet(out_episodes_parquet, index=False)

    summary = {
        "episodes": int(out[["user_id", "traj_id"]].drop_duplicates().shape[0]),
        "users": int(out["user_id"].nunique()),
        "rows": int(len(out)),
        "lat_min": lat_min, "lat_max": lat_max,
        "lon_min": lon_min, "lon_max": lon_max,
    }
    print("GeoLife episodes saved:", summary)
    return summary

def load_episodes_for_users(episodes_parquet: str) -> Dict[str, List[pd.DataFrame]]:
    """
    Retourne dict[user_id] -> liste d'épisodes (DataFrame trié t).
    """
    df = pd.read_parquet(episodes_parquet).copy()
    df = df.sort_values(["user_id", "traj_id", "t"]).reset_index(drop=True)

    episodes_by_user: Dict[str, List[pd.DataFrame]] = {}
    for (uid, tid), g in df.groupby(["user_id", "traj_id"], sort=False):
        episodes_by_user.setdefault(uid, []).append(g.reset_index(drop=True))
    return episodes_by_user
