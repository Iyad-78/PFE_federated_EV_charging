# src/env_ev.py

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

from src.utils_geo import Zone, manhattan_distance
from src.vehicles import Vehicle
from src.stations import Station
from src.config import SimConfig


@dataclass
class StepInfo:
    cost_eur: float
    wait_min: int
    detour: int
    occ: int


class EVGeoLifeEnv:
    """
    Env véhicule-centré : 1 agent (user_id) joue sur 1 épisode (traj_id).
    Episode = DataFrame avec colonnes t, zi, zj, dist_km, dt.
    """

    def __init__(
        self,
        episodes_by_user: Dict[str, List[pd.DataFrame]],
        stations: List[Station],
        sim_cfg: SimConfig,
        seed: int = 0,
    ):
        self.episodes_by_user = episodes_by_user
        self.stations = stations
        self.cfg = sim_cfg
        self.rng = random.Random(seed)

        self.vehicle: Optional[Vehicle] = None
        self.ep: Optional[pd.DataFrame] = None
        self.t = 0

        # station choisie à ce step (peut être None)
        self.current_station: Optional[int] = None

    def reset(self, user_id: str) -> np.ndarray:
        eps = self.episodes_by_user[user_id]
        if not eps:
            raise RuntimeError("User has no episodes.")

        self.ep = self.rng.choice(eps).reset_index(drop=True)
        self.t = 0
        self.current_station = None

        z0 = Zone(int(self.ep.loc[0, "zi"]), int(self.ep.loc[0, "zj"]))

        soc0 = self.rng.uniform(self.cfg.soc_init_min, self.cfg.soc_init_max)
        self.vehicle = Vehicle(
            veh_id=int(user_id),
            zone=z0,
            soc=soc0,
            battery_kwh=self.cfg.battery_kwh,
            consumption_kwh_per_km=self.cfg.consumption_kwh_per_km,
        )
        return self._get_obs()

    def _safe_t(self) -> int:
        """Clamp current t to valid dataframe row index."""
        assert self.ep is not None
        return max(0, min(int(self.t), len(self.ep) - 1))

    def _time_features(self) -> Tuple[float, float]:
        assert self.ep is not None
        idx = self._safe_t()
        dt = pd.to_datetime(self.ep.loc[idx, "dt"], errors="coerce")
        if pd.isna(dt):
            minutes = idx * self.cfg.step_minutes
        else:
            minutes = int(dt.hour) * 60 + int(dt.minute)

        angle = 2 * math.pi * (minutes % (24 * 60)) / (24 * 60)
        return math.sin(angle), math.cos(angle)

    def _estimate_avg_session_minutes(self) -> int:
        soc_mean = 0.4
        e_need = max(0.0, (self.cfg.soc_target - soc_mean) * self.cfg.battery_kwh)
        minutes = int(math.ceil((e_need / self.cfg.station_power_kw) * 60))
        return max(5, minutes)

    def _get_obs(self) -> np.ndarray:
        assert self.vehicle is not None

        soc = float(self.vehicle.soc)
        tsin, tcos = self._time_features()
        avg_session = self._estimate_avg_session_minutes()

        dists, waits, prices, occs = [], [], [], []
        for st in self.stations:
            d = manhattan_distance(self.vehicle.zone, st.zone)
            dists.append(d)
            waits.append(st.estimate_wait_minutes(avg_session))
            prices.append(st.price_kwh)
            occs.append(st.occupation)

        dists = np.array(dists, dtype=np.float32)
        waits = np.array(waits, dtype=np.float32)
        prices = np.array(prices, dtype=np.float32)
        occs = np.array(occs, dtype=np.float32)

        dists_norm = dists / (len(self.stations) * 2 + 1)
        waits_norm = waits / (4 * avg_session + 1)
        prices_norm = (prices - self.cfg.price_min) / max(
            1e-6, (self.cfg.price_max - self.cfg.price_min)
        )
        occs_norm = occs / max(1.0, float(self.cfg.n_ports_per_station))

        return np.concatenate(
            [
                np.array([soc, tsin, tcos], dtype=np.float32),
                dists_norm,
                waits_norm,
                prices_norm,
                occs_norm,
            ]
        )

    def _is_actually_charging_now(self) -> bool:
        """True si le véhicule est en charge dans la station courante."""
        if self.current_station is None:
            return False
        st = self.stations[self.current_station]
        return st.is_charging(self.vehicle.veh_id)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, StepInfo]:
        assert self.vehicle is not None
        assert self.ep is not None

        nS = len(self.stations)
        done = False

        cost = 0.0
        wait_min = 0
        detour = 0
        occ = 0

        # =========================
        # 1) Action du véhicule
        # =========================
        if action == nS:
            # WAIT = ne pas (re)demander une borne
            self.current_station = None
        else:
            self.current_station = int(action)
            st = self.stations[self.current_station]

            # déplacement (détour) vers la station
            detour = manhattan_distance(self.vehicle.zone, st.zone)
            self.vehicle.zone = st.zone

            # énergie nécessaire pour atteindre soc_target
            e_need = max(
                0.0,
                (self.cfg.soc_target - self.vehicle.soc) * self.cfg.battery_kwh,
            )

            # durée totale demandée (approximée)
            session_min = int(math.ceil((e_need / st.power_kw) * 60))
            session_min = max(5, session_min)

            accepted, wait_min = st.plug_or_queue(self.vehicle.veh_id, session_min)

            # coût basé sur l'énergie réellement chargée pendant UN PAS (réaliste)
            if accepted and e_need > 0:
                e_add = st.power_kw * (self.cfg.step_minutes / 60.0)  # kWh ajoutables ce step
                e_real = min(e_add, e_need)
                cost = e_real * st.price_kwh

        # =========================
        # 2) Avancer le temps stations
        # =========================
        for st in self.stations:
            st.step_time(self.cfg.step_minutes)

        # =========================
        # 3) Charger si réellement en charge
        # =========================
        if self._is_actually_charging_now():
            st = self.stations[self.current_station]
            self.vehicle.charge_minutes(st.power_kw, self.cfg.step_minutes)
            occ = st.occupation

        # =========================
        # 4) Avancer le temps de l’épisode
        # IMPORTANT: si on charge, on NE BOUGE PAS.
        # =========================
        is_charging_now = self._is_actually_charging_now()

        self.t += 1
        if self.t >= len(self.ep):
            done = True
            obs = self._get_obs()
            r = 0.0
            return obs, r, done, StepInfo(cost, wait_min, detour, occ)

        if not is_charging_now:
            # suit la trajectoire seulement si pas en charge
            new_zone = Zone(int(self.ep.loc[self.t, "zi"]), int(self.ep.loc[self.t, "zj"]))
            dist_km = float(self.ep.loc[self.t, "dist_km"])
            self.vehicle.zone = new_zone
            self.vehicle.consume_trip(dist_km)
        # else: reste à la station, pas de conso, pas de déplacement

        # =========================
        # 5) Reward (temps / utilité / efficacité)
        # =========================
        r_time = -0.01 * detour - 0.001 * wait_min
        r_util = -0.05 * cost

        if self.vehicle.soc < self.cfg.soc_critical:
            r_util -= 2.0

        # efficacité globale : pénaliser la congestion (signal plus réaliste)
        r_eff = 0.0
        if self.current_station is not None:
            st = self.stations[self.current_station]
            load_ratio = st.occupation / float(max(1, self.cfg.n_ports_per_station))
            r_eff -= 0.2 * load_ratio

        r = r_time + r_util + r_eff

        # fin si on dépasse la limite max de steps
        if self.t >= self.cfg.max_steps_per_episode:
            done = True

        obs = self._get_obs()
        return obs, float(r), done, StepInfo(cost, wait_min, detour, occ)
