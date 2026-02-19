# src/stations.py

from dataclasses import dataclass, field
from typing import List, Tuple
from src.utils_geo import Zone


@dataclass
class ChargingSession:
    veh_id: int
    minutes_left: int


@dataclass
class Station:
    station_id: int
    zone: Zone
    n_ports: int
    power_kw: float
    price_kwh: float

    charging: List[ChargingSession] = field(default_factory=list)
    queue: List[Tuple[int, int]] = field(default_factory=list)  # (veh_id, session_minutes)

    @property
    def occupation(self) -> int:
        return len(self.charging)

    @property
    def queue_length(self) -> int:
        return len(self.queue)

    # -----------------------
    # Helpers
    # -----------------------
    def is_charging(self, veh_id: int) -> bool:
        return any(s.veh_id == veh_id for s in self.charging)

    def queued_position(self, veh_id: int) -> int:
        """Position 0 = premier de la file. Retourne -1 si absent."""
        for idx, (v, _) in enumerate(self.queue):
            if v == veh_id:
                return idx
        return -1

    def _avg_remaining_minutes(self) -> int:
        """
        Approx réaliste du temps restant sur les ports occupés.
        Sert à éviter wait=0 quand la station est pleine.
        """
        if not self.charging:
            return 0
        return int(sum(max(0, s.minutes_left) for s in self.charging) / len(self.charging))

    def estimate_wait_minutes(self, avg_session_minutes: int) -> int:
        """
        Estimation "publique" (utilisée dans l'observation):
        - si port libre => 0
        - sinon => (temps restant moyen sur ports) + (taille file) * durée moyenne
        """
        avg_session_minutes = int(max(1, avg_session_minutes))
        if self.occupation < self.n_ports:
            return 0

        avg_remaining = self._avg_remaining_minutes()
        # une personne en queue ajoute ~ avg_session_minutes avant de commencer
        return int(avg_remaining + self.queue_length * avg_session_minutes)

    def plug_or_queue(self, veh_id: int, session_minutes: int) -> Tuple[bool, int]:
        """
        Règle finale (stable + réaliste):
        - Si déjà en charge => accepted=True, wait=0
        - Si déjà en file  => accepted=False, wait estimée réaliste (reste + personnes devant)
        - Sinon :
            - si port libre => start maintenant
            - sinon => enqueue et wait estimée réaliste (reste + personnes devant)

        IMPORTANT: si station pleine => wait_est >= 1 (jamais 0).
        """
        session_minutes = int(max(1, session_minutes))

        # 1) déjà en charge
        if self.is_charging(veh_id):
            return True, 0

        # station pleine ?
        station_full = (self.occupation >= self.n_ports)
        avg_remaining = self._avg_remaining_minutes()

        # 2) déjà en file => ne PAS ré-ajouter
        pos = self.queued_position(veh_id)
        if pos != -1:
            # personnes devant = pos
            # attente = temps restant moyen + pos * durée stockée (approx)
            _, stored_minutes = self.queue[pos]
            stored_minutes = int(max(1, stored_minutes))
            wait_est = avg_remaining + pos * stored_minutes
            if station_full:
                wait_est = max(1, wait_est)  # jamais 0 si plein
            return False, int(wait_est)

        # 3) port libre => start
        if not station_full:
            self.charging.append(
                ChargingSession(veh_id=veh_id, minutes_left=session_minutes)
            )
            return True, 0

        # 4) sinon => enqueue (une seule fois)
        self.queue.append((veh_id, session_minutes))
        pos = self.queue_length - 1  # position en file après ajout
        wait_est = avg_remaining + pos * session_minutes

        # station pleine => wait > 0 garanti
        wait_est = max(1, wait_est)
        return False, int(wait_est)

    def step_time(self, step_minutes: int) -> None:
        """
        Fait avancer le temps :
        - décrémente minutes_left
        - libère ports si session finie
        - fait entrer des véhicules de la queue si ports libres (FIFO)
        """
        step_minutes = int(max(1, step_minutes))

        # 1) avancer les sessions en charge
        new_charging: List[ChargingSession] = []
        for s in self.charging:
            s.minutes_left -= step_minutes
            if s.minutes_left > 0:
                new_charging.append(s)
        self.charging = new_charging

        # 2) remplir les ports libres avec la queue (FIFO)
        while self.queue and self.occupation < self.n_ports:
            veh_id, session_minutes = self.queue.pop(0)
            self.charging.append(
                ChargingSession(
                    veh_id=veh_id,
                    minutes_left=int(max(1, session_minutes))
                )
            )
