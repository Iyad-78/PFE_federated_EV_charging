from dataclasses import dataclass
from src.utils_geo import Zone


@dataclass
class Vehicle:
    veh_id: int
    zone: Zone
    soc: float  # 0..1
    battery_kwh: float
    consumption_kwh_per_km: float

    def consume_trip(self, dist_km: float) -> None:
        dist_km = max(0.0, float(dist_km))
        e = dist_km * float(self.consumption_kwh_per_km)  # kWh
        self.soc = max(0.0, self.soc - e / float(self.battery_kwh))

    def charge_minutes(self, power_kw: float, minutes: int) -> None:
        power_kw = max(0.0, float(power_kw))
        minutes = max(0, int(minutes))
        e = power_kw * (minutes / 60.0)  # kWh
        self.soc = min(1.0, self.soc + e / float(self.battery_kwh))
