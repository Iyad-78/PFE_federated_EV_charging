import math
from dataclasses import dataclass

@dataclass(frozen=True)
class Zone:
    i: int
    j: int

def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def gps_to_zone(lat: float, lon: float,
                lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                n_lat: int, n_lon: int) -> Zone:
    # Eviter divisions par zéro
    if lat_max <= lat_min or lon_max <= lon_min:
        raise ValueError("Invalid GPS bounds.")

    di = (lat - lat_min) / (lat_max - lat_min)
    dj = (lon - lon_min) / (lon_max - lon_min)

    i = int(math.floor(di * n_lat))
    j = int(math.floor(dj * n_lon))

    i = clamp_int(i, 0, n_lat - 1)
    j = clamp_int(j, 0, n_lon - 1)
    return Zone(i=i, j=j)

def manhattan_distance(a: Zone, b: Zone) -> int:
    return abs(a.i - b.i) + abs(a.j - b.j)
