from dataclasses import dataclass

@dataclass
class DataConfig:
    """
    GeoLife (Kaggle / Microsoft Research) : dataset composé de dossiers Data/000..181.
    On convertit d'abord les .plt en parquet "points", puis en parquet "episodes".
    """
    geolife_root: str = "data/geolife_raw"  # contient le dossier Data/
    points_parquet: str = "data/processed/geolife_points.parquet"
    episodes_parquet: str = "data/processed/geolife_episodes.parquet"

    # Pour aller vite au début (tu peux augmenter ensuite)
    max_users: int = 60
    max_trajectories_per_user: int = 40
    max_points_total: int = 2_000_000  # sécurité mémoire

@dataclass
class GridConfig:
    # simplification volontaire : zones = grille lat/lon
    n_lat: int = 20
    n_lon: int = 20

@dataclass
class SimConfig:
    # Batterie / conso (simulées)
    battery_kwh: float = 60.0
    consumption_kwh_per_km: float = 0.28
    soc_init_min: float = 0.5
    soc_init_max: float = 0.85
    soc_target: float = 0.8
    soc_critical: float = 0.1

    # Temps discret pour MDP (GeoLife est dense => on ré-échantillonne)
    step_minutes: int = 1
    max_steps_per_episode: int = 180  # 3h (tu peux augmenter)

    # Stations (générées à partir de la densité des zones)
    n_stations: int = 10
    n_ports_per_station: int = 4
    station_power_kw: float = 22.0
    price_min: float = 0.20
    price_max: float = 0.35

@dataclass
class RLConfig:
    gamma: float = 0.99
    lr: float = 2e-4
    hidden: int = 128
    episodes_local: int = 8

@dataclass
class FLConfig:
    rounds: int = 6
    clients_per_round: int = 12

    # FedDyn
    feddyn_alpha: float = 0.05

@dataclass
class EvalConfig:
    # Evaluation congestion multi-véhicules
    n_eval_vehicles: int = 20   # <-- ton choix
    n_eval_episodes: int = 30   # nb d'épisodes test
