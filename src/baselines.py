import numpy as np


def wait_action(n_stations: int) -> int:
    return n_stations


def nearest_station_policy(obs, n_stations: int, soc_threshold: float = 0.40) -> int:
    """
    Baseline défendable :
    - si SoC > threshold : WAIT (pas besoin de recharger)
    - sinon : choisir la station la plus proche
    """
    soc = float(obs[0])
    if soc > soc_threshold:
        return n_stations  # WAIT

    offset = 3
    dists = obs[offset : offset + n_stations]
    return int(np.argmin(dists))


def min_wait_price_policy(obs, n_stations: int) -> int:
    """
    Choisit la station qui minimise (attente + prix).
    """
    offset = 3
    waits = obs[offset + n_stations : offset + 2 * n_stations]
    prices = obs[offset + 2 * n_stations : offset + 3 * n_stations]
    score = waits + prices
    return int(np.argmin(score))


def smart_rule_policy(obs, n_stations: int, soc_threshold: float = 0.55) -> int:
    """
    Règle simple plus "réaliste" :
    - si SoC > threshold : WAIT (ne recharge pas)
    - sinon : choisir min(attente + prix)
    """
    soc = float(obs[0])
    if soc > soc_threshold:
        return wait_action(n_stations)
    return min_wait_price_policy(obs, n_stations)
