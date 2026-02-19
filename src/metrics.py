# src/metrics.py

from dataclasses import dataclass


@dataclass
class EpisodeMetrics:
    total_cost: float
    total_wait: int
    total_detour: int
    soc_critical_steps: int
    steps: int
    charge_requests: int  # nb de fois où on a demandé une charge (action != WAIT)


def init_metrics() -> EpisodeMetrics:
    return EpisodeMetrics(
        total_cost=0.0,
        total_wait=0,
        total_detour=0,
        soc_critical_steps=0,
        steps=0,
        charge_requests=0,
    )


def update_metrics(
    m: EpisodeMetrics,
    cost: float,
    wait_min: int,
    detour: int,
    soc: float,
    soc_critical: float,
    did_request_charge: bool,
):
    """
    did_request_charge :
      - True si l'agent a fait une demande de charge (action != WAIT)
      - False si WAIT
    """
    m.total_cost += float(cost)
    m.total_detour += int(detour)

    # Attente mesurée par demande de charge (pas par step)
    if did_request_charge:
        m.total_wait += int(wait_min)
        m.charge_requests += 1

    if soc < soc_critical:
        m.soc_critical_steps += 1

    m.steps += 1


def finalize(m: EpisodeMetrics) -> dict:
    if m.steps <= 0:
        return {"steps": 0}

    wait_mean = (m.total_wait / m.charge_requests) if m.charge_requests > 0 else 0.0

    return {
        "steps": m.steps,
        "cost_mean": m.total_cost / m.steps,
        "wait_mean": float(wait_mean),  # attente moyenne par demande de charge
        "detour_mean": m.total_detour / m.steps,
        "soc_critical_rate": m.soc_critical_steps / m.steps,
        "charge_requests": m.charge_requests,
    }
