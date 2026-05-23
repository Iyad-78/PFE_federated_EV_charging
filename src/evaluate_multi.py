# src/evaluate_multi.py

import copy
import numpy as np
import torch

from src.env_ev import EVGeoLifeEnv
from src.metrics import init_metrics, update_metrics, finalize


def evaluate_multi_vehicle(
    episodes_by_user,
    base_stations,
    sim_cfg,
    policy,
    user_ids,
    n_vehicles: int = 20,
    n_episodes: int = 30,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    policy.eval()
    results = []

    wait_action = len(base_stations)

    for _ in range(n_episodes):
        chosen_users = list(
            rng.choice(user_ids, size=min(n_vehicles, len(user_ids)), replace=False)
        )

        stations = copy.deepcopy(base_stations)

        envs = []
        mets = []
        done_flags = []

        for uid in chosen_users:
            env = EVGeoLifeEnv(episodes_by_user, stations, sim_cfg, seed=int(uid))
            env.reset(uid)
            envs.append(env)
            mets.append(init_metrics())
            done_flags.append(False)

        max_steps = sim_cfg.max_steps_per_episode

        for _t in range(max_steps):
            for i, env in enumerate(envs):
                if done_flags[i]:
                    continue

                obs = env._get_obs()
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    logits = policy(obs_t)
                    action = int(torch.argmax(logits, dim=1).item())

                _, _, done, info = env.step(action, advance_stations=False)

                did_request_charge = action != wait_action

                update_metrics(
                    mets[i],
                    cost=info.cost_eur,
                    wait_min=info.wait_min,
                    detour=info.detour,
                    soc=env.vehicle.soc,
                    soc_critical=sim_cfg.soc_critical,
                    did_request_charge=did_request_charge,
                )
                done_flags[i] = done

            for st in stations:
                st.step_time(sim_cfg.step_minutes)

            if all(done_flags):
                break

        ep = [finalize(m) for m in mets]
        keys = ["cost_mean", "wait_mean", "detour_mean", "soc_critical_rate", "charge_requests"]
        agg = {k: float(np.mean([e.get(k, 0.0) for e in ep])) for k in keys}
        results.append(agg)

    final = {k: float(np.mean([r[k] for r in results])) for k in results[0].keys()}
    return final