import os
import copy
import random
import numpy as np
import pandas as pd
import torch

from src.config import DataConfig, GridConfig, SimConfig, RLConfig, FLConfig, EvalConfig
from src.geolife_prepare import prepare_geolife_points
from src.geolife_load import build_episodes, load_episodes_for_users
from src.utils_geo import Zone
from src.stations import Station
from src.env_ev import EVGeoLifeEnv
from src.models import PolicyNet

# RL / FL helpers
from src.rl_train import train_local_from_model
from src.federated import set_param_vector, fedavg, feddyn_server_aggregate

# Multi-vehicle eval (policy réseau)
from src.evaluate_multi import evaluate_multi_vehicle

# Heuristics eval
from src.evaluate_heuristics import evaluate_heuristic_multi_vehicle
from src.baselines import nearest_station_policy, min_wait_price_policy, smart_rule_policy


def build_stations_from_zone_density(episodes_by_user, sim_cfg: SimConfig, seed: int = 0):
    counts = {}
    for eps in episodes_by_user.values():
        for ep in eps:
            for r in ep[["zi", "zj"]].itertuples(index=False):
                z = Zone(int(r.zi), int(r.zj))
                counts[z] = counts.get(z, 0) + 1

    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    chosen = [z for z, _ in top[:sim_cfg.n_stations]]
    if len(chosen) < sim_cfg.n_stations:
        chosen += [Zone(0, 0)] * (sim_cfg.n_stations - len(chosen))

    rng = random.Random(seed)
    stations = []
    for sid, z in enumerate(chosen):
        stations.append(
            Station(
                station_id=sid,
                zone=z,
                n_ports=sim_cfg.n_ports_per_station,
                power_kw=sim_cfg.station_power_kw,
                price_kwh=rng.uniform(sim_cfg.price_min, sim_cfg.price_max),
            )
        )
    return stations


def train_federated(
    episodes_by_user,
    base_stations,
    sim_cfg: SimConfig,
    rl_cfg: RLConfig,
    fl_cfg: FLConfig,
    obs_dim: int,
    act_dim: int,
    device: str = "cpu",
    method: str = "FedDyn",  # "FedAvg" or "FedDyn"
    seed: int = 0,
):
    rng = random.Random(seed)

    user_ids = sorted(list(episodes_by_user.keys()))
    global_vec = None
    h_memory = {}  # FedDyn memory per client (key=int(uid))

    for rnd in range(fl_cfg.rounds):
        # deterministic per round given seed
        user_ids_shuffled = user_ids[:]
        rng.shuffle(user_ids_shuffled)
        clients = user_ids_shuffled[: min(fl_cfg.clients_per_round, len(user_ids_shuffled))]

        local_models = {}
        local_scores = {}
        weights = {}

        for uid in clients:
            stations_clone = copy.deepcopy(base_stations)
            env = EVGeoLifeEnv(episodes_by_user, stations_clone, sim_cfg, seed=int(uid))
            env.reset(uid)

            local_policy = PolicyNet(obs_dim, act_dim, hidden=rl_cfg.hidden).to(device)
            if global_vec is not None:
                set_param_vector(local_policy, global_vec.clone())

            # FedDyn local regularizer needs feddyn_h (optional if your rl_train supports it)
            hi = h_memory.get(int(uid), None) if method == "FedDyn" else None

            local_policy, avg_return = train_local_from_model(
                env,
                local_policy,
                rl_cfg,
                device=device,
                global_vec=global_vec,
                feddyn_h=hi,
                feddyn_alpha=fl_cfg.feddyn_alpha if method == "FedDyn" else 0.0,
            )

            local_models[int(uid)] = local_policy
            local_scores[int(uid)] = avg_return
            weights[int(uid)] = 1.0

        if global_vec is None:
            global_vec = fedavg(local_models, weights)
        else:
            if method == "FedAvg":
                global_vec = fedavg(local_models, weights)
            else:
                global_vec, h_memory = feddyn_server_aggregate(
                    local_models, h_memory, alpha=fl_cfg.feddyn_alpha, weights=weights
                )

        print(
            f"[{method}] Round {rnd+1}/{fl_cfg.rounds} | mean local return = {np.mean(list(local_scores.values())):.3f}"
        )

    global_policy = PolicyNet(obs_dim, act_dim, hidden=rl_cfg.hidden).to(device)
    set_param_vector(global_policy, global_vec.clone())
    return global_policy


def train_centralized(
    episodes_by_user,
    base_stations,
    sim_cfg: SimConfig,
    rl_cfg: RLConfig,
    obs_dim: int,
    act_dim: int,
    device: str = "cpu",
    seed: int = 0,
):
    """
    Baseline centralisée (plus stable) :
    - un seul modèle
    - sample aléatoire des users
    - beaucoup d'épisodes mais 1 épisode/update (variance réduite)
    """
    rng = random.Random(seed)
    user_ids = sorted(list(episodes_by_user.keys()))
    policy = PolicyNet(obs_dim, act_dim, hidden=rl_cfg.hidden).to(device)

    total_episodes = 400  # monte à 800 si tu veux

    from copy import deepcopy
    rl_cfg_c = deepcopy(rl_cfg)
    rl_cfg_c.episodes_local = 1

    for _ in range(total_episodes):
        uid = rng.choice(user_ids)
        env = EVGeoLifeEnv(episodes_by_user, copy.deepcopy(base_stations), sim_cfg, seed=int(uid))
        env.reset(uid)
        policy, _ = train_local_from_model(env, policy, rl_cfg_c, device=device)

    return policy


def _evaluate_all_methods_once(
    run_id: int,
    seed_run: int,
    episodes_by_user,
    user_ids,
    base_stations,
    sim_cfg: SimConfig,
    rl_cfg: RLConfig,
    fl_cfg: FLConfig,
    ev_cfg: EvalConfig,
    obs_dim: int,
    act_dim: int,
    device: str,
):
    """
    Execute 1 run complet (Centralized + FedAvg + FedDyn + heuristiques).
    Retourne une liste de dicts (une ligne par méthode).
    """
    results = []

    # 1) Train policies
    policy_fedavg = train_federated(
        episodes_by_user, base_stations, sim_cfg, rl_cfg, fl_cfg, obs_dim, act_dim,
        device=device, method="FedAvg", seed=seed_run
    )
    policy_feddyn = train_federated(
        episodes_by_user, base_stations, sim_cfg, rl_cfg, fl_cfg, obs_dim, act_dim,
        device=device, method="FedDyn", seed=seed_run
    )
    policy_central = train_centralized(
        episodes_by_user, base_stations, sim_cfg, rl_cfg, obs_dim, act_dim,
        device=device, seed=seed_run
    )

    # 2) Evaluate policies (multi-vehicle congestion)
    metrics_fedavg = evaluate_multi_vehicle(
        episodes_by_user=episodes_by_user,
        base_stations=base_stations,
        sim_cfg=sim_cfg,
        policy=policy_fedavg,
        user_ids=user_ids,
        n_vehicles=ev_cfg.n_eval_vehicles,
        n_episodes=ev_cfg.n_eval_episodes,
        device=device,
        seed=seed_run,
    )
    results.append({"run": run_id, "seed": seed_run, "method": "FedAvg", **metrics_fedavg})

    metrics_feddyn = evaluate_multi_vehicle(
        episodes_by_user=episodes_by_user,
        base_stations=base_stations,
        sim_cfg=sim_cfg,
        policy=policy_feddyn,
        user_ids=user_ids,
        n_vehicles=ev_cfg.n_eval_vehicles,
        n_episodes=ev_cfg.n_eval_episodes,
        device=device,
        seed=seed_run,
    )
    results.append({"run": run_id, "seed": seed_run, "method": "FedDyn", **metrics_feddyn})

    metrics_central = evaluate_multi_vehicle(
        episodes_by_user=episodes_by_user,
        base_stations=base_stations,
        sim_cfg=sim_cfg,
        policy=policy_central,
        user_ids=user_ids,
        n_vehicles=ev_cfg.n_eval_vehicles,
        n_episodes=ev_cfg.n_eval_episodes,
        device=device,
        seed=seed_run,
    )
    results.append({"run": run_id, "seed": seed_run, "method": "Centralized", **metrics_central})

    # 3) Heuristics (mêmes seeds)
    metrics_nearest = evaluate_heuristic_multi_vehicle(
        episodes_by_user=episodes_by_user,
        base_stations=base_stations,
        sim_cfg=sim_cfg,
        policy_func=nearest_station_policy,
        user_ids=user_ids,
        n_vehicles=ev_cfg.n_eval_vehicles,
        n_episodes=ev_cfg.n_eval_episodes,
        seed=seed_run,
    )
    results.append({"run": run_id, "seed": seed_run, "method": "Heuristic_NearestWhenNeeded", **metrics_nearest})

    metrics_waitprice = evaluate_heuristic_multi_vehicle(
        episodes_by_user=episodes_by_user,
        base_stations=base_stations,
        sim_cfg=sim_cfg,
        policy_func=min_wait_price_policy,
        user_ids=user_ids,
        n_vehicles=ev_cfg.n_eval_vehicles,
        n_episodes=ev_cfg.n_eval_episodes,
        seed=seed_run,
    )
    results.append({"run": run_id, "seed": seed_run, "method": "Heuristic_Wait+Price", **metrics_waitprice})

    metrics_smart = evaluate_heuristic_multi_vehicle(
        episodes_by_user=episodes_by_user,
        base_stations=base_stations,
        sim_cfg=sim_cfg,
        policy_func=smart_rule_policy,
        user_ids=user_ids,
        n_vehicles=ev_cfg.n_eval_vehicles,
        n_episodes=ev_cfg.n_eval_episodes,
        seed=seed_run,
    )
    results.append({"run": run_id, "seed": seed_run, "method": "Heuristic_SmartRule", **metrics_smart})

    return results


def main():
    data_cfg = DataConfig()
    grid_cfg = GridConfig()
    sim_cfg = SimConfig()
    rl_cfg = RLConfig()
    fl_cfg = FLConfig()
    ev_cfg = EvalConfig()

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1) Parse GeoLife -> points parquet
    if not os.path.exists(data_cfg.points_parquet):
        prepare_geolife_points(
            geolife_root=data_cfg.geolife_root,
            out_parquet=data_cfg.points_parquet,
            max_users=data_cfg.max_users,
            max_trajectories_per_user=data_cfg.max_trajectories_per_user,
            max_points_total=data_cfg.max_points_total,
        )

    # 2) Build episodes parquet
    if not os.path.exists(data_cfg.episodes_parquet):
        build_episodes(
            points_parquet=data_cfg.points_parquet,
            out_episodes_parquet=data_cfg.episodes_parquet,
            n_lat=grid_cfg.n_lat,
            n_lon=grid_cfg.n_lon,
            step_minutes=sim_cfg.step_minutes,
            max_steps_per_episode=sim_cfg.max_steps_per_episode,
        )

    # 3) Load episodes
    episodes_by_user = load_episodes_for_users(data_cfg.episodes_parquet)
    user_ids = sorted(list(episodes_by_user.keys()))
    print(f"Users loaded: {len(user_ids)}")

    # 4) Build stations from zone density
    base_stations = build_stations_from_zone_density(episodes_by_user, sim_cfg, seed=0)

    # 5) obs_dim / act_dim
    env0 = EVGeoLifeEnv(episodes_by_user, copy.deepcopy(base_stations), sim_cfg, seed=0)
    obs0 = env0.reset(user_ids[0])
    obs_dim = int(obs0.shape[0])
    act_dim = len(base_stations) + 1  # + WAIT

    device = "cpu"

    # ===============================
    # MULTI-RUN (3 runs)
    # ===============================
    N_RUNS = 3
    all_rows = []

    for run_id in range(N_RUNS):
        seed_run = 100 + run_id
        print(f"\n================ RUN {run_id+1}/{N_RUNS} (seed={seed_run}) ================")

        rows = _evaluate_all_methods_once(
            run_id=run_id,
            seed_run=seed_run,
            episodes_by_user=episodes_by_user,
            user_ids=user_ids,
            base_stations=base_stations,
            sim_cfg=sim_cfg,
            rl_cfg=rl_cfg,
            fl_cfg=fl_cfg,
            ev_cfg=ev_cfg,
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=device,
        )
        all_rows.extend(rows)

    df_runs = pd.DataFrame(all_rows)

    # Sauvegarde tous les runs
    out_runs = "outputs/results_runs.csv"
    df_runs.to_csv(out_runs, index=False)

    # Moyenne + écart-type
    df_mean = df_runs.groupby("method").mean(numeric_only=True).reset_index()
    df_std = df_runs.groupby("method").std(numeric_only=True).reset_index()

    out_mean = "outputs/results_mean.csv"
    out_std = "outputs/results_std.csv"
    df_mean.to_csv(out_mean, index=False)
    df_std.to_csv(out_std, index=False)

    print("\n=== MEAN RESULTS (3 runs) ===")
    print(df_mean.to_string(index=False))

    print("\n=== STD RESULTS (3 runs) ===")
    print(df_std.to_string(index=False))

    print(f"\nSaved runs to:  {out_runs}")
    print(f"Saved mean to:  {out_mean}")
    print(f"Saved std to:   {out_std}")

    # Limitations & perspectives
    limitations = """
Limitations & perspectives (to include in report):
1) The battery/energy model is simulated (SoC from distance) and not based on real EV telemetry.
2) Charging stations are synthetically generated from mobility density (no real station dataset).
3) The RL algorithm is intentionally simple (policy gradient) to keep the prototype readable; advanced DRL (SAC/DDPG) could improve stability.
4) The FL stack is implemented from scratch (no Flower/FedML). This is acceptable for a PFE if clearly documented and reproducible.
5) GeoLife users are general mobility traces (not only EVs). The work uses them as mobility proxies.

Future work:
- Integrate real EV energy datasets (e.g., NREL Fleet DNA) and real station locations.
- Add reservation mechanisms (Book/Keep/Switch) and more realistic charging duration constraints.
- Use multi-agent RL or game-theoretic coordination for collective efficiency.
- Add FedProx / personalized FL and stronger privacy/security assumptions.
""".strip()
    print("\n" + limitations)


if __name__ == "__main__":
    main()
