from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import torch
from torch.distributions import Categorical

from src.config import RLConfig
from src.federated import get_param_vector


def _safe_normalize(x: torch.Tensor) -> torch.Tensor:
    if x.numel() <= 1:
        return x * 0.0

    mean = x.mean()
    std = x.std(unbiased=False)

    if torch.isnan(std) or std.item() < 1e-8:
        return x - mean

    return (x - mean) / (std + 1e-6)


def _discounted_returns(rewards, gamma: float, device: str) -> torch.Tensor:
    G = 0.0
    out = []

    for r in reversed(rewards):
        G = float(r) + gamma * G
        out.append(G)

    out.reverse()
    returns = torch.tensor(out, dtype=torch.float32, device=device)
    return _safe_normalize(returns)


def run_episode_reinforce(
    env,
    policy: torch.nn.Module,
    device: str,
    gamma: float,
) -> Tuple[torch.Tensor, float]:
    if not hasattr(env, "user_id") or env.user_id is None:
        raise RuntimeError("Environment must be reset with env.reset(user_id) before training.")

    obs = env.reset(env.user_id)

    logps = []
    rewards = []

    done = False
    steps = 0

    while not done:
        obs = np.asarray(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits = policy(obs_t)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.zeros_like(logits)

        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)

        obs, r, done, _ = env.step(int(action.item()))
        logps.append(logp)
        rewards.append(float(r))

        steps += 1
        if steps >= env.cfg.max_steps_per_episode:
            done = True

    returns = _discounted_returns(rewards, gamma=gamma, device=device)

    loss = torch.tensor(0.0, device=device)

    for logp, R in zip(logps, returns):
        if torch.isnan(R) or torch.isinf(R):
            continue
        loss = loss + (-logp * R)

    ep_return = float(np.sum(rewards)) if rewards else 0.0
    return loss, ep_return


def _feddyn_regularizer(
    policy: torch.nn.Module,
    global_vec: Optional[torch.Tensor],
    feddyn_h: Optional[torch.Tensor],
    alpha: float,
) -> torch.Tensor:
    w = get_param_vector(policy)

    if alpha <= 0:
        return torch.zeros((), dtype=w.dtype, device=w.device)

    reg = torch.zeros((), dtype=w.dtype, device=w.device)

    if global_vec is not None:
        gv = global_vec.to(device=w.device, dtype=w.dtype)
        reg = reg + 0.5 * alpha * torch.sum((w - gv) ** 2)

    if feddyn_h is not None:
        hi = feddyn_h.to(device=w.device, dtype=w.dtype)
        reg = reg - torch.dot(hi, w)

    return reg


def train_local_from_model(
    env,
    policy: torch.nn.Module,
    rl_cfg: RLConfig,
    device: str = "cpu",
    global_vec: Optional[torch.Tensor] = None,
    feddyn_h: Optional[torch.Tensor] = None,
    feddyn_alpha: float = 0.0,
) -> Tuple[torch.nn.Module, float]:
    policy = policy.to(device)
    policy.train()

    lr = min(rl_cfg.lr, 5e-5)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    ep_returns = []

    for _ in range(rl_cfg.episodes_local):
        opt.zero_grad()

        loss_pg, ep_ret = run_episode_reinforce(
            env,
            policy,
            device=device,
            gamma=rl_cfg.gamma,
        )

        reg = _feddyn_regularizer(
            policy,
            global_vec=global_vec,
            feddyn_h=feddyn_h,
            alpha=feddyn_alpha,
        )

        loss = loss_pg + reg

        if torch.isnan(loss) or torch.isinf(loss):
            ep_returns.append(ep_ret)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=2.0)
        opt.step()

        with torch.no_grad():
            for p in policy.parameters():
                if torch.isnan(p).any() or torch.isinf(p).any():
                    p.data = torch.nan_to_num(
                        p.data,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )

        ep_returns.append(ep_ret)

    avg_return = float(np.mean(ep_returns)) if ep_returns else 0.0
    return policy, avg_return