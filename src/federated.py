from __future__ import annotations

from typing import Dict, Tuple
import torch


def get_param_vector(model: torch.nn.Module) -> torch.Tensor:
    """
    Flatten all model parameters into a single 1D tensor.
    """
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_param_vector(model: torch.nn.Module, vec: torch.Tensor) -> None:
    """
    Load a 1D parameter vector back into model parameters.
    """
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec[offset:offset + n].view_as(p))
        offset += n


def get_param_vector_grad(model: torch.nn.Module) -> torch.Tensor:
    """
    Flatten all gradients into a single 1D tensor.
    (Useful if you want to debug / inspect.)
    """
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p.data).view(-1))
        else:
            grads.append(p.grad.data.view(-1))
    return torch.cat(grads)


def fedavg(models: Dict[int, torch.nn.Module], weights: Dict[int, float]) -> torch.Tensor:
    """
    Standard FedAvg aggregation:
        w_global = sum_i p_i * w_i
    Returns: global parameter vector
    """
    keys = list(models.keys())
    if not keys:
        raise ValueError("fedavg: empty models dict.")

    vecs = [get_param_vector(models[k]) for k in keys]
    w = torch.tensor([weights.get(k, 1.0) for k in keys], dtype=torch.float32)
    w = w / (w.sum() + 1e-12)

    global_vec = sum(wi * vi for wi, vi in zip(w, vecs))
    return global_vec


def feddyn_server_aggregate(
    models: Dict[int, torch.nn.Module],
    h_memory: Dict[int, torch.Tensor],
    alpha: float,
    weights: Dict[int, float],
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    FedDyn server aggregation (practical implementation):

    Given local model params w_i and per-client memory h_i, compute:
        w_new = sum_i p_i * (w_i - (1/alpha) * h_i)

    Then update memory:
        h_i <- h_i + alpha * (w_i - w_new)

    Notes:
    - alpha must be > 0
    - h_memory is a dict {client_id: tensor} same dim as params
    """
    if alpha <= 0:
        raise ValueError("feddyn_server_aggregate: alpha must be > 0.")

    keys = list(models.keys())
    if not keys:
        raise ValueError("feddyn_server_aggregate: empty models dict.")

    # local vectors
    local_vecs = {k: get_param_vector(models[k]) for k in keys}

    # init memories if absent
    any_key = keys[0]
    dim = local_vecs[any_key].numel()
    for k in keys:
        if k not in h_memory:
            h_memory[k] = torch.zeros(dim, dtype=local_vecs[k].dtype, device=local_vecs[k].device)

    # weights
    w = torch.tensor([weights.get(k, 1.0) for k in keys], dtype=torch.float32)
    w = w / (w.sum() + 1e-12)

    # aggregate corrected
    corrected = []
    for k in keys:
        corrected.append(local_vecs[k] - (1.0 / alpha) * h_memory[k])

    new_global = sum(wi * vi for wi, vi in zip(w, corrected))

    # update h_i
    for k in keys:
        h_memory[k] = h_memory[k] + alpha * (local_vecs[k] - new_global)

    return new_global, h_memory
