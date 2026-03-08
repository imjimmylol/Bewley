import torch
from torch import Tensor


def flow_utility(
    consumption: Tensor,
    labor: Tensor,
    theta: float,
    gamma: float,
    eps: float = 1e-8,
) -> Tensor:
    """Compute per-agent flow utility: u(c,l) = c^(1-θ)/(1-θ) - l^(1+γ)/(1+γ)."""
    c = consumption.clamp(min=eps)
    l = labor.clamp(min=eps)
    if theta == 1.0:
        u_c = torch.log(c)
    else:
        u_c = c ** (1 - theta) / (1 - theta)
    u_l = l ** (1 + gamma) / (1 + gamma)
    return u_c - u_l
