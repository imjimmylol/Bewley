# src/shocks.py
"""
Ability shock transitions for Bewley model.

Implements:
1. AR(1) persistence with normal shocks
2. Superstar dynamics (rare high-ability events)
3. Bounded ability constraints [v_min, v_max]
4. Ability history tracking (rolling window)
"""

from typing import Optional, Tuple
import torch
from torch import Tensor


def transition_superstar(
    is_superstar_t: Tensor,
    p: float,
    q: float,
    *,
    deterministic: bool = False
) -> Tensor:
    """
    Transition superstar status using two-state Markov chain.

    Transition probabilities:
    - P(become superstar | not superstar) = p
    - P(remain superstar | superstar) = q

    Args:
        is_superstar_t: Current superstar status (B, A) bool tensor
        p: Probability of becoming superstar (typically very small, e.g., 2.2e-6)
        q: Probability of remaining superstar (typically high, e.g., 0.99)
        deterministic: If True, always transitions to expected state (for validation)

    Returns:
        is_superstar_tp1: Next period superstar status (B, A) bool tensor
    """
    B, A = is_superstar_t.shape
    device = is_superstar_t.device

    if deterministic:
        # Deterministic: use expected values
        # Non-stars become stars with prob p (always False if p < 0.5)
        # Stars remain stars if q >= 0.5
        return is_superstar_t.clone() if q >= 0.5 else torch.zeros_like(is_superstar_t)

    # Stochastic transitions
    rand = torch.rand(B, A, device=device)

    # Non-superstar → superstar with prob p
    become_superstar = (~is_superstar_t) & (rand < p)

    # Superstar → remains superstar with prob q
    remain_superstar = is_superstar_t & (rand < q)

    is_superstar_tp1 = become_superstar | remain_superstar

    return is_superstar_tp1


def transition_ability(
    ability_t: Tensor,
    is_superstar_t: Tensor,
    rho_v: float,
    sigma_v: float,
    v_bar: float,
    v_min: float,
    v_max: float,
    p: float,
    q: float,
    superstar_multiplier: float = 10.0,
    *,
    deterministic: bool = False
) -> Tuple[Tensor, Tensor]:
    """
    Transition ability using AR(1) process with superstar dynamics.

    Process:
    1. Transition superstar status (Markov chain)
    2. AR(1): log(v[t+1]) = (1-rho)*log(v_bar) + rho*log(v[t]) + eps
       where eps ~ N(0, sigma_v^2)
    3. If superstar, multiply by superstar_multiplier
    4. Clamp to [v_min, v_max]

    Args:
        ability_t: Current ability (B, A)
        is_superstar_t: Current superstar status (B, A) bool
        rho_v: AR(1) persistence parameter (e.g., 0.95)
        sigma_v: Standard deviation of innovation (e.g., 0.2)
        v_bar: Long-run mean of ability (e.g., 1.5)
        v_min: Minimum ability (computed from steady state)
        v_max: Maximum ability (computed from steady state)
        p: Probability of becoming superstar
        q: Probability of remaining superstar
        superstar_multiplier: Multiplier for superstar ability (default: 10.0)
        deterministic: If True, no random shocks (for validation)

    Returns:
        ability_tp1: Next period ability (B, A)
        is_superstar_tp1: Next period superstar status (B, A) bool
    """
    B, A = ability_t.shape
    device = ability_t.device

    # 1. Transition superstar status
    is_superstar_tp1 = transition_superstar(is_superstar_t, p, q, deterministic=deterministic)

    # 2. AR(1) transition in log space
    # log(v[t+1]) = (1-rho)*log(v_bar) + rho*log(v[t]) + eps
    log_v_bar = torch.log(torch.tensor(v_bar, device=device))
    log_v_t = torch.log(ability_t.clamp(min=1e-8))  # Avoid log(0)

    if deterministic:
        # No shock: use conditional mean
        log_v_tp1 = (1 - rho_v) * log_v_bar + rho_v * log_v_t
    else:
        # Stochastic: add normal shock
        eps = torch.randn(B, A, device=device) * sigma_v
        log_v_tp1 = (1 - rho_v) * log_v_bar + rho_v * log_v_t + eps

    # 3. Transform back to level
    ability_tp1 = torch.exp(log_v_tp1)

    # 4. Apply superstar multiplier
    ability_tp1 = torch.where(
        is_superstar_tp1,
        ability_tp1 * superstar_multiplier,
        ability_tp1
    )

    # 5. Clamp to bounds
    ability_tp1 = ability_tp1.clamp(min=v_min, max=v_max)

    return ability_tp1, is_superstar_tp1


def update_ability_history(
    ability_history: Optional[Tensor],
    ability_new: Tensor,
    max_length: int
) -> Tensor:
    """
    Update ability history with rolling window.

    If history is None, initialize with current ability repeated.
    Otherwise, append new ability and drop oldest if exceeds max_length.

    Args:
        ability_history: Current history (L, B, A) or None
        ability_new: New ability to append (B, A)
        max_length: Maximum history length to keep

    Returns:
        ability_history_new: Updated history (L', B, A) where L' <= max_length
    """
    if ability_history is None:
        # Initialize: repeat current ability max_length times
        # Shape: (max_length, B, A)
        return ability_new.unsqueeze(0).repeat(max_length, 1, 1)

    # Append new ability
    # ability_new: (B, A) → (1, B, A)
    ability_new_expanded = ability_new.unsqueeze(0)

    # Concatenate: (L, B, A) + (1, B, A) → (L+1, B, A)
    history_extended = torch.cat([ability_history, ability_new_expanded], dim=0)

    # Keep only last max_length entries
    if history_extended.shape[0] > max_length:
        history_extended = history_extended[-max_length:]

    return history_extended


def transition_ability_with_history(
    ability_t: Tensor,
    is_superstar_t: Tensor,
    ability_history_t: Optional[Tensor],
    config,
    history_length: int,
    *,
    deterministic: bool = False
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Convenience function: transition ability and update history in one call.

    Args:
        ability_t: Current ability (B, A)
        is_superstar_t: Current superstar status (B, A) bool
        ability_history_t: Current ability history (L, B, A) or None
        config: Config namespace with shock parameters (rho_v, sigma_v, v_bar, etc.)
        history_length: Maximum history length to maintain
        deterministic: If True, no random shocks

    Returns:
        ability_tp1: Next period ability (B, A)
        is_superstar_tp1: Next period superstar status (B, A) bool
        ability_history_tp1: Updated ability history (L', B, A)
    """
    # Get shock parameters from config
    shock_params = {
        'rho_v': config.shock.rho_v,
        'sigma_v': config.shock.sigma_v,
        'v_bar': config.shock.v_bar,
        'p': config.shock.p,
        'q': config.shock.q,
    }

    # Get bounds (computed in derived params)
    if hasattr(config.shock, 'v_min') and hasattr(config.shock, 'v_max'):
        shock_params['v_min'] = config.shock.v_min
        shock_params['v_max'] = config.shock.v_max
    else:
        # Fallback: compute simple bounds if not in config
        shock_params['v_min'] = config.shock.v_bar * 0.1
        shock_params['v_max'] = config.shock.v_bar * 100.0

    # Optional: superstar multiplier (can be added to config)
    if hasattr(config.shock, 'superstar_multiplier'):
        shock_params['superstar_multiplier'] = config.shock.superstar_multiplier

    # Transition ability
    ability_tp1, is_superstar_tp1 = transition_ability(
        ability_t,
        is_superstar_t,
        deterministic=deterministic,
        **shock_params
    )

    # Update history
    ability_history_tp1 = update_ability_history(
        ability_history_t,
        ability_tp1,
        max_length=history_length
    )

    return ability_tp1, is_superstar_tp1, ability_history_tp1
