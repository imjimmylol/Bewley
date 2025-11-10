# src/market.py
"""
Market equilibrium calculations for Bewley model.

Implements:
1. Aggregate capital and labor computation
2. Wage determination from production function
3. Return to capital determination
4. Full market clearing
"""

from typing import Tuple
import torch
from torch import Tensor


def compute_price(
    savings: Tensor,
    ability: Tensor, 
    labor: Tensor,
    A: float,
    alpha: float

) -> Tuple[Tensor, Tensor]:
    """
    Args:
        savings: Capital holdings (B, A)
        labor: Labor supply (B, A)
        ability: Ability/productivity (B, A)

    Returns:
        wage
        return 
    """
    # mean over agent dimension (dim=1)
    savings_agg = savings.mean(dim=1, keepdim=True)
    labor_eff_agg = (labor * ability).mean(dim=1, keepdim=True)
    
    # avoid zero or negative denominators
    labor_eff_agg = torch.clamp(labor_eff_agg, min=1e-8)
    
    ratio = savings_agg / labor_eff_agg
    ratio = torch.clamp(ratio, min=1e-8)
    
    wage = A * (1 - alpha) * (ratio ** alpha)
    ret = A * alpha * (ratio ** (alpha - 1))  # corrected exponent
    return wage, ret




def compute_aggregates(
    savings: Tensor,
    labor: Tensor,
    ability: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute aggregate capital and effective labor from agent decisions.

    Aggregates are computed as cross-agent means (averaging over the A dimension).
    Each batch can have different aggregates.

    Args:
        savings: Capital holdings (B, A)
        labor: Labor supply (B, A)
        ability: Ability/productivity (B, A)

    Returns:
        K_agg: Aggregate capital (B,) - mean across agents
        L_eff: Aggregate effective labor (B,) - mean of labor * ability
        capital_labor_ratio: K/L ratio (B,)
    """
    # Aggregate capital: mean across agents (dim=1)
    K_agg = savings.mean(dim=1)  # (B,)

    # Effective labor: labor weighted by ability, then averaged
    labor_eff = labor * ability  # (B, A)
    L_eff = labor_eff.mean(dim=1)  # (B,)

    # Capital-labor ratio
    capital_labor_ratio = K_agg / (L_eff + 1e-8)  # Avoid division by zero

    return K_agg, L_eff, capital_labor_ratio


def compute_wage(
    capital_labor_ratio: Tensor,
    A: float,
    alpha: float
) -> Tensor:
    """
    Compute wage rate from production function.

    From Cobb-Douglas Y = A * K^α * L^(1-α):
    wage = ∂Y/∂L = A(1-α)(K/L)^α

    Args:
        capital_labor_ratio: K/L ratio (B,)
        A: Total factor productivity (technology parameter)
        alpha: Capital share of income (elasticity)

    Returns:
        wage: Wage rate (B,)
    """
    wage = A * (1 - alpha) * (capital_labor_ratio ** alpha)
    return wage


def compute_return(
    capital_labor_ratio: Tensor,
    A: float,
    alpha: float,
    delta: float
) -> Tensor:
    """
    Compute net return to capital from production function.

    From Cobb-Douglas Y = A * K^α * L^(1-α):
    marginal_product = ∂Y/∂K = A * α * (K/L)^(α-1)
    net_return = marginal_product - δ

    Args:
        capital_labor_ratio: K/L ratio (B,)
        A: Total factor productivity (technology parameter)
        alpha: Capital share of income (elasticity)
        delta: Depreciation rate

    Returns:
        ret: Net return to capital (B,)
    """
    # Marginal product of capital
    marginal_product = A * alpha * (capital_labor_ratio ** (alpha - 1))

    # Net return after depreciation
    ret = marginal_product - delta

    return ret


def compute_market_prices(
    savings: Tensor,
    labor: Tensor,
    ability: Tensor,
    A: float,
    alpha: float,
    delta: float
) -> Tuple[Tensor, Tensor]:
    """
    Compute market-clearing wage and return from agent decisions.

    This is the main function for market equilibrium.
    Aggregates agent decisions and determines prices.

    Args:
        savings: Capital holdings (B, A)
        labor: Labor supply (B, A)
        ability: Ability/productivity (B, A)
        A: Total factor productivity
        alpha: Capital share of income
        delta: Depreciation rate

    Returns:
        wage: Market wage rate (B,) - same for all agents in each batch
        ret: Net return to capital (B,) - same for all agents in each batch
    """
    # 1. Compute aggregates
    K_agg, L_eff, kl_ratio = compute_aggregates(savings, labor, ability)

    # 2. Compute prices from aggregates
    wage = compute_wage(kl_ratio, A, alpha)
    ret = compute_return(kl_ratio, A, alpha, delta)

    return wage, ret


def broadcast_prices_to_agents(
    wage: Tensor,
    ret: Tensor,
    n_agents: int
) -> Tuple[Tensor, Tensor]:
    """
    Broadcast aggregate prices to agent dimension.

    Converts (B,) prices to (B, A) so each agent sees the same price.

    Args:
        wage: Wage rate (B,)
        ret: Return to capital (B,)
        n_agents: Number of agents (A)

    Returns:
        wage_broadcast: (B, A) - same wage for all agents
        ret_broadcast: (B, A) - same return for all agents
    """
    # (B,) → (B, 1) → (B, A)
    wage_broadcast = wage.unsqueeze(1).expand(-1, n_agents)
    ret_broadcast = ret.unsqueeze(1).expand(-1, n_agents)

    return wage_broadcast, ret_broadcast


def compute_market_equilibrium(
    savings: Tensor,
    labor: Tensor,
    ability: Tensor,
    config,
    *,
    broadcast: bool = True
) -> Tuple[Tensor, Tensor]:
    """
    Convenience function: compute market equilibrium from config.

    Args:
        savings: Capital holdings (B, A)
        labor: Labor supply (B, A)
        ability: Ability/productivity (B, A)
        config: Config namespace with bewley_model parameters (A, alpha, delta)
        broadcast: If True, broadcast (B,) prices to (B, A). Default True.

    Returns:
        wage: Wage rate - (B, A) if broadcast=True, else (B,)
        ret: Return to capital - (B, A) if broadcast=True, else (B,)
    """
    # Extract parameters
    A = config.bewley_model.A
    alpha = config.bewley_model.alpha

    # Delta might be in milf_inputs or bewley_model
    if hasattr(config.bewley_model, 'delta'):
        delta = config.bewley_model.delta
    elif hasattr(config, 'milf_inputs') and hasattr(config.milf_inputs, 'delta'):
        delta = config.milf_inputs.delta
    else:
        raise ValueError("Delta (depreciation rate) not found in config")

    # Compute prices
    wage, ret = compute_market_prices(savings, labor, ability, A, alpha, delta)

    # Broadcast to agent dimension if requested
    if broadcast:
        n_agents = savings.shape[1]
        wage, ret = broadcast_prices_to_agents(wage, ret, n_agents)

    return wage, ret


def compute_income_multiplier(
    wage: Tensor,
    labor: Tensor,
    ability: Tensor,
    savings: Tensor,
    ret_lagged: Tensor,
    delta: float
) -> Tensor:
    """
    Compute total income multiplier (before tax).

    This is the income before tax (ibt):
    ibt[t] = wage[t]*labor[t]*ability[t] + (1-δ+ret[t-1])*savings[t]

    Note: This function is here for convenience but will likely move to income_tax.py

    Args:
        wage: Wage rate (B, A) or (B,)
        labor: Labor supply (B, A)
        ability: Ability (B, A)
        savings: Capital holdings (B, A)
        ret_lagged: Return from previous period ret[t-1] (B, A) or (B,) or scalar
        delta: Depreciation rate

    Returns:
        ibt: Income before tax (B, A)
    """
    # Labor income
    labor_income = wage * labor * ability

    # Capital income (gross return = 1 - δ + ret[t-1])
    gross_return = 1.0 - delta + ret_lagged
    capital_income = gross_return * savings

    # Total income before tax
    ibt = labor_income + capital_income

    return ibt
