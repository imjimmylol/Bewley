# src/ability_init.py
"""
Ability initialization strategies for Bewley model.

Provides several initialization methods inspired by real-world distributions:
1. Stationary AR(1) - matches long-run distribution
2. Truncated log-normal (IQ-like) - bounded realistic distribution
3. Narrow uniform - minimal heterogeneity for testing
"""

import numpy as np
from typing import Tuple, Literal


def initialize_ability_stationary(
    batch_size: int,
    n_agents: int,
    rho_v: float,
    sigma_v: float,
    v_bar: float,
    clip_sigma: float = 2.0
) -> np.ndarray:
    """
    Initialize ability from stationary distribution of AR(1) process WITH CLIPPING.

    AR(1): log(v[t+1]) = (1-rho)*log(v_bar) + rho*log(v[t]) + eps
    Stationary: mean = log(v_bar), variance = sigma_v^2 / (1 - rho_v^2)

    IMPORTANT: Clips to ±clip_sigma standard deviations to prevent extreme outliers.

    Args:
        batch_size: Number of batches
        n_agents: Number of agents per batch
        rho_v: AR(1) persistence parameter (e.g., 0.98)
        sigma_v: Standard deviation of innovation (e.g., 0.36)
        v_bar: Long-run mean of ability (e.g., 5.0)
        clip_sigma: Clip to ±clip_sigma standard deviations (default: 2.0)

    Returns:
        ability: (batch_size, n_agents) array
    """
    # Stationary distribution in log space
    ability_log_mean = np.log(v_bar)
    ability_log_std = sigma_v / np.sqrt(1 - rho_v**2)

    # Sample from normal in log space
    log_ability = np.random.normal(
        ability_log_mean,
        ability_log_std,
        size=(batch_size, n_agents)
    )

    # CRITICAL: Clip to prevent extreme outliers
    # This keeps 95% of distribution if clip_sigma=2.0
    log_ability = np.clip(
        log_ability,
        ability_log_mean - clip_sigma * ability_log_std,
        ability_log_mean + clip_sigma * ability_log_std
    )

    # Transform to level
    ability = np.exp(log_ability)

    return ability


def initialize_ability_iq_like(
    batch_size: int,
    n_agents: int,
    mean: float = 1.0,
    cv: float = 0.3,
    min_ability: float = 0.3,
    max_ability: float = 3.0
) -> np.ndarray:
    """
    Initialize ability like real-world IQ distribution.

    Uses truncated log-normal distribution:
    - Most people clustered near mean
    - Moderate right tail (high performers)
    - Hard bounds prevent extreme outliers
    - CV (coefficient of variation) = std/mean controls dispersion

    Real-world analogy:
    - IQ: mean=100, std=15, CV=0.15 (very tight)
    - Earnings: CV ≈ 0.4-0.6 (moderate dispersion)
    - Wealth: CV > 1.0 (high inequality)

    Args:
        batch_size: Number of batches
        n_agents: Number of agents per batch
        mean: Target mean ability (e.g., 1.0)
        cv: Coefficient of variation = std/mean (e.g., 0.3)
        min_ability: Minimum ability (e.g., 0.3 = 30% of mean)
        max_ability: Maximum ability (e.g., 3.0 = 300% of mean)

    Returns:
        ability: (batch_size, n_agents) array

    Example:
        # Conservative (like IQ): cv=0.15, range [0.7, 1.5]
        # Moderate (like earnings): cv=0.3, range [0.3, 3.0]
        # High inequality (like wealth): cv=0.6, range [0.1, 10.0]
    """
    # For log-normal: if X ~ LogNormal(μ, σ), then E[X] = exp(μ + σ²/2)
    # We want E[X] = mean and CV[X] = cv
    # CV² = exp(σ²) - 1, so σ² = log(1 + CV²)
    target_cv = cv
    sigma_log = np.sqrt(np.log(1 + target_cv**2))
    mu_log = np.log(mean) - 0.5 * sigma_log**2

    # Sample from log-normal
    ability = np.random.lognormal(
        mu_log,
        sigma_log,
        size=(batch_size, n_agents)
    )

    # Clip to bounds
    ability = np.clip(ability, min_ability, max_ability)

    return ability


def initialize_ability_narrow_uniform(
    batch_size: int,
    n_agents: int,
    mean: float = 1.0,
    half_width: float = 0.2
) -> np.ndarray:
    """
    Initialize ability with narrow uniform distribution (for testing/debugging).

    Minimal heterogeneity to isolate other model mechanisms.

    Args:
        batch_size: Number of batches
        n_agents: Number of agents per batch
        mean: Center of distribution (e.g., 1.0)
        half_width: Half-width of uniform range (e.g., 0.2 → [0.8, 1.2])

    Returns:
        ability: (batch_size, n_agents) array
    """
    ability = np.random.uniform(
        mean - half_width,
        mean + half_width,
        size=(batch_size, n_agents)
    )
    return ability


def initialize_ability(
    batch_size: int,
    n_agents: int,
    method: Literal["stationary", "iq_like", "narrow_uniform"] = "iq_like",
    config=None,
    **kwargs
) -> np.ndarray:
    """
    Unified interface for ability initialization.

    Args:
        batch_size: Number of batches
        n_agents: Number of agents per batch
        method: Initialization method
            - "stationary": Match AR(1) stationary distribution (requires config)
            - "iq_like": Realistic bounded log-normal (default)
            - "narrow_uniform": Minimal heterogeneity for testing
        config: Configuration object (required for "stationary" method)
        **kwargs: Additional parameters passed to specific initializers

    Returns:
        ability: (batch_size, n_agents) array

    Examples:
        # Recommended: IQ-like distribution with moderate dispersion
        ability = initialize_ability(4, 128, method="iq_like",
                                      mean=1.0, cv=0.3, min_ability=0.3, max_ability=3.0)

        # For testing: minimal heterogeneity
        ability = initialize_ability(4, 128, method="narrow_uniform",
                                      mean=1.0, half_width=0.1)

        # Match model's stationary distribution (old method)
        ability = initialize_ability(4, 128, method="stationary", config=config)
    """
    if method == "stationary":
        if config is None:
            raise ValueError("config is required for 'stationary' initialization")
        return initialize_ability_stationary(
            batch_size=batch_size,
            n_agents=n_agents,
            rho_v=config.shock.rho_v,
            sigma_v=config.shock.sigma_v,
            v_bar=config.shock.v_bar,
            clip_sigma=kwargs.get("clip_sigma", 2.0)
        )
    elif method == "iq_like":
        return initialize_ability_iq_like(
            batch_size=batch_size,
            n_agents=n_agents,
            mean=kwargs.get("mean", 1.0),
            cv=kwargs.get("cv", 0.3),
            min_ability=kwargs.get("min_ability", 0.3),
            max_ability=kwargs.get("max_ability", 3.0)
        )
    elif method == "narrow_uniform":
        return initialize_ability_narrow_uniform(
            batch_size=batch_size,
            n_agents=n_agents,
            mean=kwargs.get("mean", 1.0),
            half_width=kwargs.get("half_width", 0.2)
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def get_ability_init_stats(ability: np.ndarray) -> dict:
    """
    Compute statistics for ability distribution.

    Args:
        ability: (batch_size, n_agents) array

    Returns:
        stats: Dictionary with mean, std, cv, min, max, percentiles
    """
    return {
        "mean": float(np.mean(ability)),
        "std": float(np.std(ability)),
        "cv": float(np.std(ability) / np.mean(ability)),
        "min": float(np.min(ability)),
        "max": float(np.max(ability)),
        "p5": float(np.percentile(ability, 5)),
        "p25": float(np.percentile(ability, 25)),
        "p50": float(np.percentile(ability, 50)),
        "p75": float(np.percentile(ability, 75)),
        "p95": float(np.percentile(ability, 95)),
        "p99": float(np.percentile(ability, 99)),
    }
