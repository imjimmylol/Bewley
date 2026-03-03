#!/usr/bin/env python3
"""
Test script for checking environment transitions and shock clipping.

Tests:
1. Ability shock distribution (check if shocks are extreme)
2. Ability bounds (check if agents hit v_min/v_max)
3. Transition dynamics over multiple steps

Usage:
    python test_environment_transitions.py --configs config/default.yaml
    python test_environment_transitions.py --configs config/default.yaml config/lossweights.yaml
"""

import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.environment import EconomyEnv
from src.env_state import MainState
from src.normalizer import RunningPerAgentWelford
from src.models.model import FiLMResNet2In
from src.utils.configloader import load_configs, dict_to_namespace
from src.train import initialize_env_state


def test_shock_distribution(config, n_steps=100):
    """Test the distribution of ability shocks over multiple transitions."""
    print("="*60)
    print("TEST 1: Ability Shock Distribution")
    print("="*60)

    device = "cpu"
    batch_size = config.training.batch_size
    n_agents = config.training.agents

    # Initialize environment
    normalizer = RunningPerAgentWelford(batch_dim=0, agent_dim=1)
    env = EconomyEnv(config, normalizer, device=device)
    state = initialize_env_state(config, device)

    # Dummy policy network (not used for shock testing)
    policy_net = FiLMResNet2In(
        state_dim=2*n_agents+2,
        cond_dim=5,
        output_dim=3
    ).to(device)

    # Collect ability values over transitions
    ability_history = []
    ability_changes = []

    initial_ability = state.ability.clone()

    print(f"\nRunning {n_steps} transitions...")
    for step in tqdm(range(n_steps), desc="Simulating transitions"):
        prev_ability = state.ability.clone()

        # Step environment
        state, temp_state, (parallel_A, outcomes_A), (parallel_B, outcomes_B) = env.step(
            main_state=state,
            policy_net=policy_net,
            deterministic=False,
            fix=False,
            update_normalizer=True,
            commit_strategy="random"
        )

        # Record ability and changes
        ability_history.append(state.ability.cpu().numpy().flatten())
        ability_change = (state.ability - prev_ability).cpu().numpy().flatten()
        ability_changes.append(ability_change)

    ability_history = np.array(ability_history)  # Shape: (n_steps, B*A)
    ability_changes = np.array(ability_changes)  # Shape: (n_steps, B*A)

    # Compute shock bounds
    rho_v = config.shock.rho_v
    sigma_v = config.shock.sigma_v
    v_min = config.shock.v_min
    v_max = config.shock.v_max

    # Statistics
    print(f"\nShock Parameters:")
    print(f"  rho_v = {rho_v}")
    print(f"  sigma_v = {sigma_v}")
    print(f"  v_bar = {config.shock.v_bar}")
    print(f"\nComputed Bounds (±2σ of stationary dist):")
    print(f"  v_min = {v_min:.4f}")
    print(f"  v_max = {v_max:.4f}")

    # Check how many agents hit bounds
    n_total = n_steps * batch_size * n_agents
    n_hit_min = np.sum(ability_history <= v_min * 1.001)  # Small tolerance
    n_hit_max = np.sum(ability_history >= v_max * 0.999)

    print(f"\nBound Violations:")
    print(f"  Hit v_min: {n_hit_min} / {n_total} ({100*n_hit_min/n_total:.2f}%)")
    print(f"  Hit v_max: {n_hit_max} / {n_total} ({100*n_hit_max/n_total:.2f}%)")

    # Ability distribution statistics
    final_ability = ability_history[-1]
    print(f"\nFinal Ability Distribution:")
    print(f"  Mean: {final_ability.mean():.4f}")
    print(f"  Std: {final_ability.std():.4f}")
    print(f"  Min: {final_ability.min():.4f}")
    print(f"  Max: {final_ability.max():.4f}")
    print(f"  95th percentile: {np.percentile(final_ability, 95):.4f}")
    print(f"  99th percentile: {np.percentile(final_ability, 99):.4f}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Ability distribution over time
    for percentile in [5, 25, 50, 75, 95]:
        values = np.percentile(ability_history, percentile, axis=1)
        axes[0, 0].plot(values, label=f'{percentile}th percentile')
    axes[0, 0].axhline(v_min, color='red', linestyle='--', label='v_min')
    axes[0, 0].axhline(v_max, color='red', linestyle='--', label='v_max')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Ability')
    axes[0, 0].set_title('Ability Distribution Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Final ability distribution
    axes[0, 1].hist(final_ability, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(v_min, color='red', linestyle='--', linewidth=2, label='v_min')
    axes[0, 1].axvline(v_max, color='red', linestyle='--', linewidth=2, label='v_max')
    axes[0, 1].axvline(final_ability.mean(), color='blue', linestyle='--', linewidth=2, label='mean')
    axes[0, 1].set_xlabel('Ability')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'Final Ability Distribution (step {n_steps})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Ability changes (proxy for shocks)
    all_changes = ability_changes.flatten()
    # Estimate log-shocks (reverse engineer from changes)
    # This is approximate since we don't have direct access to eps
    axes[1, 0].hist(all_changes, bins=100, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].set_xlabel('Ability Change')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Distribution of Ability Changes (Δv)')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Extreme values check
    extreme_changes = all_changes[np.abs(all_changes) > 3 * all_changes.std()]
    axes[1, 1].hist(extreme_changes, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1, 1].set_xlabel('Extreme Ability Change')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'Extreme Changes (|Δv| > 3σ): {len(extreme_changes)} / {len(all_changes)}')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_shock_distribution.png', dpi=150)
    print(f"\n✓ Plot saved to: test_shock_distribution.png")

    return ability_history, ability_changes


def test_with_fixed_ability(config):
    """Test environment with fix=True (complete markets, no shocks)."""
    print("\n" + "="*60)
    print("TEST 2: Fixed Ability (Complete Markets)")
    print("="*60)

    # Modify config to set fix=True
    config.bewley_model.fix = True

    device = "cpu"
    normalizer = RunningPerAgentWelford(batch_dim=0, agent_dim=1)
    env = EconomyEnv(config, normalizer, device=device)
    state = initialize_env_state(config, device)

    policy_net = FiLMResNet2In(
        state_dim=2*config.training.agents+2,
        cond_dim=5,
        output_dim=3
    ).to(device)

    initial_ability = state.ability.clone()

    # Run 10 steps
    print("\nRunning 10 transitions with fixed ability...")
    for step in tqdm(range(10), desc="Testing fixed ability"):
        state, temp_state, (parallel_A, outcomes_A), (parallel_B, outcomes_B) = env.step(
            main_state=state,
            policy_net=policy_net,
            deterministic=False,
            fix=False,
            update_normalizer=True,
            commit_strategy="random"
        )

    final_ability = state.ability

    # Check if ability changed
    ability_diff = (final_ability - initial_ability).abs().max().item()

    print(f"\nInitial ability: mean={initial_ability.mean():.4f}, std={initial_ability.std():.4f}")
    print(f"Final ability:   mean={final_ability.mean():.4f}, std={final_ability.std():.4f}")
    print(f"Max absolute change: {ability_diff:.10f}")

    if ability_diff < 1e-6:
        print("✓ PASS: Ability is fixed when fix=True")
    else:
        print("✗ FAIL: Ability changed when fix=True!")

    return ability_diff < 1e-6


def main():
    """Run all tests."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test environment transitions and shock distribution.")
    parser.add_argument(
        '--configs',
        nargs='+',
        default=['config/default.yaml'],
        help='Paths to one or more config files. They are merged in the given order.'
    )
    parser.add_argument(
        '--n-steps',
        type=int,
        default=100,
        help='Number of transition steps to simulate (default: 100)'
    )
    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {', '.join(args.configs)}")
    config_dict = load_configs(args.configs)
    config = dict_to_namespace(config_dict)

    # Display shock parameters
    print("\n" + "="*60)
    print("SHOCK PARAMETERS")
    print("="*60)
    print(f"  rho_v (persistence): {config.shock.rho_v}")
    print(f"  sigma_v (std dev):   {config.shock.sigma_v}")
    print(f"  v_bar (mean):        {config.shock.v_bar}")
    print(f"  p (become star):     {config.shock.p}")
    print(f"  q (stay star):       {config.shock.q}")
    if hasattr(config.shock, 'v_min') and hasattr(config.shock, 'v_max'):
        print(f"  v_min (lower bound): {config.shock.v_min:.4f}")
        print(f"  v_max (upper bound): {config.shock.v_max:.4f}")
    if hasattr(config.bewley_model, 'fix'):
        print(f"  fix (complete markets): {config.bewley_model.fix}")

    # Test 1: Shock distribution with normal transitions
    print("\n" + "="*60)
    print("RUNNING TESTS")
    print("="*60)

    ability_history, ability_changes = test_shock_distribution(config, n_steps=args.n_steps)

    # Test 2: Fixed ability (complete markets)
    test_with_fixed_ability(config)

    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)
    print("\nRecommendations:")
    print("1. If many agents hit v_min/v_max bounds, consider:")
    print("   - Clipping shocks to ±3σ in src/shocks.py")
    print("   - Increasing v_min/v_max bounds in configloader.py")
    print("2. Check test_shock_distribution.png for visual analysis")
    print(f"3. Config used: {', '.join(args.configs)}")


if __name__ == "__main__":
    main()
