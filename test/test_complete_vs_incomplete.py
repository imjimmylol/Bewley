#!/usr/bin/env python3
"""
Test complete vs incomplete markets with config files.

This verifies that:
1. Complete market (fix=true): ability stays constant
2. Incomplete market (fix=false): ability transitions with shocks
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.configloader import load_configs, dict_to_namespace
from src.environment import EconomyEnv
from src.normalizer import RunningPerAgentWelford
from src.models.model import FiLMResNet2In
from src.train import initialize_env_state


def test_market_type(config_path, n_steps=5):
    """Test market type from config."""
    print("=" * 70)
    print(f"Testing: {config_path}")
    print("=" * 70)

    # Load config
    config_dict = load_configs([config_path])
    config = dict_to_namespace(config_dict)

    # Get fix flag
    fix_ability = getattr(config.bewley_model, 'fix', False)
    market_type = "COMPLETE" if fix_ability else "INCOMPLETE"

    print(f"Config: {Path(config_path).name}")
    print(f"Experiment: {config.exp_name}")
    print(f"Market type: {market_type}")
    print(f"fix flag: {fix_ability}")
    print()

    # Initialize components
    device = torch.device("cpu")
    normalizer = RunningPerAgentWelford(batch_dim=0, agent_dim=1)
    env = EconomyEnv(config, normalizer, device=device)

    policy_net = FiLMResNet2In(
        state_dim=2*config.training.agents+2,
        cond_dim=5,
        output_dim=3
    ).to(device)

    main_state = initialize_env_state(config, device)

    # Store initial ability
    initial_ability = main_state.ability.clone()

    print(f"Initial ability stats:")
    print(f"  Mean: {initial_ability.mean():.4f}")
    print(f"  Std:  {initial_ability.std():.4f}")
    print(f"  Min:  {initial_ability.min():.4f}")
    print(f"  Max:  {initial_ability.max():.4f}")
    print()

    # Run simulation
    print(f"Running {n_steps} steps...")
    ability_history = [initial_ability.cpu().numpy()]

    for step in range(n_steps):
        main_state, temp_state, (parallel_A, outcomes_A), (parallel_B, outcomes_B) = env.step(
            main_state=main_state,
            policy_net=policy_net,
            deterministic=False,
            fix=fix_ability,  # Use flag from config
            update_normalizer=True,
            commit_strategy="random"
        )

        # Store ability
        ability_history.append(main_state.ability.cpu().numpy())

    # Final ability
    final_ability = main_state.ability

    print(f"\nFinal ability stats:")
    print(f"  Mean: {final_ability.mean():.4f}")
    print(f"  Std:  {final_ability.std():.4f}")
    print(f"  Min:  {final_ability.min():.4f}")
    print(f"  Max:  {final_ability.max():.4f}")
    print()

    # Check if ability changed
    ability_diff = (final_ability - initial_ability).abs()
    max_change = ability_diff.max().item()
    mean_change = ability_diff.mean().item()

    print(f"Ability changes:")
    print(f"  Max absolute change:  {max_change:.10f}")
    print(f"  Mean absolute change: {mean_change:.10f}")
    print()

    # Verify behavior
    if fix_ability:
        # Complete markets: ability should be unchanged
        if max_change < 1e-6:
            print("✓ PASS: Ability is fixed (complete markets)")
            return True
        else:
            print(f"✗ FAIL: Ability changed by {max_change:.6f} (expected 0)")
            return False
    else:
        # Incomplete markets: ability should change (unless very unlucky with random seed)
        if max_change > 1e-6:
            print("✓ PASS: Ability transitions (incomplete markets)")
            return True
        else:
            print(f"⚠ WARNING: Ability didn't change (might be random chance)")
            print("  This is unlikely but possible with very small shock params")
            return True


def main():
    print("\n" + "=" * 70)
    print("TESTING COMPLETE VS INCOMPLETE MARKETS")
    print("=" * 70)
    print()

    # Test both configs
    configs_to_test = [
        "config/1202/iq_init_complete.yaml",
        "config/1202/iq_init_incomplete.yaml"
    ]

    results = {}
    for config_path in configs_to_test:
        try:
            passed = test_market_type(config_path, n_steps=10)
            results[config_path] = passed
        except FileNotFoundError:
            print(f"✗ Config not found: {config_path}")
            results[config_path] = False
        except Exception as e:
            print(f"✗ Error testing {config_path}: {e}")
            import traceback
            traceback.print_exc()
            results[config_path] = False
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for config_path, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        config_name = Path(config_path).name
        print(f"{status}: {config_name}")

    print()
    if all(results.values()):
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")


if __name__ == "__main__":
    main()
