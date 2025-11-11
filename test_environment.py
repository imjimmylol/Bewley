# test_environment.py
"""
Test file for EconomyEnv class.

This file demonstrates how to use the EconomyEnv and helps test its functionality.
"""

import torch
import torch.nn as nn
from src.environment import EconomyEnv
from src.env_state import MainState
from src.train import initialize_env_state
from src.normalizer import RunningPerAgentWelford
from src.utils.configloader import load_configs, dict_to_namespace, compute_derived_params
from src.models.model import FiLMResNet2In

def test_environment_initialization():
    """Test 1: Initialize environment and check basic setup."""
    print("\n" + "="*70)
    print("TEST 1: Environment Initialization")
    print("="*70)

    # Load config
    config_dict = load_configs(['config/default.yaml'])
    config_dict = compute_derived_params(config_dict)
    config = dict_to_namespace(config_dict)

    # Initialize components
    device = torch.device("cpu")
    normalizer = RunningPerAgentWelford(batch_dim=0, agent_dim=1)
    env = EconomyEnv(config, normalizer, device)

    print(f"✓ EconomyEnv initialized")
    print(f"  - Device: {device}")
    print(f"  - Batch size: {env.batch_size}")
    print(f"  - Number of agents: {env.n_agents}")
    print(f"  - History length: {env.history_length}")

    return env, config, device


def test_main_state_initialization(config, device):
    """Test 2: Initialize MainState."""
    print("\n" + "="*70)
    print("TEST 2: MainState Initialization")
    print("="*70)

    main_state = initialize_env_state(config, device)

    print(f"✓ MainState initialized")
    print(f"  - moneydisposable shape: {main_state.moneydisposable.shape}")
    print(f"  - savings shape: {main_state.savings.shape}")
    print(f"  - ability shape: {main_state.ability.shape}")
    print(f"  - ret: {main_state.ret}")
    print(f"  - is_superstar_vA shape: {main_state.is_superstar_vA.shape}")
    print(f"  - is_superstar_vB shape: {main_state.is_superstar_vB.shape}")

    return main_state


def test_temporary_state_creation(env, main_state, policy_net):
    """Test 3: Create TemporaryState (will fail on NotImplementedError)."""
    print("\n" + "="*70)
    print("TEST 3: TemporaryState Creation")
    print("="*70)

    try:
        temp_state = env.create_temporary_state(
            main_state=main_state,
            policy_net=policy_net,
            update_normalizer=True
        )

        print(f"✓ TemporaryState created")
        print(f"  - consumption shape: {temp_state.consumption.shape}")
        print(f"  - labor shape: {temp_state.labor.shape}")
        print(f"  - savings_ratio shape: {temp_state.savings_ratio.shape}")
        print(f"  - wage shape: {temp_state.wage.shape}")
        print(f"  - ret shape: {temp_state.ret.shape}")

        return temp_state

    except NotImplementedError as e:
        print(f"✗ NotImplementedError: {e}")
        print(f"  → You need to implement the helper methods:")
        print(f"     - _prepare_features()")
        print(f"     - _compute_agent_actions()")
        print(f"     - _compute_market_equilibrium()")
        print(f"     - _compute_income_and_taxes()")
        return None


def test_parallel_transition(env, temp_state):
    """Test 4: Transition to ParallelState (will fail if temp_state is None)."""
    print("\n" + "="*70)
    print("TEST 4: Transition to ParallelState")
    print("="*70)

    if temp_state is None:
        print("✗ Skipping (temp_state is None)")
        return None, None

    try:
        parallel_A = env.transition_to_parallel(
            temp_state=temp_state,
            branch="A",
            deterministic=False
        )

        parallel_B = env.transition_to_parallel(
            temp_state=temp_state,
            branch="B",
            deterministic=False
        )

        print(f"✓ ParallelStates created")
        print(f"  Branch A:")
        print(f"    - ability shape: {parallel_A.ability.shape}")
        print(f"    - savings shape: {parallel_A.savings.shape}")
        print(f"  Branch B:")
        print(f"    - ability shape: {parallel_B.ability.shape}")
        print(f"    - savings shape: {parallel_B.savings.shape}")

        return parallel_A, parallel_B

    except NotImplementedError as e:
        print(f"✗ NotImplementedError: {e}")
        print(f"  → You need to implement:")
        print(f"     - _transition_ability()")
        return None, None


def test_parallel_outcomes(env, parallel_A, parallel_B,  policy_net):
    """Test 5: Compute outcomes for ParallelStates."""
    print("\n" + "="*70)
    print("TEST 5: Compute ParallelState Outcomes")
    print("="*70)

    if parallel_A is None or parallel_B is None:
        print("✗ Skipping (parallel states are None)")
        return None, None

    try:
        parallel_A, outcomes_A = env.compute_parallel_outcomes(
            parallel_state=parallel_A,
            policy_net=policy_net,
            update_normalizer=False
        )

        parallel_B, outcomes_B = env.compute_parallel_outcomes(
            parallel_state=parallel_B,
            policy_net=policy_net,
            update_normalizer=False
        )

        print(f"✓ Outcomes computed")
        print(f"  Branch A outcomes keys: {list(outcomes_A.keys())}")
        print(f"  Branch B outcomes keys: {list(outcomes_B.keys())}")

        return outcomes_A, outcomes_B

    except NotImplementedError as e:
        print(f"✗ NotImplementedError: {e}")
        print(f"  → You need to implement:")
        print(f"     - compute_parallel_outcomes()")
        return None, None


def test_full_step(env, main_state, policy_net):
    """Test 6: Execute full environment step."""
    print("\n" + "="*70)
    print("TEST 6: Full Environment Step")
    print("="*70)

    try:
        # Make a copy of main_state to preserve original
        main_state_copy = MainState(
            moneydisposable=main_state.moneydisposable.clone(),
            savings=main_state.savings.clone(),
            ability=main_state.ability.clone(),
            ret=main_state.ret,
            tax_params=main_state.tax_params.clone(),
            is_superstar_vA=main_state.is_superstar_vA.clone(),
            is_superstar_vB=main_state.is_superstar_vB.clone(),
            ability_history_vA=main_state.ability_history_vA,
            ability_history_vB=main_state.ability_history_vB,
        )

        main_state_updated, temp_state, (parallel_A, outcomes_A), (parallel_B, outcomes_B) = env.step(
            main_state=main_state_copy,
            policy_net=policy_net,
            deterministic=False,
            update_normalizer=True,
            commit_strategy="random"
        )

        print(f"✓ Full step executed successfully!")
        print(f"  - TemporaryState created")
        print(f"  - ParallelState A and B created")
        print(f"  - Outcomes A and B computed")
        print(f"  - MainState updated")
        print(f"\n  MainState before step:")
        print(f"    - savings mean: {main_state.savings.mean():.3f}")
        print(f"    - ability mean: {main_state.ability.mean():.3f}")
        print(f"    - moneydisposable mean: {main_state.moneydisposable.mean():.5f}")
        print(f"\n  MainState after step:")
        print(f"    - savings mean: {main_state_updated.savings.mean():.3f}")
        print(f"    - ability mean: {main_state_updated.ability.mean():.3f}")
        print(f"    - moneydisposable mean: {main_state_updated.moneydisposable.mean():.5f}")
        return True

    except NotImplementedError as e:
        print(f"✗ NotImplementedError: {e}")
        print(f"  → Complete all helper method implementations first")
        return False


def test_rollout(env, main_state, policy_net):
    """Test 7: Execute multi-step rollout."""
    print("\n" + "="*70)
    print("TEST 7: Multi-Step Rollout")
    print("="*70)

    try:
        # Make a copy
        from src.env_state import MainState as MS
        main_state_copy = MS(
            moneydisposable=main_state.moneydisposable.clone(),
            savings=main_state.savings.clone(),
            ability=main_state.ability.clone(),
            ret=main_state.ret,
            tax_params=main_state.tax_params.clone(),
            is_superstar_vA=main_state.is_superstar_vA.clone(),
            is_superstar_vB=main_state.is_superstar_vB.clone(),
            ability_history_vA=main_state.ability_history_vA,
            ability_history_vB=main_state.ability_history_vB,
        )

        final_state, trajectory = env.rollout(
            main_state=main_state_copy,
            policy_net=policy_net,
            n_steps=10,
            deterministic=False,
            update_normalizer=True,
            commit_strategy="random"
        )

        print(f"✓ Rollout completed!")
        print(f"  - Number of steps: 10")
        print(f"  - Trajectory length: {len(trajectory)}")
        print(f"  - Final savings mean: {final_state.savings.mean():.3f}")
        print(f"  - Final ability mean: {final_state.ability.mean():.3f}")

        return True

    except NotImplementedError as e:
        print(f"✗ NotImplementedError: {e}")
        print(f"  → Complete all helper method implementations first")
        return False


def main():
    """Run all tests."""
    print("\n" + "#"*70)
    print("# ECONOMYENV TEST SUITE")
    print("#"*70)

    # Test 1: Initialize environment and policy network
    env, config, device = test_environment_initialization()
    net = FiLMResNet2In(
        state_dim=2*config.training.agents+2,
        cond_dim=5,
        output_dim=3
    )
    # Test 2: Initialize MainState
    main_state = test_main_state_initialization(config, device)

    # Test 3: Create TemporaryState
    temp_state = test_temporary_state_creation(env, main_state, net)

    # Test 4: Transition to ParallelState
    parallel_A, parallel_B = test_parallel_transition(env, temp_state)

    # Test 5: Compute parallel outcomes
    outcomes_A, outcomes_B = test_parallel_outcomes(env, parallel_A, parallel_B, net)

    # Test 6: Full step
    step_success = test_full_step(env, main_state, net)

    # Test 7: Rollout (only if step succeeded)
    if step_success:
        test_rollout(env, main_state, net)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("✓ Tests 1-2: Basic initialization works")
    print("✗ Tests 3-7: Need to implement helper methods")
    print("\nNext steps:")
    print("1. Implement _prepare_features()")
    print("2. Implement _compute_agent_actions()")
    print("3. Implement _compute_market_equilibrium()")
    print("4. Implement _compute_income_and_taxes()")
    print("5. Implement _transition_ability()")
    print("6. Implement compute_parallel_outcomes()")
    print("\nOnce all methods are implemented, all tests should pass!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
