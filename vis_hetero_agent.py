#!/usr/bin/env python3
"""
Experiment 3: Decision Rule Visualization on Training Manifold

Instead of fitting artificial relationships, this script:
1. Loads checkpoint state
2. Runs one environment transition
3. Visualizes decision rules from the naturally-evolved state

This ensures we visualize on states the network actually encounters during training.
"""

import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy import stats

from src.valid import load_checkpoint
from src.environment import EconomyEnv
from src.train import initialize_env_state
from src.env_state import MainState
from src.visualization import (
    compute_statistics,
    plot_decision_rules_scatter,
    plot_binned_decision_rules,
    plot_state_distributions
)


def dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d


def run_transition_and_collect(state, policy_net, env, n_transitions=1):
    """
    Run environment transitions and collect state/action data.

    Args:
        state: Initial MainState
        policy_net: Trained policy network
        env: Environment instance
        n_transitions: Number of transitions to run

    Returns:
        dict with collected data, structured as (n_transitions, batch_size, n_agents)
    """
    collected_data = {
        # State variables (inputs)
        'ability': [],
        'money_disposable': [],
        'savings_input': [],  # savings from previous period

        # Action variables (outputs)
        'consumption': [],
        'labor': [],
        'savings': [],  # savings for next period
        'mu': [],
        'savings_ratio': [],

        # Derived variables
        'wage': [],
        'income_before_tax': [],
    }

    current_state = state

    print(f"\nRunning {n_transitions} environment transition(s)...")

    with torch.no_grad():
        for t in range(n_transitions):
            # Run one environment step
            next_state, temp_state, (parallel_A, outcomes_A), (parallel_B, outcomes_B) = env.step(
                main_state=current_state,
                policy_net=policy_net,
                deterministic=True,  # Deterministic for visualization
                update_normalizer=False,
                commit_strategy="A"
            )

            # Collect data from this transition
            # temp_state contains the realized outcomes for current period
            collected_data['ability'].append(temp_state.ability.cpu().numpy())
            collected_data['money_disposable'].append(temp_state.money_disposable.cpu().numpy())
            collected_data['savings_input'].append(current_state.savings.cpu().numpy())

            collected_data['consumption'].append(temp_state.consumption.cpu().numpy())
            collected_data['labor'].append(temp_state.labor.cpu().numpy())
            collected_data['savings'].append(temp_state.savings.cpu().numpy())
            collected_data['mu'].append(temp_state.mu.cpu().numpy())
            collected_data['savings_ratio'].append(temp_state.savings_ratio.cpu().numpy())

            collected_data['wage'].append(temp_state.wage.cpu().numpy())
            collected_data['income_before_tax'].append(temp_state.income_before_tax.cpu().numpy())

            if t == 0:
                print(f"  Transition {t+1}:")
                print(f"    Batch size: {temp_state.ability.shape[0]}")
                print(f"    Num agents: {temp_state.ability.shape[1]}")
                print(f"    Mean ability: {temp_state.ability.mean().item():.4f}")
                print(f"    Mean money_disposable: {temp_state.money_disposable.mean().item():.4f}")
                print(f"    Mean consumption: {temp_state.consumption.mean().item():.4f}")
                print(f"    Mean labor: {temp_state.labor.mean().item():.4f}")
                print(f"    Mean savings_ratio: {temp_state.savings_ratio.mean().item():.4f}")

            current_state = next_state

    # Convert to numpy arrays: (n_transitions, batch_size, n_agents)
    for key in collected_data:
        collected_data[key] = np.array(collected_data[key])

    shape = collected_data['ability'].shape
    n_points = shape[0] * shape[1] * shape[2]
    print(f"✓ Collected data shape: {shape} (transitions, batches, agents)")
    print(f"✓ Total data points: {n_points}")

    return collected_data


# Note: plotting functions (compute_statistics, plot_decision_rules_scatter,
# plot_binned_decision_rules, plot_state_distributions) are now imported
# from src.visualization module


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Decision Rules on Training Manifold"
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Path to checkpoint directory")
    parser.add_argument("--step", type=int, required=True,
                       help="Training step to load")
    parser.add_argument("--state_step", type=int, default=None,
                       help="Training step to load state from (default: same as --step)")
    parser.add_argument("--config", type=str, default="config/baseline.yaml",
                       help="Path to config file")
    parser.add_argument("--n_transitions", type=int, default=1,
                       help="Number of environment transitions to run")
    parser.add_argument("--n_bins", type=int, default=10,
                       help="Number of bins for binned visualization")
    parser.add_argument("--batch_idx", type=int, default=0,
                       help="Which batch to visualize (default: 0)")

    args = parser.parse_args()

    state_step = args.state_step if args.state_step is not None else args.step

    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = dict_to_namespace(config_dict)

    # Determine device
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    print(f"\n{'='*60}")
    print("EXPERIMENT 3: DECISION RULES ON TRAINING MANIFOLD")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Model step: {args.step}")
    print(f"State step: {state_step}")
    print(f"Transitions: {args.n_transitions}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Load checkpoint
    policy_net, normalizer, _ = load_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        step=args.step,
        config=config,
        device=device
    )

    # Load state
    _, _, state = load_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        step=state_step,
        config=config,
        device=device
    )

    if state is None:
        print("⚠ No saved state found, initializing new state")
        state = initialize_env_state(config, device)

    # Initialize environment
    env = EconomyEnv(config, normalizer, device=device)

    # ========================================================================
    # STEP 0: Plot initial state distributions
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 0: Plotting Initial State Distributions")
    print("="*60)

    plot_state_distributions(
        state=state,
        save_path="exp3_initial_state_distributions.png"
    )

    # ========================================================================
    # STEP 1: Run transition and collect data
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 1: Running Environment Transition")
    print("="*60)

    data = run_transition_and_collect(
        state=state,
        policy_net=policy_net,
        env=env,
        n_transitions=args.n_transitions
    )

    # ========================================================================
    # STEP 2: Compute and print statistics
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 2: Computing Statistics")
    print("="*60)

    # Extract batch data for statistics
    flat_data = {
        'ability': data['ability'][:, args.batch_idx, :].flatten(),
        'money_disposable': data['money_disposable'][:, args.batch_idx, :].flatten(),
        'consumption': data['consumption'][:, args.batch_idx, :].flatten(),
        'labor': data['labor'][:, args.batch_idx, :].flatten(),
        'savings': data['savings'][:, args.batch_idx, :].flatten(),
        'savings_ratio': data['savings_ratio'][:, args.batch_idx, :].flatten(),
    }

    stats_dict = compute_statistics(flat_data)

    print(f"\nCorrelations on Training Manifold (Batch {args.batch_idx}):")
    print(f"  Ability-Money: {stats_dict['corr_ability_money']:.4f}")
    print(f"  Ability-Consumption: {stats_dict['corr_ability_consumption']:.4f}")
    print(f"  Ability-Labor: {stats_dict['corr_ability_labor']:.4f}")
    print(f"  Ability-SavingsRatio: {stats_dict['corr_ability_savings_ratio']:.4f}")
    print(f"  Money-Consumption: {stats_dict['corr_money_consumption']:.4f}")

    print(f"\nConsumption vs Ability regression:")
    print(f"  Slope: {stats_dict['consumption_vs_ability_slope']:.4f}")
    print(f"  R²: {stats_dict['consumption_vs_ability_r2']:.4f}")

    # ========================================================================
    # STEP 3: Plot results
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 3: Plotting Results")
    print("="*60)

    # Scatter plot
    plot_decision_rules_scatter(
        data,
        batch_idx=args.batch_idx,
        save_path=f"exp3_scatter_decision_rules_batch{args.batch_idx}.png"
    )

    # Binned plot (clearer trends)
    plot_binned_decision_rules(
        data,
        batch_idx=args.batch_idx,
        n_bins=args.n_bins,
        save_path=f"exp3_binned_decision_rules_batch{args.batch_idx}.png"
    )

    print("\n" + "="*60)
    print("EXPERIMENT 3 COMPLETE")
    print("="*60)
    print(f"\nVisualized batch {args.batch_idx}")
    print(f"Each point in the scatter plots represents one agent's actual decision.")
    print(f"\nKey insight: If consumption INCREASES with ability,")
    print(f"the wealth effect is working correctly on the training manifold.")
    print(f"\nIf consumption still DECREASES, the issue is in the model,")
    print(f"not the visualization methodology.")


if __name__ == "__main__":
    main()
