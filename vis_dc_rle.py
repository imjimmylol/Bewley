#!/usr/bin/env python3
"""
Visualize decision rules from saved checkpoints.

This script loads a checkpoint and visualizes the policy network's decision rules.
"""

import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

from src.valid import load_checkpoint
from src.environment import EconomyEnv
from src.train import initialize_env_state


def dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d


def visualize_decision_rules(policy_net, normalizer, env, config, device, initial_state):
    """
    Visualize decision rules.

    TODO: Implement your own visualization logic here.

    Args:
        policy_net: Loaded policy network
        normalizer: Loaded normalizer
        env: Environment instance
        config: Configuration object
        device: Device
        initial_state: Initial MainState (either loaded from checkpoint or newly initialized)
    """
    print("\n" + "="*60)
    print("VISUALIZING DECISION RULES")
    print("="*60)

    # Use provided initial state
    state = initial_state

    # Run a few steps to collect data
    n_steps = 20
    trajectories = {
        'consumption': [],
        'labor': [],
        'savings': [],
        'mu': [],
        'savings_ratio': [],
        'ability': [],
        'money_disposable': [],
        'wage': [],
        'income_before_tax': [],
    }

    print(f"\nRunning {n_steps} environment steps...")
    print(f"Initial state:")
    print(f"  Mean money_disposable: {state.moneydisposable.mean().item():.4f}")
    print(f"  Mean savings: {state.savings.mean().item():.4f}")
    print(f"  Mean ability: {state.ability.mean().item():.4f}")
    print()

    with torch.no_grad():
        for t in range(n_steps):
            state, temp_state, (parallel_A, outcomes_A), (parallel_B, outcomes_B) = env.step(
                main_state=state,
                policy_net=policy_net,
                deterministic=True,
                update_normalizer=False,
                commit_strategy="A"
            )

            # Collect trajectories
            trajectories['consumption'].append(temp_state.consumption.cpu().numpy())
            trajectories['labor'].append(temp_state.labor.cpu().numpy())
            trajectories['savings'].append(state.savings.cpu().numpy())
            trajectories['mu'].append(temp_state.mu.cpu().numpy())
            trajectories['savings_ratio'].append(temp_state.savings_ratio.cpu().numpy())
            trajectories['ability'].append(state.ability.cpu().numpy())
            trajectories['money_disposable'].append(temp_state.money_disposable.cpu().numpy())
            trajectories['wage'].append(temp_state.wage.cpu().numpy())
            trajectories['income_before_tax'].append(temp_state.income_before_tax.cpu().numpy())

            # Print diagnostic every 5 steps
            if t % 5 == 0 or t == 0:
                print(f"Step {t}:")
                print(f"  Mean consumption: {temp_state.consumption.mean().item():.4f}")
                print(f"  Mean money_disposable: {temp_state.money_disposable.mean().item():.4f}")
                print(f"  Mean savings (t+1): {state.savings.mean().item():.4f}")
                print(f"  Mean ability: {state.ability.mean().item():.4f}")
                print(f"  Mean wage: {temp_state.wage.mean().item():.4f}")

    # Convert to numpy arrays
    for key in trajectories:
        trajectories[key] = np.array(trajectories[key])

    print("✓ Data collection complete")

    # ========================================================================
    # TODO: Add your custom visualization here
    # ========================================================================

    # Example: Basic time series plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Plot mean consumption over time
    axes[0, 0].plot(trajectories['consumption'].mean(axis=(1, 2)))
    axes[0, 0].set_title('Mean Consumption Over Time')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Consumption')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot mean labor over time
    axes[0, 1].plot(trajectories['labor'].mean(axis=(1, 2)))
    axes[0, 1].set_title('Mean Labor Over Time')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Labor')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot mean savings over time
    axes[0, 2].plot(trajectories['savings'].mean(axis=(1, 2)))
    axes[0, 2].set_title('Mean Savings Over Time')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Savings')
    axes[0, 2].grid(True, alpha=0.3)

    # Plot mean money_disposable over time
    axes[1, 0].plot(trajectories['money_disposable'].mean(axis=(1, 2)))
    axes[1, 0].set_title('Mean Money Disposable Over Time')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Money Disposable')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot mean ability over time
    axes[1, 1].plot(trajectories['ability'].mean(axis=(1, 2)))
    axes[1, 1].set_title('Mean Ability Over Time')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Ability')
    axes[1, 1].grid(True, alpha=0.3)

    # Plot mean wage over time
    axes[1, 2].plot(trajectories['wage'].mean(axis=(1, 2)))
    axes[1, 2].set_title('Mean Wage Over Time')
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('Wage')
    axes[1, 2].grid(True, alpha=0.3)

    # Plot distribution of mu (final step)
    axes[2, 0].hist(trajectories['mu'][-1].flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[2, 0].set_title(f'Distribution of mu (final step)')
    axes[2, 0].set_xlabel('mu')
    axes[2, 0].set_ylabel('Frequency')
    axes[2, 0].grid(True, alpha=0.3)

    # Plot distribution of savings ratio (final step)
    axes[2, 1].hist(trajectories['savings_ratio'][-1].flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[2, 1].set_title(f'Distribution of Savings Ratio (final step)')
    axes[2, 1].set_xlabel('Savings Ratio')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].grid(True, alpha=0.3)

    # Plot distribution of consumption (final step)
    axes[2, 2].hist(trajectories['consumption'][-1].flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[2, 2].set_title(f'Distribution of Consumption (final step)')
    axes[2, 2].set_xlabel('Consumption')
    axes[2, 2].set_ylabel('Frequency')
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    output_path = "decision_rules_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    plt.close()

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (Final Step)")
    print("="*60)
    print(f"  Mean consumption:     {trajectories['consumption'][-1].mean():.4f}")
    print(f"  Mean labor:           {trajectories['labor'][-1].mean():.4f}")
    print(f"  Mean savings:         {trajectories['savings'][-1].mean():.4f}")
    print(f"  Mean mu:              {trajectories['mu'][-1].mean():.4f}")
    print(f"  Mean savings ratio:   {trajectories['savings_ratio'][-1].mean():.4f}")
    print("="*60 + "\n")

    return trajectories


def main():
    parser = argparse.ArgumentParser(description="Visualize decision rules from checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Path to checkpoint directory (e.g., checkpoints/run_name)")
    parser.add_argument("--step", type=int, required=True,
                       help="Training step to load")
    parser.add_argument("--config", type=str, default="config/baseline.yaml",
                       help="Path to config file")

    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

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
    print(f"DECISION RULE VISUALIZATION")
    print(f"{'='*60}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Step:           {args.step}")
    print(f"Config:         {args.config}")
    print(f"Device:         {device}")
    print(f"{'='*60}\n")

    # Load checkpoint
    policy_net, normalizer, state = load_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        step=args.step,
        config=config,
        device=device
    )

    # Initialize environment
    env = EconomyEnv(config, normalizer, device=device)

    # Use saved state if available, otherwise initialize new state
    if state is not None:
        print(f"✓ Using saved state from checkpoint")
        print(f"  State batch size: {state.moneydisposable.shape[0]}")
        print(f"  State n_agents: {state.moneydisposable.shape[1]}")
        initial_state = state
    else:
        print("⚠ No saved state found, initializing new state")
        initial_state = initialize_env_state(config, device)

    # Visualize decision rules
    trajectories = visualize_decision_rules(policy_net, normalizer, env, config, device, initial_state)

    print("✓ Done!")


if __name__ == "__main__":
    main()
