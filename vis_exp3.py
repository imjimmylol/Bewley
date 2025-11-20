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
        dict with collected data from all agents across all batches
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
                print(f"    Mean ability: {temp_state.ability.mean().item():.4f}")
                print(f"    Mean money_disposable: {temp_state.money_disposable.mean().item():.4f}")
                print(f"    Mean consumption: {temp_state.consumption.mean().item():.4f}")
                print(f"    Mean labor: {temp_state.labor.mean().item():.4f}")
                print(f"    Mean savings_ratio: {temp_state.savings_ratio.mean().item():.4f}")

            current_state = next_state

    # Flatten all data across batches and agents
    # Each array was (n_transitions, batch_size, n_agents)
    # We want (n_transitions * batch_size * n_agents,)
    for key in collected_data:
        collected_data[key] = np.array(collected_data[key]).flatten()

    n_points = len(collected_data['ability'])
    print(f"✓ Collected {n_points} data points")

    return collected_data


def compute_statistics(data):
    """
    Compute correlation statistics from collected data.

    Returns dict with correlations and regression fits.
    """
    ability = data['ability']
    money = data['money_disposable']
    consumption = data['consumption']
    labor = data['labor']
    savings_ratio = data['savings_ratio']

    stats_dict = {
        'corr_ability_money': np.corrcoef(ability, money)[0, 1],
        'corr_ability_consumption': np.corrcoef(ability, consumption)[0, 1],
        'corr_ability_labor': np.corrcoef(ability, labor)[0, 1],
        'corr_ability_savings_ratio': np.corrcoef(ability, savings_ratio)[0, 1],
        'corr_money_consumption': np.corrcoef(money, consumption)[0, 1],
        'corr_money_labor': np.corrcoef(money, labor)[0, 1],
        'corr_money_savings_ratio': np.corrcoef(money, savings_ratio)[0, 1],
    }

    # Linear regression: consumption = f(ability)
    slope, intercept, r_value, _, _ = stats.linregress(ability, consumption)
    stats_dict['consumption_vs_ability_slope'] = slope
    stats_dict['consumption_vs_ability_r2'] = r_value ** 2

    # Linear regression: money = f(ability)
    slope, intercept, r_value, _, _ = stats.linregress(ability, money)
    stats_dict['money_vs_ability_slope'] = slope
    stats_dict['money_vs_ability_r2'] = r_value ** 2

    return stats_dict


def plot_decision_rules_scatter(data, save_path="exp3_decision_rules.png"):
    """
    Plot decision rules as scatter plots from naturally-evolved state.

    This shows the actual relationship between variables on the training manifold.
    """
    ability = data['ability']
    money = data['money_disposable']
    consumption = data['consumption']
    labor = data['labor']
    savings = data['savings']
    savings_ratio = data['savings_ratio']

    # Compute statistics
    stats_dict = compute_statistics(data)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Subsample for cleaner visualization if too many points
    n_points = len(ability)
    if n_points > 2000:
        idx = np.random.choice(n_points, 2000, replace=False)
    else:
        idx = np.arange(n_points)

    # Plot 1: Consumption vs Ability
    axes[0, 0].scatter(ability[idx], consumption[idx], alpha=0.3, s=10, c='blue')
    # Add trend line
    z = np.polyfit(ability, consumption, 1)
    p = np.poly1d(z)
    x_line = np.linspace(ability.min(), ability.max(), 100)
    axes[0, 0].plot(x_line, p(x_line), 'r-', linewidth=2,
                    label=f"slope={z[0]:.3f}")
    axes[0, 0].set_xlabel("Ability", fontsize=11)
    axes[0, 0].set_ylabel("Consumption", fontsize=11)
    axes[0, 0].set_title(f"Consumption vs Ability\n(corr={stats_dict['corr_ability_consumption']:.3f})",
                         fontsize=12)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Labor vs Ability
    axes[0, 1].scatter(ability[idx], labor[idx], alpha=0.3, s=10, c='red')
    z = np.polyfit(ability, labor, 1)
    p = np.poly1d(z)
    axes[0, 1].plot(x_line, p(x_line), 'darkred', linewidth=2,
                    label=f"slope={z[0]:.3f}")
    axes[0, 1].set_xlabel("Ability", fontsize=11)
    axes[0, 1].set_ylabel("Labor", fontsize=11)
    axes[0, 1].set_title(f"Labor vs Ability\n(corr={stats_dict['corr_ability_labor']:.3f})",
                         fontsize=12)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Savings vs Ability
    axes[0, 2].scatter(ability[idx], savings[idx], alpha=0.3, s=10, c='green')
    z = np.polyfit(ability, savings, 1)
    p = np.poly1d(z)
    axes[0, 2].plot(x_line, p(x_line), 'darkgreen', linewidth=2,
                    label=f"slope={z[0]:.3f}")
    axes[0, 2].set_xlabel("Ability", fontsize=11)
    axes[0, 2].set_ylabel("Savings", fontsize=11)
    axes[0, 2].set_title(f"Savings vs Ability\n(corr={stats_dict['corr_ability_savings_ratio']:.3f})",
                         fontsize=12)
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Savings Ratio vs Ability
    axes[1, 0].scatter(ability[idx], savings_ratio[idx], alpha=0.3, s=10, c='purple')
    z = np.polyfit(ability, savings_ratio, 1)
    p = np.poly1d(z)
    axes[1, 0].plot(x_line, p(x_line), 'darkviolet', linewidth=2,
                    label=f"slope={z[0]:.4f}")
    axes[1, 0].set_xlabel("Ability", fontsize=11)
    axes[1, 0].set_ylabel("Savings Ratio", fontsize=11)
    axes[1, 0].set_title(f"Savings Ratio vs Ability\n(corr={stats_dict['corr_ability_savings_ratio']:.3f})",
                         fontsize=12)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Money Disposable vs Ability (the conditioning relationship)
    axes[1, 1].scatter(ability[idx], money[idx], alpha=0.3, s=10, c='cyan')
    z = np.polyfit(ability, money, 1)
    p = np.poly1d(z)
    axes[1, 1].plot(x_line, p(x_line), 'darkcyan', linewidth=2,
                    label=f"slope={z[0]:.3f}")
    axes[1, 1].set_xlabel("Ability", fontsize=11)
    axes[1, 1].set_ylabel("Money Disposable", fontsize=11)
    axes[1, 1].set_title(f"Money vs Ability (Training Manifold)\n(corr={stats_dict['corr_ability_money']:.3f})",
                         fontsize=12)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Consumption vs Money Disposable
    axes[1, 2].scatter(money[idx], consumption[idx], alpha=0.3, s=10, c='orange')
    z = np.polyfit(money, consumption, 1)
    p = np.poly1d(z)
    x_line_money = np.linspace(money.min(), money.max(), 100)
    axes[1, 2].plot(x_line_money, p(x_line_money), 'darkorange', linewidth=2,
                    label=f"slope={z[0]:.3f}")
    axes[1, 2].set_xlabel("Money Disposable", fontsize=11)
    axes[1, 2].set_ylabel("Consumption", fontsize=11)
    axes[1, 2].set_title(f"Consumption vs Money\n(corr={stats_dict['corr_money_consumption']:.3f})",
                         fontsize=12)
    axes[1, 2].legend(fontsize=9)
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle("Decision Rules on Training Manifold (After Transition)",
                 fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Decision rules plot saved to: {save_path}")
    plt.close()

    return stats_dict


def plot_binned_decision_rules(data, n_bins=10, save_path="exp3_binned_rules.png"):
    """
    Plot decision rules with binned averages for clearer trends.

    Groups agents by ability bins and shows mean decisions in each bin.
    """
    ability = data['ability']
    money = data['money_disposable']
    consumption = data['consumption']
    labor = data['labor']
    savings = data['savings']
    savings_ratio = data['savings_ratio']

    # Create ability bins
    ability_bins = np.linspace(ability.min(), ability.max(), n_bins + 1)
    bin_centers = (ability_bins[:-1] + ability_bins[1:]) / 2
    bin_indices = np.digitize(ability, ability_bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute means and stds for each bin
    def compute_bin_stats(values):
        means = np.array([values[bin_indices == i].mean() if (bin_indices == i).any() else np.nan
                         for i in range(n_bins)])
        stds = np.array([values[bin_indices == i].std() if (bin_indices == i).any() else np.nan
                        for i in range(n_bins)])
        return means, stds

    consumption_mean, consumption_std = compute_bin_stats(consumption)
    labor_mean, labor_std = compute_bin_stats(labor)
    savings_mean, savings_std = compute_bin_stats(savings)
    savings_ratio_mean, savings_ratio_std = compute_bin_stats(savings_ratio)
    money_mean, money_std = compute_bin_stats(money)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Consumption vs Ability (binned)
    axes[0, 0].errorbar(bin_centers, consumption_mean, yerr=consumption_std,
                        fmt='o-', capsize=3, color='blue', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel("Ability (binned)", fontsize=11)
    axes[0, 0].set_ylabel("Mean Consumption", fontsize=11)
    axes[0, 0].set_title("Consumption vs Ability", fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Labor vs Ability (binned)
    axes[0, 1].errorbar(bin_centers, labor_mean, yerr=labor_std,
                        fmt='o-', capsize=3, color='red', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel("Ability (binned)", fontsize=11)
    axes[0, 1].set_ylabel("Mean Labor", fontsize=11)
    axes[0, 1].set_title("Labor vs Ability", fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Savings vs Ability (binned)
    axes[0, 2].errorbar(bin_centers, savings_mean, yerr=savings_std,
                        fmt='o-', capsize=3, color='green', linewidth=2, markersize=6)
    axes[0, 2].set_xlabel("Ability (binned)", fontsize=11)
    axes[0, 2].set_ylabel("Mean Savings", fontsize=11)
    axes[0, 2].set_title("Savings vs Ability", fontsize=12)
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Savings Ratio vs Ability (binned)
    axes[1, 0].errorbar(bin_centers, savings_ratio_mean, yerr=savings_ratio_std,
                        fmt='o-', capsize=3, color='purple', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel("Ability (binned)", fontsize=11)
    axes[1, 0].set_ylabel("Mean Savings Ratio", fontsize=11)
    axes[1, 0].set_title("Savings Ratio vs Ability", fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Money vs Ability (binned) - shows the natural correlation
    axes[1, 1].errorbar(bin_centers, money_mean, yerr=money_std,
                        fmt='o-', capsize=3, color='cyan', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel("Ability (binned)", fontsize=11)
    axes[1, 1].set_ylabel("Mean Money Disposable", fontsize=11)
    axes[1, 1].set_title("Money vs Ability (Natural Correlation)", fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Summary statistics
    axes[1, 2].axis('off')
    stats_dict = compute_statistics(data)
    summary_text = (
        f"Binned Decision Rules Summary\n"
        f"{'='*35}\n\n"
        f"Number of bins: {n_bins}\n"
        f"Total data points: {len(ability)}\n\n"
        f"Correlations:\n"
        f"  Ability-Money: {stats_dict['corr_ability_money']:.3f}\n"
        f"  Ability-Consumption: {stats_dict['corr_ability_consumption']:.3f}\n"
        f"  Ability-Labor: {stats_dict['corr_ability_labor']:.3f}\n"
        f"  Ability-SavingsRatio: {stats_dict['corr_ability_savings_ratio']:.3f}\n\n"
        f"Expected patterns:\n"
        f"  Consumption: Should INCREASE\n"
        f"  Labor: Could increase/decrease\n"
        f"  Savings Ratio: Should be STABLE"
    )
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle("Binned Decision Rules (Mean +/- Std)", fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Binned decision rules plot saved to: {save_path}")
    plt.close()


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

    stats_dict = compute_statistics(data)

    print(f"\nCorrelations on Training Manifold:")
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
        save_path="exp3_scatter_decision_rules.png"
    )

    # Binned plot (clearer trends)
    plot_binned_decision_rules(
        data,
        n_bins=args.n_bins,
        save_path="exp3_binned_decision_rules.png"
    )

    print("\n" + "="*60)
    print("EXPERIMENT 3 COMPLETE")
    print("="*60)
    print(f"\nKey insight: If consumption INCREASES with ability,")
    print(f"the wealth effect is working correctly on the training manifold.")
    print(f"\nIf consumption still DECREASES, the issue is in the model,")
    print(f"not the visualization methodology.")


if __name__ == "__main__":
    main()
