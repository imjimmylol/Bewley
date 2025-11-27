#!/usr/bin/env python3
"""
Visualization module for decision rules and state distributions.

Functions for plotting agent decision rules, correlations, and state distributions.
Used by both training (train.py) and offline analysis (vis_hetero_agent.py).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import wandb
import torch


def prepare_data_for_plotting(main_state, temp_state):
    """
    Convert training step data into format expected by plotting functions.

    This matches the structure from vis_hetero_agent.py's run_transition_and_collect().

    Args:
        main_state: MainState instance
        temp_state: TemporaryState instance

    Returns:
        dict: Data dictionary with shape (1, batch_size, n_agents)
              Keys: ability, money_disposable, savings_input, consumption, labor,
                    savings, mu, savings_ratio, wage, income_before_tax
    """
    # Add a "transition" dimension to match (n_transitions, batch_size, n_agents)
    # Since we only have one step, n_transitions = 1
    data = {
        # State variables (inputs)
        'ability': temp_state.ability.detach().cpu().numpy()[np.newaxis, ...],
        'money_disposable': temp_state.money_disposable.detach().cpu().numpy()[np.newaxis, ...],
        'savings_input': main_state.savings.detach().cpu().numpy()[np.newaxis, ...],

        # Action variables (outputs)
        'consumption': temp_state.consumption.detach().cpu().numpy()[np.newaxis, ...],
        'labor': temp_state.labor.detach().cpu().numpy()[np.newaxis, ...],
        'savings': temp_state.savings.detach().cpu().numpy()[np.newaxis, ...],
        'mu': temp_state.mu.detach().cpu().numpy()[np.newaxis, ...],
        'savings_ratio': temp_state.savings_ratio.detach().cpu().numpy()[np.newaxis, ...],

        # Derived variables
        'wage': temp_state.wage.detach().cpu().numpy()[np.newaxis, ...],
        'income_before_tax': temp_state.income_before_tax.detach().cpu().numpy()[np.newaxis, ...],
    }

    return data


def compute_statistics(data):
    """
    Compute correlation statistics from collected data.

    Args:
        data: Dictionary with flattened arrays (ability, money_disposable, consumption, etc.)

    Returns:
        dict: Correlations and regression fits
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


def plot_decision_rules_scatter(data, batch_idx=None, save_path="decision_rules.png",
                                 log_to_wandb=False, step=None):
    """
    Plot decision rules as scatter plots of actual agent decisions.

    Each point represents one agent's actual state and decision at a given timestep.

    Args:
        data: Dictionary with shape (n_transitions, batch_size, n_agents)
        batch_idx: Which batch to visualize. If None, uses all batches (default: None)
        save_path: Where to save the plot
        log_to_wandb: If True, log image to wandb
        step: Training step (for W&B logging)

    Returns:
        dict: Statistics dictionary
    """
    # Extract data - flatten across all dimensions to get full distribution
    if batch_idx is None:
        # Use ALL batches - flatten across (n_transitions, batch_size, n_agents)
        ability = data['ability'].flatten()
        money = data['money_disposable'].flatten()
        consumption = data['consumption'].flatten()
        labor = data['labor'].flatten()
        savings = data['savings'].flatten()
        savings_ratio = data['savings_ratio'].flatten()
        batch_label = "All Batches"
    else:
        # Use only specified batch
        ability = data['ability'][:, batch_idx, :].flatten()
        money = data['money_disposable'][:, batch_idx, :].flatten()
        consumption = data['consumption'][:, batch_idx, :].flatten()
        labor = data['labor'][:, batch_idx, :].flatten()
        savings = data['savings'][:, batch_idx, :].flatten()
        savings_ratio = data['savings_ratio'][:, batch_idx, :].flatten()
        batch_label = f"Batch {batch_idx}"

    # Create a flat version for compute_statistics
    flat_data = {
        'ability': ability,
        'money_disposable': money,
        'consumption': consumption,
        'labor': labor,
        'savings': savings,
        'savings_ratio': savings_ratio,
    }

    # Compute statistics
    stats_dict = compute_statistics(flat_data)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    n_points = len(ability)
    n_agents = data['ability'].shape[2]
    n_transitions = data['ability'].shape[0]

    # Use all points (each point is a real agent decision)
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

    fig.suptitle(f"Decision Rules: Actual Agent Decisions ({batch_label}, {n_points} points)",
                 fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save to file
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Decision rules scatter plot saved to: {save_path}")

    # Log to wandb
    if log_to_wandb and wandb.run:
        wandb.log({
            "visualization/decision_rules_scatter": wandb.Image(fig),
            "step": step
        })

    plt.close()

    return stats_dict


def plot_binned_decision_rules(data, batch_idx=None, n_bins=10,
                                save_path="binned_rules.png",
                                log_to_wandb=False, step=None):
    """
    Plot decision rules with binned averages for clearer trends.

    Groups agents by ability bins and shows mean decisions in each bin.

    Args:
        data: Dictionary with shape (n_transitions, batch_size, n_agents)
        batch_idx: Which batch to visualize. If None, uses all batches (default: None)
        n_bins: Number of bins for grouping
        save_path: Where to save the plot
        log_to_wandb: If True, log image to wandb
        step: Training step (for W&B logging)
    """
    # Extract data - flatten to get full distribution
    if batch_idx is None:
        # Use ALL batches
        ability = data['ability'].flatten()
        money = data['money_disposable'].flatten()
        consumption = data['consumption'].flatten()
        labor = data['labor'].flatten()
        savings = data['savings'].flatten()
        savings_ratio = data['savings_ratio'].flatten()
        batch_label = "All Batches"
    else:
        # Use only specified batch
        ability = data['ability'][:, batch_idx, :].flatten()
        money = data['money_disposable'][:, batch_idx, :].flatten()
        consumption = data['consumption'][:, batch_idx, :].flatten()
        labor = data['labor'][:, batch_idx, :].flatten()
        savings = data['savings'][:, batch_idx, :].flatten()
        savings_ratio = data['savings_ratio'][:, batch_idx, :].flatten()
        batch_label = f"Batch {batch_idx}"

    # Create flat data dict for compute_statistics
    flat_data = {
        'ability': ability,
        'money_disposable': money,
        'consumption': consumption,
        'labor': labor,
        'savings': savings,
        'savings_ratio': savings_ratio,
    }

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
    stats_dict = compute_statistics(flat_data)
    summary_text = (
        f"Binned Decision Rules Summary\n"
        f"{'='*35}\n\n"
        f"Data: {batch_label}\n"
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

    fig.suptitle(f"Binned Decision Rules ({batch_label}: {len(ability)} data points)",
                 fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save to file
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Binned decision rules plot saved to: {save_path}")

    # Log to wandb
    if log_to_wandb and wandb.run:
        wandb.log({
            "visualization/binned_decision_rules": wandb.Image(fig),
            "step": step
        })

    plt.close()


def plot_state_distributions(state, save_path="state_distributions.png",
                             log_to_wandb=False, step=None):
    """
    Plot distributions of all state variables in the loaded state.

    Shows the heterogeneity across agents before any transitions.

    Args:
        state: MainState instance
        save_path: Where to save the plot
        log_to_wandb: If True, log image to wandb
        step: Training step (for W&B logging)
    """
    # Extract state variables and flatten across batches
    ability = state.ability.cpu().numpy().flatten()
    money = state.moneydisposable.cpu().numpy().flatten()
    savings = state.savings.cpu().numpy().flatten()
    ret = state.ret.cpu().numpy().flatten()

    # Check if v_bar exists (heterogeneous ability means)
    has_v_bar = hasattr(state, 'v_bar') and state.v_bar is not None
    if has_v_bar:
        v_bar = state.v_bar.cpu().numpy().flatten()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Ability distribution
    axes[0, 0].hist(ability, bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(ability.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean={ability.mean():.3f}')
    axes[0, 0].axvline(np.median(ability), color='orange', linestyle='--', linewidth=2,
                       label=f'Median={np.median(ability):.3f}')
    axes[0, 0].set_xlabel("Ability", fontsize=11)
    axes[0, 0].set_ylabel("Count", fontsize=11)
    axes[0, 0].set_title(f"Ability Distribution\nStd={ability.std():.3f}", fontsize=12)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Money Disposable distribution
    axes[0, 1].hist(money, bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(money.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean={money.mean():.3f}')
    axes[0, 1].axvline(np.median(money), color='orange', linestyle='--', linewidth=2,
                       label=f'Median={np.median(money):.3f}')
    axes[0, 1].set_xlabel("Money Disposable", fontsize=11)
    axes[0, 1].set_ylabel("Count", fontsize=11)
    axes[0, 1].set_title(f"Money Distribution\nStd={money.std():.3f}", fontsize=12)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Savings distribution
    axes[0, 2].hist(savings, bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(savings.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean={savings.mean():.3f}')
    axes[0, 2].axvline(np.median(savings), color='orange', linestyle='--', linewidth=2,
                       label=f'Median={np.median(savings):.3f}')
    axes[0, 2].set_xlabel("Savings", fontsize=11)
    axes[0, 2].set_ylabel("Count", fontsize=11)
    axes[0, 2].set_title(f"Savings Distribution\nStd={savings.std():.3f}", fontsize=12)
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Return distribution
    axes[1, 0].hist(ret, bins=30, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(ret.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean={ret.mean():.3f}')
    axes[1, 0].set_xlabel("Return to Capital", fontsize=11)
    axes[1, 0].set_ylabel("Count", fontsize=11)
    axes[1, 0].set_title(f"Return Distribution\nStd={ret.std():.3f}", fontsize=12)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: v_bar distribution (if exists) or Money vs Ability scatter
    if has_v_bar:
        axes[1, 1].hist(v_bar, bins=30, color='cyan', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(v_bar.mean(), color='red', linestyle='--', linewidth=2,
                           label=f'Mean={v_bar.mean():.3f}')
        axes[1, 1].axvline(np.median(v_bar), color='orange', linestyle='--', linewidth=2,
                           label=f'Median={np.median(v_bar):.3f}')
        axes[1, 1].set_xlabel("v_bar (Long-run Mean Ability)", fontsize=11)
        axes[1, 1].set_ylabel("Count", fontsize=11)
        axes[1, 1].set_title(f"v_bar Distribution\nStd={v_bar.std():.3f}", fontsize=12)
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Scatter: Money vs Ability
        idx = np.random.choice(len(ability), min(500, len(ability)), replace=False)
        axes[1, 1].scatter(ability[idx], money[idx], alpha=0.5, s=20, c='cyan')
        corr = np.corrcoef(ability, money)[0, 1]
        axes[1, 1].set_xlabel("Ability", fontsize=11)
        axes[1, 1].set_ylabel("Money Disposable", fontsize=11)
        axes[1, 1].set_title(f"Money vs Ability\n(corr={corr:.3f})", fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Summary statistics
    axes[1, 2].axis('off')

    summary_text = (
        f"State Distribution Summary\n"
        f"{'='*35}\n\n"
        f"Number of agents: {len(ability)}\n\n"
        f"Ability:\n"
        f"  Mean: {ability.mean():.4f}\n"
        f"  Std: {ability.std():.4f}\n"
        f"  Min: {ability.min():.4f}\n"
        f"  Max: {ability.max():.4f}\n\n"
        f"Money Disposable:\n"
        f"  Mean: {money.mean():.4f}\n"
        f"  Std: {money.std():.4f}\n\n"
        f"Savings:\n"
        f"  Mean: {savings.mean():.4f}\n"
        f"  Std: {savings.std():.4f}\n\n"
        f"Correlation:\n"
        f"  Ability-Money: {np.corrcoef(ability, money)[0,1]:.4f}\n"
        f"  Ability-Savings: {np.corrcoef(ability, savings)[0,1]:.4f}"
    )

    if has_v_bar:
        summary_text += (
            f"\n\nv_bar (heterogeneous):\n"
            f"  Mean: {v_bar.mean():.4f}\n"
            f"  Std: {v_bar.std():.4f}\n"
            f"  Ability-v_bar corr: {np.corrcoef(ability, v_bar)[0,1]:.4f}"
        )

    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    fig.suptitle("Loaded State Distributions (Before Transitions)",
                 fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save to file
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ State distributions plot saved to: {save_path}")

    # Log to wandb
    if log_to_wandb and wandb.run:
        wandb.log({
            "visualization/state_distributions": wandb.Image(fig),
            "step": step
        })

    plt.close()
