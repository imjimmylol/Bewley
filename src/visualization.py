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
from typing import Optional, Dict, List


# =============================================================================
# Color palette for quantile-based conditioning (5 levels)
# =============================================================================
QUANTILE_COLORS = {
    "q5": "#1f77b4",    # blue
    "q25": "#2ca02c",   # green
    "q50": "#ff7f0e",   # orange
    "q75": "#d62728",   # red
    "q95": "#9467bd",   # purple
    # For discrete s_t
    "Normal": "#1f77b4",
    "Superstar": "#d62728",
    # Fallback
    "mean": "#333333",
}


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

    # Compute 95th percentile for axis clipping (to handle outliers)
    ability_95 = np.percentile(ability, 95)
    money_95 = np.percentile(money, 95)
    consumption_95 = np.percentile(consumption, 95)
    labor_95 = np.percentile(labor, 95)
    savings_95 = np.percentile(savings, 95)
    savings_ratio_95 = np.percentile(savings_ratio, 95)

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
    x_line = np.linspace(ability.min(), min(ability.max(), ability_95), 100)
    axes[0, 0].plot(x_line, p(x_line), 'r-', linewidth=2,
                    label=f"slope={z[0]:.3f}")
    axes[0, 0].set_xlim(0, ability_95)
    axes[0, 0].set_ylim(0, consumption_95)
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
    axes[0, 1].set_xlim(0, ability_95)
    axes[0, 1].set_ylim(0, labor_95)
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
    axes[0, 2].set_xlim(0, ability_95)
    axes[0, 2].set_ylim(0, savings_95)
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
    axes[1, 0].set_xlim(0, ability_95)
    axes[1, 0].set_ylim(0, 1)  # Savings ratio is naturally bounded [0, 1]
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
    axes[1, 1].set_xlim(0, ability_95)
    axes[1, 1].set_ylim(0, money_95)
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
    x_line_money = np.linspace(money.min(), min(money.max(), money_95), 100)
    axes[1, 2].plot(x_line_money, p(x_line_money), 'darkorange', linewidth=2,
                    label=f"slope={z[0]:.3f}")
    axes[1, 2].set_xlim(0, money_95)
    axes[1, 2].set_ylim(0, consumption_95)
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

    # Compute 95th percentile for axis clipping
    ability_95 = np.percentile(ability, 95)

    # Create ability bins (using 95th percentile to avoid outliers)
    ability_bins = np.linspace(ability.min(), ability_95, n_bins + 1)
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


# =============================================================================
# New Decision Rule Visualization (Synthetic Grid Evaluation)
# Following CLAUDE.local.md methodology
# =============================================================================

def _format_axis_label(var_name: str, use_latex: bool = True) -> str:
    """Convert variable name to formatted axis label."""
    labels = {
        "m_t": r"$m_t$ (Money Disposable)" if use_latex else "m_t (Money Disposable)",
        "a_t": r"$a_t$ (Assets)" if use_latex else "a_t (Assets)",
        "v_t": r"$v_t$ (Ability)" if use_latex else "v_t (Ability)",
        "s_t": r"$s_t$ (Regime)" if use_latex else "s_t (Regime)",
        "y_t": r"$y_t$ (Income)" if use_latex else "y_t (Income)",
        "c_t": r"$c_t$ (Consumption)" if use_latex else "c_t (Consumption)",
        "a_tp1": r"$a_{t+1}$ (Next Assets)" if use_latex else "a_{t+1} (Next Assets)",
        "zeta_t": r"$\zeta_t$ (Savings Ratio)" if use_latex else "zeta_t (Savings Ratio)",
        "da_t": r"$\Delta a_t$ (Net Saving)" if use_latex else "da_t (Net Saving)",
        "l_t": r"$l_t$ (Labor)" if use_latex else "l_t (Labor)",
        "mu_t": r"$\mu_t$ (Multiplier)" if use_latex else "mu_t (Multiplier)",
        "I_bind": r"$\mathbb{1}[a_{t+1}=0]$ (Binding)" if use_latex else "I[a_{t+1}=0] (Binding)",
    }
    return labels.get(var_name, var_name)


def _add_reference_lines(ax, ref_lines: List[str], x_values: np.ndarray) -> None:
    """Add reference lines to axis."""
    for ref in ref_lines:
        if ref == "y=0":
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7, label='y=0')
        elif ref == "y=x":
            ax.plot(x_values, x_values, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='y=x')


def _add_log1p_validity_line(ax, threshold: float = 1.0) -> None:
    """
    Add a vertical line marking where log1p(x) ≈ x approximation breaks down.

    For small x, log(1+x) ≈ x. Beyond the threshold, the transformation
    compresses the scale significantly, affecting slope interpretation.

    Args:
        ax: matplotlib axis
        threshold: x value where log1p(x) starts to differ meaningfully from x.
                   Default is 1.0, where log(1+1) = 0.693 vs linear 1.0 (~30% diff).
    """
    log1p_threshold = np.log1p(threshold)
    ax.axvline(x=log1p_threshold, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)

    # Add annotation
    y_min, y_max = ax.get_ylim()
    y_pos = y_min + 0.95 * (y_max - y_min)
    ax.annotate(
        f'x={threshold:.0f}',
        xy=(log1p_threshold, y_pos),
        xytext=(log1p_threshold + 0.1, y_pos),
        fontsize=8,
        color='gray',
        verticalalignment='top'
    )


def plot_decision_rule(
    evaluator,  # PolicyEvaluator
    x_var: str,
    y_var: str,
    color_var: Optional[str] = None,
    fixed_vars: Optional[Dict] = None,
    use_log1p_x: bool = True,
    ref_lines: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    log_to_wandb: bool = False,
    step: Optional[int] = None,
    n_points: int = 100,
    debug: bool = False
) -> plt.Figure:
    """
    Plot a decision rule using synthetic grid evaluation.

    This is the master plotting function that uses PolicyEvaluator to generate
    synthetic grid data with controlled conditioning variables.

    Args:
        evaluator: PolicyEvaluator instance
        x_var: Variable for x-axis (e.g., "m_t", "a_t")
        y_var: Variable for y-axis (e.g., "c_t", "a_tp1")
        color_var: Optional variable for color conditioning (5 quantile levels)
        fixed_vars: Optional dict of {var_name: value} for fixed variables
        use_log1p_x: If True, apply log1p transform to x-axis
        ref_lines: List of reference lines to add (e.g., ["y=0", "y=x"])
        title: Optional custom title
        save_path: Where to save the plot
        log_to_wandb: If True, log to wandb
        step: Training step for wandb
        n_points: Number of grid points

    Returns:
        matplotlib Figure
    """
    if ref_lines is None:
        ref_lines = []

    # Evaluate on grid
    results = evaluator.evaluate_on_grid(
        x_var=x_var,
        y_var=y_var,
        color_var=color_var,
        fixed_vars=fixed_vars,
        n_points=n_points,
        debug=debug
    )

    x_values = results["x_values"]
    y_values = results["y_values"]
    color_levels = results["color_levels"]
    color_values = results["color_values"]
    fixed_values = results["fixed_values"]

    # Transform x if needed
    if use_log1p_x:
        x_plot = np.log1p(x_values)
        x_label = f"log(1 + {_format_axis_label(x_var)})"
    else:
        x_plot = x_values
        x_label = _format_axis_label(x_var)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot each color level
    for c_label, c_val in zip(color_levels, color_values):
        y_arr = y_values[c_label]
        color = QUANTILE_COLORS.get(c_label, "#333333")

        # Format legend label
        if color_var and c_val is not None:
            if color_var == "s_t":
                legend_label = c_label
            else:
                legend_label = f"{color_var}={c_val:.2f} ({c_label})"
        else:
            legend_label = "Policy"

        ax.plot(x_plot, y_arr, color=color, linewidth=2, label=legend_label)

    # Add reference lines
    _add_reference_lines(ax, ref_lines, x_plot if not use_log1p_x else x_values)

    # Add log1p validity line when using log transform
    if use_log1p_x:
        _add_log1p_validity_line(ax, threshold=1.0)

    # Labels and title
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(_format_axis_label(y_var), fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        title_str = f"Decision Rule: {x_var} → {y_var}"
        if color_var:
            title_str += f" (color by {color_var})"
        ax.set_title(title_str, fontsize=14, fontweight='bold')

    # Add fixed variables info
    fixed_str = ", ".join([f"{k}={v:.2f}" for k, v in fixed_values.items()])
    ax.text(0.02, 0.98, f"Fixed: {fixed_str}", transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Decision rule plot saved to: {save_path}")

    # Log to wandb
    if log_to_wandb and wandb.run:
        wandb.log({
            f"decision_rules/{x_var}_to_{y_var}": wandb.Image(fig),
            "step": step
        })

    return fig


def plot_A1_resources_to_assets(
    evaluator,
    color_var: str = "v_t",
    save_path: Optional[str] = None,
    log_to_wandb: bool = False,
    step: Optional[int] = None,
    debug: bool = False
) -> plt.Figure:
    """
    Plot A1: m_t → a_{t+1} (Resources to Next Assets)

    Shows how agents allocate resources to savings across different ability levels.

    Args:
        evaluator: PolicyEvaluator instance
        color_var: Variable for color conditioning (default: "v_t")
        save_path: Where to save
        log_to_wandb: Log to wandb
        step: Training step
        debug: If True, print debug information

    Returns:
        matplotlib Figure
    """
    fig = plot_decision_rule(
        evaluator=evaluator,
        x_var="m_t",
        y_var="a_tp1",
        color_var=color_var,
        use_log1p_x=False,
        ref_lines=["y=0"],
        title=r"A1: $m_t \rightarrow a_{t+1}$ (Resources to Next Assets)",
        save_path=save_path,
        log_to_wandb=log_to_wandb,
        step=step,
        debug=debug
    )
    return fig


def plot_B1_assets_to_assets(
    evaluator,
    color_var: str = "v_t",
    save_path: Optional[str] = None,
    log_to_wandb: bool = False,
    step: Optional[int] = None
) -> plt.Figure:
    """
    Plot B1: a_t → a_{t+1} (Assets Today to Assets Tomorrow)

    Shows history dependence - how current assets predict future assets.
    The 45-degree line (y=x) shows where assets would remain unchanged.

    Args:
        evaluator: PolicyEvaluator instance
        color_var: Variable for color conditioning (default: "v_t")
        save_path: Where to save
        log_to_wandb: Log to wandb
        step: Training step

    Returns:
        matplotlib Figure
    """
    fig = plot_decision_rule(
        evaluator=evaluator,
        x_var="a_t",
        y_var="a_tp1",
        color_var=color_var,
        use_log1p_x=False,
        ref_lines=["y=0", "y=x"],
        title=r"B1: $a_t \rightarrow a_{t+1}$ (Assets Today to Tomorrow)",
        save_path=save_path,
        log_to_wandb=log_to_wandb,
        step=step
    )
    return fig


def plot_all_decision_rules(
    evaluator,
    save_dir: str,
    log_to_wandb: bool = False,
    step: Optional[int] = None,
    plots: Optional[List[str]] = None
) -> Dict[str, plt.Figure]:
    """
    Generate all (or selected) decision rule plots.

    Args:
        evaluator: PolicyEvaluator instance
        save_dir: Directory to save plots
        log_to_wandb: Log to wandb
        step: Training step
        plots: List of plot IDs to generate (default: ["A1", "B1"])

    Returns:
        Dict mapping plot ID to Figure
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    if plots is None:
        plots = ["A1", "B1"]

    figures = {}

    plot_funcs = {
        "A1": plot_A1_resources_to_assets,
        "B1": plot_B1_assets_to_assets,
    }

    for plot_id in plots:
        if plot_id in plot_funcs:
            save_path = os.path.join(save_dir, f"fig_{plot_id}_step_{step}.png") if step else os.path.join(save_dir, f"fig_{plot_id}.png")
            fig = plot_funcs[plot_id](
                evaluator=evaluator,
                save_path=save_path,
                log_to_wandb=log_to_wandb,
                step=step
            )
            figures[plot_id] = fig
            plt.close(fig)

    return figures
