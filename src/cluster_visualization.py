"""
src/cluster_visualization.py

Visualization functions for GMM-based regime tracking.
Companion to src/cluster_analysis.py.

All functions accept save_path and log_to_wandb options.
Color scheme: blue = regime_low, orange/red = regime_high, gray = NA.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional

REGIME_COLORS = {
    'regime_low': '#2196F3',   # blue
    'regime_high': '#F44336',  # red
    'NA': '#9E9E9E',           # gray
}
REGIME_LABELS = {
    'regime_low': 'Low regime',
    'regime_high': 'High regime',
    'NA': 'Outside sample',
}


def _save_and_log(fig: plt.Figure, save_path: Optional[str], log_to_wandb: bool, step: Optional[int], key: str) -> None:
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=120)
    if log_to_wandb:
        try:
            import wandb
            log_dict = {key: wandb.Image(fig)}
            if step is not None:
                wandb.log(log_dict, step=step)
            else:
                wandb.log(log_dict)
        except Exception:
            pass


def plot_regime_scatter(
    enriched_df: pd.DataFrame,
    save_path: Optional[str] = None,
    log_to_wandb: bool = False,
    step: Optional[int] = None,
    alpha: float = 0.4,
    s: float = 8.0,
) -> plt.Figure:
    """
    Step 10: Three scatter panels (saving_ratio, mu, labor) vs ability,
    colored by regime. Matches the original scatter plots from the user.

    Args:
        enriched_df: Output from RegimeAnalyzer.fit().
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    y_vars = [
        ('saving_ratio', r'$\zeta_t$ (savings ratio)'),
        ('mu', r'$\mu_t$ (multiplier)'),
        ('labor', r'$l_t$ (labor)'),
    ]

    for ax, (y_col, y_label) in zip(axes, y_vars):
        if y_col not in enriched_df.columns:
            ax.set_visible(False)
            continue

        for regime, color in REGIME_COLORS.items():
            if regime == 'NA':
                sub = enriched_df[enriched_df['regime'].isna()]
                label = REGIME_LABELS['NA']
            else:
                sub = enriched_df[enriched_df['regime'] == regime]
                label = REGIME_LABELS[regime]

            if len(sub) == 0:
                continue
            ax.scatter(sub['ability'], sub[y_col],
                       c=color, alpha=alpha, s=s, label=label, linewidths=0)

        ax.set_xlabel(r'$v_t$ (ability)')
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, markerscale=2)

    title = 'Regime-colored decision rules'
    if step is not None:
        title += f' (step {step})'
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    _save_and_log(fig, save_path, log_to_wandb, step, 'regime/scatter')
    return fig


def plot_regime_paths(
    enriched_df: pd.DataFrame,
    agent_ids: Optional[List[int]] = None,
    n_agents: int = 5,
    save_path: Optional[str] = None,
    log_to_wandb: bool = False,
    step: Optional[int] = None,
) -> plt.Figure:
    """
    Step 14: Time series of regime label + key variables for representative agents.

    Args:
        enriched_df: Output from compute_transitions() (has regime_prev, switch_flag).
        agent_ids: Specific agents to plot. If None, picks n_agents with most switches.
        n_agents: Number of agents to plot (used if agent_ids is None).
    """
    df = enriched_df.copy()

    if agent_ids is None:
        if 'switch_flag' in df.columns:
            switches = df.groupby('agent_id')['switch_flag'].sum()
            agent_ids = switches.nlargest(n_agents).index.tolist()
        else:
            agent_ids = sorted(df['agent_id'].unique())[:n_agents]

    regime_to_num = {'regime_low': 0, 'regime_high': 1}
    overlay_vars = [v for v in ['ability', 'saving_ratio', 'mu', 'labor'] if v in df.columns]
    n_rows = 1 + len(overlay_vars)

    fig, axes = plt.subplots(n_rows, len(agent_ids),
                             figsize=(4 * len(agent_ids), 2.5 * n_rows),
                             sharex='col')
    if len(agent_ids) == 1:
        axes = axes.reshape(-1, 1)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for col_idx, agent_id in enumerate(agent_ids):
        agent_df = df[df['agent_id'] == agent_id].sort_values('time')
        times = agent_df['time'].values

        # Row 0: regime as step plot
        ax = axes[0, col_idx]
        regimes_num = agent_df['regime'].map(regime_to_num).values.astype(float)
        ax.step(times, regimes_num, where='post', color='black', linewidth=1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Low', 'High'], fontsize=8)
        ax.set_ylabel('Regime', fontsize=8)
        ax.set_title(f'Agent {agent_id}', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Color background by regime
        for i in range(len(times)):
            r = agent_df['regime'].iloc[i]
            color = REGIME_COLORS.get(r, REGIME_COLORS['NA'])
            t_start = times[i]
            t_end = times[i + 1] if i + 1 < len(times) else times[i] + 1
            ax.axvspan(t_start, t_end, alpha=0.15, color=color, linewidth=0)

        # Overlay rows
        for row_idx, var in enumerate(overlay_vars, start=1):
            ax2 = axes[row_idx, col_idx]
            ax2.plot(times, agent_df[var].values, color='black', linewidth=1)
            ax2.set_ylabel(var, fontsize=8)
            ax2.grid(True, alpha=0.3)
            # Color background
            for i in range(len(times)):
                r = agent_df['regime'].iloc[i]
                color = REGIME_COLORS.get(r, REGIME_COLORS['NA'])
                t_start = times[i]
                t_end = times[i + 1] if i + 1 < len(times) else times[i] + 1
                ax2.axvspan(t_start, t_end, alpha=0.10, color=color, linewidth=0)

        axes[-1, col_idx].set_xlabel('Time (step)')

    title = 'Representative agent regime paths'
    if step is not None:
        title += f' (step {step})'
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    _save_and_log(fig, save_path, log_to_wandb, step, 'regime/paths')
    return fig


def plot_transition_matrix(
    transition_matrix: np.ndarray,
    transition_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    log_to_wandb: bool = False,
    step: Optional[int] = None,
) -> plt.Figure:
    """
    Step 13: Heatmap of 2x2 transition probability matrix.

    Args:
        transition_matrix: (2,2) array, rows=from, cols=to.
        transition_labels: Labels for each row/col. Default ['regime_low','regime_high'].
    """
    if transition_labels is None:
        transition_labels = ['regime_low', 'regime_high']

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(transition_matrix, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax, label='Probability')

    ax.set_xticks(range(len(transition_labels)))
    ax.set_yticks(range(len(transition_labels)))
    short = [l.replace('regime_', '') for l in transition_labels]
    ax.set_xticklabels([f'→ {s}' for s in short])
    ax.set_yticklabels([f'{s} →' for s in short])
    ax.set_xlabel('To regime')
    ax.set_ylabel('From regime')

    for i in range(len(transition_labels)):
        for j in range(len(transition_labels)):
            val = transition_matrix[i, j]
            color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=13, color=color, fontweight='bold')

    title = 'Regime transition matrix'
    if step is not None:
        title += f' (step {step})'
    ax.set_title(title)
    fig.tight_layout()

    _save_and_log(fig, save_path, log_to_wandb, step, 'regime/transition_matrix')
    return fig


def plot_switch_distribution(
    switch_counts: pd.Series,
    save_path: Optional[str] = None,
    log_to_wandb: bool = False,
    step: Optional[int] = None,
) -> plt.Figure:
    """
    Step 12: Histogram of regime switch counts across agents.

    Args:
        switch_counts: pd.Series indexed by agent_id, values = total_switches.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    max_val = int(switch_counts.max()) if len(switch_counts) > 0 else 1
    bins = np.arange(-0.5, max_val + 1.5, 1)
    ax.hist(switch_counts.values, bins=bins, color='#5C6BC0', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Number of regime switches per agent')
    ax.set_ylabel('Count')
    ax.axvline(switch_counts.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean = {switch_counts.mean():.2f}')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    title = 'Regime switch distribution'
    if step is not None:
        title += f' (step {step})'
    ax.set_title(title)
    fig.tight_layout()

    _save_and_log(fig, save_path, log_to_wandb, step, 'regime/switch_distribution')
    return fig


def plot_regime_summary(
    summary_df: pd.DataFrame,
    save_path: Optional[str] = None,
    log_to_wandb: bool = False,
    step: Optional[int] = None,
) -> plt.Figure:
    """
    Step 15: Side-by-side bar plots comparing mean ± std by regime for key variables.

    Args:
        summary_df: Output from RegimeAnalyzer.compute_summary_stats().
    """
    plot_vars = [v for v in ['saving_ratio', 'mu', 'labor', 'ability',
                              'consumption', 'savings', 'money_disposable']
                 if v in summary_df['variable'].values]

    n_vars = len(plot_vars)
    if n_vars == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    ncols = min(4, n_vars)
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).ravel()

    for ax_idx, var in enumerate(plot_vars):
        ax = axes[ax_idx]
        sub = summary_df[summary_df['variable'] == var]
        regimes = ['regime_low', 'regime_high']
        means = []
        stds = []
        colors = []
        for r in regimes:
            row = sub[sub['regime'] == r]
            if len(row) == 0:
                means.append(0)
                stds.append(0)
            else:
                means.append(float(row['mean'].iloc[0]))
                stds.append(float(row['std'].iloc[0]))
            colors.append(REGIME_COLORS[r])

        x = np.arange(len(regimes))
        bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor='white',
                      linewidth=0.5, capsize=5, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(['Low', 'High'], fontsize=9)
        ax.set_title(var, fontsize=10)
        ax.set_ylabel('Mean ± Std', fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

    for ax in axes[n_vars:]:
        ax.set_visible(False)

    title = 'Regime summary statistics'
    if step is not None:
        title += f' (step {step})'
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    _save_and_log(fig, save_path, log_to_wandb, step, 'regime/summary')
    return fig


def plot_robustness_bic(
    robustness_df: pd.DataFrame,
    save_path: Optional[str] = None,
    log_to_wandb: bool = False,
    step: Optional[int] = None,
) -> plt.Figure:
    """
    Step 16: BIC and AIC comparison across K values and filter cutoffs.

    Args:
        robustness_df: Output from RegimeAnalyzer.robustness_check().
    """
    if len(robustness_df) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    k_values = sorted(robustness_df['K'].unique())
    configs = robustness_df.drop_duplicates(subset=['ability_cutoff_quantile', 'labor_upper'])
    config_labels = [
        f"q={row['ability_cutoff_quantile']:.1f}, l<{row['labor_upper']:.2f}"
        for _, row in configs.iterrows()
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, len(config_labels)))

    for metric, ax in zip(['BIC', 'AIC'], axes):
        for i, (_, cfg_row) in enumerate(configs.iterrows()):
            sub = robustness_df[
                (robustness_df['ability_cutoff_quantile'] == cfg_row['ability_cutoff_quantile']) &
                (robustness_df['labor_upper'] == cfg_row['labor_upper'])
            ].sort_values('K')
            if len(sub) == 0:
                continue
            ax.plot(sub['K'], sub[metric], marker='o', color=colors[i],
                    label=config_labels[i], linewidth=1.5)

        ax.set_xlabel('Number of components K')
        ax.set_ylabel(metric)
        ax.set_xticks(k_values)
        ax.set_title(f'{metric} by K and filter settings')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    title = 'Robustness check: BIC/AIC'
    if step is not None:
        title += f' (step {step})'
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    _save_and_log(fig, save_path, log_to_wandb, step, 'regime/robustness_bic')
    return fig
