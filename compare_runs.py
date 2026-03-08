"""
Compare decision rules and input-output plots across multiple training runs.

Usage:
    # Compare decision rule grid plots (default)
    python compare_runs.py checkpoints/RunA checkpoints/RunB

    # Specific step and plots
    python compare_runs.py checkpoints/RunA checkpoints/RunB --step 20000 --plots A1 B1

    # Show all quantile lines (default: q50 only)
    python compare_runs.py checkpoints/RunA checkpoints/RunB --color-level all

    # Compare input-output plots
    python compare_runs.py checkpoints/RunA checkpoints/RunB --type input_output

    # Grid layout instead of overlay
    python compare_runs.py checkpoints/RunA checkpoints/RunB --layout grid
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

from src.plot_data_io import (
    load_grid_data,
    load_input_output_data,
    load_run_meta,
    discover_runs,
)

# Quantile color scheme (matches visualization.py)
QUANTILE_COLORS = {
    "q5": "#1f77b4",
    "q25": "#2ca02c",
    "q50": "#ff7f0e",
    "q75": "#d62728",
    "q95": "#9467bd",
}

# Run colors for overlay mode
RUN_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
RUN_MARKERS = ["o", "^", "s", "D", "v", "P"]
Q_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
Q_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]
RUN_LINESTYLES = ["-", "--", ":", "-."]

# Plot ID to reference line mapping
PLOT_REF_LINES = {
    "A1": ["y=0"],
    "A1-1": [],
    "A1-1h": [],
    "B1": ["y=0", "y=x"],
    "H1": [],
    "H1-h": [],
}

# Variable label formatting
VAR_LABELS = {
    "m_t": r"$m_t$",
    "a_t": r"$a_t$",
    "a_tp1": r"$a_{t+1}$",
    "c_t": r"$c_t$",
    "l_t": r"$l_t$",
    "v_t": r"$v_t$",
    "zeta_t": r"$\zeta_t$",
    "mu_t": r"$\mu_t$",
    "mps": "MPS",
}


def format_label(var: str) -> str:
    return VAR_LABELS.get(var, var)


def find_latest_common_step(run_dirs: List[str]) -> Optional[int]:
    """Find the latest step available in all runs."""
    step_sets = []
    for d in run_dirs:
        grid_dir = os.path.join(d, "plot_data", "grid")
        if not os.path.isdir(grid_dir):
            continue
        steps = set()
        for f in os.listdir(grid_dir):
            if f.endswith(".npz"):
                parts = f[:-4].rsplit("_step_", 1)
                if len(parts) == 2:
                    steps.add(int(parts[1]))
        step_sets.append(steps)

    if not step_sets:
        return None
    common = step_sets[0]
    for s in step_sets[1:]:
        common = common & s
    return max(common) if common else None


def find_common_plot_ids(run_dirs: List[str], step: int) -> List[str]:
    """Find plot IDs available in all runs at the given step."""
    id_sets = []
    for d in run_dirs:
        grid_dir = os.path.join(d, "plot_data", "grid")
        ids = set()
        if os.path.isdir(grid_dir):
            for f in os.listdir(grid_dir):
                if f.endswith(f"_step_{step}.npz"):
                    plot_id = f[: -len(f"_step_{step}.npz")]
                    ids.add(plot_id)
        id_sets.append(ids)

    if not id_sets:
        return []
    common = id_sets[0]
    for s in id_sets[1:]:
        common = common & s
    return sorted(common)


def get_run_label(run_dir: str) -> str:
    """Get a short label for a run from its meta or directory name."""
    meta_path = os.path.join(run_dir, "plot_data", "meta.npz")
    if os.path.exists(meta_path):
        meta = load_run_meta(meta_path)
        return meta.get("exp_name", os.path.basename(run_dir))
    return os.path.basename(run_dir)


def add_reference_lines(ax, plot_id: str, x_values: np.ndarray):
    """Add reference lines based on plot type."""
    refs = PLOT_REF_LINES.get(plot_id, [])
    for ref in refs:
        if ref == "y=0":
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        elif ref == "y=x":
            ax.plot(x_values, x_values, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="y=x")


def plot_grid_overlay(
    datasets: List[Dict],
    plot_id: str,
    color_level: str = "q50",
    shared_xrange: bool = False,
) -> plt.Figure:
    """
    Overlay decision rule curves from multiple runs on the same axes.

    Args:
        datasets: List of loaded grid data dicts (from load_grid_data), each with "label" key
        plot_id: Plot identifier
        color_level: "q50" for median only, "all" for all quantile lines
        shared_xrange: If True, clip to intersection of x ranges
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, data in enumerate(datasets):
        x = data["x_values"]
        ls = RUN_LINESTYLES[i % len(RUN_LINESTYLES)]
        run_color = RUN_COLORS[i % len(RUN_COLORS)]
        label = data["label"]

        if color_level == "all":
            for level in data["color_levels"]:
                y = data["y_values"].get(level)
                if y is None:
                    continue
                q_color = QUANTILE_COLORS.get(level, run_color)
                ax.plot(x, y, linestyle=ls, color=q_color, linewidth=1.5, alpha=0.7,
                        label=f"{label} ({level})")
        else:
            # Plot single color level
            y = data["y_values"].get(color_level)
            if y is None:
                # Fallback: try "mean" key
                y = data["y_values"].get("mean")
            if y is None:
                print(f"Warning: color level '{color_level}' not found in {label}, skipping")
                continue
            ax.plot(x, y, linestyle=ls, color=run_color, linewidth=2, label=label)

    # Reference lines using the last dataset's x range
    if datasets:
        all_x = np.concatenate([d["x_values"] for d in datasets])
        add_reference_lines(ax, plot_id, np.linspace(all_x.min(), all_x.max(), 100))

    x_var = datasets[0]["x_var"] if datasets else ""
    y_var = datasets[0]["y_var"] if datasets else ""
    ax.set_xlabel(format_label(x_var), fontsize=12)
    ax.set_ylabel(format_label(y_var), fontsize=12)
    ax.set_title(f"Comparison: {plot_id} ({x_var} → {y_var})", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    if shared_xrange and len(datasets) > 1:
        x_min = max(d["x_values"].min() for d in datasets)
        x_max = min(d["x_values"].max() for d in datasets)
        ax.set_xlim(x_min, x_max)

    plt.tight_layout()
    return fig


def plot_grid_layout(
    datasets: List[Dict],
    plot_id: str,
    color_level: str = "q50",
) -> plt.Figure:
    """
    Grid layout: one subplot per run, shared y-axis.
    """
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    for i, (ax, data) in enumerate(zip(axes, datasets)):
        x = data["x_values"]

        if color_level == "all":
            for level in data["color_levels"]:
                y = data["y_values"].get(level)
                if y is None:
                    continue
                q_color = QUANTILE_COLORS.get(level, "#333333")
                ax.plot(x, y, color=q_color, linewidth=1.5, label=level)
        else:
            y = data["y_values"].get(color_level, data["y_values"].get("mean"))
            if y is not None:
                ax.plot(x, y, color=RUN_COLORS[i % len(RUN_COLORS)], linewidth=2)

        add_reference_lines(ax, plot_id, x)
        ax.set_title(data["label"], fontsize=11)
        ax.set_xlabel(format_label(data["x_var"]), fontsize=10)
        if i == 0:
            ax.set_ylabel(format_label(data["y_var"]), fontsize=10)
        ax.grid(True, alpha=0.3)
        if color_level == "all":
            ax.legend(fontsize=8)

    fig.suptitle(f"Comparison: {plot_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_io_comparison(
    datasets: List[Dict],
) -> plt.Figure:
    """
    Compare input-output pairwise data across runs.
    Overlay runs on the same subplots with quintile coloring by agent type.
    Grid: rows = (money, ability) × columns = (zeta, mu, labor).
    Optional utility row and loss row below.
    Each run uses a different marker shape; colors indicate quintile of the OTHER input.
    """
    inputs = ["own_money", "own_ability"]
    input_labels = ["Own Money (normalized)", "Own Ability (normalized)"]
    outputs = ["zeta", "mu", "labor"]
    output_labels = [r"$\zeta_t$ (savings ratio)", r"$\mu_t$ (multiplier)", r"$l_t$ (labor)"]

    has_utility = any(d.get("utility") is not None for d in datasets)
    has_losses = any(d.get("per_agent_losses") for d in datasets)
    n_rows = 2 + (1 if has_utility else 0) + (1 if has_losses else 0)

    height_ratios = [3, 3]
    if has_utility:
        height_ratios.append(3)
    if has_losses:
        height_ratios.append(2)

    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 5 * n_rows),
                             squeeze=False,
                             gridspec_kw={"height_ratios": height_ratios})

    for row, (inp, inp_label) in enumerate(zip(inputs, input_labels)):
        # Color by the OTHER input's quintile (same as original plot)
        other_inp = "own_ability" if row == 0 else "own_money"
        other_label = "ability" if row == 0 else "money"

        for col, (out_key, out_label) in enumerate(zip(outputs, output_labels)):
            ax = axes[row, col]

            for i, data in enumerate(datasets):
                x = data[inp]
                y = data[out_key]
                marker = RUN_MARKERS[i % len(RUN_MARKERS)]

                # Compute quintile bins for the OTHER input
                other_data = data[other_inp]
                edges = np.percentile(other_data, [0, 20, 40, 60, 80, 100])
                bins = np.digitize(other_data, edges[1:-1])

                # First run more opaque, subsequent runs more transparent
                alpha = 0.8 if i == 0 else 0.2

                for qi in range(5):
                    mask = bins == qi
                    if mask.sum() == 0:
                        continue
                    # Legend: show run name + quintile only in first subplot
                    label = None
                    if row == 0 and col == 0:
                        label = f"{data['exp_name']} {other_label} {Q_LABELS[qi]}"
                    ax.scatter(
                        x[mask], y[mask],
                        c=Q_COLORS[qi], marker=marker,
                        alpha=alpha, s=2, rasterized=True,
                        label=label,
                    )

            ax.set_xlabel(inp_label, fontsize=9)
            ax.set_ylabel(out_label, fontsize=9)
            ax.grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=6, markerscale=5, loc="best", ncol=len(datasets))

    # ---- Utility row: own_money vs utility, own_ability vs utility ----
    if has_utility:
        utility_row = 2
        utility_inputs = [
            ("own_money", "Own Money (normalized)", "own_ability", "ability"),
            ("own_ability", "Own Ability (normalized)", "own_money", "money"),
        ]
        for col, (inp_key, inp_label, other_key, other_label) in enumerate(utility_inputs):
            ax = axes[utility_row, col]
            for i, data in enumerate(datasets):
                if data.get("utility") is None:
                    continue
                x = data[inp_key]
                y = data["utility"]
                marker = RUN_MARKERS[i % len(RUN_MARKERS)]
                other_data = data[other_key]
                edges = np.percentile(other_data, [0, 20, 40, 60, 80, 100])
                bins = np.digitize(other_data, edges[1:-1])
                alpha = 0.8 if i == 0 else 0.2

                for qi in range(5):
                    mask = bins == qi
                    if mask.sum() == 0:
                        continue
                    label = None
                    if col == 0:
                        label = f"{data['exp_name']} {other_label} {Q_LABELS[qi]}"
                    ax.scatter(
                        x[mask], y[mask],
                        c=Q_COLORS[qi], marker=marker,
                        alpha=alpha, s=2, rasterized=True,
                        label=label,
                    )
            ax.set_xlabel(inp_label, fontsize=9)
            ax.set_ylabel(r"$u(c,l)$ (flow utility)", fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[utility_row, 0].legend(fontsize=6, markerscale=5, loc="best", ncol=len(datasets))
        axes[utility_row, 2].set_visible(False)

    # ---- Loss row: Mean losses per ability quintile ----
    if has_losses:
        loss_row = 2 + (1 if has_utility else 0)
        # Collect common loss names across datasets
        all_loss_names = set()
        for data in datasets:
            if data.get("per_agent_losses"):
                all_loss_names.update(data["per_agent_losses"].keys())
        loss_names = sorted(all_loss_names)[:3]

        bar_width = 0.8 / max(len(datasets), 1)
        for col, loss_name in enumerate(loss_names):
            ax = axes[loss_row, col]
            for i, data in enumerate(datasets):
                losses = data.get("per_agent_losses", {})
                loss_vals = losses.get(loss_name)
                if loss_vals is None:
                    continue
                ability_data = data["own_ability"]
                edges = np.percentile(ability_data, [0, 20, 40, 60, 80, 100])
                bins = np.digitize(ability_data, edges[1:-1])

                means_per_q = []
                for qi in range(5):
                    mask = bins == qi
                    if mask.sum() > 0:
                        means_per_q.append(np.mean(np.abs(loss_vals[mask])))
                    else:
                        means_per_q.append(0.0)

                x_pos = np.arange(5) + i * bar_width
                ax.bar(x_pos, means_per_q, width=bar_width, alpha=0.7,
                       label=data["exp_name"], color=RUN_COLORS[i % len(RUN_COLORS)])

            ax.set_xticks(np.arange(5) + bar_width * (len(datasets) - 1) / 2)
            ax.set_xticklabels(Q_LABELS)
            ax.set_ylabel(f"Mean |{loss_name}|", fontsize=9)
            ax.set_xlabel("Ability quintile", fontsize=9)
            ax.set_title(f"{loss_name} by agent type", fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")
            ax.legend(fontsize=7)

        for col in range(len(loss_names), 3):
            axes[loss_row, col].set_visible(False)

    fig.suptitle("Input-Output Comparison (overlaid)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Compare decision rules across training runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("run_dirs", nargs="+", help="Checkpoint directories to compare")
    parser.add_argument("--step", type=int, default=None, help="Training step (default: latest common)")
    parser.add_argument("--plots", nargs="+", default=None, help="Plot IDs to compare (default: all common)")
    parser.add_argument("--type", choices=["grid", "input_output"], default="grid",
                        help="Data type to compare (default: grid)")
    parser.add_argument("--layout", choices=["overlay", "grid"], default=None,
                        help="Plot layout (default: overlay for <=3 runs, grid for more)")
    parser.add_argument("--color-level", default="q50",
                        help="Quantile level to show: q5/q25/q50/q75/q95/all (default: q50)")
    parser.add_argument("--shared-xrange", action="store_true",
                        help="Clip x-axis to intersection of all runs' ranges")
    parser.add_argument("--output-dir", default="comparison_plots",
                        help="Output directory (default: comparison_plots)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Validate run directories
    for d in args.run_dirs:
        if not os.path.isdir(d):
            print(f"Error: {d} is not a directory")
            sys.exit(1)

    if args.type == "input_output":
        # Input-output comparison
        step = args.step
        if step is None:
            # Find latest common step from input_output dirs
            step_sets = []
            for d in args.run_dirs:
                io_dir = os.path.join(d, "plot_data", "input_output")
                steps = set()
                if os.path.isdir(io_dir):
                    for f in os.listdir(io_dir):
                        if f.startswith("pairwise_step_") and f.endswith(".npz"):
                            steps.add(int(f[len("pairwise_step_"):-4]))
                step_sets.append(steps)
            common = step_sets[0] if step_sets else set()
            for s in step_sets[1:]:
                common = common & s
            step = max(common) if common else None

        if step is None:
            print("Error: No common input-output steps found across runs")
            sys.exit(1)

        print(f"Comparing input-output data at step {step}")
        datasets = []
        for d in args.run_dirs:
            path = os.path.join(d, "plot_data", "input_output", f"pairwise_step_{step}.npz")
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping")
                continue
            data = load_input_output_data(path)
            datasets.append(data)

        if len(datasets) < 2:
            print("Error: Need at least 2 runs with data to compare")
            sys.exit(1)

        fig = plot_io_comparison(datasets)
        out_path = os.path.join(args.output_dir, f"compare_input_output_step_{step}.pdf")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
        return

    # Grid (decision rule) comparison
    step = args.step or find_latest_common_step(args.run_dirs)
    if step is None:
        print("Error: No common grid steps found across runs")
        sys.exit(1)

    plot_ids = args.plots or find_common_plot_ids(args.run_dirs, step)
    if not plot_ids:
        print(f"Error: No common plot IDs found at step {step}")
        sys.exit(1)

    layout = args.layout
    if layout is None:
        layout = "overlay" if len(args.run_dirs) <= 3 else "grid"

    print(f"Comparing {len(args.run_dirs)} runs at step {step}")
    print(f"Plot IDs: {plot_ids}")
    print(f"Layout: {layout}, Color level: {args.color_level}")

    for plot_id in plot_ids:
        datasets = []
        for d in args.run_dirs:
            path = os.path.join(d, "plot_data", "grid", f"{plot_id}_step_{step}.npz")
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping")
                continue
            data = load_grid_data(path)
            data["label"] = get_run_label(d)
            datasets.append(data)

        if len(datasets) < 2:
            print(f"Skipping {plot_id}: fewer than 2 runs have data")
            continue

        if layout == "overlay":
            fig = plot_grid_overlay(datasets, plot_id,
                                     color_level=args.color_level,
                                     shared_xrange=args.shared_xrange)
        else:
            fig = plot_grid_layout(datasets, plot_id,
                                    color_level=args.color_level)

        out_path = os.path.join(args.output_dir, f"compare_{plot_id}_step_{step}.pdf")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    print(f"\nAll comparison plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
