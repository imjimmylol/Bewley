"""
run_cluster_analysis.py

Standalone post-hoc analysis script for GMM-based regime tracking.

Usage examples:
    # Load a saved panel NPZ directly
    python run_cluster_analysis.py --panel_npz checkpoints/my_run/plot_data/panel/panel_step_10000.npz

    # Auto-discover latest panel in a checkpoint directory (requires panel data from new training runs)
    python run_cluster_analysis.py --checkpoint_dir checkpoints/my_run

    # Simulate from an existing checkpoint (no re-training needed)
    python run_cluster_analysis.py \\
        --checkpoint_dir checkpoints/my_run \\
        --config config/my_config.yaml \\
        --sim_steps 500

    # Customize clustering parameters
    python run_cluster_analysis.py \\
        --checkpoint_dir checkpoints/my_run \\
        --config config/my_config.yaml \\
        --ability_cutoff_quantile 0.4 \\
        --labor_upper 0.90
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Add project root to path if running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cluster_analysis import PanelBuffer, RegimeAnalyzer
from src.cluster_visualization import (
    plot_regime_scatter,
    plot_regime_paths,
    plot_transition_matrix,
    plot_switch_distribution,
    plot_regime_summary,
    plot_robustness_bic,
)


def simulate_panel_from_checkpoint(
    checkpoint_dir: str,
    config_path: str,
    sim_steps: int = 500,
    state_step: int = None,
) -> PanelBuffer:
    """
    Load a trained model from checkpoint and run sim_steps inference steps
    (no gradient updates) to collect a panel of agent data.

    Args:
        checkpoint_dir: Path to checkpoint directory (has weights/, states/, normalizer/).
        config_path: Path to the original YAML config used for training.
        sim_steps: Number of simulation steps to run.
        state_step: Which saved step to load (e.g. 50000). If None, uses the latest.

    Returns:
        PanelBuffer containing sim_steps rows of agent data.
    """
    import torch
    from src.train import initialize_env_state
    from src.environment import EconomyEnv
    from src.normalizer import RunningPerAgentWelford, HardNormalizer
    from src.models.model import FiLMResNet2In
    from src.utils.configloader import load_configs, dict_to_namespace, compute_derived_params

    # --- Load config ---
    print(f"Loading config from: {config_path}")
    config_dict = load_configs([config_path])
    config_dict = compute_derived_params(config_dict)
    config = dict_to_namespace(config_dict)

    # --- Discover checkpoint step ---
    weights_dir = os.path.join(checkpoint_dir, "weights")
    states_dir = os.path.join(checkpoint_dir, "states")
    normalizer_dir = os.path.join(checkpoint_dir, "normalizer")

    if state_step is None:
        # Find the latest available step
        pt_files = [f for f in os.listdir(weights_dir) if f.startswith("model_step_") and f.endswith(".pt")]
        pt_files.sort(key=lambda f: int(f[len("model_step_"):-3]))
        if not pt_files:
            raise FileNotFoundError(f"No model checkpoint files found in: {weights_dir}")
        state_step = int(pt_files[-1][len("model_step_"):-3])
    print(f"Loading checkpoint at step {state_step}...")

    model_path = os.path.join(weights_dir, f"model_step_{state_step}.pt")
    state_path = os.path.join(states_dir, f"state_step_{state_step}.pt")
    norm_path = os.path.join(normalizer_dir, f"norm_step_{state_step}.pt")

    # --- Device ---
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    # --- Normalizer ---
    norm_type = getattr(config.training, 'normalizer', 'welford')
    if norm_type == 'hard':
        # v_min/v_max are computed as derived params from shock.rho_v / shock.sigma_v
        v_min = getattr(config.shock, 'v_min', getattr(config.initial_state, 'v_min', 0))
        v_max = getattr(config.shock, 'v_max', getattr(config.initial_state, 'v_max', 100))
        normalizer = HardNormalizer(fixed_bounds={"ability": (v_min, v_max)})
    else:
        normalizer = RunningPerAgentWelford(batch_dim=0, agent_dim=None)
    normalizer.load(norm_path)

    # --- Policy network ---
    policy_net = FiLMResNet2In(
        state_dim=2 * config.training.agents + 2,
        cond_dim=5,
        output_dim=3,
    ).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    # --- Environment ---
    env = EconomyEnv(config, normalizer, device=device)

    # --- Load saved state ---
    main_state = torch.load(state_path, map_location=device)

    # --- Fix flag ---
    fix_ability = getattr(config.bewley_model, 'fix', False)

    # --- Panel buffer ---
    panel_buffer = PanelBuffer(
        max_steps=sim_steps,
        n_agents=config.training.agents,
    )

    print(f"Running {sim_steps} simulation steps (no gradient updates)...")
    with torch.no_grad():
        from tqdm import tqdm
        for step in tqdm(range(sim_steps), desc="Simulating", ncols=80):
            main_state, temp_state, _, _ = env.step(
                main_state=main_state,
                policy_net=policy_net,
                deterministic=False,
                fix=fix_ability,
                update_normalizer=False,
                commit_strategy="random",
            )
            panel_buffer.append(step, {
                'ability':           temp_state.ability[0].cpu().numpy(),
                'saving_ratio':      temp_state.savings_ratio[0].cpu().numpy(),
                'mu':                temp_state.mu[0].cpu().numpy(),
                'labor':             temp_state.labor[0].cpu().numpy(),
                'money_disposable':  temp_state.money_disposable[0].cpu().numpy(),
                'consumption':       temp_state.consumption[0].cpu().numpy(),
                'savings':           temp_state.savings[0].cpu().numpy(),
                'income_before_tax': temp_state.income_before_tax[0].cpu().numpy(),
                'wage':              temp_state.wage[0].cpu().numpy(),
                'ret':               temp_state.ret[0].cpu().numpy(),
            })
            del temp_state

    print(f"Simulation complete. Panel: {sim_steps} steps x {config.training.agents} agents.")
    return panel_buffer


def discover_latest_panel(checkpoint_dir: str) -> str:
    """Find the most recent panel NPZ in a checkpoint directory."""
    panel_dir = os.path.join(checkpoint_dir, "plot_data", "panel")
    if not os.path.isdir(panel_dir):
        raise FileNotFoundError(
            f"No panel data directory found at: {panel_dir}\n"
            "Make sure the training run saved panel data (requires cluster tracking integration)."
        )
    npz_files = [f for f in os.listdir(panel_dir) if f.startswith("panel_step_") and f.endswith(".npz")]
    if not npz_files:
        raise FileNotFoundError(f"No panel .npz files found in: {panel_dir}")
    # Sort by step number
    npz_files.sort(key=lambda f: int(f[len("panel_step_"):-4]))
    latest = os.path.join(panel_dir, npz_files[-1])
    print(f"Auto-discovered latest panel: {latest}")
    return latest


def print_summary_table(summary_df: pd.DataFrame) -> None:
    """Print regime summary statistics as a formatted table."""
    print("\n" + "=" * 60)
    print("REGIME SUMMARY STATISTICS")
    print("=" * 60)
    pivot = summary_df.pivot_table(
        index='variable',
        columns='regime',
        values=['mean', 'std', 'q50'],
        aggfunc='first',
    )
    print(pivot.to_string(float_format='{:.4f}'.format))
    print()


def print_transition_matrix(trans_matrix: np.ndarray, labels: list) -> None:
    """Print the transition matrix as a formatted table."""
    print("\n" + "=" * 60)
    print("TRANSITION MATRIX  P(regime[t+1] | regime[t])")
    print("=" * 60)
    short = [l.replace('regime_', '') for l in labels]
    from_to = 'From \\ To'
    header = f"{from_to:<15}" + "".join(f"{s:>12}" for s in short)
    print(header)
    for i, label in enumerate(short):
        row_str = f"{label:<15}" + "".join(f"{trans_matrix[i, j]:>12.4f}" for j in range(len(labels)))
        print(row_str)
    print()


def print_switch_stats(switch_counts: pd.Series) -> None:
    """Print switch count statistics."""
    print("\n" + "=" * 60)
    print("REGIME SWITCH STATISTICS")
    print("=" * 60)
    print(f"  Total agents tracked: {len(switch_counts)}")
    print(f"  Mean switches/agent:  {switch_counts.mean():.3f}")
    print(f"  Median:               {switch_counts.median():.1f}")
    print(f"  Max:                  {switch_counts.max()}")
    print(f"  Agents with 0 switches: {(switch_counts == 0).sum()} ({100*(switch_counts==0).mean():.1f}%)")
    print(f"  Agents with ≥1 switch:  {(switch_counts >= 1).sum()} ({100*(switch_counts>=1).mean():.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc GMM regime tracking analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input: panel NPZ or checkpoint directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--panel_npz', type=str,
        help="Path to a saved panel .npz file (from PanelBuffer.to_npz)."
    )
    input_group.add_argument(
        '--checkpoint_dir', type=str,
        help="Checkpoint directory. If --config is also given, simulates from checkpoint "
             "(no re-training). Otherwise, auto-discovers a saved panel .npz inside plot_data/panel/."
    )

    # Simulation mode (used when checkpoint_dir is provided without existing panel data)
    parser.add_argument('--config', type=str, default=None,
                        help="Path to the original YAML config. Required for --checkpoint_dir "
                             "simulation mode.")
    parser.add_argument('--sim_steps', type=int, default=500,
                        help="Number of inference steps to run when simulating from checkpoint.")
    parser.add_argument('--state_step', type=int, default=None,
                        help="Which saved checkpoint step to load (e.g. 50000). "
                             "Defaults to the latest available.")

    # Clustering parameters
    parser.add_argument('--ability_cutoff_quantile', type=float, default=0.1,
                        help="Filter: keep ability >= this quantile.")
    parser.add_argument('--labor_upper', type=float, default=0.95,
                        help="Filter: keep labor < this threshold (interior region).")
    parser.add_argument('--n_components', type=int, default=2,
                        help="Number of GMM components.")
    parser.add_argument('--detrend_method', type=str, default='lowess',
                        choices=['lowess', 'poly'],
                        help="Detrending method for ability trend removal.")
    parser.add_argument('--lowess_frac', type=float, default=0.3,
                        help="LOWESS bandwidth fraction.")

    # Robustness check
    parser.add_argument('--skip_robustness', action='store_true',
                        help="Skip robustness check (faster).")

    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Directory to save output plots and CSV. "
                             "Defaults to <panel_dir>/regime_analysis/.")
    parser.add_argument('--n_path_agents', type=int, default=5,
                        help="Number of representative agents in regime path plot.")
    parser.add_argument('--no_plots', action='store_true',
                        help="Skip plot generation (print tables only).")

    args = parser.parse_args()

    # ----------------------------------------------------------------
    # 1. Load / build panel data
    # ----------------------------------------------------------------
    if args.panel_npz:
        print(f"Loading panel data from: {args.panel_npz}")
        buf = PanelBuffer.from_npz(args.panel_npz)

    elif args.checkpoint_dir and args.config:
        # Simulate from checkpoint — no re-training needed
        buf = simulate_panel_from_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            config_path=args.config,
            sim_steps=args.sim_steps,
            state_step=args.state_step,
        )
        # Save the generated panel for reuse
        panel_save_dir = os.path.join(args.checkpoint_dir, "plot_data", "panel")
        os.makedirs(panel_save_dir, exist_ok=True)
        panel_path = os.path.join(panel_save_dir, f"panel_sim_{args.sim_steps}steps.npz")
        buf.to_npz(panel_path)
        print(f"Panel saved to: {panel_path}")

    else:
        # Try to find an existing panel NPZ in the checkpoint directory
        try:
            panel_path = discover_latest_panel(args.checkpoint_dir)
            buf = PanelBuffer.from_npz(panel_path)
        except FileNotFoundError:
            print(
                "\nNo saved panel data found in checkpoint directory.\n"
                "To generate panel data without re-training, add --config:\n\n"
                f"  python run_cluster_analysis.py \\\n"
                f"      --checkpoint_dir {args.checkpoint_dir} \\\n"
                f"      --config config/your_config.yaml\n"
            )
            sys.exit(1)

    print(f"  Buffer size: {len(buf)} steps x {buf.n_agents} agents")

    panel_df = buf.to_dataframe()
    print(f"  Panel shape: {panel_df.shape} (long format)")
    print(f"  Time range: {panel_df['time'].min()} \u2013 {panel_df['time'].max()}")
    print(f"  Unique agents: {panel_df['agent_id'].nunique()}")

    # ----------------------------------------------------------------
    # 2. Output directory
    # ----------------------------------------------------------------
    if args.output_dir is None:
        if args.checkpoint_dir:
            args.output_dir = os.path.join(args.checkpoint_dir, "plot_data", "regime_analysis")
        else:
            args.output_dir = os.path.join(os.path.dirname(args.panel_npz), "regime_analysis")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to: {args.output_dir}")

    # ----------------------------------------------------------------
    # 3. Run RegimeAnalyzer
    # ----------------------------------------------------------------
    print("\nRunning RegimeAnalyzer...")
    analyzer = RegimeAnalyzer(
        ability_cutoff_quantile=args.ability_cutoff_quantile,
        labor_upper=args.labor_upper,
        n_components=args.n_components,
        detrend_method=args.detrend_method,
        lowess_frac=args.lowess_frac,
    )

    try:
        enriched_df = analyzer.fit(panel_df)
    except ValueError as e:
        print(f"ERROR in RegimeAnalyzer.fit(): {e}")
        print("Try adjusting --ability_cutoff_quantile or --labor_upper.")
        sys.exit(1)

    n_regime = enriched_df['regime'].notna().sum()
    n_total = len(enriched_df)
    print(f"  Classified {n_regime}/{n_total} observations ({100*n_regime/n_total:.1f}%)")
    for r in ['regime_low', 'regime_high']:
        n_r = (enriched_df['regime'] == r).sum()
        print(f"  {r}: {n_r} ({100*n_r/n_total:.1f}%)")

    # ----------------------------------------------------------------
    # 4. Compute transitions
    # ----------------------------------------------------------------
    print("\nComputing transitions...")
    trans_result = analyzer.compute_transitions(enriched_df)
    enriched_df = trans_result['enriched_df']
    trans_matrix = trans_result['transition_matrix']
    trans_labels = trans_result['transition_labels']
    switch_counts = trans_result['switch_counts']
    time_in_regime = trans_result['time_in_regime']

    print_transition_matrix(trans_matrix, trans_labels)
    print_switch_stats(switch_counts)

    # ----------------------------------------------------------------
    # 5. Summary statistics
    # ----------------------------------------------------------------
    print("Computing regime summary statistics...")
    summary_df = analyzer.compute_summary_stats(enriched_df)
    print_summary_table(summary_df)

    # ----------------------------------------------------------------
    # 6. Robustness check
    # ----------------------------------------------------------------
    robustness_df = None
    if not args.skip_robustness:
        print("Running robustness check (BIC/AIC across K and filter settings)...")
        robustness_df = analyzer.robustness_check(panel_df)
        print("\nBIC/AIC comparison:")
        print(robustness_df.to_string(index=False, float_format='{:.1f}'.format))

    # ----------------------------------------------------------------
    # 7. Save enriched panel CSV
    # ----------------------------------------------------------------
    csv_path = os.path.join(args.output_dir, "enriched_panel.csv")
    enriched_df.to_csv(csv_path, index=False)
    print(f"\nEnriched panel saved to: {csv_path}")

    summary_csv = os.path.join(args.output_dir, "regime_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Regime summary saved to: {summary_csv}")

    if robustness_df is not None:
        rob_csv = os.path.join(args.output_dir, "robustness_bic.csv")
        robustness_df.to_csv(rob_csv, index=False)
        print(f"Robustness table saved to: {rob_csv}")

    # ----------------------------------------------------------------
    # 8. Generate plots
    # ----------------------------------------------------------------
    if not args.no_plots:
        print("\nGenerating plots...")
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for script use

        fig = plot_regime_scatter(
            enriched_df,
            save_path=os.path.join(args.output_dir, "regime_scatter.png"),
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
        print("  Saved: regime_scatter.png")

        fig = plot_regime_paths(
            enriched_df,
            n_agents=args.n_path_agents,
            save_path=os.path.join(args.output_dir, "regime_paths.png"),
        )
        plt.close(fig)
        print("  Saved: regime_paths.png")

        fig = plot_transition_matrix(
            trans_matrix,
            transition_labels=trans_labels,
            save_path=os.path.join(args.output_dir, "transition_matrix.png"),
        )
        plt.close(fig)
        print("  Saved: transition_matrix.png")

        fig = plot_switch_distribution(
            switch_counts,
            save_path=os.path.join(args.output_dir, "switch_distribution.png"),
        )
        plt.close(fig)
        print("  Saved: switch_distribution.png")

        fig = plot_regime_summary(
            summary_df,
            save_path=os.path.join(args.output_dir, "regime_summary.png"),
        )
        plt.close(fig)
        print("  Saved: regime_summary.png")

        if robustness_df is not None and len(robustness_df) > 0:
            fig = plot_robustness_bic(
                robustness_df,
                save_path=os.path.join(args.output_dir, "robustness_bic.png"),
            )
            plt.close(fig)
            print("  Saved: robustness_bic.png")

    print(f"\nDone. All outputs in: {args.output_dir}")


if __name__ == '__main__':
    main()
