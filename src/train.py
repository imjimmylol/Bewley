# src/train.py
import os
import wandb
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn.functional as F
from src.env_state import MainState
from src.environment import EconomyEnv
from src.normalizer import RunningPerAgentWelford, HardNormalizer
from src.models.model import FiLMResNet2In
from src.calloss import LossCalculator
from src.monitoring import TrainingMonitor
from src.visualization import (
    prepare_data_for_plotting,
    plot_decision_rules_scatter,
    plot_binned_decision_rules,
    plot_state_distributions,
    plot_all_decision_rules,
    plot_input_output_pairwise,
    plot_input_output_pca
)
from src.policy_evaluation import (
    HistoricalRanges,
    PolicyEvaluator,
    collect_ranges_from_step
)
from src.plot_data_io import save_input_output_data, save_run_meta, save_panel_data
from src.utils.economics import flow_utility
from src.cluster_analysis import PanelBuffer, RegimeAnalyzer, lightweight_cluster_snapshot
from src.cluster_visualization import (
    plot_regime_scatter,
    plot_regime_paths,
    plot_transition_matrix,
    plot_switch_distribution,
    plot_regime_summary,
)
import numpy as np
import threading

def initialize_env_state(config, device="cpu"):
    """
    Initialize the environment state based on configuration.

    Args:
        config: Configuration namespace containing training and model parameters
        device: Device to place tensors on ("cpu" or "cuda")

    Returns:
        MainState: Initialized environment state
    """
    # Get dimensions from config
    batch_size = config.training.batch_size
    n_agents = config.training.agents  # Ability grid dimension

    # History length for ability tracking (can be made configurable)
    history_length = getattr(config.training, 'history_length', 50)

    # Initialize tax parameters tensor [tax_consumption, tax_income, tax_saving, ...]
    tax_params_values = {
        "tax_consumption": config.tax_params.tax_consumption,
        "tax_income": config.tax_params.tax_income,
        "tax_saving": config.tax_params.tax_saving,
        "incomew_tax_elasticity": config.tax_params.income_tax_elasticity,
        "saving_tax_elasticity": config.tax_params.saving_tax_elasticity,
    }

    # Expand tax_params to (batch_size, n_params) for batch processing
    tax_params = torch.tensor(list(tax_params_values.values()), dtype=torch.float32, device=device)
    tax_params = tax_params.repeat(batch_size, 1)

    moneydisposable=np.random.lognormal(np.log(config.initial_state.moneydisposable_mean), config.initial_state.moneydisposable_std, batch_size * n_agents).reshape(batch_size, n_agents)
    savings= np.random.lognormal(np.log(config.initial_state.assets_mean), config.initial_state.assets_std, batch_size * n_agents).reshape(batch_size, n_agents)

    # Initialize ability using bounded distribution (prevents explosion)
    # Use IQ-like distribution: realistic, bounded, prevents model collapse
    from src.ability_init import initialize_ability

    # IQ-like distribution parameters (used for both ability init and v_bar)
    ability_mean = getattr(config.initial_state, 'ability_mean', 1.0)
    ability_cv = getattr(config.initial_state, 'ability_cv', 0.3)
    ability_min = getattr(config.initial_state, 'ability_min', 0.3)
    ability_max = getattr(config.initial_state, 'ability_max', 3.0)

    # Get initialization method from config (default: iq_like)
    init_method = getattr(config.initial_state, 'ability_init_method', 'iq_like')

    if init_method == 'iq_like':
        # Recommended: bounded log-normal like real ability distribution
        ability = initialize_ability(
            batch_size, n_agents,
            method='iq_like',
            mean=ability_mean,
            cv=ability_cv,
            min_ability=ability_min,
            max_ability=ability_max
        )
    else:
        # Fallback: stationary AR(1) with clipping (old method)
        ability = initialize_ability(
            batch_size, n_agents,
            method='stationary',
            config=config,
            clip_sigma=2.0  # Clip to ±2σ to prevent explosion
        )

    # Initialize heterogeneous v_bar (per-agent long-run ability mean, fixed for entire simulation)
    # Drawn independently from same IQ-like distribution
    v_bar_array = initialize_ability(
        batch_size, n_agents,
        method='iq_like',
        mean=ability_mean,
        cv=ability_cv,
        min_ability=ability_min,
        max_ability=ability_max
    )

    is_superstar_vA = np.zeros((batch_size, n_agents), dtype=bool)
    is_superstar_vB = np.zeros((batch_size, n_agents), dtype=bool)

    # Create ret as a tensor with shape (batch_size, n_agents) for consistency
    # This ensures ret is a tensor from the start (same as after first training step)
    ret_value = config.bewley_model.r
    ret_tensor = torch.full((batch_size, n_agents), ret_value, dtype=torch.float32, device=device)

    state = MainState(
        moneydisposable = torch.tensor(moneydisposable, dtype=torch.float32, device=device),
        savings = torch.tensor(savings, dtype=torch.float32, device=device),
        ability = torch.tensor(ability, dtype=torch.float32, device=device),
        v_bar = torch.tensor(v_bar_array, dtype=torch.float32, device=device),
        ret = ret_tensor,
        tax_params=tax_params,
        is_superstar_vA = torch.tensor(is_superstar_vA, dtype=torch.bool, device=device),
        is_superstar_vB = torch.tensor(is_superstar_vB, dtype=torch.bool, device=device),
        ability_history_vA=None,
        ability_history_vB=None
    )
    return state


def _run_regime_analysis_background(panel_df, step, output_dir):
    """Run full RegimeAnalyzer pipeline in a background thread."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
        analyzer = RegimeAnalyzer(
            ability_cutoff_quantile=0.5,
            labor_upper=0.95,
            n_components=2,
        )
        enriched_df = analyzer.fit(panel_df)

        n_regime = enriched_df['regime'].notna().sum()
        n_total = len(enriched_df)
        frac_high = (enriched_df['regime'] == 'regime_high').sum() / max(n_total, 1)
        frac_low = (enriched_df['regime'] == 'regime_low').sum() / max(n_total, 1)

        # Transitions
        trans_result = analyzer.compute_transitions(enriched_df)
        enriched_df = trans_result['enriched_df']
        trans_matrix = trans_result['transition_matrix']
        trans_labels = trans_result['transition_labels']
        switch_counts = trans_result['switch_counts']

        # Summary stats (skip robustness check — too slow for training)
        summary_df = analyzer.compute_summary_stats(enriched_df)

        # Log to wandb
        wandb.log({
            'regime_full/fraction_high': frac_high,
            'regime_full/fraction_low': frac_low,
            'regime_full/classified': n_regime,
            'regime_full/total': n_total,
        }, step=step)

        # Save CSVs
        step_dir = os.path.join(output_dir, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)
        enriched_df.to_csv(os.path.join(step_dir, "enriched_panel.csv"), index=False)
        summary_df.to_csv(os.path.join(step_dir, "regime_summary.csv"), index=False)

        # Generate plots
        for plot_fn, fname, kwargs in [
            (plot_regime_scatter, "regime_scatter.png", {"enriched_df": enriched_df}),
            (plot_regime_paths, "regime_paths.png", {"enriched_df": enriched_df, "n_agents": 5}),
            (plot_transition_matrix, "transition_matrix.png",
             {"transition_matrix": trans_matrix, "transition_labels": trans_labels}),
            (plot_switch_distribution, "switch_distribution.png", {"switch_counts": switch_counts}),
            (plot_regime_summary, "regime_summary.png", {"summary_df": summary_df}),
        ]:
            try:
                fig = plot_fn(**kwargs, save_path=os.path.join(step_dir, fname))
                plt.close(fig)
            except Exception:
                pass

        print(f"\n[regime] Step {step}: {n_regime}/{n_total} classified, "
              f"high={frac_high:.1%}, low={frac_low:.1%}")

    except Exception as e:
        print(f"\n[regime] Background analysis failed at step {step}: {e}")


def train(config, run):
    """
    The main training loop.
    
    Args:
        config (SimpleNamespace): The complete configuration object from wandb.
        run (wandb.sdk.wandb_run.Run): The current wandb run object.
    """
    # --- 1. Get run name and create checkpoint directories ---
    # The run name is determined by the rules in GEMINI.md
    if run and run.name:
        run_name = run.name
    elif hasattr(config, 'exp_name'):
        run_name = config.exp_name
    else:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"Starting run: {run_name}")

    base_checkpoint_dir = os.path.join("checkpoints", run_name)
    weights_dir = os.path.join(base_checkpoint_dir, "weights")
    states_dir = os.path.join(base_checkpoint_dir, "states")
    normalizer_dir = os.path.join(base_checkpoint_dir, "normalizer")

    plot_data_grid_dir = os.path.join(base_checkpoint_dir, "plot_data", "grid")
    plot_data_io_dir = os.path.join(base_checkpoint_dir, "plot_data", "input_output")

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(states_dir, exist_ok=True)
    os.makedirs(normalizer_dir, exist_ok=True)
    os.makedirs(plot_data_grid_dir, exist_ok=True)
    os.makedirs(plot_data_io_dir, exist_ok=True)

    save_run_meta(config, os.path.join(base_checkpoint_dir, "plot_data"))

    print(f"Checkpoints will be saved in: {base_checkpoint_dir}")

    # --- 2. Initialize components ---

    device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
    )
    print(f"Using device: {device}")
    # --- Normalizer selection (controlled by config.training.normalizer) ---
    norm_type = getattr(config.training, 'normalizer', 'welford')
    if norm_type == 'hard':
        # Hard normalization: fixed bounds → [0, 1]
        # ability uses config bounds; money_disposable tracks running max
        v_min = getattr(config.initial_state, 'v_min', 0)
        v_max = getattr(config.initial_state, 'v_max', 100)
        normalizer = HardNormalizer(fixed_bounds={"ability": (v_min, v_max)})
        print(f"Using HardNormalizer (ability: [{v_min:.4f}, {v_max:.4f}], moneydisposable: running)")
    else:
        # Welford (running mean/std, maps to ~N(0,1))
        normalizer = RunningPerAgentWelford(batch_dim=0, agent_dim=None)
        print(f"Using Welford normalizer (global mode)")
    env = EconomyEnv(config, normalizer, device=device)
    policy_net = FiLMResNet2In(
        state_dim=2*config.training.agents+2,
        cond_dim=5,
        output_dim=3
    ).to(device)
    print(f"✓ EconomyEnv initialized")
    print(f"  - Device: {device}")
    print(f"  - Batch size: {env.batch_size}")
    print(f"  - Number of agents: {env.n_agents}")
    print(f"  - History length: {env.history_length}")

    # Initialize state
    main_state = initialize_env_state(config, device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=float(config.training.learning_rate))

    # Initialize loss
    loss_calculator = LossCalculator(config=config, device=device)

    # Initialize training monitor for metrics and logging
    monitor = TrainingMonitor(config=config, normalizer=normalizer)

    # Initialize historical ranges for synthetic grid evaluation
    # Enable per-agent tracking so we can use agent-specific x-axis ranges
    historical_ranges = HistoricalRanges(track_per_agent=True)

    # Initialize panel buffer for regime tracking (tracks batch 0 across time)
    panel_buffer_size = getattr(config.training, 'panel_buffer_size', 200)
    cluster_interval = getattr(config.training, 'cluster_interval', 5000)
    panel_buffer = PanelBuffer(
        max_steps=panel_buffer_size,
        n_agents=config.training.agents,
    )
    plot_data_panel_dir = os.path.join(base_checkpoint_dir, "plot_data", "panel")
    regime_analysis_dir = os.path.join(base_checkpoint_dir, "plot_data", "regime_analysis")
    os.makedirs(plot_data_panel_dir, exist_ok=True)
    os.makedirs(regime_analysis_dir, exist_ok=True)
    
    # --- 3.5 Plot initial state distributions before training ---
    print("Plotting initial state distributions...")
    plot_state_distributions(
        main_state,
        save_path=os.path.join(base_checkpoint_dir, "state_distributions_initial.png"),
        log_to_wandb=True,
        step=0
    )

    # --- 4. Training Loop ---
    print("Starting training loop with environment stepping...")

    # Get market completeness flag from config
    fix_ability = getattr(config.bewley_model, 'fix', False)
    market_type = "COMPLETE" if fix_ability else "INCOMPLETE"
    print(f"  - Market type: {market_type} (fix={fix_ability})")

    total_steps = config.training.training_steps
    for step in tqdm(range(1, total_steps + 1), total=total_steps, desc="Training", ncols=100):
        # ==== STEP THE ENVIRONMENT ====
        # This performs the full 4-step workflow:
        # 1. Agents observe MainState[t] and act
        # 2. Create TemporaryState with realized outcomes
        # 3. Transition to ParallelState A and B with different shocks
        # 4. Compute outcomes for both branches, choose one to commit

        main_state, temp_state, (parallel_A, outcomes_A), (parallel_B, outcomes_B) = env.step(
            main_state=main_state,
            policy_net=policy_net,
            deterministic=False,
            fix=fix_ability,  # Read from config: complete vs incomplete markets
            update_normalizer=True,
            commit_strategy="random"
        )

        # ==== EXTRACT VARIABLES FOR LOSS COMPUTATION ====
        # Current period (t) outcomes from TemporaryState
        consumption_t = temp_state.consumption              # (B, A)
        labor_t = temp_state.labor                          # (B, A)
        savings_ratio_t = temp_state.savings_ratio          # (B, A)
        ibt = temp_state.income_before_tax
        mu_t = temp_state.mu                                # (B, A)
        wage_t = temp_state.wage                            # (B, A)
        ret_t = temp_state.ret                              # (B, A)
        money_disposable_t = temp_state.money_disposable    # (B, A)
        ability_t = temp_state.ability
        savings_t = temp_state.savings  # Savings for t+1 (allocated from budget at t)

        # Next period (t+1) outcomes from parallel branches
        consumption_A_tp1 = outcomes_A["consumption"]       # (B, A)
        income_before_tax_A_tp1 = outcomes_A["income_before_tax"]
        consumption_B_tp1 = outcomes_B["consumption"]       # (B, A)
        income_before_tax_B_tp1 = outcomes_B["income_before_tax"]

        # ==== EXTRACT NORMALIZED FEATURES FOR DEBUGGING ====
        # Get normalized features that were fed to the policy network
        # These are computed internally during env.step()
        with torch.no_grad():
            normalized_features, _ = env._prepare_features(main_state, update_normalizer=False)
            # Extract individual normalized features from the stacked tensor
            # normalized_features shape: (B, A, 2A+2)
            # Structure: [all_money (2A), money_self (1), all_ability (2A), ability_self (1)]
            # Wait, need to check actual structure from buildipnuts.py
            # From buildipnuts: features = [sum_info_rep (2A), money_self (1), ability_self (1)]
            # So shape is (B, A, 2A+2)

            normalized_money_mean = normalized_features[..., -2].mean().item()  # money_self
            normalized_ability_mean = normalized_features[..., -1].mean().item()  # ability_self
            normalized_money_std = normalized_features[..., -2].std().item()
            normalized_ability_std = normalized_features[..., -1].std().item()

        # ==== COMPUTE LOSSES (PLACEHOLDER - TO BE IMPLEMENTED) ====
        
        loss = loss_calculator.compute_all_losses(
            # Current period (t)
            consumption_t = consumption_t,
            labor_t=labor_t,
            ibt=ibt,
            savings_ratio_t=savings_ratio_t,
            mu_t=mu_t,
            wage_t=wage_t,
            ret_t=ret_t,
            money_disposable_t=money_disposable_t,
            ability_t=ability_t,
            # Next period (t+1) - two branches
            consumption_A_tp1=consumption_A_tp1,
            consumption_B_tp1=consumption_B_tp1,
            ibt_A_tp1=income_before_tax_A_tp1,
            ibt_B_tp1=income_before_tax_B_tp1
        )


        # ==== BACKWARD PASS AND PARAMETER UPDATE ====
        optimizer.zero_grad()
        loss["total"].backward()

        # Optional: gradient clipping
        # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)

        optimizer.step()

        # ==== MONITORING: Log metrics, correlations, and debug info ====
        monitor.log_step(step, main_state, temp_state, loss)

        # ==== COLLECT HISTORICAL RANGES for synthetic grid evaluation ====
        # track_per_agent=True enables agent-specific x-axis ranges in visualization
        historical_ranges = collect_ranges_from_step(
            temp_state, main_state, historical_ranges, track_per_agent=True
        )

        # ==== VISUALIZATION: Generate decision rule plots at plot intervals ====
        # Use frequent interval early, then switch to save_interval after 30k steps
        plot_interval = getattr(config.training, 'plot_interval', config.training.save_interval)
        if step >= 15000:
            current_plot_interval = plot_interval
        else:
            current_plot_interval = config.training.save_interval
        if step % current_plot_interval == 0:
            print(f"\nGenerating decision rule visualizations at step {step}...")

            # Prepare data for plotting (converts current step data to plotting format)
            plot_data = prepare_data_for_plotting(main_state, temp_state)

            # Generate scatter plot with W&B logging (uses all batches by default)
            plot_decision_rules_scatter(
                plot_data,
                save_path=os.path.join(base_checkpoint_dir, f"decision_rules_step_{step}.png"),
                log_to_wandb=True,
                step=step
            )

            # Generate binned plot with W&B logging (uses all batches by default)
            plot_binned_decision_rules(
                plot_data,
                n_bins=10,
                save_path=os.path.join(base_checkpoint_dir, f"binned_rules_step_{step}.png"),
                log_to_wandb=True,
                step=step
            )

            # Generate state distribution plot with W&B logging
            plot_state_distributions(
                main_state,
                save_path=os.path.join(base_checkpoint_dir, f"state_distributions_step_{step}.png"),
                log_to_wandb=True,
                step=step
            )

            # ==== NEW: Synthetic Grid Decision Rule Plots (A1, B1) ====
            # Only generate if we have collected enough data
            if historical_ranges.m_t.count > 0:
                print(f"Generating synthetic grid decision rule plots...")
                # Create reference state from current simulation for GE-aware evaluation
                # This uses actual tensors as background population (detached from graph)
                reference_state = {
                    "money": money_disposable_t.detach(),  # (B, A)
                    "ability": ability_t.detach(),  # (B, A)
                }
                evaluator = PolicyEvaluator(
                    policy_net=policy_net,
                    normalizer=normalizer,
                    ranges=historical_ranges,
                    tax_params=main_state.tax_params[0],  # Use first batch's tax params
                    n_agents=config.training.agents,  # Pass n_agents for proper input shape
                    device=device,
                    reference_state=reference_state,  # Pass actual GE simulation data
                    config=config,  # Pass config for computing m_t from a_t
                    agent_idx=0,  # Focus on agent 0 for visualization
                    use_agent_specific_range=True  # Use agent 0's explored range for x-axis
                )
                plot_all_decision_rules(
                    evaluator=evaluator,
                    save_dir=os.path.join(base_checkpoint_dir, "decision_rules"),
                    log_to_wandb=True,
                    step=step,
                    plots=["A1", "A1-1", "B1", "A1-1h", "H1", "H1-h"],  # Specify which plots to generate
                    v_bar=main_state.v_bar.detach().cpu().numpy().flatten(),
                    data_save_dir=plot_data_grid_dir,
                    exp_name=run_name
                )

            # ==== Direct Input-Output Visualization ====
            print(f"Generating direct input-output plots...")
            with torch.no_grad():
                features_snap, condi_snap = env._prepare_features(main_state, update_normalizer=False)
                out_snap = policy_net(features_snap, condi_snap)
                zeta_snap = torch.sigmoid(out_snap[..., 0]) * 0.98 + 0.01
                mu_snap = F.softplus(out_snap[..., 1]) + 1e-6
                labor_snap = torch.sigmoid(out_snap[..., 2]) * 0.98 + 0.01

                # Compute per-agent losses (reduce=False) using existing loss classes
                fb_per_agent = loss_calculator.fb_loss_fn(
                    savings_ratio=savings_ratio_t, mu=mu_t, reduce=False
                ).detach()
                euler_per_agent = loss_calculator.aux_loss_mu_fn(
                    c0=consumption_t, mu0=mu_t, ret0=ret_t,
                    savings_tp=savings_ratio_t * money_disposable_t,
                    c1_A=consumption_A_tp1, c1_B=consumption_B_tp1,
                    ibt_A=income_before_tax_A_tp1, ibt_B=income_before_tax_B_tp1,
                    reduce=False
                ).detach()
                labor_foc_per_agent = loss_calculator.labor_loss_fn(
                    consumption=consumption_t, ibt=ibt, wage=wage_t,
                    ability=ability_t, labor=labor_t, reduce=False
                ).detach()

            per_agent_losses = {
                "FB": fb_per_agent.cpu().numpy().flatten(),
                "Euler": euler_per_agent.cpu().numpy().flatten(),
                "Labor FOC": labor_foc_per_agent.cpu().numpy().flatten(),
            }

            # Compute utility from snapshot (consistent with plotted zeta/labor)
            # Denormalize money from snapshot features, then c = m * (1 - zeta)
            money_snap = normalizer.denormalize("moneydisposalbe", features_snap[..., -2])  # (B, A)
            consumption_snap = money_snap * (1.0 - zeta_snap)
            utility = flow_utility(
                consumption_snap, labor_snap,
                theta=loss_calculator.theta, gamma=loss_calculator.gamma,
            ).detach().cpu().numpy().flatten()

            plot_input_output_pairwise(
                features_snap, zeta_snap, mu_snap, labor_snap, normalizer,
                per_agent_losses=per_agent_losses,
                utility=utility,
                save_path=os.path.join(base_checkpoint_dir, f"input_output_pairwise_step_{step}.png"),
                log_to_wandb=True, step=step
            )
            plot_input_output_pca(
                features_snap, zeta_snap, mu_snap, labor_snap,
                save_path=os.path.join(base_checkpoint_dir, f"input_output_pca_step_{step}.png"),
                log_to_wandb=True, step=step
            )

            # Save numerical input-output data for cross-run comparison
            save_input_output_data(
                own_money=features_snap[..., -2].detach().cpu().numpy().flatten(),
                own_ability=features_snap[..., -1].detach().cpu().numpy().flatten(),
                zeta=zeta_snap.detach().cpu().numpy().flatten(),
                mu=mu_snap.detach().cpu().numpy().flatten(),
                labor=labor_snap.detach().cpu().numpy().flatten(),
                per_agent_losses=per_agent_losses,
                normalizer_stats={},
                step=step, exp_name=run_name, save_dir=plot_data_io_dir,
                utility=utility
            )

        # ==== PANEL BUFFER: Collect batch-0 data for regime tracking ====
        # Must happen before the del block below.
        # Batch 0 agents are persistent across steps (MainState carries forward),
        # giving a genuine panel: agent_id × time.
        panel_buffer.append(step, {
            'ability':           ability_t[0].detach().cpu().numpy(),
            'saving_ratio':      savings_ratio_t[0].detach().cpu().numpy(),
            'mu':                mu_t[0].detach().cpu().numpy(),
            'labor':             labor_t[0].detach().cpu().numpy(),
            'money_disposable':  money_disposable_t[0].detach().cpu().numpy(),
            'consumption':       consumption_t[0].detach().cpu().numpy(),
            'savings':           savings_t[0].detach().cpu().numpy(),
            'income_before_tax': ibt[0].detach().cpu().numpy(),
            'wage':              wage_t[0].detach().cpu().numpy(),
            'ret':               ret_t[0].detach().cpu().numpy(),
        })

        # ==== LIGHTWEIGHT CLUSTERING: Periodic cross-sectional GMM snapshot ====
        if step % cluster_interval == 0 and step > 0:
            try:
                cluster_result = lightweight_cluster_snapshot(
                    ability=ability_t.detach().cpu().numpy().flatten(),
                    saving_ratio=savings_ratio_t.detach().cpu().numpy().flatten(),
                    mu=mu_t.detach().cpu().numpy().flatten(),
                    labor=labor_t.detach().cpu().numpy().flatten(),
                )
                wandb.log({
                    'regime/fraction_high':    cluster_result['regime_fraction_high'],
                    'regime/fraction_low':     cluster_result['regime_fraction_low'],
                    'regime/separation_score': cluster_result['separation_score'],
                    'regime/bic_k2':           cluster_result['bic'],
                    'regime/aic_k2':           cluster_result['aic'],
                }, step=step)
            except Exception as e:
                print(f"[regime tracking] Lightweight clustering failed at step {step}: {e}")

            # ==== FULL REGIME ANALYSIS: Run in background thread ====
            if panel_buffer._count >= 50:  # Need enough panel history
                panel_df_snapshot = panel_buffer.to_dataframe()
                t = threading.Thread(
                    target=_run_regime_analysis_background,
                    args=(panel_df_snapshot, step, regime_analysis_dir),
                    daemon=True,
                )
                t.start()

        # CRITICAL: Clear temporary variables to prevent memory leaks
        # Delete tensors that have computational graphs attached
        del temp_state, parallel_A, parallel_B, outcomes_A, outcomes_B
        del consumption_t, labor_t, savings_ratio_t, mu_t, wage_t, ret_t, money_disposable_t, savings_t
        del consumption_A_tp1, consumption_B_tp1, ibt, ability_t
        del income_before_tax_A_tp1, income_before_tax_B_tp1
        # Delete all individual losses from the loss dict
        del loss

        # --- 5. Save checkpoints periodically ---
        if step % config.training.save_interval == 0:
            print(f"Saving checkpoint at step {step}...")
            torch.save(policy_net.state_dict(), os.path.join(weights_dir, f"model_step_{step}.pt"))
            torch.save(main_state, os.path.join(states_dir, f"state_step_{step}.pt"))
            normalizer.save(os.path.join(normalizer_dir, f"norm_step_{step}.pt"))
            save_panel_data(panel_buffer, step, run_name, plot_data_panel_dir)

    print("\nTraining loop finished.")
    # --- 6. Final save (Example) ---
    print("Saving final model...")
    torch.save(policy_net.state_dict(), os.path.join(weights_dir, "model_final.pt"))
