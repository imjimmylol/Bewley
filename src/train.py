# src/train.py
import os
import wandb
from datetime import datetime
from tqdm import tqdm
import torch
from src.env_state import MainState
from src.environment import EconomyEnv
from src.normalizer import RunningPerAgentWelford
from src.models.model import FiLMResNet2In
from src.calloss import LossCalculator
import numpy as np

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

    # Initialize ability from stationary distribution of AR(1) process
    # AR(1): log(v[t+1]) = (1-rho)*log(v_bar) + rho*log(v[t]) + eps
    # Stationary: mean = log(v_bar), variance = sigma_v^2 / (1 - rho_v^2)
    rho_v = config.shock.rho_v
    sigma_v = config.shock.sigma_v
    v_bar = config.shock.v_bar
    ability_log_mean = np.log(v_bar)
    ability_log_std = sigma_v / np.sqrt(1 - rho_v**2)
    ability = np.random.lognormal(ability_log_mean, ability_log_std, batch_size * n_agents).reshape(batch_size, n_agents)

    is_superstar_vA = np.zeros((batch_size, n_agents), dtype=bool)
    is_superstar_vB = np.zeros((batch_size, n_agents), dtype=bool)

    state = MainState(
        moneydisposable = torch.tensor(moneydisposable, dtype=torch.float32, device=device),
        savings = torch.tensor(savings, dtype=torch.float32, device=device),
        ability = torch.tensor(ability, dtype=torch.float32, device=device),
        ret = config.bewley_model.r,
        tax_params=tax_params,
        is_superstar_vA = torch.tensor(is_superstar_vA, dtype=torch.bool, device=device),
        is_superstar_vB = torch.tensor(is_superstar_vB, dtype=torch.bool, device=device),
        ability_history_vA=None,
        ability_history_vB=None
    )
    return state


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

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(states_dir, exist_ok=True)
    os.makedirs(normalizer_dir, exist_ok=True)

    print(f"Checkpoints will be saved in: {base_checkpoint_dir}")

    # --- 2. Initialize components ---

    device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
    )
    print(f"Using device: {device}")
    normalizer = RunningPerAgentWelford(batch_dim=0, agent_dim=1)
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

    # --- EXPERIMENT 2: Track ability-money correlation ---
    # Store previous step's data for lagged correlation analysis
    prev_ability = None
    prev_money = None

    # --- 4. Training Loop ---
    print("Starting training loop with environment stepping...")
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

        # ==== LOGGING ====
        # Extract scalar values BEFORE deleting tensors
        if step % config.training.display_step == 0 or step == 1 or wandb.run:
            # ==== EXPERIMENT 2: Extract state for correlation analysis ====
            # Extract current state data (only during logging to avoid overhead)
            current_ability = main_state.ability.detach().cpu().numpy().flatten()
            current_money = main_state.moneydisposable.detach().cpu().numpy().flatten()
            current_labor = labor_t.detach().cpu().numpy().flatten()  # From temporary state
            # Convert to Python scalars to avoid holding tensor references
            loss_total_val = loss["total"].item()
            loss_fb_val = loss["fb"].item()
            loss_aux_mu_val = loss["aux_mu"].item()
            loss_labor_val = loss["labor"].item()
            consumption_mean_val = consumption_t.mean().item()
            labor_mean_val = labor_t.mean().item()
            savings_mean_val = main_state.savings.mean().item()
            ability_mean_val = main_state.ability.mean().item()
            wage_mean_val = wage_t.mean().item()
            ret_mean_val = ret_t.mean().item()

            # Raw (unnormalized) state values for debugging
            money_raw_mean = main_state.moneydisposable.mean().item()
            ability_raw_mean = main_state.ability.mean().item()
            money_raw_std = main_state.moneydisposable.std().item()
            ability_raw_std = main_state.ability.std().item()

            # Extract normalizer statistics for comparison
            if "moneydisposalbe" in normalizer._stats:  # Note: typo in key from environment.py
                money_norm_stats = normalizer._stats["moneydisposalbe"]
                money_norm_mean = money_norm_stats.mean.mean().item()
                money_norm_std = torch.sqrt(
                    money_norm_stats.M2 / torch.clamp(money_norm_stats.count - 1, min=1.0)
                ).mean().item()
                money_norm_count = money_norm_stats.count.mean().item()
            else:
                money_norm_mean = 0.0
                money_norm_std = 1.0
                money_norm_count = 0.0

            if "ability" in normalizer._stats:
                ability_norm_stats = normalizer._stats["ability"]
                ability_norm_mean = ability_norm_stats.mean.mean().item()
                ability_norm_std = torch.sqrt(
                    ability_norm_stats.M2 / torch.clamp(ability_norm_stats.count - 1, min=1.0)
                ).mean().item()
                ability_norm_count = ability_norm_stats.count.mean().item()
            else:
                ability_norm_mean = 0.0
                ability_norm_std = 1.0
                ability_norm_count = 0.0

            # Compute discrepancy between normalizer and reality
            money_mean_discrepancy = money_raw_mean - money_norm_mean
            money_std_discrepancy = money_raw_std - money_norm_std
            ability_mean_discrepancy = ability_raw_mean - ability_norm_mean
            ability_std_discrepancy = ability_raw_std - ability_norm_std

            # ==== EXPERIMENT 2: Ability-Money Correlation Analysis ====
            # Compute correlation: ability[t] vs money[t] (contemporaneous)
            corr_ability_money_current = np.corrcoef(current_ability, current_money)[0, 1]
            # Compute correlation: ability[t] vs labor[t] (contemporaneous)
            corr_ability_labor_current = np.corrcoef(current_ability, current_labor)[0, 1]

            # Compute lagged correlation: ability[t-1] vs money[t]
            # This captures: "high ability yesterday → high income yesterday → high money today"
            if prev_ability is not None and prev_money is not None:
                corr_ability_money_lagged = np.corrcoef(prev_ability, current_money)[0, 1]
                # Also check: ability[t] vs money[t-1] (reverse lag)
                corr_ability_current_money_lagged = np.corrcoef(current_ability, prev_money)[0, 1]
            else:
                corr_ability_money_lagged = 0.0
                corr_ability_current_money_lagged = 0.0

            # Prepare scatter plot data for wandb (subsample for visualization)
            # Sample 500 points to avoid overwhelming wandb
            n_samples = min(500, len(current_ability))
            sample_indices = np.random.choice(len(current_ability), n_samples, replace=False)
            ability_sample = current_ability[sample_indices]
            money_sample = current_money[sample_indices]
            labor_sample = current_labor[sample_indices]

            # VERIFY BUDGET CONSTRAINT: consumption[t] + savings[t+1] = money_disposable[t]
            money_disposable_t_mean_val = money_disposable_t.mean().item()
            savings_t_mean_val = savings_t.mean().item()
            sav_plus_cons_val = savings_t_mean_val + consumption_mean_val
            budget_diff_val = money_disposable_t_mean_val - sav_plus_cons_val
            budget_ratio_val = sav_plus_cons_val / money_disposable_t_mean_val if abs(money_disposable_t_mean_val) > 1e-8 else 0.0

            # Agent actions for wandb histograms (extract before deletion)
            mu_t_numpy = mu_t.detach().cpu().numpy().flatten()
            savings_ratio_t_numpy = savings_ratio_t.detach().cpu().numpy().flatten()
            mu_t_mean_val = mu_t.mean().item()
            savings_ratio_t_mean_val = savings_ratio_t.mean().item()

        # CRITICAL: Clear temporary variables to prevent memory leaks
        # Delete tensors that have computational graphs attached
        del temp_state, parallel_A, parallel_B, outcomes_A, outcomes_B
        del consumption_t, labor_t, savings_ratio_t, mu_t, wage_t, ret_t, money_disposable_t, savings_t
        del consumption_A_tp1, consumption_B_tp1, ibt, ability_t
        del income_before_tax_A_tp1, income_before_tax_B_tp1
        # Delete normalized features (no gradient, but still takes memory)
        del normalized_features
        # Delete all individual losses from the loss dict
        del loss 

        # Now log using the extracted scalar values
        if step % config.training.display_step == 0 or step == 1:
            print(f"\nStep {step}/{config.training.training_steps}")
            print(f"  Loss: {loss_total_val:.4f} (fb={loss_fb_val:.4f}, euler={loss_aux_mu_val:.4f}, labor={loss_labor_val:.4f})")
            print(f"  Agent Actions:")
            print(f"    - Mean mu (consumption/money): {mu_t_mean_val:.3f}")
            print(f"    - Mean savings ratio: {savings_ratio_t_mean_val:.3f}")
            print(f"  State Statistics:")
            print(f"    - Mean consumption: {consumption_mean_val:.3f}")
            print(f"    - Mean labor: {labor_mean_val:.3f}")
            print(f"    - Mean savings: {savings_mean_val:.3f}")
            print(f"    - Mean ability: {ability_mean_val:.3f}")
            print(f"    - Market wage: {wage_mean_val:.3f}")
            print(f"    - Market return: {ret_mean_val:.4f}")
            print(f"  Normalized Network Inputs (debugging):")
            print(f"    - Normalized money: mean={normalized_money_mean:.3f}, std={normalized_money_std:.3f}")
            print(f"    - Normalized ability: mean={normalized_ability_mean:.3f}, std={normalized_ability_std:.3f}")
            print(f"    - Raw money: mean={money_raw_mean:.3f}, std={money_raw_std:.3f}")
            print(f"    - Raw ability: mean={ability_raw_mean:.3f}, std={ability_raw_std:.3f}")
            print(f"  Normalizer Health Check:")
            print(f"    - Money: normalizer_mean={money_norm_mean:.3f}, actual_mean={money_raw_mean:.3f}, discrepancy={money_mean_discrepancy:.3f}")
            print(f"    - Money: normalizer_std={money_norm_std:.3f}, actual_std={money_raw_std:.3f}, discrepancy={money_std_discrepancy:.3f}")
            print(f"    - Ability: normalizer_mean={ability_norm_mean:.3f}, actual_mean={ability_raw_mean:.3f}, discrepancy={ability_mean_discrepancy:.3f}")
            print(f"    - Ability: normalizer_std={ability_norm_std:.3f}, actual_std={ability_raw_std:.3f}, discrepancy={ability_std_discrepancy:.3f}")
            print(f"    - Normalizer update count: money={money_norm_count:.0f}, ability={ability_norm_count:.0f}")
            print(f"  Experiment 2: Ability-Money & Ability-Labor Correlation:")
            print(f"    - Corr(ability[t], money[t]): {corr_ability_money_current:.4f}")
            print(f"    - Corr(ability[t], labor[t]): {corr_ability_labor_current:.4f}")
            if prev_ability is not None:
                print(f"    - Corr(ability[t-1], money[t]): {corr_ability_money_lagged:.4f}")
                print(f"    - Corr(ability[t], money[t-1]): {corr_ability_current_money_lagged:.4f}")
            print(f"  Budget Constraint Verification:")
            print(f"    - Money disposable[t]: {money_disposable_t_mean_val:.3f}")
            print(f"    - Consumption[t]: {consumption_mean_val:.3f}")
            print(f"    - Savings[t+1]: {savings_t_mean_val:.3f}")
            print(f"    - Savings + Consumption: {sav_plus_cons_val:.3f}")
            print(f"    - Difference (should be ~0): {budget_diff_val:.6f}")
            print(f"    - Ratio (should be ~1.0): {budget_ratio_val:.6f}")

        # Log metrics to wandb
        if wandb.run:
            wandb.log({
                "loss/total": loss_total_val,
                "loss/fb": loss_fb_val,
                "loss/aux_mu": loss_aux_mu_val,
                "loss/labor_foc": loss_labor_val,
                "state/consumption_mean": consumption_mean_val,
                "state/labor_mean": labor_mean_val,
                "state/savings_mean": savings_mean_val,
                "state/ability_mean": ability_mean_val,
                "state/money_disposable_mean": money_disposable_t_mean_val,
                "state/savings_plus_consumption": sav_plus_cons_val,
                "state/budget_difference": budget_diff_val,
                "state/budget_ratio": budget_ratio_val,
                "market/wage": wage_mean_val,
                "market/return": ret_mean_val,
                # Agent actions
                "actions/mu_mean": mu_t_mean_val,
                "actions/savings_ratio_mean": savings_ratio_t_mean_val,
                "actions/mu_distribution": wandb.Histogram(mu_t_numpy),
                "actions/savings_ratio_distribution": wandb.Histogram(savings_ratio_t_numpy),
                # Normalized network inputs (for debugging)
                "debug/normalized_money_mean": normalized_money_mean,
                "debug/normalized_ability_mean": normalized_ability_mean,
                "debug/normalized_money_std": normalized_money_std,
                "debug/normalized_ability_std": normalized_ability_std,
                "debug/raw_money_mean": money_raw_mean,
                "debug/raw_ability_mean": ability_raw_mean,
                "debug/raw_money_std": money_raw_std,
                "debug/raw_ability_std": ability_raw_std,
                # Normalizer statistics
                "normalizer/money_mean": money_norm_mean,
                "normalizer/money_std": money_norm_std,
                "normalizer/money_count": money_norm_count,
                "normalizer/ability_mean": ability_norm_mean,
                "normalizer/ability_std": ability_norm_std,
                "normalizer/ability_count": ability_norm_count,
                # Discrepancy metrics (RED FLAG if these are large!)
                "normalizer/money_mean_discrepancy": money_mean_discrepancy,
                "normalizer/money_std_discrepancy": money_std_discrepancy,
                "normalizer/ability_mean_discrepancy": ability_mean_discrepancy,
                "normalizer/ability_std_discrepancy": ability_std_discrepancy,
                # Relative discrepancy (percentage)
                "normalizer/money_mean_drift_percent": abs(money_mean_discrepancy) / max(abs(money_norm_mean), 1e-6) * 100,
                "normalizer/ability_mean_drift_percent": abs(ability_mean_discrepancy) / max(abs(ability_norm_mean), 1e-6) * 100,
                # Experiment 2: Ability-Money & Ability-Labor Correlation
                "experiment2/corr_ability_money_current": corr_ability_money_current,
                "experiment2/corr_ability_money_lagged": corr_ability_money_lagged,
                "experiment2/corr_ability_current_money_lagged": corr_ability_current_money_lagged,
                "experiment2/corr_ability_labor_current": corr_ability_labor_current,
                "step": step
            })

            # Scatter plots: Create less frequently to avoid overhead (every save_interval steps)
            if step % config.training.save_interval == 0 or step == 1:
                wandb.log({
                    "experiment2/ability_vs_money_scatter": wandb.plot.scatter(
                        wandb.Table(
                            data=[[a, m] for a, m in zip(ability_sample, money_sample)],
                            columns=["ability[t]", "money[t]"]
                        ),
                        "ability[t]", "money[t]",
                        title=f"Ability vs Money (Step {step})"
                    ),
                    "experiment2/ability_vs_labor_scatter": wandb.plot.scatter(
                        wandb.Table(
                            data=[[a, l] for a, l in zip(ability_sample, labor_sample)],
                            columns=["ability[t]", "labor[t]"]
                        ),
                        "ability[t]", "labor[t]",
                        title=f"Ability vs Labor (Step {step})"
                    ),
                    "step": step
                })

            # Store current state for next iteration's lagged correlation
            prev_ability = current_ability.copy()
            prev_money = current_money.copy()

        # --- 5. Save checkpoints periodically (Example) ---
        if step % config.training.save_interval == 0:
            print(f"Saving checkpoint at step {step}...")
            torch.save(policy_net.state_dict(), os.path.join(weights_dir, f"model_step_{step}.pt"))
            torch.save(main_state, os.path.join(states_dir, f"state_step_{step}.pt"))
            normalizer.save(os.path.join(normalizer_dir, f"norm_step_{step}.pt"))

    print("\nTraining loop finished.")
    # --- 6. Final save (Example) ---
    print("Saving final model...")
    torch.save(policy_net.state_dict(), os.path.join(weights_dir, "model_final.pt"))
