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

    moneydisposable=np.random.lognormal(0.1, 2.0, batch_size * n_agents).reshape(batch_size, n_agents)
    savings= np.random.lognormal(0.1, 2.0, batch_size * n_agents).reshape(batch_size, n_agents)

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

    # --- 4. Training Loop ---
    print("Starting training loop with environment stepping...")
    # total_steps = config.training.training_steps
    print(config.tax_params)
    total_steps = 5
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

        # Next period (t+1) outcomes from parallel branches
        consumption_A_tp1 = outcomes_A["consumption"]       # (B, A)
        income_before_tax_A_tp1 = outcomes_A["income_before_tax"]
        consumption_B_tp1 = outcomes_B["consumption"]       # (B, A)
        income_before_tax_B_tp1 = outcomes_B["income_before_tax"]

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
        total_loss.backward()

        # Optional: gradient clipping
        # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)

        optimizer.step()

        # ==== LOGGING ====
        # Extract scalar values BEFORE deleting tensors
        if step % config.training.display_step == 0 or step == 1 or wandb.run:
            # Convert to Python scalars to avoid holding tensor references
            loss_total_val = loss["total"].item()
            loss_fb_val = loss["fb"].item()
            loss_euler_val = loss["total"].item()
            loss_labor_val = labor_foc_loss.item()
            consumption_mean_val = consumption_t.mean().item()
            labor_mean_val = labor_t.mean().item()
            savings_mean_val = main_state.savings.mean().item()
            ability_mean_val = main_state.ability.mean().item()
            wage_mean_val = wage_t.mean().item()
            ret_mean_val = ret_t.mean().item()

        # CRITICAL: Clear temporary variables to prevent memory leaks
        # Delete tensors that have computational graphs attached
        del temp_state, parallel_A, parallel_B, outcomes_A, outcomes_B
        del consumption_t, labor_t, savings_ratio_t, mu_t, wage_t, ret_t, money_disposable_t
        del consumption_A_tp1, consumption_B_tp1
        del fb_loss, euler_loss, labor_foc_loss, total_loss

        # Now log using the extracted scalar values
        if step % config.training.display_step == 0 or step == 1:
            print(f"\nStep {step}/{config.training.training_steps}")
            print(f"  Loss: {loss_total_val:.4f} (fb={loss_fb_val:.4f}, euler={loss_euler_val:.4f}, labor={loss_labor_val:.4f})")
            print(f"  State Statistics:")
            print(f"    - Mean consumption: {consumption_mean_val:.3f}")
            print(f"    - Mean labor: {labor_mean_val:.3f}")
            print(f"    - Mean savings: {savings_mean_val:.3f}")
            print(f"    - Mean ability: {ability_mean_val:.3f}")
            print(f"    - Market wage: {wage_mean_val:.3f}")
            print(f"    - Market return: {ret_mean_val:.4f}")

        # Log metrics to wandb
        if wandb.run:
            wandb.log({
                "loss/total": loss_total_val,
                "loss/fb": loss_fb_val,
                "loss/euler": loss_euler_val,
                "loss/labor_foc": loss_labor_val,
                "state/consumption_mean": consumption_mean_val,
                "state/labor_mean": labor_mean_val,
                "state/savings_mean": savings_mean_val,
                "state/ability_mean": ability_mean_val,
                "market/wage": wage_mean_val,
                "market/return": ret_mean_val,
                "step": step
            })

        # --- 5. Save checkpoints periodically (Example) ---
        # if step % config.training.save_interval == 0:
        #     print(f"Saving checkpoint at step {step}...")
        #     torch.save(policy_net.state_dict(), os.path.join(weights_dir, f"model_step_{step}.pt"))
        #     torch.save(main_state, os.path.join(states_dir, f"state_step_{step}.pt"))
        #     normalizer.save(os.path.join(normalizer_dir, f"norm_step_{step}.pt"))

    print("\nTraining loop finished.")
    # --- 6. Final save (Example) ---
    # print("Saving final model...")
    # torch.save(model.state_dict(), os.path.join(weights_dir, "model_final.pt"))
