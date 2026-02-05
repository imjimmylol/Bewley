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
from src.monitoring import TrainingMonitor
from src.visualization import (
    prepare_data_for_plotting,
    plot_decision_rules_scatter,
    plot_binned_decision_rules,
    plot_state_distributions,
    plot_all_decision_rules
)
from src.policy_evaluation import (
    HistoricalRanges,
    PolicyEvaluator,
    collect_ranges_from_step
)
from src.replay_buffer.prioritized_buffer import PrioritizedReplayBuffer
from src.replay_buffer.experience import snapshot_main_state, pack_experience, replay_step
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

    # Initialize ability using bounded distribution (prevents explosion)
    # Use IQ-like distribution: realistic, bounded, prevents model collapse
    from src.ability_init import initialize_ability

    # Get initialization method from config (default: iq_like)
    init_method = getattr(config.initial_state, 'ability_init_method', 'iq_like')

    if init_method == 'iq_like':
        # Recommended: bounded log-normal like real ability distribution
        ability_mean = getattr(config.initial_state, 'ability_mean', 1.0)
        ability_cv = getattr(config.initial_state, 'ability_cv', 0.3)
        ability_min = getattr(config.initial_state, 'ability_min', 0.3)
        ability_max = getattr(config.initial_state, 'ability_max', 3.0)

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
        ret = ret_tensor,
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
    # Per-agent mode: each agent has its own normalization statistics
    # normalizer = RunningPerAgentWelford(batch_dim=0, agent_dim=1)

    # Global mode: all agents share the same mean/std statistics
    # This ensures decision rule evaluation works correctly for any (m_t, v_t) combination
    normalizer = RunningPerAgentWelford(batch_dim=0, agent_dim=None)
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

    # --- Initialize Prioritized Experience Replay Buffer ---
    per_config = getattr(config, 'prioritized_exp_replay', None)
    use_per = per_config is not None and getattr(per_config, 'enabled', False)

    if use_per:
        replay_buffer = PrioritizedReplayBuffer(
            capacity=per_config.buffer_size,
            alpha=per_config.alpha,
            beta_start=per_config.beta_start,
            beta_end=per_config.beta_end,
            beta_annealing_steps=per_config.beta_annealing_steps,
            epsilon=per_config.epsilon
        )
        replay_period = per_config.replay_period
        replay_batch_size = per_config.batch_size
        replay_buffer_dir = os.path.join(base_checkpoint_dir, "replay_buffer")
        os.makedirs(replay_buffer_dir, exist_ok=True)
        print(f"✓ Prioritized Experience Replay enabled")
        print(f"  - Buffer capacity: {per_config.buffer_size}")
        print(f"  - Alpha: {per_config.alpha}, Beta: {per_config.beta_start} → {per_config.beta_end}")
        print(f"  - Replay period: every {replay_period} steps")
    else:
        replay_buffer = None
        print("✓ Prioritized Experience Replay disabled")
    
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
        # ==== SNAPSHOT MAIN STATE (for PER) ====
        # Capture state BEFORE env.step() for experience replay
        if use_per:
            main_state_snapshot = snapshot_main_state(main_state)

        # ==== STEP THE ENVIRONMENT ====
        # This performs the full 4-step workflow:
        # 1. Agents observe MainState[t] and act
        # 2. Create TemporaryState with realized outcomes
        # 3. Transition to ParallelState A and B with different shocks
        # 4. Compute outcomes for both branches, choose one to commit

        main_state, temp_state, (parallel_A, outcomes_A), (parallel_B, outcomes_B), chosen_branch = env.step(
            main_state=main_state,
            policy_net=policy_net,
            deterministic=False,
            fix=fix_ability,  # Read from config: complete vs incomplete markets
            update_normalizer=True,
            commit_strategy="random"
        )
        # ==== STORE EXPERIENCE IN REPLAY BUFFER ====
        if use_per:
            experience = pack_experience(
                main_state_snapshot,
                parallel_A,
                parallel_B,
                committed_branch=chosen_branch,
                step=step
            )
            replay_buffer.add(experience)

        if use_per and step % replay_period == 0 and len(replay_buffer) >= replay_batch_size:
            
            beta = replay_buffer.compute_beta(step)
            optimizer.zero_grad()
            # _replay_iter Loop corresponding to gradient_steps (Line 8 of Algorithm 1)
            for _replay_iter in range(per_config.gradient_steps):
                experiences, indices, weights = replay_buffer.sample(replay_batch_size, beta)
                weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

                # Compute losses for the mini-batch of experiences
                updated_priorities = []
                for exp, is_weight in zip(experiences, weights_tensor):
                    replay_temp_state, replay_outcomes_A, replay_outcomes_B = replay_step(
                        exp, env, policy_net
                    )

                    replay_loss = loss_calculator.compute_all_losses(
                        consumption_t=replay_temp_state.consumption,
                        labor_t=replay_temp_state.labor,
                        ibt=replay_temp_state.income_before_tax,
                        savings_ratio_t=replay_temp_state.savings_ratio,
                        mu_t=replay_temp_state.mu,
                        wage_t=replay_temp_state.wage,
                        ret_t=replay_temp_state.ret,
                        money_disposable_t=replay_temp_state.money_disposable,
                        ability_t=replay_temp_state.ability,
                        consumption_A_tp1=replay_outcomes_A["consumption"],
                        consumption_B_tp1=replay_outcomes_B["consumption"],
                        ibt_A_tp1=replay_outcomes_A["income_before_tax"],
                        ibt_B_tp1=replay_outcomes_B["income_before_tax"]
                    )

                    weithed_loss = (replay_loss["total"]*is_weight) / (replay_batch_size*per_config.gradient_steps)
                    weithed_loss.backward()
                    updated_priorities.append(replay_loss["total"].item() + replay_buffer.epsilon)

                # Update priorities in the replay buffer
                replay_buffer.update_priorities(indices, updated_priorities)
            # Backpropagate and optimize the policy network
            optimizer.step()
            

        # ==== MONITORING: Log metrics, correlations, and debug info ====
        monitor.log_step(step, main_state, temp_state, loss)

        # ==== COLLECT HISTORICAL RANGES for synthetic grid evaluation ====
        # track_per_agent=True enables agent-specific x-axis ranges in visualization
        historical_ranges = collect_ranges_from_step(
            temp_state, main_state, historical_ranges, track_per_agent=True
        )

        # ==== VISUALIZATION: Generate decision rule plots at checkpoint intervals ====
        if step % config.training.save_interval == 0:
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
                    plots=["A1", "A1-1", "B1"]
                )

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

    print("\nTraining loop finished.")
    # --- 6. Final save (Example) ---
    print("Saving final model...")
    torch.save(policy_net.state_dict(), os.path.join(weights_dir, "model_final.pt"))
