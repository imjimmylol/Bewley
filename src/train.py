# src/train.py
import os
import wandb
from datetime import datetime
import time
import torch
from src.env_state import MainState
from src.environment import EconomyEnv
from src.normalizer import RunningPerAgentWelford
from src.models.model import FiLMResNet2In
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
    tax_params = torch.tensor(list(tax_params_values.values()), dtype=torch.float32)
    tax_params = tax_params.repeat(batch_size, 1)

    moneydisposable=np.random.lognormal(0.1, 2.0, batch_size * n_agents).reshape(batch_size, n_agents)
    savings= np.random.lognormal(0.1, 2.0, batch_size * n_agents).reshape(batch_size, n_agents)
    ability = np.random.lognormal(0.5, 1.5, batch_size * n_agents).reshape(batch_size, n_agents)

    is_superstar_vA = np.zeros((batch_size, n_agents), dtype=bool)
    is_superstar_vB = np.zeros((batch_size, n_agents), dtype=bool)

    state = MainState(
        moneydisposable = torch.tensor(moneydisposable, dtype=torch.float32),
        savings = torch.tensor(savings, dtype=torch.float32),
        ability = torch.tensor(ability, dtype=torch.float32),
        ret = config.bewley_model.r,
        tax_params=tax_params,
        is_superstar_vA = torch.tensor(is_superstar_vA, dtype=torch.bool),
        is_superstar_vB = torch.tensor(is_superstar_vB, dtype=torch.bool),
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

    # --- 2. Initialize models, normalizers, etc. (Example) ---
    # from src.models import MyModel
    # from src.normalizer import RunningNormalizer
    # 
    # model = MyModel(config.model)
    # normalizer = RunningNormalizer()
    normalizer = RunningPerAgentWelford(batch_dim=0, agent_dim=1)
    env = BewleyEnvironment(config, normalizer, device="cpu")
    print("\n--- Final Configuration ---")
    print(f"Learning Rate: {config.training.learning_rate}")
    print(f"Beta: {config.bewley_model.beta}")
    if hasattr(config, 'shock') and hasattr(config.shock, 'v_min'):
        print(f"Shock v_min (computed): {config.shock.v_min:.4f}")
        print(f"Shock v_max (computed): {config.shock.v_max:.4f}")
    print("---------------------------\n")

    # --- 3. Initialize Environment State ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize state
    main_state = initialize_env_state(config, device)
    policy_net = FiLMResNet2In(
        state_dim=2*config.training.agents + 2, cond_dim=5,
        hidden_dim=128, num_res_blocks=3, dropout=0.1,
        output_dim=3)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=config.training.learning_rate)

    
    # --- 4. Training Loop ---
    print("Starting dummy training loop...")
    for step in range(1, config.training.training_steps + 1):
        # Dummy loss calculation
        loss = 1.0 / (step + 1e-5) + (1 - config.bewley_model.beta)
        
        if step % config.training.display_step == 0 or step == 1:
            print(f"Step {step}/{config.training.training_steps}, Loss: {loss:.4f}")

        # Log metrics to wandb
        if wandb.run:
            wandb.log({"loss": loss, "step": step})

        # --- 5. Save checkpoints periodically (Example) ---
        # if step % config.training.save_interval == 0:
        #     print(f"Saving checkpoint at step {step}...")
        #     # torch.save(model.state_dict(), os.path.join(weights_dir, f"model_step_{step}.pt"))
        #     # torch.save(normalizer.state_dict(), os.path.join(normalizer_dir, f"norm_step_{step}.pt"))
        
        time.sleep(0.001) # Simulate work

    print("\nDummy training loop finished.")
    # --- 6. Final save (Example) ---
    # print("Saving final model...")
    # torch.save(model.state_dict(), os.path.join(weights_dir, "model_final.pt"))
