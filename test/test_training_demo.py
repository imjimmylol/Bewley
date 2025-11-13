# test_training_demo.py
"""
Demo script to test the training loop with environment stepping.
This shows how the training loop works without implementing actual loss functions.
"""

import torch
from tqdm import tqdm 
from src.utils.configloader import load_configs, dict_to_namespace, compute_derived_params
from src.train import initialize_env_state, train
from src.environment import EconomyEnv
from src.normalizer import RunningPerAgentWelford
from src.models.model import FiLMResNet2In


def test_training_step():
    """Test a single training step without wandb."""
    print("\n" + "="*70)
    print("DEMO: Training Loop with Environment Stepping")
    print("="*70)

    # Load config
    config_dict = load_configs(['config/default.yaml'])
    config_dict = compute_derived_params(config_dict)

    # Override for faster demo
    config_dict['training']['training_steps'] = 100000
    config_dict['training']['display_step'] = 1000
    config_dict['training']['batch_size'] = 256  # Smaller batch for demo
    config_dict['training']['agents'] = 87     # Fewer agents for demo

    config = dict_to_namespace(config_dict)

    # Initialize components
    device = torch.device("cpu")
    normalizer = RunningPerAgentWelford(batch_dim=0, agent_dim=1)
    env = EconomyEnv(config, normalizer, device)

    policy_net = FiLMResNet2In(
        state_dim=2*config.training.agents+2,
        cond_dim=5,
        output_dim=3
    ).to(device)

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=float(config.training.learning_rate))

    print(f"\n✓ Setup complete")
    print(f"  - Batch size: {config.training.batch_size}")
    print(f"  - Agents: {config.training.agents}")
    print(f"  - Training steps: {config.training.training_steps}")

    # Initialize state
    main_state = initialize_env_state(config, device)

    print(f"\n✓ Initial state:")
    print(f"  - Mean money: {main_state.moneydisposable.mean():.3f}")
    print(f"  - Mean savings: {main_state.savings.mean():.3f}")
    print(f"  - Mean ability: {main_state.ability.mean():.3f}")

    # Training loop
    print("\n" + "-"*70)
    print("Starting training loop...")
    print("-"*70)
    total_steps = config.training.training_steps

    for step in tqdm(range(1, total_steps + 1), total=total_steps, desc="Training", ncols=100):
        # Step the environment
        main_state, temp_state, (parallel_A, outcomes_A), (parallel_B, outcomes_B) = env.step(
            main_state=main_state,
            policy_net=policy_net,
            deterministic=False,
            update_normalizer=True,
            commit_strategy="random"
        )

        # Extract variables
        consumption_t = temp_state.consumption
        labor_t = temp_state.labor
        savings_ratio_t = temp_state.savings_ratio
        mu_t = temp_state.mu
        wage_t = temp_state.wage
        ret_t = temp_state.ret
        money_disposable_t = temp_state.money_disposable

        # Placeholder losses (zeros with gradient)
        fb_loss = torch.tensor(0.0, device=device, requires_grad=True)
        euler_loss = torch.tensor(0.0, device=device, requires_grad=True)
        labor_foc_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_loss = fb_loss + euler_loss + labor_foc_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Extract scalars for logging BEFORE deleting tensors
        if step % config.training.display_step == 0 or step == 1:
            loss_val = total_loss.item()
            consumption_mean = consumption_t.mean().item()
            consumption_std = consumption_t.std().item()
            labor_mean = labor_t.mean().item()
            labor_std = labor_t.std().item()
            savings_mean = main_state.savings.mean().item()
            savings_std = main_state.savings.std().item()
            ability_mean = main_state.ability.mean().item()
            ability_std = main_state.ability.std().item()
            wage_mean = wage_t.mean().item()
            ret_mean = ret_t.mean().item()

        # Clear temporary variables to prevent memory leaks
        del consumption_t, labor_t, savings_ratio_t, mu_t, wage_t, ret_t, money_disposable_t
        del fb_loss, euler_loss, labor_foc_loss, total_loss

        # Logging
        if step % config.training.display_step == 0 or step == 1:
            print(f"\nStep {step}/{config.training.training_steps}")
            print(f"  Loss: {loss_val:.4f}")
            print(f"  State:")
            print(f"    - Consumption: {consumption_mean:.3f} (std: {consumption_std:.3f})")
            print(f"    - Labor: {labor_mean:.3f} (std: {labor_std:.3f})")
            print(f"    - Savings: {savings_mean:.3f} (std: {savings_std:.3f})")
            print(f"    - Ability: {ability_mean:.3f} (std: {ability_std:.3f})")
            print(f"  Market:")
            print(f"    - Wage: {wage_mean:.3f}")
            print(f"    - Return: {ret_mean:.4f}")

    print("\n" + "="*70)
    print("✓ Demo completed successfully!")
    print("="*70)
    print("\nKey observations:")
    print("  1. Environment steps through MainState → TemporaryState → ParallelStates")
    print("  2. Policy network receives normalized features and outputs actions")
    print("  3. Market equilibrium (wage, return) is computed from aggregate decisions")
    print("  4. Losses would be computed from these variables (currently placeholder zeros)")
    print("  5. Gradients flow back through policy network and parameters are updated")
    print(f"\nFinal state after {config.training.training_steps} steps:")
    print(f"  - Mean savings: {main_state.savings.mean():.3f}")
    print(f"  - Mean ability: {main_state.ability.mean():.3f}")
    print(f"  - Mean money: {main_state.moneydisposable.mean():.3f}")


if __name__ == "__main__":
    test_training_step()
