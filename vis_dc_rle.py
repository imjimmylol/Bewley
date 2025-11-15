#!/usr/bin/env python3
"""
Visualize decision rules from saved checkpoints.

This script loads a checkpoint and visualizes the policy network's decision rules.
"""

import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

from src.valid import load_checkpoint
from src.environment import EconomyEnv
from src.train import initialize_env_state


def dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d


def simulation(policy_net, normalizer, env, config, device, initial_state):
    """
    Visualize decision rules.

    TODO: Implement your own visualization logic here.

    Args:
        policy_net: Loaded policy network
        normalizer: Loaded normalizer
        env: Environment instance
        config: Configuration object
        device: Device
        initial_state: Initial MainState (either loaded from checkpoint or newly initialized)
    """
    print("\n" + "="*60)
    print("VISUALIZING DECISION RULES")
    print("="*60)

    # Use provided initial state
    state = initial_state

    # Run a few steps to collect data
    n_steps = 20
    trajectories = {
        'consumption': [],
        'labor': [],
        'savings': [],
        'mu': [],
        'savings_ratio': [],
        'ability': [],
        'money_disposable': [],
        'wage': [],
        'income_before_tax': [],
    }

    print(f"\nRunning {n_steps} environment steps...")
    print(f"Initial state:")
    print(f"  Mean money_disposable: {state.moneydisposable.mean().item():.4f}")
    print(f"  Mean savings: {state.savings.mean().item():.4f}")
    print(f"  Mean ability: {state.ability.mean().item():.4f}")
    print()

    with torch.no_grad():
        for t in range(n_steps):
            state, temp_state, (parallel_A, outcomes_A), (parallel_B, outcomes_B) = env.step(
                main_state=state,
                policy_net=policy_net,
                deterministic=True,
                update_normalizer=False,
                commit_strategy="A"
            )

            # Collect trajectories
            trajectories['consumption'].append(temp_state.consumption.cpu().numpy())
            trajectories['labor'].append(temp_state.labor.cpu().numpy())
            trajectories['savings'].append(state.savings.cpu().numpy())
            trajectories['mu'].append(temp_state.mu.cpu().numpy())
            trajectories['savings_ratio'].append(temp_state.savings_ratio.cpu().numpy())
            trajectories['ability'].append(state.ability.cpu().numpy())
            trajectories['money_disposable'].append(temp_state.money_disposable.cpu().numpy())
            trajectories['wage'].append(temp_state.wage.cpu().numpy())
            trajectories['income_before_tax'].append(temp_state.income_before_tax.cpu().numpy())

            # Print diagnostic every 5 steps
            if t % 5 == 0 or t == 0:
                print(f"Step {t}:")
                print(f"  Mean consumption: {temp_state.consumption.mean().item():.4f}")
                print(f"  Mean money_disposable: {temp_state.money_disposable.mean().item():.4f}")
                print(f"  Mean savings (t+1): {state.savings.mean().item():.4f}")
                print(f"  Mean ability: {state.ability.mean().item():.4f}")
                print(f"  Mean wage: {temp_state.wage.mean().item():.4f}")

    # Convert to numpy arrays
    for key in trajectories:
        trajectories[key] = np.array(trajectories[key])

    print("✓ Data collection complete")

    # ========================================================================
    # TODO: Add your custom visualization here
    # ========================================================================

    # Example: Basic time series plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Plot mean consumption over time
    axes[0, 0].plot(trajectories['consumption'].mean(axis=(1, 2)))
    axes[0, 0].set_title('Mean Consumption Over Time')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Consumption')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot mean labor over time
    axes[0, 1].plot(trajectories['labor'].mean(axis=(1, 2)))
    axes[0, 1].set_title('Mean Labor Over Time')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Labor')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot mean savings over time
    axes[0, 2].plot(trajectories['savings'].mean(axis=(1, 2)))
    axes[0, 2].set_title('Mean Savings Over Time')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Savings')
    axes[0, 2].grid(True, alpha=0.3)

    # Plot mean money_disposable over time
    axes[1, 0].plot(trajectories['money_disposable'].mean(axis=(1, 2)))
    axes[1, 0].set_title('Mean Money Disposable Over Time')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Money Disposable')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot mean ability over time
    axes[1, 1].plot(trajectories['ability'].mean(axis=(1, 2)))
    axes[1, 1].set_title('Mean Ability Over Time')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Ability')
    axes[1, 1].grid(True, alpha=0.3)

    # Plot mean wage over time
    axes[1, 2].plot(trajectories['wage'].mean(axis=(1, 2)))
    axes[1, 2].set_title('Mean Wage Over Time')
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('Wage')
    axes[1, 2].grid(True, alpha=0.3)

    # Plot distribution of mu (final step)
    axes[2, 0].hist(trajectories['mu'][-1].flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[2, 0].set_title(f'Distribution of mu (final step)')
    axes[2, 0].set_xlabel('mu')
    axes[2, 0].set_ylabel('Frequency')
    axes[2, 0].grid(True, alpha=0.3)

    # Plot distribution of savings ratio (final step)
    axes[2, 1].hist(trajectories['savings_ratio'][-1].flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[2, 1].set_title(f'Distribution of Savings Ratio (final step)')
    axes[2, 1].set_xlabel('Savings Ratio')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].grid(True, alpha=0.3)

    # Plot distribution of consumption (final step)
    axes[2, 2].hist(trajectories['consumption'][-1].flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[2, 2].set_title(f'Distribution of Consumption (final step)')
    axes[2, 2].set_xlabel('Consumption')
    axes[2, 2].set_ylabel('Frequency')
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    output_path = "decision_rules_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    plt.close()

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (Final Step)")
    print("="*60)
    print(f"  Mean consumption:     {trajectories['consumption'][-1].mean():.4f}")
    print(f"  Mean labor:           {trajectories['labor'][-1].mean():.4f}")
    print(f"  Mean savings:         {trajectories['savings'][-1].mean():.4f}")
    print(f"  Mean mu:              {trajectories['mu'][-1].mean():.4f}")
    print(f"  Mean savings ratio:   {trajectories['savings_ratio'][-1].mean():.4f}")
    print("="*60 + "\n")

    return trajectories


def make_decision_rules_inputs(
    state,
    batch_idx=0,
    agent_idx=0,
    vary_var="money_disposable",
    vary_range=(0.1, 10.0),
    n_points=50
):
    """
    Create a grid of states for decision rule visualization.

    Fixes one world (batch_idx) and varies one agent's characteristic
    while keeping all other agents in that world unchanged.

    Args:
        state: MainState object from checkpoint
        batch_idx: Which world/batch to analyze (default: 0)
        agent_idx: Which agent in that world to analyze (default: 0)
        vary_var: Variable to vary. Options:
                  - "money_disposable" (current wealth)
                  - "ability" (current productivity)
                  - "savings" (previous period savings)
        vary_range: (min, max) tuple for the variable range
        n_points: Number of grid points

    Returns:
        dict with keys:
            - 'states': List of MainState objects (length n_points)
            - 'values': numpy array of shape (n_points,) with varied values
            - 'metadata': dict with batch_idx, agent_idx, varied_var info
    """
    from src.env_state import MainState

    # Step 1: Validate inputs
    batch_size = state.moneydisposable.shape[0]
    n_agents = state.moneydisposable.shape[1]

    if batch_idx >= batch_size:
        raise ValueError(f"batch_idx={batch_idx} out of range. Only {batch_size} batches available.")

    if agent_idx >= n_agents:
        raise ValueError(f"agent_idx={agent_idx} out of range. Only {n_agents} agents available.")

    # Variable mapping: user-friendly names -> MainState attribute names
    var_mapping = {
        "money_disposable": "moneydisposable",  # Note: different naming!
        "ability": "ability",
        "savings": "savings"
    }

    if vary_var not in var_mapping:
        raise ValueError(f"vary_var must be one of {list(var_mapping.keys())}, got '{vary_var}'")

    attr_name = var_mapping[vary_var]

    # Step 2: Create grid of values to vary
    grid_values = np.linspace(vary_range[0], vary_range[1], n_points)

    # Step 3: Create grid of states
    grid_states = []

    for value in grid_values:
        # Step 3a: Clone the original state (deep copy all tensor attributes)
        new_state = _clone_main_state(state)

        # Step 3b: Modify ONLY the target agent in target batch
        # Get the attribute tensor (e.g., new_state.moneydisposable)
        attr_tensor = getattr(new_state, attr_name)

        # Modify the specific [batch_idx, agent_idx] entry
        attr_tensor[batch_idx, agent_idx] = torch.tensor(
            value,
            device=attr_tensor.device,
            dtype=attr_tensor.dtype
        )

        # Step 3c: Keep everything else unchanged (done automatically via cloning)

        grid_states.append(new_state)

    # Return with metadata
    return {
        'states': grid_states,
        'values': grid_values,
        'metadata': {
            'batch_idx': batch_idx,
            'agent_idx': agent_idx,
            'varied_var': vary_var,
            'vary_range': vary_range,
            'n_points': n_points,
            'batch_size': batch_size,
            'n_agents': n_agents
        }
    }



def evaluate_policy_on_grid(grid_result, policy_net, env):
    """
    Evaluate policy network on all grid states to extract decision rules.

    Args:
        grid_result: Dict returned by make_decision_rules_inputs
        policy_net: Trained policy network
        env: Environment instance (for feature preparation)

    Returns:
        dict with keys:
            # Policy outputs (decisions)
            - 'consumption': numpy array (n_points,) - consumption decisions
            - 'labor': numpy array (n_points,) - labor decisions
            - 'savings': numpy array (n_points,) - next period savings
            - 'mu': numpy array (n_points,) - Lagrange multipliers
            - 'savings_ratio': numpy array (n_points,) - savings ratios

            # Input state variables (for x-axis plotting)
            - 'money_disposable': numpy array (n_points,) - current wealth
            - 'ability': numpy array (n_points,) - current productivity
            - 'input_savings': numpy array (n_points,) - previous period savings

            - 'varied_values': numpy array (n_points,) - the specific variable that was varied
            - 'metadata': dict - metadata from grid_result
    """
    grid_states = grid_result['states']
    grid_values = grid_result['values']
    metadata = grid_result['metadata']

    batch_idx = metadata['batch_idx']
    agent_idx = metadata['agent_idx']

    # Storage for policy decisions (outputs)
    consumption_decisions = []
    labor_decisions = []
    savings_decisions = []
    mu_decisions = []
    savings_ratio_decisions = []

    # Storage for input state variables
    money_disposable_values = []
    ability_values = []
    input_savings_values = []

    print(f"\nEvaluating policy on {len(grid_states)} grid points...")

    with torch.no_grad():
        for i, state in enumerate(grid_states):
            # Extract input state variables for this grid point
            money_disposable = state.moneydisposable[batch_idx, agent_idx].item()
            ability = state.ability[batch_idx, agent_idx].item()
            input_savings = state.savings[batch_idx, agent_idx].item()

            money_disposable_values.append(money_disposable)
            ability_values.append(ability)
            input_savings_values.append(input_savings)

            # Use environment's _compute_agent_actions to get policy decisions
            # This handles feature preparation and normalization automatically
            actions = env._compute_agent_actions(
                state=state,
                policy_net=policy_net,
                update_normalizer=False  # Don't update during evaluation
            )

            # Extract decisions for the target agent in target batch
            # actions shape: (batch_size, n_agents, 3) where 3 = [mu, labor, savings_ratio]
            print(actions.keys())
            mu = actions["mu"][batch_idx, agent_idx].item()
            labor = actions["labor"][batch_idx, agent_idx].item()
            savings_ratio = actions["savings_ratio"][batch_idx, agent_idx].item()

            # Compute implied consumption and savings
            # From environment logic: consumption = mu * money_disposable
            consumption = money_disposable * (1-savings_ratio)
            savings = savings_ratio * money_disposable

            # Store policy decisions
            consumption_decisions.append(consumption)
            labor_decisions.append(labor)
            savings_decisions.append(savings)
            mu_decisions.append(mu)
            savings_ratio_decisions.append(savings_ratio)

    print(f"✓ Policy evaluation complete")

    return {
        # Policy outputs
        'consumption': np.array(consumption_decisions),
        'labor': np.array(labor_decisions),
        'savings': np.array(savings_decisions),
        'mu': np.array(mu_decisions),
        'savings_ratio': np.array(savings_ratio_decisions),

        # Input state variables
        'money_disposable': np.array(money_disposable_values),
        'ability': np.array(ability_values),
        'input_savings': np.array(input_savings_values),

        # Varied values (for convenience)
        'varied_values': grid_values,
        'metadata': metadata
    }


def plot_decision_rules(policy_results, save_path="decision_rules.png"):
    """
    Plot decision rules as a function of the varied variable.

    Args:
        policy_results: Dict returned by evaluate_policy_on_grid
        save_path: Path to save the plot (default: "decision_rules.png")
    """
    metadata = policy_results['metadata']
    varied_var = metadata['varied_var']
    x_values = policy_results['varied_values']

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Consumption policy
    axes[0, 0].plot(x_values, policy_results['consumption'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel(f"{varied_var}", fontsize=11)
    axes[0, 0].set_ylabel("Consumption", fontsize=11)
    axes[0, 0].set_title(f"Consumption Policy\n(Agent {metadata['agent_idx']}, Batch {metadata['batch_idx']})", fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Labor supply policy
    axes[0, 1].plot(x_values, policy_results['labor'], 'r-', linewidth=2)
    axes[0, 1].set_xlabel(f"{varied_var}", fontsize=11)
    axes[0, 1].set_ylabel("Labor", fontsize=11)
    axes[0, 1].set_title(f"Labor Supply Policy", fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Savings policy
    axes[1, 0].plot(x_values, policy_results['savings'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel(f"{varied_var}", fontsize=11)
    axes[1, 0].set_ylabel("Savings (next period)", fontsize=11)
    axes[1, 0].set_title(f"Savings Policy", fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Savings ratio
    axes[1, 1].plot(x_values, policy_results['savings_ratio'], 'm-', linewidth=2)
    axes[1, 1].set_xlabel(f"{varied_var}", fontsize=11)
    axes[1, 1].set_ylabel("Savings Ratio", fontsize=11)
    axes[1, 1].set_title(f"Savings Ratio Policy", fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)

    # Add overall title
    fig.suptitle(f"Decision Rules: Varying {varied_var}", fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle

    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Decision rules plot saved to: {save_path}")

    plt.close()


def _clone_main_state(state):
    """
    Clone a MainState by cloning all its tensor attributes.

    Args:
        state: MainState object

    Returns:
        new_state: Cloned MainState with independent tensor copies
    """
    from src.env_state import MainState

    # Clone all tensor attributes
    # Use .clone() for tensors, handle None for optional fields
    def clone_tensor(t):
        if t is None:
            return None
        elif torch.is_tensor(t):
            return t.clone()
        else:
            # Handle scalar (like ret if it's a float)
            return t

    new_state = MainState(
        moneydisposable=clone_tensor(state.moneydisposable),
        savings=clone_tensor(state.savings),
        ability=clone_tensor(state.ability),
        ret=clone_tensor(state.ret),
        tax_params=clone_tensor(state.tax_params),
        is_superstar_vA=clone_tensor(state.is_superstar_vA),
        is_superstar_vB=clone_tensor(state.is_superstar_vB),
        ability_history_vA=clone_tensor(state.ability_history_vA),
        ability_history_vB=clone_tensor(state.ability_history_vB)
    )

    return new_state

def main():
    parser = argparse.ArgumentParser(description="Visualize decision rules from checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Path to checkpoint directory (e.g., checkpoints/run_name)")
    parser.add_argument("--step", type=int, required=True,
                       help="Training step to load model weights from")
    parser.add_argument("--state_step", type=int, default=None,
                       help="Training step to load state from (default: same as --step)")
    parser.add_argument("--config", type=str, default="config/baseline.yaml",
                       help="Path to config file")

    args = parser.parse_args()

    # If state_step not specified, use the same as model step
    state_step = args.state_step if args.state_step is not None else args.step

    # Load config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = dict_to_namespace(config_dict)

    # Determine device
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    print(f"\n{'='*60}")
    print(f"DECISION RULE VISUALIZATION")
    print(f"{'='*60}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Model step:     {args.step}")
    print(f"State step:     {state_step}")
    print(f"Config:         {args.config}")
    print(f"Device:         {device}")
    print(f"{'='*60}\n")

    # Load model weights and normalizer from specified step
    policy_net, normalizer, _ = load_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        step=args.step,
        config=config,
        device=device
    )

    # Load state from potentially different step
    if state_step != args.step:
        print(f"Loading state from different step: {state_step}")
        _, _, state = load_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            step=state_step,
            config=config,
            device=device
        )
    else:
        # State already loaded with the model
        _, _, state = load_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            step=args.step,
            config=config,
            device=device
        )

    # Initialize environment
    env = EconomyEnv(config, normalizer, device=device)

    # Use saved state if available, otherwise initialize new state
    if state is not None:
        print(f"✓ Using saved state from checkpoint")
        print(f"  State batch size: {state.moneydisposable.shape[0]}")
        print(f"  State n_agents: {state.moneydisposable.shape[1]}")
        initial_state = state
    else:
        print("⚠ No saved state found, initializing new state")
        initial_state = initialize_env_state(config, device)

    # Example: Create grid of states varying money_disposable for Agent 0 in Batch 0
    print("\n" + "="*60)
    print("CREATING DECISION RULE GRID")
    print("="*60)

    grid_result = make_decision_rules_inputs(
        state=initial_state,
        batch_idx=0,
        agent_idx=0,
        vary_var="money_disposable",
        vary_range=(0.1, 10),
        n_points=50
    )

    print(f"✓ Created {len(grid_result['states'])} grid states")
    print(f"  Varying: {grid_result['metadata']['varied_var']}")
    print(f"  Range: {grid_result['metadata']['vary_range']}")
    print(f"  Batch: {grid_result['metadata']['batch_idx']}")
    print(f"  Agent: {grid_result['metadata']['agent_idx']}")
    print(f"  Grid values (first 5): {grid_result['values'][:5]}")
    print("="*60 + "\n")

    # Evaluate policy on all grid states
    policy_results = evaluate_policy_on_grid(grid_result, policy_net, env)

    # Print sample results
    print("\n" + "="*60)
    print("POLICY EVALUATION RESULTS (First 5 points)")
    print("="*60)
    print(f"{'Varied Var':<15} {'Consumption':<12} {'Labor':<12} {'Savings':<12}")
    print("-"*60)
    for i in range(min(5, len(policy_results['varied_values']))):
        print(f"{policy_results['varied_values'][i]:<15.4f} "
              f"{policy_results['consumption'][i]:<12.4f} "
              f"{policy_results['labor'][i]:<12.4f} "
              f"{policy_results['savings'][i]:<12.4f}")
    print("="*60 + "\n")

    # Plot the decision rules
    plot_decision_rules(
        policy_results,
        save_path=f"decision_rules_agent{grid_result['metadata']['agent_idx']}_batch{grid_result['metadata']['batch_idx']}.png"
    )

    # Run Simulation (optional)
    # trajectories = simulation(policy_net, normalizer, env, config, device, initial_state)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
