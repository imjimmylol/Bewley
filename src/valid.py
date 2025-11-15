# src/valid.py
"""
Validation utilities for loading and verifying saved checkpoints.
"""

import os
import torch
from src.normalizer import RunningPerAgentWelford
from src.models.model import FiLMResNet2In


def load_checkpoint(checkpoint_dir: str, step: int, config, device="cpu"):
    """
    Load model, normalizer, and optionally state from checkpoint.

    Args:
        checkpoint_dir: Base checkpoint directory (e.g., "checkpoints/run_name")
        step: Training step to load
        config: Configuration object
        device: Device to load to

    Returns:
        Tuple of (policy_net, normalizer, state)
        state will be None if not found
    """
    # Load policy network
    weights_path = os.path.join(checkpoint_dir, "weights", f"model_step_{step}.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    policy_net = FiLMResNet2In(
        state_dim=2*config.training.agents+2,
        cond_dim=5,
        output_dim=3
    ).to(device)
    policy_net.load_state_dict(torch.load(weights_path, map_location=device))
    policy_net.eval()
    print(f"✓ Loaded model from {weights_path}")

    # Load normalizer
    normalizer_path = os.path.join(checkpoint_dir, "normalizer", f"norm_step_{step}.pt")
    if not os.path.exists(normalizer_path):
        raise FileNotFoundError(f"Normalizer not found at {normalizer_path}")

    normalizer = RunningPerAgentWelford.from_file(normalizer_path, device=device)
    print(f"✓ Loaded normalizer from {normalizer_path}")

    # Try to load state if available
    state_path = os.path.join(checkpoint_dir, "states", f"state_step_{step}.pt")
    state = None
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)

        # Move all tensors in MainState to the correct device
        # MainState is a dataclass, so we need to move each tensor attribute
        if hasattr(state, 'moneydisposable'):
            state.moneydisposable = state.moneydisposable.to(device)
        if hasattr(state, 'savings'):
            state.savings = state.savings.to(device)
        if hasattr(state, 'ability'):
            state.ability = state.ability.to(device)
        if hasattr(state, 'ret'):
            # ret might be a scalar, handle both cases
            if torch.is_tensor(state.ret):
                state.ret = state.ret.to(device)
        if hasattr(state, 'tax_params'):
            state.tax_params = state.tax_params.to(device)
        if hasattr(state, 'is_superstar_vA'):
            state.is_superstar_vA = state.is_superstar_vA.to(device)
        if hasattr(state, 'is_superstar_vB'):
            state.is_superstar_vB = state.is_superstar_vB.to(device)
        if hasattr(state, 'ability_history_vA') and state.ability_history_vA is not None:
            state.ability_history_vA = state.ability_history_vA.to(device)
        if hasattr(state, 'ability_history_vB') and state.ability_history_vB is not None:
            state.ability_history_vB = state.ability_history_vB.to(device)

        print(f"✓ Loaded state from {state_path}")
    else:
        print(f"⚠ No saved state found at {state_path}")

    return policy_net, normalizer, state
