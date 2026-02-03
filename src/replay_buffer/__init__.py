# src/replay_buffer/__init__.py
"""
Prioritized Experience Replay (PER) buffer implementation.

Exports:
    - PrioritizedReplayBuffer: Main buffer class for prioritized sampling
    - Experience: Dataclass for storing experiences
    - ShockOutcomes: Container for shock outcomes from one branch
    - snapshot_main_state: Create deep copy of MainState before env.step()
    - pack_experience: Convert MainState + shock outcomes to Experience
    - unpack_to_main_state: Convert Experience back to MainState
    - unpack_shock_outcomes: Extract shock outcomes from Experience
    - replay_step: Replay env.step() using stored Experience
"""

from src.replay_buffer.experience import (
    Experience,
    ShockOutcomes,
    snapshot_main_state,
    pack_experience,
    unpack_to_main_state,
    unpack_shock_outcomes,
    replay_step,
    compute_experience_memory_bytes,
    estimate_buffer_memory_gb,
)
from src.replay_buffer.prioritized_buffer import PrioritizedReplayBuffer, compute_beta
from src.replay_buffer.sum_tree import SumTree

__all__ = [
    "PrioritizedReplayBuffer",
    "compute_beta",
    "Experience",
    "ShockOutcomes",
    "snapshot_main_state",
    "pack_experience",
    "unpack_to_main_state",
    "unpack_shock_outcomes",
    "replay_step",
    "compute_experience_memory_bytes",
    "estimate_buffer_memory_gb",
    "SumTree",
]
