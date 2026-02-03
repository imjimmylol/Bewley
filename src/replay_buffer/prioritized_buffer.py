# src/replay_buffer/prioritized_buffer.py
"""
Prioritized Experience Replay Buffer.

Implements Algorithm 1 from Schaul et al. (2016) "Prioritized Experience Replay".
Uses a SumTree for O(log N) proportional sampling.
"""

import pickle
from typing import List, Tuple, Optional
import numpy as np

from src.replay_buffer.sum_tree import SumTree
from src.replay_buffer.experience import Experience, pack_experience, snapshot_main_state
from src.env_state import MainState, ParallelState


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer with proportional prioritization.

    Key features:
    - O(log N) sampling using SumTree
    - New experiences added with max priority (ensures they get sampled)
    - Priorities updated based on TD-error (loss) after replay
    - Importance sampling weights for unbiased gradient updates

    Usage:
        buffer = PrioritizedReplayBuffer(capacity=100000, alpha=0.6)

        # Online step: snapshot before env.step, add after with parallel states
        snapshot = snapshot_main_state(main_state)
        main_state, temp_state, (parallel_A, outcomes_A), (parallel_B, outcomes_B) = env.step(...)
        buffer.add(snapshot, parallel_A, parallel_B, committed_branch, step)

        # Replay phase: sample, replay_step, compute loss, update priorities
        experiences, indices, is_weights = buffer.sample(batch_size, beta)
        for exp, is_weight in zip(experiences, is_weights):
            _, temp_state, (pA, oA), (pB, oB) = replay_step(exp, env, policy_net, device)
            loss = loss_calculator.compute_all_losses(...)
            # accumulate weighted loss
        buffer.update_priorities(indices, losses)
    """

    def __init__(
        self,
        capacity: int,
        alpha: float,
        epsilon: float = 1e-6,
        device: str = "cpu"
    ):
        """
        Initialize PrioritizedReplayBuffer.

        Args:
            capacity: Maximum number of experiences to store
            alpha: Prioritization exponent (0=uniform, 1=full priority)
                   P(i) = p_i^alpha / sum_j(p_j^alpha)
            epsilon: Small constant to prevent zero priority
            device: Device for tensor operations during replay
        """
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.device = device

        # SumTree for priority-based sampling
        self.tree = SumTree(capacity)

        # Circular buffer for experiences
        self.data: List[Optional[Experience]] = [None] * capacity
        self.position = 0
        self.size = 0

    def add(
        self,
        main_state_snapshot: MainState,
        parallel_A: ParallelState,
        parallel_B: ParallelState,
        committed_branch: str,
        step: int
    ) -> None:
        """
        Add experience with MAX PRIORITY (Line 6 of Algorithm 1).

        New experiences are assigned: p_t = max_{i<t} p_i
        This ensures they get sampled at least once.
        Priority will be updated when experience is replayed.

        Args:
            main_state_snapshot: MainState snapshot taken BEFORE env.step()
            parallel_A: ParallelState from branch A (contains shock outcomes)
            parallel_B: ParallelState from branch B (contains shock outcomes)
            committed_branch: Which branch was committed ("A" or "B")
            step: Training step when collected
        """
        # Pack MainState + shock outcomes into Experience
        exp = pack_experience(main_state_snapshot, parallel_A, parallel_B, committed_branch, step)

        # Get max priority for new experience (Line 6 of Algorithm 1)
        # First experience gets priority 1.0 (Line 2)
        max_priority = self.tree.max_priority

        # Apply alpha exponent to priority
        priority = max_priority ** self.alpha

        # Add to tree (returns the data index where it was stored)
        data_idx = self.tree.add(priority)

        # Store experience in circular buffer
        self.data[data_idx] = exp

        # Update size
        if self.size < self.capacity:
            self.size += 1

    def sample(
        self,
        batch_size: int,
        beta: float
    ) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample batch proportional to priorities (Lines 9-10 of Algorithm 1).

        Sampling probability: P(i) = p_i^alpha / sum_j(p_j^alpha)
        IS weight: w_i = (N * P(i))^(-beta) / max_j(w_j)

        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent (anneals from beta_start to 1.0)

        Returns:
            experiences: List of Experience objects
            indices: np.ndarray of buffer indices (for priority update)
            is_weights: np.ndarray of importance sampling weights (normalized to [0,1])
        """
        experiences = []
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float64)

        # Divide priority range into segments for stratified sampling
        total_priority = self.tree.total
        segment_size = total_priority / batch_size

        for i in range(batch_size):
            # Sample uniformly within segment
            low = segment_size * i
            high = segment_size * (i + 1)
            value = np.random.uniform(low, high)

            # Get leaf from tree
            data_idx, priority = self.tree.get(value)

            # Store
            indices[i] = data_idx
            priorities[i] = priority
            experiences.append(self.data[data_idx])

        # Compute importance sampling weights (Line 10 of Algorithm 1)
        # P(i) = p_i / total (note: tree already stores p_i^alpha)
        sampling_probs = priorities / (total_priority + 1e-10)

        # w_i = (N * P(i))^(-beta)
        is_weights = (self.size * sampling_probs) ** (-beta)

        # Normalize by max weight for stability
        is_weights = is_weights / (is_weights.max() + 1e-10)

        # Clip for numerical stability
        is_weights = np.clip(is_weights, 0.0, 100.0)

        return experiences, indices, is_weights.astype(np.float32)

    def update_priorities(
        self,
        indices: np.ndarray,
        losses: np.ndarray
    ) -> None:
        """
        Update priorities after recomputing losses (Line 12 of Algorithm 1).

        New priority: p_i = |loss_i| + epsilon

        Args:
            indices: Buffer indices from sample()
            losses: TD-errors (losses) computed during replay
        """
        for idx, loss in zip(indices, losses):
            # New priority = |loss| + epsilon (Line 12)
            new_priority = (abs(loss) + self.epsilon) ** self.alpha
            self.tree.update(int(idx), new_priority)

    def __len__(self) -> int:
        """Return number of experiences stored."""
        return self.size

    def save(self, path: str) -> None:
        """
        Save buffer to file.

        Args:
            path: Path to save file
        """
        state = {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "tree_data": self.tree.tree.copy(),
            "tree_pointer": self.tree.data_pointer,
            "tree_n_entries": self.tree.n_entries,
            "tree_max_priority": self.tree.max_priority,
            "data": self.data,
            "size": self.size,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        """
        Load buffer from file.

        Args:
            path: Path to load file
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.capacity = state["capacity"]
        self.alpha = state["alpha"]
        self.epsilon = state["epsilon"]
        self.tree = SumTree(self.capacity)
        self.tree.tree = state["tree_data"]
        self.tree.data_pointer = state["tree_pointer"]
        self.tree.n_entries = state["tree_n_entries"]
        self.tree.max_priority = state["tree_max_priority"]
        self.data = state["data"]
        self.size = state["size"]


def compute_beta(step: int, config) -> float:
    """
    Linear annealing of beta from beta_start to beta_end.

    Beta controls the importance sampling correction.
    Starts low (more prioritization bias) and anneals to 1.0 (unbiased).

    Args:
        step: Current training step
        config: Config namespace with prioritized_exp_replay section

    Returns:
        beta: Current beta value
    """
    per = config.prioritized_exp_replay
    progress = min(1.0, step / per.beta_annealing_steps)
    return per.beta_start + progress * (per.beta_end - per.beta_start)
