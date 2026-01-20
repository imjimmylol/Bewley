# normalizer.py
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import torch
from torch import Tensor

@dataclass
class _Stats:
    # Shape is (A, *feat_shape) for per-agent mode, or (*feat_shape) for global mode
    count: Tensor  # 用 float32 比較穩
    mean: Tensor
    M2: Tensor

# Keep old name as alias for backward compatibility
_PerAgentStats = _Stats

class RunningPerAgentWelford:
    """
    Running normalization with Welford's algorithm.

    Supports two modes:
    - Per-agent mode (agent_dim != None): Statistics shape is (A, F...)
      Each agent has its own mean/std for each feature.
    - Global mode (agent_dim = None): Statistics shape is (F...)
      Single mean/std across all agents. Use this when you want the policy
      to be evaluated consistently regardless of agent index.

    Input x shape: (B, A, F...) where B=batch, A=agents, F=features
    """
    def __init__(self, batch_dim: int = 0, agent_dim: Optional[int] = 1, eps: float = 1e-4,
                 min_std: float = 0.01, momentum: float = 0.99, clip_range: float = 10.0):
        """
        Args:
            batch_dim: Dimension index for batch
            agent_dim: Dimension index for agents. Set to None for global normalization
                       (aggregates stats across all agents).
            eps: Small constant added to variance before sqrt (increased from 1e-6 to 1e-4 for stability)
            min_std: Minimum standard deviation floor to prevent division by tiny numbers
            momentum: EMA momentum for statistics (0.99 = slow adaptation, 0.9 = faster)
            clip_range: Clip normalized outputs to [-clip_range, +clip_range]
        """
        self.batch_dim = batch_dim
        self.agent_dim = agent_dim  # None means global mode
        self.eps = eps
        self.min_std = min_std
        self.momentum = momentum
        self.clip_range = clip_range
        self._stats: Dict[str, _Stats] = {}

        # Track mode for clarity
        self.global_mode = (agent_dim is None)

    # ----------------- public -----------------
    def transform(self, name: str, x: Tensor, *, update: bool) -> Tensor:
        """
        Normalize input tensor x.

        Args:
            x: Input tensor, typically (B, A, F) for per-agent or (B, A, F) for global
            update: Whether to update running statistics with this batch

        Returns:
            Normalized tensor with same shape as x
        """
        stats = self._ensure_stats(name, x)

        if update:
            self._update_stats(stats, x)

        mean, var = self._current_mean_var(stats)

        if self.global_mode:
            # Global mode: mean/var shape is (*feat_shape)
            # x shape is (B, A, F...), need to broadcast mean/var to match
            # Add dimensions for batch and agent: (1, 1, *feat_shape)
            mean_b = mean.unsqueeze(0).unsqueeze(0)
            std_b = torch.sqrt(torch.clamp(var, min=0.0) + self.eps).unsqueeze(0).unsqueeze(0)

            # Apply minimum std floor
            std_b = torch.clamp(std_b, min=self.min_std)

            # Normalize directly (no dimension reordering needed)
            y = (x - mean_b) / std_b
        else:
            # Per-agent mode: mean/var shape is (A, *feat_shape)
            # 把 x 搬成 (B, A, ...)
            x_ba = self._move_to_ba(x)
            # broadcast: (1, A, *feat) 去減 (B, A, *feat)
            mean_b = mean.unsqueeze(0)
            std_b = torch.sqrt(torch.clamp(var, min=0.0) + self.eps).unsqueeze(0)

            # Apply minimum std floor to prevent division by tiny numbers
            std_b = torch.clamp(std_b, min=self.min_std)

            y = (x_ba - mean_b) / std_b

            # 再搬回原本的 dim 排序
            y = self._move_from_ba(y, x)

        # Clip normalized values to prevent extreme inputs to network
        y = torch.clamp(y, min=-self.clip_range, max=self.clip_range)

        return y

    def state_dict(self) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {
            "batch_dim": torch.tensor(self.batch_dim),
            # Store -1 for None (global mode) since tensors can't hold None
            "agent_dim": torch.tensor(self.agent_dim if self.agent_dim is not None else -1),
            "eps": torch.tensor(self.eps),
            "min_std": torch.tensor(self.min_std),
            "momentum": torch.tensor(self.momentum),
            "clip_range": torch.tensor(self.clip_range),
            "global_mode": torch.tensor(self.global_mode),
        }
        for k, s in self._stats.items():
            out[f"{k}.count"] = s.count
            out[f"{k}.mean"]  = s.mean
            out[f"{k}.M2"]    = s.M2
        return out

    def load_state_dict(self, sd: Dict[str, Tensor]) -> None:
        self.batch_dim = int(sd["batch_dim"])
        # Handle global mode: -1 means None (global mode)
        agent_dim_val = int(sd["agent_dim"])
        self.agent_dim = None if agent_dim_val == -1 else agent_dim_val
        self.global_mode = bool(sd.get("global_mode", torch.tensor(self.agent_dim is None)).item())
        self.eps = float(sd["eps"])
        self.min_std = float(sd.get("min_std", 0.01))  # Default for backward compatibility
        self.momentum = float(sd.get("momentum", 0.99))
        self.clip_range = float(sd.get("clip_range", 10.0))

        # Reconstruct stats from state dict
        self._stats = {}
        # Find all unique stat names by looking for keys ending in .count, .mean, .M2
        stat_names = set()
        for key in sd.keys():
            if key.endswith(".count") or key.endswith(".mean") or key.endswith(".M2"):
                stat_name = key.rsplit(".", 1)[0]
                stat_names.add(stat_name)

        # Reconstruct each stat
        for name in stat_names:
            self._stats[name] = _Stats(
                count=sd[f"{name}.count"],
                mean=sd[f"{name}.mean"],
                M2=sd[f"{name}.M2"]
            )

    def save(self, path: str) -> None:
        """
        Save normalizer state to file.

        Args:
            path: File path to save to (e.g., 'checkpoints/run_name/normalizer/norm_step_100.pt')
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Load normalizer state from file.

        Args:
            path: File path to load from
        """
        state_dict = torch.load(path, map_location='cpu')  # Load to CPU first for device flexibility
        self.load_state_dict(state_dict)

    def to(self, device):
        """
        Move all internal statistics to the specified device.

        Args:
            device: Target device (e.g., 'cpu', 'cuda', 'mps')

        Returns:
            self (for method chaining)
        """
        for name, stats in self._stats.items():
            stats.count = stats.count.to(device)
            stats.mean = stats.mean.to(device)
            stats.M2 = stats.M2.to(device)
        return self

    @classmethod
    def from_file(cls, path: str, device='cpu') -> 'RunningPerAgentWelford':
        """
        Create a new normalizer instance from a saved file.

        Args:
            path: File path to load from
            device: Device to load tensors to (default: 'cpu')

        Returns:
            RunningPerAgentWelford: Loaded normalizer instance
        """
        state_dict = torch.load(path, map_location=device)

        # Handle global mode: -1 means None
        agent_dim_val = int(state_dict["agent_dim"])
        agent_dim = None if agent_dim_val == -1 else agent_dim_val

        # Create new instance with saved hyperparameters
        normalizer = cls(
            batch_dim=int(state_dict["batch_dim"]),
            agent_dim=agent_dim,
            eps=float(state_dict["eps"]),
            min_std=float(state_dict.get("min_std", 0.01)),
            momentum=float(state_dict.get("momentum", 0.99)),
            clip_range=float(state_dict.get("clip_range", 10.0))
        )

        # Load the statistics
        normalizer.load_state_dict(state_dict)

        # Ensure all stats are on the correct device
        normalizer.to(device)

        return normalizer

    # ----------------- internal helpers -----------------
    def _move_to_ba(self, x: Tensor) -> Tensor:
        # 把 batch_dim, agent_dim 搬到前兩個 (only used in per-agent mode)
        if self.global_mode:
            raise RuntimeError("_move_to_ba should not be called in global mode")
        return x.movedim([self.batch_dim, self.agent_dim], [0, 1])

    def _move_from_ba(self, x_ba: Tensor, ref: Tensor) -> Tensor:
        # 把 (B, A, ...) 搬回 ref 的 dim 排序 (only used in per-agent mode)
        if self.global_mode:
            raise RuntimeError("_move_from_ba should not be called in global mode")
        return x_ba.movedim([0, 1], [self.batch_dim, self.agent_dim])

    def _ensure_stats(self, name: str, x: Tensor) -> _Stats:
        if name in self._stats:
            return self._stats[name]

        device, dtype = x.device, x.dtype

        if self.global_mode:
            # Global mode: stats shape is (*feat_shape)
            # x shape is (B, A, F...), feat_shape is everything after batch and agent dims
            # Assuming standard layout (B, A, F...)
            feat_shape = x.shape[2:]

            count = torch.zeros(feat_shape, device=device, dtype=torch.float32)
            mean = torch.zeros(feat_shape, device=device, dtype=dtype)
            M2 = torch.zeros(feat_shape, device=device, dtype=dtype)
        else:
            # Per-agent mode: stats shape is (A, *feat_shape)
            x_ba = self._move_to_ba(x)        # (B, A, ...)
            A = x_ba.shape[1]
            feat_shape = x_ba.shape[2:]       # 這就是你想要 per-feature 的那塊

            # Use float32 instead of float64 for MPS compatibility
            count = torch.zeros((A, *feat_shape), device=device, dtype=torch.float32)
            mean = torch.zeros((A, *feat_shape), device=device, dtype=dtype)
            M2 = torch.zeros((A, *feat_shape), device=device, dtype=dtype)

        s = _Stats(count=count, mean=mean, M2=M2)
        self._stats[name] = s
        return s

    def _update_stats(self, s: _Stats, x: Tensor) -> None:
        # CRITICAL: Detach to prevent gradient accumulation in normalizer stats
        m, mean_b, M2_b = self._reduce_over_batch(x)
        mean_b = mean_b.detach()  # Statistics should not have gradients
        M2_b = M2_b.detach()

        # Check if this is the first update
        is_first_update = (s.count.sum() == 0).item()

        if is_first_update:
            # First update: initialize with batch statistics
            s.mean = mean_b
            s.M2 = M2_b
            s.count = torch.ones_like(s.count) * m
        else:
            # Use Exponential Moving Average (EMA) to prevent freezing
            # alpha = 1 - momentum (e.g., 0.01 for momentum=0.99)
            alpha = 1.0 - self.momentum

            # Update mean and M2 with EMA
            s.mean = self.momentum * s.mean + alpha * mean_b
            s.M2 = self.momentum * s.M2 + alpha * M2_b

            # Cap count at a maximum to prevent it from growing forever
            # This ensures the statistics remain responsive to distribution changes
            s.count = torch.clamp(s.count + 1, max=1000.0)

    def _reduce_over_batch(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Reduce a batch to compute mean and M2 statistics.

        For per-agent mode:
            Input: (B, A, F...)
            Output: mean_b (A, F...), M2_b (A, F...)

        For global mode:
            Input: (B, A, F...)
            Output: mean_b (F...), M2_b (F...) - aggregated across both B and A
        """
        if self.global_mode:
            # Global mode: aggregate over both batch and agent dimensions
            # x shape: (B, A, F...)
            B, A = x.shape[:2]
            feat_shape = x.shape[2:]

            # Flatten batch and agent dims: (B*A, F...)
            x_flat = x.reshape(B * A, -1)  # (B*A, Ff)

            # Compute mean across all samples
            mean_flat = x_flat.mean(dim=0)  # (Ff,)

            # Compute M2 (sum of squared deviations)
            diff = x_flat - mean_flat.unsqueeze(0)  # (B*A, Ff)
            M2_flat = (diff ** 2).sum(dim=0)  # (Ff,)

            # Reshape back to feature shape
            mean_b = mean_flat.reshape(feat_shape)
            M2_b = M2_flat.reshape(feat_shape)

            # Total sample count is B * A
            m = torch.tensor(float(B * A), device=x.device, dtype=torch.float32)
        else:
            # Per-agent mode: aggregate only over batch dimension
            x_ba = self._move_to_ba(x)          # (B, A, ...)
            B, A = x_ba.shape[:2]
            feat_shape = x_ba.shape[2:]

            # 攤成 (B, A, F_flat)
            x_flat = x_ba.reshape(B, A, -1)     # (B, A, Ff)

            # 這批的平均 (沿 batch 維度)
            mean_flat = x_flat.mean(dim=0)      # (A, Ff)

            # 這批的平方偏差
            diff = x_flat - mean_flat.unsqueeze(0)   # (B, A, Ff)
            M2_flat = (diff ** 2).sum(dim=0)         # (A, Ff)

            # reshape 回去
            mean_b = mean_flat.reshape(A, *feat_shape)
            M2_b = M2_flat.reshape(A, *feat_shape)

            # Use float32 instead of float64 for MPS compatibility
            m = torch.tensor(float(B), device=x.device, dtype=torch.float32)

        return m, mean_b, M2_b

    @staticmethod
    def _current_mean_var(s: _Stats) -> Tuple[Tensor, Tensor]:
        # var = M2 / (n - 1)
        var = s.M2 / torch.clamp(s.count - 1.0, min=1.0)
        return s.mean, var
