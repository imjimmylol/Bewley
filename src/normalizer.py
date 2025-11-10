# normalizer.py
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
from torch import Tensor

@dataclass
class _PerAgentStats:
    # 這三個的形狀會是 (A, *feat_shape)
    count: Tensor  # 用 float64 比較穩
    mean: Tensor
    M2: Tensor

class RunningPerAgentWelford:
    """
    做的事：
    - x 通常長這樣： (B, A, F...)   B= batch / step 進來的那一批
    - 我們要維持的統計是： (A, F...)
      也就是「每個 agent、每個 feature、一條跨 step 的 running 統計線」
    - 每次呼叫 transform(update=True) 就會把這一批 B 吃進統計裡
    """
    def __init__(self, batch_dim: int = 0, agent_dim: int = 1, eps: float = 1e-6):
        self.batch_dim = batch_dim
        self.agent_dim = agent_dim
        self.eps = eps
        self._stats: Dict[str, _PerAgentStats] = {}

    # ----------------- public -----------------
    def transform(self, name: str, x: Tensor, *, update: bool) -> Tensor:
        """
        x: 例如 (B, A, F)
        return: 同形狀的標準化結果
        """
        stats = self._ensure_stats(name, x)

        if update:
            self._update_stats(stats, x)

        mean, var = self._current_mean_var(stats)   # (A, *feat_shape)

        # 把 x 搬成 (B, A, ...)
        x_ba = self._move_to_ba(x)
        # broadcast: (1, A, *feat) 去減 (B, A, *feat)
        mean_b = mean.unsqueeze(0)
        std_b  = torch.sqrt(torch.clamp(var, min=0.0) + self.eps).unsqueeze(0)

        y = (x_ba - mean_b) / std_b
        # 再搬回原本的 dim 排序
        y = self._move_from_ba(y, x)
        return y

    def state_dict(self) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {
            "batch_dim": torch.tensor(self.batch_dim),
            "agent_dim": torch.tensor(self.agent_dim),
            "eps": torch.tensor(self.eps),
        }
        for k, s in self._stats.items():
            out[f"{k}.count"] = s.count
            out[f"{k}.mean"]  = s.mean
            out[f"{k}.M2"]    = s.M2
        return out

    def load_state_dict(self, sd: Dict[str, Tensor]) -> None:
        self.batch_dim = int(sd["batch_dim"])
        self.agent_dim = int(sd["agent_dim"])
        self.eps = float(sd["eps"])
        # 真正的 stats 形狀要等到下一次看到同名的 x 才能初始化，
        # 所以這裡先放著
        self._stats = {}
        # 如果你想要連 stats 一起塞回去，也可以在這裡把 key 掃一遍組回去

    # ----------------- internal helpers -----------------
    def _move_to_ba(self, x: Tensor) -> Tensor:
        # 把 batch_dim, agent_dim 搬到前兩個
        return x.movedim([self.batch_dim, self.agent_dim], [0, 1])

    def _move_from_ba(self, x_ba: Tensor, ref: Tensor) -> Tensor:
        # 把 (B, A, ...) 搬回 ref 的 dim 排序
        return x_ba.movedim([0, 1], [self.batch_dim, self.agent_dim])

    def _ensure_stats(self, name: str, x: Tensor) -> _PerAgentStats:
        if name in self._stats:
            return self._stats[name]

        x_ba = self._move_to_ba(x)        # (B, A, ...)
        A = x_ba.shape[1]
        feat_shape = x_ba.shape[2:]       # 這就是你想要 per-feature 的那塊
        device, dtype = x.device, x.dtype

        count = torch.zeros((A, *feat_shape), device=device, dtype=torch.float64)
        mean  = torch.zeros((A, *feat_shape), device=device, dtype=dtype)
        M2    = torch.zeros((A, *feat_shape), device=device, dtype=dtype)

        s = _PerAgentStats(count=count, mean=mean, M2=M2)
        self._stats[name] = s
        return s

    def _update_stats(self, s: _PerAgentStats, x: Tensor) -> None:
        m, mean_b, M2_b = self._reduce_over_batch(x)   # m: scalar (這一批的 B)
        n = s.count                                    # (A, *feat)
        tot = n + m                                    # broadcast scalar m

        delta = (mean_b - s.mean).to(s.mean.dtype)
        new_mean = s.mean + delta * (m / torch.clamp(tot, min=1.0))

        new_M2 = s.M2 + M2_b + (delta ** 2) * (n * m / torch.clamp(tot, min=1.0))

        s.count = tot
        s.mean  = new_mean
        s.M2    = new_M2

    def _reduce_over_batch(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        把一批 (B, A, F...) 收成：
        - m: 這批的樣本數 (=B)
        - mean_b: (A, F...)
        - M2_b:   (A, F...)
        """
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
        M2_b   = M2_flat.reshape(A, *feat_shape)

        m = torch.tensor(float(B), device=x.device, dtype=torch.float64)
        return m, mean_b, M2_b

    @staticmethod
    def _current_mean_var(s: _PerAgentStats) -> Tuple[Tensor, Tensor]:
        # var = M2 / (n - 1)
        var = s.M2 / torch.clamp(s.count - 1.0, min=1.0)
        return s.mean, var
