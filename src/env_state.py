# states.py
from dataclasses import dataclass
from typing import Optional, Literal
import torch
from torch import Tensor

@dataclass
class TemporaryState:
    """
    Realized outcomes at time t, before transitioning to t+1.
    This stores all variables computed AFTER agents act on MainState[t]
    but BEFORE applying ability shocks.
    """
    # Current state (before shocks)
    savings: Tensor              # (B, A) - current savings
    ability: Tensor              # (B, A) - current ability (before transition)

    # Agent decisions at time t
    consumption: Tensor          # (B, A) - consumption decision
    labor: Tensor                # (B, A) - labor supply decision
    savings_ratio: Tensor        # (B, A) - savings ratio from policy
    mu: Optional[Tensor]         # (B, A) - Lagrange multiplier (optional)

    # Market outcomes at time t
    wage: Tensor                 # (B, A) or (B,) - market wage
    ret: Tensor                  # (B, A) or (B,) - return to capital

    # Income/tax at time t
    income_before_tax: Tensor    # (B, A) - income before tax
    money_disposable: Tensor     # (B, A) - disposable money after tax
    income_tax: Tensor           # (B, A) - income tax paid
    savings_tax: Tensor          # (B, A) - savings tax paid

    # Tax parameters (constant)
    tax_params: Tensor           # (B, A, 5) - tax parameters

    # Branch memory (carried forward from MainState)
    is_superstar_vA: Tensor      # (B, A) - branch A superstar status
    is_superstar_vB: Tensor      # (B, A) - branch B superstar status
    ability_history_vA: Optional[Tensor]  # (L, B, A) - branch A history
    ability_history_vB: Optional[Tensor]  # (L, B, A) - branch B history


@dataclass
class MainState:
    # ---- realized (當期主世界) ----
    moneydisposable: Tensor
    savings: Tensor
    ability: Tensor
    # consumption: Tensor
    ret: Tensor
    tax_params: Tensor

    # ---- branch memory for path dependence (平行世界的歷程記憶) ----
    is_superstar_vA: Tensor                   # (B, A) 供下期產生 A 分支的外生狀態
    is_superstar_vB: Tensor                   # (B, A) 供下期產生 B 分支的外生狀態
    ability_history_vA: Optional[Tensor]      # (L, B, A) 或 None：A 分支的能力路徑
    ability_history_vB: Optional[Tensor]      # (L, B, A) 或 None：B 分支的能力路徑

    def commit(self, src: "ParallelState", branch: Literal["A","B"], *, detach: bool = True) -> None:
        """用選中的平行世界覆寫主世界，同步更新該分支的記憶（旗標/ability_history）。"""
        def d(x: Optional[Tensor]) -> Optional[Tensor]:
            if x is None:
                return None
            if detach and torch.is_tensor(x):
                return x.detach().clone()
            return x.clone() if torch.is_tensor(x) else x

        # 覆寫主世界（realized path）
        self.moneydisposable = d(src.moneydisposable)
        self.savings         = d(src.savings)
        self.ability         = d(src.ability)
        # self.consumption     = d(src.consumption)
        self.ret             = d(src.ret)
        self.tax_params      = d(src.tax_params)

        # 更新被選分支的「分支記憶」（另一分支保留原值，維持雙世界路徑）
        if branch == "A":
            self.is_superstar_vA  = d(src.is_superstar)
            self.ability_history_vA = d(src.ability_history)
        else:
            self.is_superstar_vB  = d(src.is_superstar)
            self.ability_history_vB = d(src.ability_history)


@dataclass
class ParallelState:
    """單一分支的平行世界，只有自己這條分支的旗標與 ability 路徑。"""
    moneydisposable: Tensor
    savings: Tensor
    ability: Tensor
    # consumption: Tensor
    ret: Tensor
    tax_params: Tensor
    is_superstar: Tensor                    # (B, A) 這條分支的外生旗標（單一份就夠）
    ability_history: Optional[Tensor]       # (L, B, A) 或 None：這條分支的能力路徑

# To create parallel state 
def make_parallel(
    main: MainState,
    *,
    # branch: Literal["A","B"],
    ability_next: Tensor,
    is_superstar_next: Tensor,
    ability_history_next: Optional[Tensor],
) -> ParallelState:
    """純工廠：由 MainState + 分支的下一期外生量，生成對應的 ParallelState。"""
    return ParallelState(
        moneydisposable=main.moneydisposable,   # 本期 money 作為下輪輸入使用
        savings=main.savings,                   # 由外部後續流程更新為 t+1（這裡僅初始化）
        ability=ability_next,                   # 分支專屬
        # consumption=main.consumption,           # 本期消費（外部可再覆寫）
        ret=main.ret,                           # 本期 ret；外部計價後覆寫為 t 的均衡值
        tax_params=main.tax_params,             # 一般沿用
        is_superstar=is_superstar_next,         # 該分支唯一旗標
        ability_history=ability_history_next,   # 該分支的路徑（main 的對應分支路徑 + append）
    )

