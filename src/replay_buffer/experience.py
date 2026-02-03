# src/replay_buffer/experience.py
"""
Experience dataclass for Prioritized Experience Replay.

Design: Store MainState + shock outcomes to enable deterministic replay.

Instead of storing all env.step() outputs (expensive), we store:
1. MainState snapshot BEFORE env.step()
2. The shock outcomes (ability, is_superstar) from both parallel branches

This allows exact replay of env.step() with gradient flow to policy network.

Memory Layout (per experience, A=800 agents):
- MainState: ~8 KB (4 float16 tensors + 2 bool + tax_params)
- Shock outcomes: ~5.2 KB (2 float16 + 2 bool per branch)
- Total: ~13 KB per experience
- Buffer (100k): ~1.3 GB
"""

from dataclasses import dataclass
from typing import Optional, Dict, TYPE_CHECKING
import torch
from torch import Tensor

if TYPE_CHECKING:
    from src.env_state import MainState, ParallelState
    from src.environment import EconomyEnv


@dataclass
class Experience:
    """
    Minimal experience storing MainState + shock outcomes for deterministic replay.

    Shock outcomes stored (from parallel states after transition):
    - ability: The transitioned ability v_{t+1}
    - is_superstar: The transitioned superstar status s_{t+1}
    """

    # =========================================================================
    # MainState Snapshot (BEFORE env.step)
    # =========================================================================
    main_moneydisposable: Tensor    # (B, A) float16
    main_savings: Tensor            # (B, A) float16
    main_ability: Tensor            # (B, A) float16
    main_ret: Tensor                # (B, A) float16
    main_tax_params: Tensor         # (B, n_params) float32

    # Branch memory for ability transitions (input to shock)
    main_is_superstar_vA: Tensor    # (B, A) bool
    main_is_superstar_vB: Tensor    # (B, A) bool
    # NOTE: ability_history NOT stored - regenerated during replay if needed

    # =========================================================================
    # Shock Outcomes (from ParallelState after transition_to_parallel)
    # =========================================================================
    # Branch A outcomes (after ability shock)
    shock_ability_A: Tensor         # (B, A) float16 - transitioned ability v_{t+1}^A
    shock_is_superstar_A: Tensor    # (B, A) bool - transitioned superstar s_{t+1}^A

    # Branch B outcomes (after ability shock)
    shock_ability_B: Tensor         # (B, A) float16 - transitioned ability v_{t+1}^B
    shock_is_superstar_B: Tensor    # (B, A) bool - transitioned superstar s_{t+1}^B

    # =========================================================================
    # Metadata
    # =========================================================================
    committed_branch: str           # "A" or "B" - which branch was committed
    step: int                       # Training step when collected


@dataclass
class ShockOutcomes:
    """
    Container for shock outcomes from one parallel branch.

    These are extracted from ParallelState after transition_to_parallel().
    """
    ability: Tensor         # (B, A) - transitioned ability v_{t+1}
    is_superstar: Tensor    # (B, A) - transitioned superstar status s_{t+1}


def snapshot_main_state(main_state: "MainState") -> "MainState":
    """
    Create a deep copy of MainState BEFORE env.step() modifies it.

    NOTE: Does NOT copy ability_history (too large).

    Args:
        main_state: Current MainState (will be modified by env.step())

    Returns:
        snapshot: Deep copy with core tensors cloned
    """
    from src.env_state import MainState

    return MainState(
        moneydisposable=main_state.moneydisposable.clone(),
        savings=main_state.savings.clone(),
        ability=main_state.ability.clone(),
        ret=main_state.ret.clone(),
        tax_params=main_state.tax_params.clone(),
        is_superstar_vA=main_state.is_superstar_vA.clone(),
        is_superstar_vB=main_state.is_superstar_vB.clone(),
        ability_history_vA=None,
        ability_history_vB=None,
    )


def pack_experience(
    main_state_snapshot: "MainState",
    parallel_A: "ParallelState",
    parallel_B: "ParallelState",
    committed_branch: str,
    step: int
) -> Experience:
    """
    Pack MainState snapshot and shock outcomes into Experience for buffer storage.

    Args:
        main_state_snapshot: MainState snapshot taken BEFORE env.step()
        parallel_A: ParallelState from branch A (contains shock outcomes)
        parallel_B: ParallelState from branch B (contains shock outcomes)
        committed_branch: Which branch was committed ("A" or "B")
        step: Training step when collected

    Returns:
        Experience ready for storage in replay buffer
    """
    def to_cpu_half(t: Tensor) -> Tensor:
        return t.detach().cpu().half()

    def to_cpu_float(t: Tensor) -> Tensor:
        return t.detach().cpu().float()

    def to_cpu_bool(t: Tensor) -> Tensor:
        return t.detach().cpu().bool()

    return Experience(
        # MainState snapshot
        main_moneydisposable=to_cpu_half(main_state_snapshot.moneydisposable),
        main_savings=to_cpu_half(main_state_snapshot.savings),
        main_ability=to_cpu_half(main_state_snapshot.ability),
        main_ret=to_cpu_half(main_state_snapshot.ret),
        main_tax_params=to_cpu_float(main_state_snapshot.tax_params),
        main_is_superstar_vA=to_cpu_bool(main_state_snapshot.is_superstar_vA),
        main_is_superstar_vB=to_cpu_bool(main_state_snapshot.is_superstar_vB),

        # Shock outcomes for branch A (from ParallelState)
        shock_ability_A=to_cpu_half(parallel_A.ability),
        shock_is_superstar_A=to_cpu_bool(parallel_A.is_superstar),

        # Shock outcomes for branch B (from ParallelState)
        shock_ability_B=to_cpu_half(parallel_B.ability),
        shock_is_superstar_B=to_cpu_bool(parallel_B.is_superstar),

        # Metadata
        committed_branch=committed_branch,
        step=step,
    )


def unpack_to_main_state(
    exp: Experience,
    device: str = "cpu"
) -> "MainState":
    """
    Reconstruct MainState from Experience for replay.

    Args:
        exp: Experience to unpack
        device: Device to place tensors on

    Returns:
        MainState ready for replay_step() call
    """
    from src.env_state import MainState

    return MainState(
        moneydisposable=exp.main_moneydisposable.float().to(device),
        savings=exp.main_savings.float().to(device),
        ability=exp.main_ability.float().to(device),
        ret=exp.main_ret.float().to(device),
        tax_params=exp.main_tax_params.to(device),
        is_superstar_vA=exp.main_is_superstar_vA.to(device),
        is_superstar_vB=exp.main_is_superstar_vB.to(device),
        ability_history_vA=None,
        ability_history_vB=None,
    )


def unpack_shock_outcomes(
    exp: Experience,
    device: str = "cpu"
) -> tuple[ShockOutcomes, ShockOutcomes]:
    """
    Unpack stored shock outcomes from Experience.

    Args:
        exp: Experience to unpack
        device: Device to place tensors on

    Returns:
        (outcomes_A, outcomes_B): ShockOutcomes for each branch
    """
    outcomes_A = ShockOutcomes(
        ability=exp.shock_ability_A.float().to(device),
        is_superstar=exp.shock_is_superstar_A.to(device),
    )
    outcomes_B = ShockOutcomes(
        ability=exp.shock_ability_B.float().to(device),
        is_superstar=exp.shock_is_superstar_B.to(device),
    )
    return outcomes_A, outcomes_B


def compute_experience_memory_bytes(n_agents: int, n_tax_params: int = 5) -> int:
    """
    Estimate memory usage per Experience in bytes.

    Args:
        n_agents: Number of agents (A dimension)
        n_tax_params: Number of tax parameters

    Returns:
        Estimated bytes per experience
    """
    # MainState: 4 float16 + 1 float32 (tax_params) + 2 bool
    main_bytes = (
        4 * n_agents * 2 +          # 4 float16 tensors
        n_tax_params * 4 +          # tax_params float32
        2 * n_agents * 1            # 2 bool tensors
    )

    # Shock outcomes: 2 float16 + 2 bool (1 ability + 1 is_superstar per branch)
    shock_bytes = (
        2 * n_agents * 2 +          # 2 float16 abilities
        2 * n_agents * 1            # 2 bool is_superstar
    )

    # Metadata
    metadata_bytes = 16

    return main_bytes + shock_bytes + metadata_bytes


def estimate_buffer_memory_gb(capacity: int, n_agents: int) -> float:
    """
    Estimate total buffer memory in GB.
    """
    bytes_per_exp = compute_experience_memory_bytes(n_agents)
    return (capacity * bytes_per_exp) / (1024 ** 3)


# =============================================================================
# REPLAY FUNCTION
# =============================================================================

def replay_step(
    exp: Experience,
    env: "EconomyEnv",
    policy_net,
    device: str = "cpu"
) -> tuple:
    """
    Replay env.step() using stored MainState and shock outcomes.

    This function reproduces the exact same env.step() transition by:
    1. Reconstructing MainState from Experience
    2. Running create_temporary_state() with policy_net (gradient flows here)
    3. Creating ParallelStates with STORED shock outcomes (deterministic)
    4. Running compute_parallel_outcomes() with policy_net (gradient flows here)

    The key insight: shocks are exogenous and don't need gradients.
    Policy decisions (consumption, labor, savings_ratio) need gradients.
    By storing shock outcomes, we skip the random sampling but preserve
    all gradient flow through policy_net.

    Args:
        exp: Experience containing MainState snapshot and shock outcomes
        env: EconomyEnv instance
        policy_net: Policy network (gradients will flow through this)
        device: Device to place tensors on

    Returns:
        Same as env.step():
        - main_state: Updated MainState (NOT committed, for replay we don't modify)
        - temp_state: TemporaryState with time t outcomes
        - (parallel_A, outcomes_A): Branch A state and outcomes
        - (parallel_B, outcomes_B): Branch B state and outcomes
    """
    from src.env_state import ParallelState

    # 1. Reconstruct MainState from Experience
    main_state = unpack_to_main_state(exp, device)

    # 2. Unpack stored shock outcomes
    shock_A, shock_B = unpack_shock_outcomes(exp, device)

    # 3. Create TemporaryState (policy_net forward pass - gradients flow here)
    temp_state = env.create_temporary_state(
        main_state=main_state,
        policy_net=policy_net,
        update_normalizer=False  # Don't update normalizer during replay
    )

    # 4. Create ParallelStates with STORED shock outcomes (skip transition_to_parallel)
    # This is the key: we inject the stored abilities instead of sampling new shocks
    parallel_A = ParallelState(
        moneydisposable=temp_state.money_disposable,
        savings=temp_state.savings,
        ability=shock_A.ability,  # STORED shock outcome
        ret=temp_state.ret,
        tax_params=temp_state.tax_params,
        is_superstar=shock_A.is_superstar,  # STORED shock outcome
        ability_history=None  # Not needed for loss computation
    )

    parallel_B = ParallelState(
        moneydisposable=temp_state.money_disposable,
        savings=temp_state.savings,
        ability=shock_B.ability,  # STORED shock outcome
        ret=temp_state.ret,
        tax_params=temp_state.tax_params,
        is_superstar=shock_B.is_superstar,  # STORED shock outcome
        ability_history=None
    )

    # 5. Compute outcomes for both branches (policy_net forward pass - gradients flow here)
    parallel_A, outcomes_A = env.compute_parallel_outcomes(
        parallel_state=parallel_A,
        policy_net=policy_net,
        update_normalizer=False
    )

    parallel_B, outcomes_B = env.compute_parallel_outcomes(
        parallel_state=parallel_B,
        policy_net=policy_net,
        update_normalizer=False
    )

    # Note: We don't commit to main_state during replay
    # The caller can use temp_state and outcomes for loss computation

    return main_state, temp_state, (parallel_A, outcomes_A), (parallel_B, outcomes_B)
