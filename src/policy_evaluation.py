#!/usr/bin/env python3
"""
Policy Evaluation Module for Decision Rule Visualization.

This module provides tools for:
1. Collecting historical ranges of state variables during training
2. Evaluating policy networks on synthetic grids with controlled conditioning
3. Supporting the "Synthetic Grid Evaluation" methodology from CLAUDE.local.md

Key insight: To visualize true decision rules (not equilibrium outcomes),
we must fix conditioning variables and vary only the x-axis variable systematically.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union
import numpy as np
import torch
from torch import Tensor

from src.utils.buildipnuts import build_inputs


# =============================================================================
# Historical Ranges Collection
# =============================================================================

@dataclass
class VariableRanges:
    """
    Stores running statistics for a single continuous variable.

    Uses reservoir sampling approximation for percentiles to keep memory bounded.
    """
    min_val: float = float('inf')
    max_val: float = float('-inf')
    sum_val: float = 0.0
    sum_sq: float = 0.0
    count: int = 0
    # Reservoir for percentile estimation (fixed size)
    reservoir: np.ndarray = field(default_factory=lambda: np.array([]))
    reservoir_size: int = 10000

    def update(self, values: np.ndarray) -> None:
        """Update statistics with new values (flattens input)."""
        values = values.flatten()
        self._update_from_flat(values)

    def _update_from_flat(self, values: np.ndarray) -> None:
        """Update statistics from already-flattened values."""
        # Update min/max
        self.min_val = min(self.min_val, float(np.min(values)))
        self.max_val = max(self.max_val, float(np.max(values)))

        # Update mean/variance statistics
        self.sum_val += float(np.sum(values))
        self.sum_sq += float(np.sum(values ** 2))
        self.count += len(values)

        # Update reservoir for percentile estimation
        self._update_reservoir(values)

    def _update_reservoir(self, values: np.ndarray) -> None:
        """Reservoir sampling for percentile estimation."""
        if len(self.reservoir) < self.reservoir_size:
            # Still filling reservoir
            space_left = self.reservoir_size - len(self.reservoir)
            to_add = values[:space_left]
            self.reservoir = np.concatenate([self.reservoir, to_add])
            values = values[space_left:]

        # Reservoir sampling for remaining values
        if len(values) > 0:
            for val in values:
                j = np.random.randint(0, self.count)
                if j < self.reservoir_size:
                    self.reservoir[j] = val

    @property
    def mean(self) -> float:
        """Return running mean."""
        if self.count == 0:
            return 0.0
        return self.sum_val / self.count

    @property
    def std(self) -> float:
        """Return running standard deviation."""
        if self.count < 2:
            return 0.0
        variance = (self.sum_sq / self.count) - (self.mean ** 2)
        return float(np.sqrt(max(0.0, variance)))

    def get_percentiles(self, percentiles: List[float] = None) -> np.ndarray:
        """
        Return percentile values.

        Args:
            percentiles: List of percentiles to compute (default: [5, 25, 50, 75, 95])

        Returns:
            Array of percentile values
        """
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]

        if len(self.reservoir) == 0:
            return np.zeros(len(percentiles))

        return np.percentile(self.reservoir, percentiles)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        percentiles = self.get_percentiles()
        return {
            "min": self.min_val,
            "max": self.max_val,
            "mean": self.mean,
            "std": self.std,
            "percentiles": percentiles,  # [q5, q25, q50, q75, q95]
            "count": self.count
        }


@dataclass
class PerAgentVariableRanges:
    """
    Stores running statistics for a variable, tracked per agent.

    Each agent has its own VariableRanges instance to track the values
    that specific agent has explored during training.
    """
    n_agents: int = 0
    agent_ranges: List[VariableRanges] = field(default_factory=list)

    def __post_init__(self):
        """Initialize per-agent ranges if n_agents is set."""
        if self.n_agents > 0 and len(self.agent_ranges) == 0:
            self.agent_ranges = [VariableRanges() for _ in range(self.n_agents)]

    def initialize(self, n_agents: int) -> None:
        """Initialize ranges for n_agents (call once when n_agents is known)."""
        if self.n_agents == 0:
            self.n_agents = n_agents
            self.agent_ranges = [VariableRanges() for _ in range(n_agents)]

    def update(self, values: np.ndarray) -> None:
        """
        Update per-agent statistics.

        Args:
            values: Array of shape (..., n_agents) - last dimension is agents.
                   All leading dimensions are flattened per agent.
        """
        if self.n_agents == 0:
            # Auto-initialize based on last dimension
            self.initialize(values.shape[-1])

        # Reshape to (n_samples, n_agents)
        n_agents = values.shape[-1]
        values_2d = values.reshape(-1, n_agents)

        for agent_idx in range(n_agents):
            agent_values = values_2d[:, agent_idx]
            self.agent_ranges[agent_idx]._update_from_flat(agent_values)

    def get_agent_range(self, agent_idx: int) -> Dict:
        """Get range info for a specific agent."""
        if agent_idx >= len(self.agent_ranges):
            raise ValueError(f"Agent index {agent_idx} out of range (have {len(self.agent_ranges)} agents)")
        return self.agent_ranges[agent_idx].to_dict()

    def get_agent_quantile_value(self, agent_idx: int, quantile: str) -> float:
        """
        Get a specific quantile value for a specific agent.

        Args:
            agent_idx: Which agent
            quantile: Quantile specification (e.g., "q5", "q50", "mean")

        Returns:
            The value at the specified quantile for this agent
        """
        range_info = self.get_agent_range(agent_idx)

        if quantile == "mean":
            return range_info["mean"]
        elif quantile == "min":
            return range_info["min"]
        elif quantile == "max":
            return range_info["max"]
        elif quantile.startswith("q"):
            q_idx_map = {"q5": 0, "q25": 1, "q50": 2, "q75": 3, "q95": 4}
            if quantile not in q_idx_map:
                raise ValueError(f"Unknown quantile: {quantile}. Use q5, q25, q50, q75, q95")
            return range_info["percentiles"][q_idx_map[quantile]]
        else:
            raise ValueError(f"Unknown quantile specification: {quantile}")


@dataclass
class HistoricalRanges:
    """
    Collects and stores historical ranges for all relevant continuous variables.

    Used to determine realistic input ranges for synthetic grid evaluation.

    Variables tracked (all continuous):
    - m_t: money_disposable
    - a_t: savings/assets (input)
    - v_t: ability
    - y_t: income_before_tax
    - c_t: consumption (for reference)
    - zeta_t: savings_ratio (for reference)
    - l_t: labor (for reference)

    Note: s_t (is_superstar) is discrete (True/False) and doesn't need range tracking.
    When conditioning on s_t, just pass boolean value directly.

    Per-agent tracking:
    When track_per_agent=True, also maintains per-agent ranges for key state variables
    (m_t, a_t, v_t). This allows generating x-axis grids based on what a specific
    agent has actually explored, rather than the global population range.
    """
    # Global ranges (aggregated across all agents)
    m_t: VariableRanges = field(default_factory=VariableRanges)
    a_t: VariableRanges = field(default_factory=VariableRanges)
    v_t: VariableRanges = field(default_factory=VariableRanges)
    y_t: VariableRanges = field(default_factory=VariableRanges)

    # Output variables (for reference, not for conditioning)
    c_t: VariableRanges = field(default_factory=VariableRanges)
    zeta_t: VariableRanges = field(default_factory=VariableRanges)
    l_t: VariableRanges = field(default_factory=VariableRanges)

    # Per-agent ranges (optional, for agent-specific x-axis grids)
    track_per_agent: bool = False
    m_t_per_agent: PerAgentVariableRanges = field(default_factory=PerAgentVariableRanges)
    a_t_per_agent: PerAgentVariableRanges = field(default_factory=PerAgentVariableRanges)
    v_t_per_agent: PerAgentVariableRanges = field(default_factory=PerAgentVariableRanges)

    def get_range(self, var_name: str) -> Dict:
        """Get range info for a variable by name."""
        var_map = {
            "m_t": self.m_t,
            "a_t": self.a_t,
            "v_t": self.v_t,
            "y_t": self.y_t,
            "c_t": self.c_t,
            "zeta_t": self.zeta_t,
            "l_t": self.l_t,
        }
        if var_name not in var_map:
            raise ValueError(f"Unknown variable: {var_name}. Valid: {list(var_map.keys())}")
        return var_map[var_name].to_dict()

    def get_quantile_value(self, var_name: str, quantile: str) -> float:
        """
        Get a specific quantile value for a variable (global, across all agents).

        Args:
            var_name: Variable name (e.g., "m_t", "v_t")
            quantile: Quantile specification (e.g., "q5", "q25", "q50", "q75", "q95", "mean")

        Returns:
            The value at the specified quantile
        """
        range_info = self.get_range(var_name)

        if quantile == "mean":
            return range_info["mean"]
        elif quantile == "min":
            return range_info["min"]
        elif quantile == "max":
            return range_info["max"]
        elif quantile.startswith("q"):
            q_idx_map = {"q5": 0, "q25": 1, "q50": 2, "q75": 3, "q95": 4}
            if quantile not in q_idx_map:
                raise ValueError(f"Unknown quantile: {quantile}. Use q5, q25, q50, q75, q95")
            return range_info["percentiles"][q_idx_map[quantile]]
        else:
            raise ValueError(f"Unknown quantile specification: {quantile}")

    def get_agent_range(self, var_name: str, agent_idx: int) -> Dict:
        """
        Get range info for a specific agent.

        Args:
            var_name: Variable name (must be "m_t", "a_t", or "v_t")
            agent_idx: Agent index

        Returns:
            Dict with min, max, mean, std, percentiles for this agent

        Raises:
            ValueError: If per-agent tracking is not enabled or var_name not supported
        """
        if not self.track_per_agent:
            raise ValueError("Per-agent tracking is not enabled. Set track_per_agent=True.")

        per_agent_map = {
            "m_t": self.m_t_per_agent,
            "a_t": self.a_t_per_agent,
            "v_t": self.v_t_per_agent,
        }
        if var_name not in per_agent_map:
            raise ValueError(
                f"Per-agent ranges only available for {list(per_agent_map.keys())}, got: {var_name}"
            )
        return per_agent_map[var_name].get_agent_range(agent_idx)

    def get_agent_quantile_value(self, var_name: str, agent_idx: int, quantile: str) -> float:
        """
        Get a specific quantile value for a specific agent.

        Args:
            var_name: Variable name (must be "m_t", "a_t", or "v_t")
            agent_idx: Agent index
            quantile: Quantile specification (e.g., "q5", "q50", "mean")

        Returns:
            The value at the specified quantile for this agent
        """
        if not self.track_per_agent:
            raise ValueError("Per-agent tracking is not enabled. Set track_per_agent=True.")

        per_agent_map = {
            "m_t": self.m_t_per_agent,
            "a_t": self.a_t_per_agent,
            "v_t": self.v_t_per_agent,
        }
        if var_name not in per_agent_map:
            raise ValueError(
                f"Per-agent ranges only available for {list(per_agent_map.keys())}, got: {var_name}"
            )
        return per_agent_map[var_name].get_agent_quantile_value(agent_idx, quantile)

    def to_dict(self) -> Dict:
        """Convert all ranges to dictionary."""
        return {
            "m_t": self.m_t.to_dict(),
            "a_t": self.a_t.to_dict(),
            "v_t": self.v_t.to_dict(),
            "y_t": self.y_t.to_dict(),
            "c_t": self.c_t.to_dict(),
            "zeta_t": self.zeta_t.to_dict(),
            "l_t": self.l_t.to_dict(),
        }

    def summary(self) -> str:
        """Return a human-readable summary of all ranges."""
        lines = ["Historical Ranges Summary", "=" * 40]

        for name in ["m_t", "a_t", "v_t", "y_t", "c_t", "zeta_t", "l_t"]:
            info = self.get_range(name)
            lines.append(f"\n{name}:")
            lines.append(f"  Range: [{info['min']:.4f}, {info['max']:.4f}]")
            lines.append(f"  Mean: {info['mean']:.4f}, Std: {info['std']:.4f}")
            pct = info['percentiles']
            lines.append(f"  Percentiles [5,25,50,75,95]: [{pct[0]:.3f}, {pct[1]:.3f}, {pct[2]:.3f}, {pct[3]:.3f}, {pct[4]:.3f}]")

        lines.append(f"\ns_t (regime): discrete, use True/False when conditioning")

        return "\n".join(lines)


def collect_ranges_from_step(
    temp_state,  # TemporaryState
    main_state,  # MainState
    existing_ranges: Optional[HistoricalRanges] = None,
    track_per_agent: bool = False
) -> HistoricalRanges:
    """
    Update historical ranges from a single training step.

    This function should be called every training step to accumulate
    statistics about variable distributions. It's designed to be lightweight.

    Args:
        temp_state: TemporaryState instance with current step data
        main_state: MainState instance (for input savings a_t)
        existing_ranges: Existing HistoricalRanges to update (or None to create new)
        track_per_agent: If True, also track per-agent ranges for m_t, a_t, v_t.
                        This enables agent-specific x-axis grids in visualization.

    Returns:
        Updated HistoricalRanges instance
    """
    if existing_ranges is None:
        existing_ranges = HistoricalRanges(track_per_agent=track_per_agent)
    elif track_per_agent and not existing_ranges.track_per_agent:
        # Enable per-agent tracking if requested but not yet enabled
        existing_ranges.track_per_agent = True

    # Helper to convert tensor to numpy
    def to_np(x: Tensor) -> np.ndarray:
        if isinstance(x, Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # Get numpy arrays
    m_t_np = to_np(temp_state.money_disposable)
    v_t_np = to_np(temp_state.ability)
    a_t_np = to_np(main_state.savings)

    # Update global ranges (aggregated across all agents)
    existing_ranges.m_t.update(m_t_np)
    existing_ranges.v_t.update(v_t_np)
    existing_ranges.y_t.update(to_np(temp_state.income_before_tax))
    existing_ranges.c_t.update(to_np(temp_state.consumption))
    existing_ranges.zeta_t.update(to_np(temp_state.savings_ratio))
    existing_ranges.l_t.update(to_np(temp_state.labor))
    existing_ranges.a_t.update(a_t_np)

    # Update per-agent ranges if enabled
    if existing_ranges.track_per_agent:
        existing_ranges.m_t_per_agent.update(m_t_np)
        existing_ranges.v_t_per_agent.update(v_t_np)
        existing_ranges.a_t_per_agent.update(a_t_np)

    return existing_ranges


# =============================================================================
# Validation Functions
# =============================================================================

# Define variable categories for validation
STATE_VARS = {"m_t", "a_t", "v_t", "s_t", "y_t"}  # Can be x-axis or conditioning
DECISION_VARS = {"c_t", "zeta_t", "l_t", "a_tp1", "mu_t", "da_t", "I_bind"}  # Can only be y-axis


def validate_plot_config(
    x_var: str,
    y_var: str,
    condition_vars: Optional[Dict[str, Union[str, float, bool]]] = None
) -> None:
    """
    Validate plot configuration to ensure no variable conflicts.

    Rules:
    1. x_var cannot be in condition_vars
    2. y_var cannot be in condition_vars
    3. x_var must be a valid state variable
    4. y_var must be a valid decision variable

    Args:
        x_var: Variable for x-axis
        y_var: Variable for y-axis
        condition_vars: Dict of {var_name: value/quantile} for conditioning

    Raises:
        ValueError: If configuration is invalid
    """
    if condition_vars is None:
        condition_vars = {}

    # Check x_var not in condition_vars
    if x_var in condition_vars:
        raise ValueError(
            f"x_var '{x_var}' cannot be in condition_vars. "
            f"You cannot both vary and fix the same variable."
        )

    # Check y_var not in condition_vars
    if y_var in condition_vars:
        raise ValueError(
            f"y_var '{y_var}' cannot be in condition_vars. "
            f"You cannot both plot and fix the same variable."
        )

    # Validate x_var is plottable on x-axis
    if x_var not in STATE_VARS:
        raise ValueError(
            f"x_var '{x_var}' is not a valid x-axis variable. "
            f"Valid options: {STATE_VARS}"
        )

    # Validate y_var is plottable on y-axis
    if y_var not in DECISION_VARS:
        raise ValueError(
            f"y_var '{y_var}' is not a valid y-axis variable. "
            f"Valid options: {DECISION_VARS}"
        )


def resolve_condition_value(
    var_name: str,
    spec: Union[str, float, bool, int],
    ranges: HistoricalRanges
) -> float:
    """
    Resolve a condition specification to an actual value.

    Args:
        var_name: Variable name
        spec: Specification - can be:
            - float/int: Use directly
            - str like "q50": Use quantile from ranges
            - str like "mean": Use mean from ranges
            - bool: For discrete vars (s_t), True=1, False=0

    Returns:
        Resolved float value
    """
    # Handle boolean for discrete variable s_t
    if isinstance(spec, bool):
        return 1.0 if spec else 0.0

    # Handle direct numeric value
    if isinstance(spec, (int, float)):
        return float(spec)

    # Handle quantile string specification
    if isinstance(spec, str):
        # s_t doesn't have ranges, just accept "normal" or "superstar" as aliases
        if var_name == "s_t":
            if spec.lower() in ("normal", "false", "0"):
                return 0.0
            elif spec.lower() in ("superstar", "true", "1"):
                return 1.0
            else:
                raise ValueError(f"For s_t, use True/False or 'normal'/'superstar', got: {spec}")
        return ranges.get_quantile_value(var_name, spec)

    raise ValueError(f"Unknown specification type for {var_name}: {type(spec)}")


# =============================================================================
# PolicyEvaluator Class
# =============================================================================

class PolicyEvaluator:
    """
    Evaluates policy network on synthetic grids for decision rule visualization.

    Uses a GE-aware evaluation approach: takes actual simulation data as the
    "background" population and only modifies the target agent's inputs.
    This preserves realistic aggregate features (sum_info_rep) while allowing
    us to vary individual state variables.

    Three types of variables:
    1. x-axis: varies systematically on a grid (range from q5 to q95 of training data)
    2. color/panel: conditioning variable evaluated at 5 quantile levels (q5, q25, q50, q75, q95)
    3. fixed: held constant at MEAN from training data (for non-featured variables like a_t, s_t)

    Per-agent x-axis range:
    When use_agent_specific_range=True and agent_idx is specified, the x-axis grid
    is based on the range that the specific agent has explored during training,
    rather than the global population range. This requires track_per_agent=True
    in HistoricalRanges.

    Usage:
        evaluator = PolicyEvaluator(
            policy_net, normalizer, ranges, tax_params, n_agents, device,
            reference_state={"money": money_tensor, "ability": ability_tensor},
            agent_idx=0,  # Focus on agent 0
            use_agent_specific_range=True  # Use agent 0's explored range for x-axis
        )
        results = evaluator.evaluate_on_grid(
            x_var="m_t",
            color_var="v_t",
            fixed_vars=["a_t", "s_t"],  # Will use mean values
            n_points=100
        )
    """

    # Quantile levels for color variable
    COLOR_QUANTILES = ["q5", "q25", "q50", "q75", "q95"]

    def __init__(
        self,
        policy_net: torch.nn.Module,
        normalizer,  # RunningPerAgentWelford
        ranges: HistoricalRanges,
        tax_params: Tensor,  # (Z,) or (B, Z) - tax parameters to use
        n_agents: int,  # Number of agents (needed to construct proper input shape)
        device: str = "cpu",
        reference_state: Optional[Dict[str, Tensor]] = None,  # Actual tensors from training
        # Economic parameters for computing m_t from a_t
        config = None,  # Full config object with tax_params and bewley_model
        # Agent-specific range settings
        agent_idx: int = 0,  # Which agent to evaluate (and use for per-agent ranges)
        use_agent_specific_range: bool = False  # If True, use agent_idx's range for x-axis
    ):
        """
        Initialize PolicyEvaluator.

        Args:
            policy_net: Trained policy network (FiLMResNet2In)
            normalizer: RunningPerAgentWelford normalizer for inputs
            ranges: HistoricalRanges with collected statistics from training
            tax_params: Tax parameters tensor (Z,) - will be used for all evaluations
            n_agents: Number of agents per batch (needed for input shape: 2*n_agents+2)
            device: Device to run evaluation on
            reference_state: Dict with actual tensors from training simulation:
                - "money": (B, A) money_disposable tensor
                - "ability": (B, A) ability tensor
                If provided, uses these as the background population for GE-aware evaluation.
                If None, falls back to homogeneous population (may produce identical curves).
            config: Full configuration object containing:
                - config.bewley_model.delta: depreciation rate
                - config.tax_params.*: tax function parameters
                Used for computing m_t from a_t when a_t is the x-axis variable.
            agent_idx: Which agent to focus on for evaluation (default: 0).
                      This agent's inputs are modified while others serve as background.
            use_agent_specific_range: If True, use agent_idx's explored range for x-axis
                      grid instead of the global range. Requires ranges.track_per_agent=True.
        """
        self.policy_net = policy_net
        self.normalizer = normalizer
        self.ranges = ranges
        self.device = device
        self.n_agents = n_agents
        self.config = config
        self.agent_idx = agent_idx
        self.use_agent_specific_range = use_agent_specific_range

        # Validate per-agent range settings
        if use_agent_specific_range and not ranges.track_per_agent:
            raise ValueError(
                "use_agent_specific_range=True requires ranges.track_per_agent=True. "
                "Enable per-agent tracking in collect_ranges_from_step()."
            )

        # Ensure tax_params is (Z,) shape
        tax_params = torch.as_tensor(tax_params, dtype=torch.float32, device=device)
        if tax_params.dim() == 2:
            tax_params = tax_params[0]  # Take first batch if (B, Z)
        self.tax_params = tax_params  # (Z,)

        # Store reference state for GE-aware evaluation
        # Use batch 0 from the reference state
        if reference_state is not None:
            self.ref_money = reference_state["money"][0:1].clone().to(device)  # (1, A)
            self.ref_ability = reference_state["ability"][0:1].clone().to(device)  # (1, A)
        else:
            # Fallback: create homogeneous population at median values
            m_ref = self.ranges.get_quantile_value("m_t", "q50")
            v_ref = self.ranges.get_quantile_value("v_t", "q50")
            self.ref_money = torch.full((1, n_agents), m_ref, dtype=torch.float32, device=device)
            self.ref_ability = torch.full((1, n_agents), v_ref, dtype=torch.float32, device=device)

        # Put model in eval mode
        self.policy_net.eval()

    def _get_quantile_value(self, var_name: str, quantile: str) -> float:
        """
        Get quantile value, using agent-specific range if enabled and available.

        Args:
            var_name: Variable name (e.g., "m_t", "v_t")
            quantile: Quantile specification (e.g., "q5", "q50", "mean")

        Returns:
            The quantile value (agent-specific if enabled, else global)
        """
        use_agent_range = (
            self.use_agent_specific_range
            and self.ranges.track_per_agent
            and var_name in ["m_t", "a_t", "v_t"]
        )

        if use_agent_range:
            return self.ranges.get_agent_quantile_value(var_name, self.agent_idx, quantile)
        else:
            return self.ranges.get_quantile_value(var_name, quantile)

    def _compute_m_from_a(
        self,
        a_t: float,
        v_t: float,
        l_t: float = 1.0,
        wage: float = 1.0,
        ret: float = 0.04
    ) -> float:
        """
        Compute money disposable (m_t) from assets (a_t) using the economic model.

        Formula (from environment.py):
            ibt = wage * labor * ability + (1 - delta + ret) * savings
            it, at = taxfunc(ibt, savings)
            m_t = (ibt - it) + (savings - at)

        Args:
            a_t: Current assets/savings
            v_t: Ability
            l_t: Labor supply (default: 1.0, assuming full employment)
            wage: Market wage (default: 1.0)
            ret: Return on capital (default: 0.04)

        Returns:
            m_t: Money disposable
        """
        if self.config is None:
            # Fallback: simple approximation m_t ≈ a_t * (1 + ret) + wage * v_t * l_t
            return a_t * (1 + ret) + wage * v_t * l_t

        # Get parameters from config
        delta = self.config.bewley_model.delta
        tax_income = self.config.tax_params.tax_income
        income_tax_elasticity = self.config.tax_params.income_tax_elasticity
        tax_saving = self.config.tax_params.tax_saving
        saving_tax_elasticity = self.config.tax_params.saving_tax_elasticity

        # Compute income before tax
        ibt = wage * l_t * v_t + (1 - delta + ret) * a_t

        # Apply tax function (from environment.py _taxfunc)
        # it = ibt - (1 - tax_income) * (ibt^(1-income_tax_elasticity) / (1-income_tax_elasticity))
        if abs(1 - income_tax_elasticity) > 1e-6:
            it = ibt - (1 - tax_income) * (ibt ** (1 - income_tax_elasticity) / (1 - income_tax_elasticity))
        else:
            it = ibt * tax_income  # Linear approximation when elasticity ≈ 1

        # at = abt - ((1-tax_saving)/(1-saving_tax_elasticity))
        # Note: This seems to be a constant independent of abt in the original code
        at = a_t - ((1 - tax_saving) / (1 - saving_tax_elasticity))

        # Compute money disposable
        m_t = (ibt - it) + (a_t - at)

        return max(0.0, m_t)  # Ensure non-negative

    def evaluate_on_grid(
        self,
        x_var: str,
        y_var: str,
        color_var: Optional[str] = None,
        fixed_vars: Optional[Dict[str, Union[str, float, bool]]] = None,
        n_points: int = 100,
        debug: bool = False
    ) -> Dict[str, any]:
        """
        Evaluate policy on a synthetic grid.

        Args:
            x_var: Variable to vary on x-axis (grid from q5 to q95)
            y_var: Output variable for y-axis (e.g., "c_t", "a_tp1", "zeta_t")
            color_var: Optional variable for color conditioning (5 quantile levels).
                       If None, single line is plotted.
            fixed_vars: Optional dict of {var_name: value} for fixed variables.
                       If a variable is not in x_var, color_var, or fixed_vars,
                       it defaults to MEAN from training data.
            n_points: Number of grid points for x-axis

        Returns:
            Dict with:
                - "x_values": np.ndarray of grid points
                - "y_values": dict mapping color_level -> np.ndarray of y values
                              (single key "mean" if no color_var)
                - "color_var": name of color variable (or None)
                - "color_levels": list of quantile labels used
                - "color_values": list of actual values for each color level
                - "fixed_values": dict of fixed variable values used
                - "x_var": x variable name
                - "y_var": y variable name
        """
        # Validate configuration
        condition_vars_for_validation = {}
        if color_var:
            condition_vars_for_validation[color_var] = "q50"  # dummy for validation
        if fixed_vars:
            condition_vars_for_validation.update(fixed_vars)
        validate_plot_config(x_var, y_var, condition_vars_for_validation)

        # Determine x-axis grid range
        # Use agent-specific range if enabled and available for this variable
        use_agent_range = (
            self.use_agent_specific_range
            and self.ranges.track_per_agent
            and x_var in ["m_t", "a_t", "v_t"]  # Only these have per-agent tracking
        )

        if use_agent_range:
            # Use the range that this specific agent has explored
            x_range_info = self.ranges.get_agent_range(x_var, self.agent_idx)
            if debug:
                print(f"Using agent {self.agent_idx}'s range for {x_var}")
        else:
            # Use global range (aggregated across all agents)
            x_range_info = self.ranges.get_range(x_var)
            if debug and self.use_agent_specific_range:
                print(f"Warning: Per-agent range not available for {x_var}, using global range")

        x_min = x_range_info["percentiles"][0]  # q5
        x_max = x_range_info["percentiles"][4]  # q95
        x_values = np.linspace(x_min, x_max, n_points)

        if debug:
            print(f"X-axis range for {x_var}: [{x_min:.4f}, {x_max:.4f}]")

        # Resolve fixed variables (use mean for unspecified vars)
        fixed_values = self._resolve_fixed_vars(x_var, color_var, fixed_vars)

        # Determine color levels
        if color_var is not None:
            if color_var == "s_t":
                # Discrete: just Normal and Superstar
                color_levels = ["Normal", "Superstar"]
                color_values = [0.0, 1.0]
            else:
                color_levels = self.COLOR_QUANTILES
                # Use agent-specific quantiles if enabled
                color_values = [
                    self._get_quantile_value(color_var, q)
                    for q in color_levels
                ]
        else:
            color_levels = ["mean"]
            color_values = [None]  # Not used

        # Evaluate at each color level
        y_values = {}

        with torch.no_grad():
            for c_idx, (c_label, c_val) in enumerate(zip(color_levels, color_values)):
                y_arr = []

                # Debug: print info for first and last color level at middle x point
                if debug and c_idx in [0, len(color_levels) - 1]:
                    print(f"\n{'#'*60}")
                    print(f"# Evaluating color_var={color_var}, level={c_label}, value={c_val}")
                    print(f"# x_var={x_var}, range=[{x_values[0]:.4f}, {x_values[-1]:.4f}]")
                    print(f"# Fixed vars: {fixed_values}")
                    print(f"{'#'*60}")

                for x_idx, x_val in enumerate(x_values):
                    # Build state dict
                    state_dict = dict(fixed_values)  # Start with fixed values
                    state_dict[x_var] = x_val
                    if color_var is not None:
                        state_dict[color_var] = c_val

                    # Debug at middle point for first and last color level
                    do_debug = debug and (c_idx in [0, len(color_levels) - 1]) and (x_idx == n_points // 2)
                    debug_label = f"{c_label}, x={x_val:.2f}" if do_debug else ""

                    # Evaluate policy
                    output = self._evaluate_single_point(state_dict, debug=do_debug, debug_label=debug_label)
                    y_arr.append(output[y_var])

                y_values[c_label] = np.array(y_arr)

                # Debug: print summary for this color level
                if debug:
                    print(f"Color {c_label}: y_var={y_var} range=[{min(y_arr):.4f}, {max(y_arr):.4f}]")

        return {
            "x_values": x_values,
            "y_values": y_values,
            "color_var": color_var,
            "color_levels": color_levels,
            "color_values": color_values,
            "fixed_values": fixed_values,
            "x_var": x_var,
            "y_var": y_var,
        }

    def _resolve_fixed_vars(
        self,
        x_var: str,
        color_var: Optional[str],
        user_fixed: Optional[Dict[str, Union[str, float, bool]]]
    ) -> Dict[str, float]:
        """
        Resolve all fixed variable values.

        Variables not on x-axis or color are fixed at their MEAN from training.

        SPECIAL CASE: When x_var="a_t", m_t is NOT fixed but computed from a_t
        using the economic model (m_t = f(a_t, v_t, l_t, wage, ret)).
        This is marked by setting fixed["m_t"] = "COMPUTED_FROM_A".

        Args:
            x_var: Variable on x-axis (excluded)
            color_var: Variable for color (excluded)
            user_fixed: User-specified fixed values (overrides mean)

        Returns:
            Dict of {var_name: float_value or "COMPUTED_FROM_A"} for all fixed variables
        """
        # All state variables that need values
        all_state_vars = ["m_t", "a_t", "v_t", "s_t"]

        fixed = {}
        for var in all_state_vars:
            # Skip x-axis and color variables
            if var == x_var:
                continue
            if var == color_var:
                continue

            # SPECIAL CASE: When a_t is x-axis, m_t should be computed from a_t
            # (not fixed) to show history dependence
            if var == "m_t" and x_var == "a_t":
                fixed[var] = "COMPUTED_FROM_A"
                continue

            # Check if user specified a value
            if user_fixed and var in user_fixed:
                # Use resolve_condition_value for user-specified values
                # (handles bool, float, and quantile strings)
                spec = user_fixed[var]
                if isinstance(spec, (bool, int, float)):
                    fixed[var] = float(spec) if isinstance(spec, (int, float)) else (1.0 if spec else 0.0)
                elif isinstance(spec, str):
                    # Use agent-specific range if enabled
                    fixed[var] = self._get_quantile_value(var, spec)
                else:
                    fixed[var] = resolve_condition_value(var, spec, self.ranges)
            else:
                # Default to mean (or False for s_t)
                if var == "s_t":
                    fixed[var] = 0.0  # Default: Normal (not superstar)
                else:
                    # Use agent-specific mean if enabled
                    fixed[var] = self._get_quantile_value(var, "mean")

        return fixed

    def _evaluate_single_point(
        self,
        state_dict: Dict[str, Union[float, str]],
        debug: bool = False,
        debug_label: str = ""
    ) -> Dict[str, float]:
        """
        Evaluate policy at a single state point using GE-aware evaluation.

        IMPORTANT: This method now follows the exact same normalization flow as training:
        1. Normalize ability and moneydisposable SEPARATELY using the normalizer
        2. Call build_inputs() with the normalized values
        3. Pass to policy network

        The model expects input shape (B, A, 2A+2) where A = n_agents.
        We use actual simulation data as the background population and only
        modify agent 0's inputs. This preserves realistic aggregate features.

        Args:
            state_dict: Dict with m_t, a_t, v_t, s_t values
            debug: If True, print debug information
            debug_label: Label for debug output

        Returns:
            Dict with all decision variables: zeta_t, c_t, a_tp1, l_t, mu_t, da_t, I_bind
        """
        # Extract target values for agent 0
        a_t = state_dict["a_t"]
        v_t = state_dict["v_t"]
        # s_t is available but not directly used in policy input

        # Handle m_t: either from state_dict or computed from a_t
        m_t_spec = state_dict["m_t"]
        if m_t_spec == "COMPUTED_FROM_A":
            # Compute m_t from a_t using economic model
            m_t = self._compute_m_from_a(a_t=a_t, v_t=v_t)
        else:
            m_t = float(m_t_spec)

        # Start from reference population (actual GE simulation data)
        # These are RAW (unnormalized) values
        moneydisposable = self.ref_money.clone()  # (1, A)
        ability = self.ref_ability.clone()  # (1, A)

        # Only modify agent 0's values to the target state
        moneydisposable[0, 0] = m_t
        ability[0, 0] = v_t

        # DEBUG: Log input before normalization
        if debug:
            print(f"\n{'='*60}")
            print(f"DEBUG [{debug_label}]: Input state: m_t={m_t:.4f}, a_t={a_t:.4f}, v_t={v_t:.4f}")
            print(f"DEBUG [{debug_label}]: Agent 0 RAW values:")
            print(f"  - moneydisposable[0,0] = {moneydisposable[0, 0].item():.4f}")
            print(f"  - ability[0,0] = {ability[0, 0].item():.4f}")

        # ============================================================
        # CRITICAL: Follow the EXACT same normalization flow as training
        # (see src/environment.py lines 76-83)
        # 1. Normalize ability separately
        # 2. Normalize moneydisposable separately (note: typo "moneydisposalbe" in training)
        # 3. Call build_inputs with normalized values
        # ============================================================

        ability_normalized = self.normalizer.transform("ability", ability, update=False)
        moneydisposable_normalized = self.normalizer.transform("moneydisposalbe", moneydisposable, update=False)

        # DEBUG: Log normalized values
        if debug:
            print(f"DEBUG [{debug_label}]: Agent 0 NORMALIZED values:")
            print(f"  - moneydisposable_norm[0,0] = {moneydisposable_normalized[0, 0].item():.4f}")
            print(f"  - ability_norm[0,0] = {ability_normalized[0, 0].item():.4f}")

            # Print normalizer statistics
            if "ability" in self.normalizer._stats:
                stats = self.normalizer._stats["ability"]
                print(f"DEBUG [{debug_label}]: Ability normalizer stats:")
                print(f"  - global_mode = {self.normalizer.global_mode}")
                if self.normalizer.global_mode:
                    print(f"  - mean = {stats.mean.item():.4f}")
                    var = stats.M2 / torch.clamp(stats.count - 1.0, min=1.0)
                    std = torch.sqrt(torch.clamp(var, min=0.0) + self.normalizer.eps)
                    print(f"  - std = {std.item():.4f}")
                else:
                    print(f"  - mean[0] = {stats.mean[0].item():.4f}")

            if "moneydisposalbe" in self.normalizer._stats:
                stats = self.normalizer._stats["moneydisposalbe"]
                print(f"DEBUG [{debug_label}]: Moneydisposable normalizer stats:")
                if self.normalizer.global_mode:
                    print(f"  - mean = {stats.mean.item():.4f}")
                    var = stats.M2 / torch.clamp(stats.count - 1.0, min=1.0)
                    std = torch.sqrt(torch.clamp(var, min=0.0) + self.normalizer.eps)
                    print(f"  - std = {std.item():.4f}")
                else:
                    print(f"  - mean[0] = {stats.mean[0].item():.4f}")

        # Build model inputs using the SAME function as training
        features, condi = build_inputs(
            moneydisposable=moneydisposable_normalized,
            ability=ability_normalized,
            tax_params=self.tax_params.unsqueeze(0),  # (1, Z)
            device=self.device
        )
        # features: (1, A, 2A+2), condi: (1, A, Z)

        # DEBUG: Log built features
        if debug:
            print(f"DEBUG [{debug_label}]: Built features shape: {features.shape}")
            print(f"DEBUG [{debug_label}]: Agent 0 features:")
            print(f"  - features[0,0,-2] (money_self) = {features[0, 0, -2].item():.4f}")
            print(f"  - features[0,0,-1] (ability_self) = {features[0, 0, -1].item():.4f}")
            print(f"  - features[0,0,:4] (sum_info first 4) = {features[0, 0, :4].cpu().numpy()}")

        # Forward pass (features are already normalized)
        zeta_raw = self.policy_net(features, condi)  # (1, A, 1)

        # DEBUG: Log raw output
        if debug:
            print(f"DEBUG [{debug_label}]: Model output:")
            print(f"  - zeta_raw[0,0,0] = {zeta_raw[0, 0, 0].item():.6f}")
            print(f"  - sigmoid(zeta_raw) = {torch.sigmoid(zeta_raw[0, 0, 0]).item():.6f}")

        # Extract decision for agent 0 (the target agent with our specified state)
        zeta_t = float(torch.sigmoid(zeta_raw[0, 0, 0]).cpu())

        # Compute derived quantities
        c_t = (1.0 - zeta_t) * m_t
        a_tp1 = zeta_t * m_t
        da_t = a_tp1 - a_t
        I_bind = 1.0 if a_tp1 < 1e-6 else 0.0

        # Placeholders for variables that need additional models
        l_t = 0.0  # Would need labor supply model
        mu_t = 0.0  # Would need complementarity computation

        return {
            "zeta_t": zeta_t,
            "c_t": c_t,
            "a_tp1": a_tp1,
            "l_t": l_t,
            "mu_t": mu_t,
            "da_t": da_t,
            "I_bind": I_bind,
        }
