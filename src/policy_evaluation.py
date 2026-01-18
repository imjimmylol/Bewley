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
        """Update statistics with new values."""
        values = values.flatten()

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
    """
    m_t: VariableRanges = field(default_factory=VariableRanges)
    a_t: VariableRanges = field(default_factory=VariableRanges)
    v_t: VariableRanges = field(default_factory=VariableRanges)
    y_t: VariableRanges = field(default_factory=VariableRanges)

    # Output variables (for reference, not for conditioning)
    c_t: VariableRanges = field(default_factory=VariableRanges)
    zeta_t: VariableRanges = field(default_factory=VariableRanges)
    l_t: VariableRanges = field(default_factory=VariableRanges)

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
        Get a specific quantile value for a variable.

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
    existing_ranges: Optional[HistoricalRanges] = None
) -> HistoricalRanges:
    """
    Update historical ranges from a single training step.

    This function should be called every training step to accumulate
    statistics about variable distributions. It's designed to be lightweight.

    Args:
        temp_state: TemporaryState instance with current step data
        main_state: MainState instance (for input savings a_t)
        existing_ranges: Existing HistoricalRanges to update (or None to create new)

    Returns:
        Updated HistoricalRanges instance
    """
    if existing_ranges is None:
        existing_ranges = HistoricalRanges()

    # Helper to convert tensor to numpy
    def to_np(x: Tensor) -> np.ndarray:
        if isinstance(x, Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # Update continuous variables from TemporaryState
    existing_ranges.m_t.update(to_np(temp_state.money_disposable))
    existing_ranges.v_t.update(to_np(temp_state.ability))
    existing_ranges.y_t.update(to_np(temp_state.income_before_tax))
    existing_ranges.c_t.update(to_np(temp_state.consumption))
    existing_ranges.zeta_t.update(to_np(temp_state.savings_ratio))
    existing_ranges.l_t.update(to_np(temp_state.labor))

    # a_t is the savings INPUT (from main_state), not the output
    existing_ranges.a_t.update(to_np(main_state.savings))

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

    Uses "Option A: Single-agent mode" - ignores aggregate effects and evaluates
    the policy network with synthetic (1, 1) shaped inputs.

    Three types of variables:
    1. x-axis: varies systematically on a grid (range from q5 to q95 of training data)
    2. color/panel: conditioning variable evaluated at 5 quantile levels (q5, q25, q50, q75, q95)
    3. fixed: held constant at MEAN from training data

    Usage:
        evaluator = PolicyEvaluator(policy_net, normalizer, ranges, tax_params, device)
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
        device: str = "cpu"
    ):
        """
        Initialize PolicyEvaluator.

        Args:
            policy_net: Trained policy network (FiLMResNet2In)
            normalizer: RunningPerAgentWelford normalizer for inputs
            ranges: HistoricalRanges with collected statistics from training
            tax_params: Tax parameters tensor (Z,) - will be used for all evaluations
            device: Device to run evaluation on
        """
        self.policy_net = policy_net
        self.normalizer = normalizer
        self.ranges = ranges
        self.device = device

        # Ensure tax_params is (Z,) shape
        tax_params = torch.as_tensor(tax_params, dtype=torch.float32, device=device)
        if tax_params.dim() == 2:
            tax_params = tax_params[0]  # Take first batch if (B, Z)
        self.tax_params = tax_params  # (Z,)

        # Put model in eval mode
        self.policy_net.eval()

    def evaluate_on_grid(
        self,
        x_var: str,
        y_var: str,
        color_var: Optional[str] = None,
        fixed_vars: Optional[Dict[str, Union[str, float, bool]]] = None,
        n_points: int = 100
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

        # Determine x-axis grid range (q5 to q95 from training data)
        x_range_info = self.ranges.get_range(x_var)
        x_min = x_range_info["percentiles"][0]  # q5
        x_max = x_range_info["percentiles"][4]  # q95
        x_values = np.linspace(x_min, x_max, n_points)

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
                color_values = [
                    self.ranges.get_quantile_value(color_var, q)
                    for q in color_levels
                ]
        else:
            color_levels = ["mean"]
            color_values = [None]  # Not used

        # Evaluate at each color level
        y_values = {}

        with torch.no_grad():
            for c_label, c_val in zip(color_levels, color_values):
                y_arr = []

                for x_val in x_values:
                    # Build state dict
                    state_dict = dict(fixed_values)  # Start with fixed values
                    state_dict[x_var] = x_val
                    if color_var is not None:
                        state_dict[color_var] = c_val

                    # Evaluate policy
                    output = self._evaluate_single_point(state_dict)
                    y_arr.append(output[y_var])

                y_values[c_label] = np.array(y_arr)

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

        Args:
            x_var: Variable on x-axis (excluded)
            color_var: Variable for color (excluded)
            user_fixed: User-specified fixed values (overrides mean)

        Returns:
            Dict of {var_name: float_value} for all fixed variables
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

            # Check if user specified a value
            if user_fixed and var in user_fixed:
                fixed[var] = resolve_condition_value(var, user_fixed[var], self.ranges)
            else:
                # Default to mean (or False for s_t)
                if var == "s_t":
                    fixed[var] = 0.0  # Default: Normal (not superstar)
                else:
                    fixed[var] = self.ranges.get_quantile_value(var, "mean")

        return fixed

    def _evaluate_single_point(self, state_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate policy at a single state point using Option A (single-agent).

        Args:
            state_dict: Dict with m_t, a_t, v_t, s_t values

        Returns:
            Dict with all decision variables: zeta_t, c_t, a_tp1, l_t, mu_t, da_t, I_bind
        """
        # Extract values
        m_t = state_dict["m_t"]
        a_t = state_dict["a_t"]
        v_t = state_dict["v_t"]
        # s_t is available but not directly used in single-agent policy input

        # Build single-agent input (Option A)
        # For single agent: B=1, A=1
        # features shape: (1, 1, 2*1+2) = (1, 1, 4)
        # The feature structure from build_inputs:
        # [sum_info_rep (all money, all ability), money_self, ability_self]
        # For A=1: [m_t, v_t, m_t, v_t] -> (1, 1, 4)

        money = torch.tensor([[m_t]], dtype=torch.float32, device=self.device)  # (1, 1)
        ability = torch.tensor([[v_t]], dtype=torch.float32, device=self.device)  # (1, 1)

        # Build features manually for single-agent case
        sum_info = torch.cat([money, ability], dim=1)  # (1, 2)
        sum_info_rep = sum_info.unsqueeze(1)  # (1, 1, 2)
        money_self = money.unsqueeze(-1)  # (1, 1, 1)
        ability_self = ability.unsqueeze(-1)  # (1, 1, 1)
        features = torch.cat([sum_info_rep, money_self, ability_self], dim=2)  # (1, 1, 4)

        # Condition (tax params)
        condi = self.tax_params.unsqueeze(0).unsqueeze(0)  # (1, 1, Z)

        # Normalize features
        features_norm = self.normalizer.transform("state", features, update=False)

        # Forward pass
        zeta_raw = self.policy_net(features_norm, condi)  # (1, 1, 1)
        zeta_t = float(torch.sigmoid(zeta_raw).squeeze().cpu())

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
