# src/monitoring.py
"""
Training monitoring module for logging metrics, correlations, and debugging info.
"""

import wandb
import numpy as np
import torch


class TrainingMonitor:
    """
    Handles metrics extraction, computation, and logging during training.

    Responsibilities:
    - Extract metrics from states, losses, and outcomes
    - Compute correlations (current + lagged)
    - Check normalizer health
    - Verify budget constraints
    - Log to console (at display_step frequency)
    - Log to wandb (every step if enabled)
    """

    def __init__(self, config, normalizer):
        """
        Initialize training monitor.

        Args:
            config: Configuration namespace
            normalizer: Running normalizer instance
        """
        self.config = config
        self.normalizer = normalizer

        # State for lagged correlation analysis
        self.prev_ability = None
        self.prev_money = None

    def log_step(self, step, main_state, temp_state, loss_dict):
        """
        Main logging method called every training step.

        Args:
            step: Current training step
            main_state: MainState instance
            temp_state: TemporaryState instance
            loss_dict: Dictionary with loss components (total, fb, aux_mu, labor)
        """
        # Determine if we should log this step
        should_display = (step % self.config.training.display_step == 0 or
                         step == 1)
        should_log_wandb = wandb.run is not None

        # Extract all metrics (only if logging is needed)
        if should_display or should_log_wandb:
            metrics = self._extract_metrics(main_state, temp_state, loss_dict)

            # Log to console
            if should_display:
                self._log_to_console(step, metrics)

            # Log to wandb
            if should_log_wandb:
                self._log_to_wandb(step, metrics)

            # Store current state for next iteration's lagged correlation
            self.prev_ability = metrics['ability_array'].copy()
            self.prev_money = metrics['money_array'].copy()

    def _extract_metrics(self, main_state, temp_state, loss_dict):
        """
        Extract and convert all metrics to scalars/numpy arrays.

        Returns:
            dict: All extracted metrics
        """
        metrics = {}

        # === LOSS COMPONENTS ===
        metrics['loss_total'] = loss_dict["total"].item()
        metrics['loss_fb'] = loss_dict["fb"].item()
        metrics['loss_aux_mu'] = loss_dict["aux_mu"].item()
        metrics['loss_labor'] = loss_dict["labor"].item()

        # === STATE STATISTICS ===
        metrics['consumption_mean'] = temp_state.consumption.mean().item()
        metrics['labor_mean'] = temp_state.labor.mean().item()
        metrics['savings_mean'] = main_state.savings.mean().item()
        metrics['ability_mean'] = main_state.ability.mean().item()
        metrics['wage_mean'] = temp_state.wage.mean().item()
        metrics['ret_mean'] = temp_state.ret.mean().item()

        # === RAW STATE VALUES (for debugging) ===
        metrics['money_raw_mean'] = main_state.moneydisposable.mean().item()
        metrics['ability_raw_mean'] = main_state.ability.mean().item()
        metrics['money_raw_std'] = main_state.moneydisposable.std().item()
        metrics['ability_raw_std'] = main_state.ability.std().item()

        # === AGENT ACTIONS ===
        metrics['mu_mean'] = temp_state.mu.mean().item()
        metrics['savings_ratio_mean'] = temp_state.savings_ratio.mean().item()

        # For wandb histograms (keep as numpy arrays)
        metrics['mu_array'] = temp_state.mu.detach().cpu().numpy().flatten()
        metrics['savings_ratio_array'] = temp_state.savings_ratio.detach().cpu().numpy().flatten()

        # === BUDGET CONSTRAINT VERIFICATION ===
        budget_metrics = self._verify_budget_constraint(temp_state)
        metrics.update(budget_metrics)

        # === NORMALIZER HEALTH CHECK ===
        normalizer_metrics = self._check_normalizer_health(main_state)
        metrics.update(normalizer_metrics)

        # === CORRELATION ANALYSIS ===
        # Extract arrays for correlation computation
        metrics['ability_array'] = main_state.ability.detach().cpu().numpy().flatten()
        metrics['money_array'] = main_state.moneydisposable.detach().cpu().numpy().flatten()
        metrics['labor_array'] = temp_state.labor.detach().cpu().numpy().flatten()

        correlation_metrics = self._compute_correlations(
            metrics['ability_array'],
            metrics['money_array'],
            metrics['labor_array']
        )
        metrics.update(correlation_metrics)

        return metrics

    def _verify_budget_constraint(self, temp_state):
        """
        Verify budget constraint: consumption[t] + savings[t+1] = money_disposable[t]

        Returns:
            dict: Budget verification metrics
        """
        money_disposable_mean = temp_state.money_disposable.mean().item()
        consumption_mean = temp_state.consumption.mean().item()
        savings_mean = temp_state.savings.mean().item()

        savings_plus_cons = savings_mean + consumption_mean
        budget_diff = money_disposable_mean - savings_plus_cons
        budget_ratio = (savings_plus_cons / money_disposable_mean
                       if abs(money_disposable_mean) > 1e-8 else 0.0)

        return {
            'money_disposable_mean': money_disposable_mean,
            'savings_plus_consumption': savings_plus_cons,
            'budget_difference': budget_diff,
            'budget_ratio': budget_ratio
        }

    def _check_normalizer_health(self, main_state):
        """
        Compare normalizer statistics vs actual state values.
        Detects normalizer drift.

        Returns:
            dict: Normalizer health metrics
        """
        metrics = {}

        # Money statistics from normalizer
        if "moneydisposalbe" in self.normalizer._stats:  # Note: typo in key from environment.py
            money_norm_stats = self.normalizer._stats["moneydisposalbe"]
            money_norm_mean = money_norm_stats.mean.mean().item()
            money_norm_std = torch.sqrt(
                money_norm_stats.M2 / torch.clamp(money_norm_stats.count - 1, min=1.0)
            ).mean().item()
            money_norm_count = money_norm_stats.count.mean().item()
        else:
            money_norm_mean = 0.0
            money_norm_std = 1.0
            money_norm_count = 0.0

        # Ability statistics from normalizer
        if "ability" in self.normalizer._stats:
            ability_norm_stats = self.normalizer._stats["ability"]
            ability_norm_mean = ability_norm_stats.mean.mean().item()
            ability_norm_std = torch.sqrt(
                ability_norm_stats.M2 / torch.clamp(ability_norm_stats.count - 1, min=1.0)
            ).mean().item()
            ability_norm_count = ability_norm_stats.count.mean().item()
        else:
            ability_norm_mean = 0.0
            ability_norm_std = 1.0
            ability_norm_count = 0.0

        # Actual state statistics
        money_raw_mean = main_state.moneydisposable.mean().item()
        money_raw_std = main_state.moneydisposable.std().item()
        ability_raw_mean = main_state.ability.mean().item()
        ability_raw_std = main_state.ability.std().item()

        # Compute discrepancies
        money_mean_discrepancy = money_raw_mean - money_norm_mean
        money_std_discrepancy = money_raw_std - money_norm_std
        ability_mean_discrepancy = ability_raw_mean - ability_norm_mean
        ability_std_discrepancy = ability_raw_std - ability_norm_std

        # Relative discrepancy (percentage)
        money_mean_drift_percent = (abs(money_mean_discrepancy) /
                                    max(abs(money_norm_mean), 1e-6) * 100)
        ability_mean_drift_percent = (abs(ability_mean_discrepancy) /
                                      max(abs(ability_norm_mean), 1e-6) * 100)

        metrics.update({
            'money_norm_mean': money_norm_mean,
            'money_norm_std': money_norm_std,
            'money_norm_count': money_norm_count,
            'ability_norm_mean': ability_norm_mean,
            'ability_norm_std': ability_norm_std,
            'ability_norm_count': ability_norm_count,
            'money_mean_discrepancy': money_mean_discrepancy,
            'money_std_discrepancy': money_std_discrepancy,
            'ability_mean_discrepancy': ability_mean_discrepancy,
            'ability_std_discrepancy': ability_std_discrepancy,
            'money_mean_drift_percent': money_mean_drift_percent,
            'ability_mean_drift_percent': ability_mean_drift_percent,
        })

        return metrics

    def _compute_correlations(self, ability, money, labor):
        """
        Compute current and lagged correlations.

        Args:
            ability: Current ability array (flattened)
            money: Current money array (flattened)
            labor: Current labor array (flattened)

        Returns:
            dict: Correlation metrics
        """
        metrics = {}

        # Current correlations
        metrics['corr_ability_money_current'] = np.corrcoef(ability, money)[0, 1]
        metrics['corr_ability_labor_current'] = np.corrcoef(ability, labor)[0, 1]

        # Lagged correlations (if previous state exists)
        if self.prev_ability is not None and self.prev_money is not None:
            # ability[t-1] vs money[t]
            metrics['corr_ability_money_lagged'] = np.corrcoef(
                self.prev_ability, money
            )[0, 1]
            # ability[t] vs money[t-1]
            metrics['corr_ability_current_money_lagged'] = np.corrcoef(
                ability, self.prev_money
            )[0, 1]
        else:
            metrics['corr_ability_money_lagged'] = 0.0
            metrics['corr_ability_current_money_lagged'] = 0.0

        return metrics

    def _log_to_console(self, step, metrics):
        """
        Print formatted metrics to console.

        Args:
            step: Current training step
            metrics: Dictionary of metrics
        """
        print(f"\nStep {step}/{self.config.training.training_steps}")
        print(f"  Loss: {metrics['loss_total']:.4f} "
              f"(fb={metrics['loss_fb']:.4f}, "
              f"euler={metrics['loss_aux_mu']:.4f}, "
              f"labor={metrics['loss_labor']:.4f})")

        print(f"  Agent Actions:")
        print(f"    - Mean mu (consumption/money): {metrics['mu_mean']:.3f}")
        print(f"    - Mean savings ratio: {metrics['savings_ratio_mean']:.3f}")

        print(f"  State Statistics:")
        print(f"    - Mean consumption: {metrics['consumption_mean']:.3f}")
        print(f"    - Mean labor: {metrics['labor_mean']:.3f}")
        print(f"    - Mean savings: {metrics['savings_mean']:.3f}")
        print(f"    - Mean ability: {metrics['ability_mean']:.3f}")
        print(f"    - Market wage: {metrics['wage_mean']:.3f}")
        print(f"    - Market return: {metrics['ret_mean']:.4f}")

        print(f"  Normalizer Health Check:")
        print(f"    - Money: normalizer_mean={metrics['money_norm_mean']:.3f}, "
              f"actual_mean={metrics['money_raw_mean']:.3f}, "
              f"discrepancy={metrics['money_mean_discrepancy']:.3f}")
        print(f"    - Money: normalizer_std={metrics['money_norm_std']:.3f}, "
              f"actual_std={metrics['money_raw_std']:.3f}, "
              f"discrepancy={metrics['money_std_discrepancy']:.3f}")
        print(f"    - Ability: normalizer_mean={metrics['ability_norm_mean']:.3f}, "
              f"actual_mean={metrics['ability_raw_mean']:.3f}, "
              f"discrepancy={metrics['ability_mean_discrepancy']:.3f}")
        print(f"    - Ability: normalizer_std={metrics['ability_norm_std']:.3f}, "
              f"actual_std={metrics['ability_raw_std']:.3f}, "
              f"discrepancy={metrics['ability_std_discrepancy']:.3f}")
        print(f"    - Normalizer update count: "
              f"money={metrics['money_norm_count']:.0f}, "
              f"ability={metrics['ability_norm_count']:.0f}")

        print(f"  Experiment 2: Ability-Money & Ability-Labor Correlation:")
        print(f"    - Corr(ability[t], money[t]): "
              f"{metrics['corr_ability_money_current']:.4f}")
        print(f"    - Corr(ability[t], labor[t]): "
              f"{metrics['corr_ability_labor_current']:.4f}")

        if self.prev_ability is not None:
            print(f"    - Corr(ability[t-1], money[t]): "
                  f"{metrics['corr_ability_money_lagged']:.4f}")
            print(f"    - Corr(ability[t], money[t-1]): "
                  f"{metrics['corr_ability_current_money_lagged']:.4f}")

        print(f"  Budget Constraint Verification:")
        print(f"    - Money disposable[t]: {metrics['money_disposable_mean']:.3f}")
        print(f"    - Consumption[t]: {metrics['consumption_mean']:.3f}")
        print(f"    - Savings[t+1]: {metrics['savings_mean']:.3f}")
        print(f"    - Savings + Consumption: {metrics['savings_plus_consumption']:.3f}")
        print(f"    - Difference (should be ~0): {metrics['budget_difference']:.6f}")
        print(f"    - Ratio (should be ~1.0): {metrics['budget_ratio']:.6f}")

    def _log_to_wandb(self, step, metrics):
        """
        Log all metrics to wandb.

        Args:
            step: Current training step
            metrics: Dictionary of metrics
        """
        wandb.log({
            # Losses
            "loss/total": metrics['loss_total'],
            "loss/fb": metrics['loss_fb'],
            "loss/aux_mu": metrics['loss_aux_mu'],
            "loss/labor_foc": metrics['loss_labor'],

            # State statistics
            "state/consumption_mean": metrics['consumption_mean'],
            "state/labor_mean": metrics['labor_mean'],
            "state/savings_mean": metrics['savings_mean'],
            "state/ability_mean": metrics['ability_mean'],
            "state/money_disposable_mean": metrics['money_disposable_mean'],
            "state/savings_plus_consumption": metrics['savings_plus_consumption'],
            "state/budget_difference": metrics['budget_difference'],
            "state/budget_ratio": metrics['budget_ratio'],

            # Market variables
            "market/wage": metrics['wage_mean'],
            "market/return": metrics['ret_mean'],

            # Agent actions
            "actions/mu_mean": metrics['mu_mean'],
            "actions/savings_ratio_mean": metrics['savings_ratio_mean'],
            "actions/mu_distribution": wandb.Histogram(metrics['mu_array']),
            "actions/savings_ratio_distribution": wandb.Histogram(metrics['savings_ratio_array']),

            # Debug: raw state values
            "debug/raw_money_mean": metrics['money_raw_mean'],
            "debug/raw_ability_mean": metrics['ability_raw_mean'],
            "debug/raw_money_std": metrics['money_raw_std'],
            "debug/raw_ability_std": metrics['ability_raw_std'],

            # Normalizer statistics
            "normalizer/money_mean": metrics['money_norm_mean'],
            "normalizer/money_std": metrics['money_norm_std'],
            "normalizer/money_count": metrics['money_norm_count'],
            "normalizer/ability_mean": metrics['ability_norm_mean'],
            "normalizer/ability_std": metrics['ability_norm_std'],
            "normalizer/ability_count": metrics['ability_norm_count'],

            # Normalizer health: discrepancies
            "normalizer/money_mean_discrepancy": metrics['money_mean_discrepancy'],
            "normalizer/money_std_discrepancy": metrics['money_std_discrepancy'],
            "normalizer/ability_mean_discrepancy": metrics['ability_mean_discrepancy'],
            "normalizer/ability_std_discrepancy": metrics['ability_std_discrepancy'],
            "normalizer/money_mean_drift_percent": metrics['money_mean_drift_percent'],
            "normalizer/ability_mean_drift_percent": metrics['ability_mean_drift_percent'],

            # Experiment 2: Correlations
            "experiment2/corr_ability_money_current": metrics['corr_ability_money_current'],
            "experiment2/corr_ability_money_lagged": metrics['corr_ability_money_lagged'],
            "experiment2/corr_ability_current_money_lagged": metrics['corr_ability_current_money_lagged'],
            "experiment2/corr_ability_labor_current": metrics['corr_ability_labor_current'],

            "step": step
        })
