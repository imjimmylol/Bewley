"""
src/cluster_analysis.py

GMM-based regime tracking for endogenous behavioral bifurcation.

Implements the 17-step procedure from Cluster.md:
  - PanelBuffer: ring buffer for per-step agent panel data
  - RegimeAnalyzer: full LOWESS detrend → z-score → GMM pipeline
  - lightweight_cluster_snapshot: fast cross-sectional GMM for training monitoring
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# PanelBuffer — ring buffer for agent panel data (batch 0)
# ---------------------------------------------------------------------------

class PanelBuffer:
    """
    Ring buffer that accumulates per-step agent outcomes for one batch.

    Agents within a batch persist across training steps (MainState carries
    forward), so batch 0 gives a genuine panel: agent_id × time.

    Args:
        max_steps: Maximum number of time steps to keep (oldest overwritten).
        n_agents: Number of agents (A dimension).
        tracked_vars: Variable names to track.
    """

    TRACKED_VARS = [
        'ability', 'saving_ratio', 'mu', 'labor',
        'money_disposable', 'consumption', 'savings',
        'income_before_tax', 'wage', 'ret'
    ]

    def __init__(
        self,
        max_steps: int = 200,
        n_agents: int = 100,
        tracked_vars: Optional[List[str]] = None,
    ):
        self.max_steps = max_steps
        self.n_agents = n_agents
        self.tracked_vars = tracked_vars or self.TRACKED_VARS

        # Pre-allocate ring buffer arrays: (max_steps, n_agents)
        self._data: Dict[str, np.ndarray] = {
            var: np.full((max_steps, n_agents), np.nan, dtype=np.float32)
            for var in self.tracked_vars
        }
        self._steps = np.full(max_steps, -1, dtype=np.int64)  # training step index
        self._ptr = 0    # write pointer
        self._count = 0  # number of filled slots

    def append(self, step: int, data: Dict[str, np.ndarray]) -> None:
        """
        Append one training step's data (from batch 0).

        Args:
            step: Training step index.
            data: Dict mapping var_name -> np.ndarray of shape (n_agents,).
        """
        idx = self._ptr % self.max_steps
        self._steps[idx] = step
        for var in self.tracked_vars:
            if var in data:
                arr = np.asarray(data[var], dtype=np.float32).ravel()
                n = min(len(arr), self.n_agents)
                self._data[var][idx, :n] = arr[:n]
        self._ptr += 1
        self._count = min(self._count + 1, self.max_steps)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the buffer to a long-format pandas DataFrame.

        Columns: agent_id, time, <tracked_vars...>
        Only includes filled slots, sorted by time.
        """
        if self._count == 0:
            cols = ['agent_id', 'time'] + self.tracked_vars
            return pd.DataFrame(columns=cols)

        # Get the valid slice of the ring buffer in chronological order
        if self._count < self.max_steps:
            # Buffer not yet full: slots 0..(count-1) in insertion order
            indices = np.arange(self._count)
        else:
            # Full buffer: oldest slot is at _ptr % max_steps
            oldest = self._ptr % self.max_steps
            indices = np.roll(np.arange(self.max_steps), -oldest)

        n_steps = len(indices)
        steps_used = self._steps[indices]  # shape (n_steps,)

        rows = []
        for j, slot in enumerate(indices):
            t = int(steps_used[j])
            for agent_id in range(self.n_agents):
                row = {'agent_id': agent_id, 'time': t}
                for var in self.tracked_vars:
                    row[var] = float(self._data[var][slot, agent_id])
                rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values(['time', 'agent_id']).reset_index(drop=True)
        return df

    def to_npz(self, path: str) -> None:
        """Save buffer state to .npz file."""
        save_dict = {
            '_steps': self._steps,
            '_ptr': np.array(self._ptr, dtype=np.int64),
            '_count': np.array(self._count, dtype=np.int64),
            '_max_steps': np.array(self.max_steps, dtype=np.int64),
            '_n_agents': np.array(self.n_agents, dtype=np.int64),
            '_tracked_vars': np.array(self.tracked_vars, dtype='U64'),
        }
        for var in self.tracked_vars:
            save_dict[f'data_{var}'] = self._data[var]
        np.savez(path, **save_dict)

    @classmethod
    def from_npz(cls, path: str) -> 'PanelBuffer':
        """Load buffer state from .npz file."""
        data = np.load(path, allow_pickle=False)
        max_steps = int(data['_max_steps'])
        n_agents = int(data['_n_agents'])
        tracked_vars = list(data['_tracked_vars'])
        buf = cls(max_steps=max_steps, n_agents=n_agents, tracked_vars=tracked_vars)
        buf._steps = data['_steps'].copy()
        buf._ptr = int(data['_ptr'])
        buf._count = int(data['_count'])
        for var in tracked_vars:
            buf._data[var] = data[f'data_{var}'].copy()
        return buf

    def __len__(self) -> int:
        return self._count


# ---------------------------------------------------------------------------
# lightweight_cluster_snapshot — fast cross-sectional GMM for training
# ---------------------------------------------------------------------------

def lightweight_cluster_snapshot(
    ability: np.ndarray,
    saving_ratio: np.ndarray,
    mu: np.ndarray,
    labor: np.ndarray,
    ability_cutoff_quantile: float = 0.5,
    labor_upper: float = 0.95,
    n_components: int = 2,
    random_state: int = 42,
) -> Dict:
    """
    Fast cross-sectional GMM for inline training monitoring.

    Skips LOWESS detrending for speed; uses raw z-scores. Suitable for
    periodic wandb logging to track whether bifurcation is forming.

    Args:
        ability, saving_ratio, mu, labor: Flattened 1-D arrays (all batches).
        ability_cutoff_quantile: Filter to ability >= this quantile.
        labor_upper: Filter to labor < this threshold (interior region).
        n_components: Number of GMM components.
        random_state: Random seed for reproducibility.

    Returns:
        Dict with keys:
            regime_fraction_high, regime_fraction_low,
            separation_score (Mahalanobis-like distance between centers),
            bic, aic, labels (array aligned to filtered observations)
    """
    from sklearn.mixture import GaussianMixture

    ability = np.asarray(ability, dtype=np.float32).ravel()
    saving_ratio = np.asarray(saving_ratio, dtype=np.float32).ravel()
    mu = np.asarray(mu, dtype=np.float32).ravel()
    labor = np.asarray(labor, dtype=np.float32).ravel()

    # Filter: ability >= cutoff quantile AND interior labor
    ability_cutoff = np.nanquantile(ability, ability_cutoff_quantile)
    mask = (ability >= ability_cutoff) & (labor < labor_upper)
    mask &= np.isfinite(saving_ratio) & np.isfinite(mu) & np.isfinite(labor)

    if mask.sum() < 2 * n_components:
        return {
            'regime_fraction_high': np.nan,
            'regime_fraction_low': np.nan,
            'separation_score': np.nan,
            'bic': np.nan,
            'aic': np.nan,
            'labels': np.full(mask.sum(), -1, dtype=np.int32),
        }

    sr = saving_ratio[mask]
    m = mu[mask]
    lb = labor[mask]

    # Z-score (skip LOWESS for speed)
    def zscore(x):
        std = np.std(x)
        if std < 1e-8:
            return np.zeros_like(x)
        return (x - np.mean(x)) / std

    X = np.column_stack([zscore(sr), zscore(m), zscore(lb)])

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=random_state,
        n_init=3,
    )
    gmm.fit(X)
    labels = gmm.predict(X)

    # Name regimes: "high" = cluster with larger mean saving_z
    means_sr = np.array([sr[labels == k].mean() for k in range(n_components)])
    high_label = int(np.argmax(means_sr))
    low_label = 1 - high_label  # only valid for n_components=2

    n_total = mask.sum()
    frac_high = float((labels == high_label).sum()) / n_total
    frac_low = float((labels == low_label).sum()) / n_total if n_components == 2 else np.nan

    # Separation: Euclidean distance between cluster centers in feature space
    if n_components >= 2:
        c0 = gmm.means_[0]
        c1 = gmm.means_[1]
        separation = float(np.linalg.norm(c0 - c1))
    else:
        separation = 0.0

    return {
        'regime_fraction_high': frac_high,
        'regime_fraction_low': frac_low,
        'separation_score': separation,
        'bic': float(gmm.bic(X)),
        'aic': float(gmm.aic(X)),
        'labels': labels.astype(np.int32),
    }


# ---------------------------------------------------------------------------
# RegimeAnalyzer — full 17-step Cluster.md pipeline
# ---------------------------------------------------------------------------

class RegimeAnalyzer:
    """
    Full regime clustering pipeline following Cluster.md.

    Steps implemented in fit():
      2. Sample selection (ability cutoff, labor < upper)
      3. LOWESS detrending per outcome
      4. Residual computation
      5. Z-score standardization
      6. Feature matrix construction
      7. GMM clustering (K=2)
      8. Regime naming
      9. Merge back to full panel

    Steps implemented in compute_transitions():
      11. Regime path construction (regime_prev, switch_flag)
      12. Switch count per agent
      13. Transition matrix

    Steps implemented in compute_summary_stats():
      15. Per-regime mean/std/quantiles

    Steps implemented in robustness_check():
      16. BIC/AIC across K values and filter cutoffs
    """

    def __init__(
        self,
        ability_cutoff_quantile: float = 0.5,
        labor_upper: float = 0.95,
        n_components: int = 2,
        detrend_method: str = 'lowess',
        lowess_frac: float = 0.3,
        random_state: int = 42,
    ):
        self.ability_cutoff_quantile = ability_cutoff_quantile
        self.labor_upper = labor_upper
        self.n_components = n_components
        self.detrend_method = detrend_method
        self.lowess_frac = lowess_frac
        self.random_state = random_state

        # Fitted attributes (populated by fit())
        self._gmm = None
        self._trend_fns: Dict[str, Callable] = {}
        self._regime_map: Dict[int, str] = {}
        self._ability_cutoff: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full clustering pipeline on the panel DataFrame.

        Args:
            panel_df: Long-format DataFrame with columns:
                agent_id, time, ability, saving_ratio, mu, labor, ...

        Returns:
            Enriched DataFrame with added columns:
                saving_resid, multiplier_resid, labor_resid,
                saving_z, multiplier_z, labor_z,
                regime, cluster_prob
            Observations outside the analysis sample get regime=NA, cluster_prob=NaN.
        """
        from sklearn.mixture import GaussianMixture

        df = panel_df.copy()

        # Step 2: Sample selection
        self._ability_cutoff = np.nanquantile(df['ability'], self.ability_cutoff_quantile)
        mask = (
            (df['ability'] >= self._ability_cutoff) &
            (df['labor'] < self.labor_upper) &
            df['saving_ratio'].notna() &
            df['mu'].notna() &
            df['labor'].notna()
        )
        df_sub = df[mask].copy()

        if len(df_sub) < 2 * self.n_components:
            raise ValueError(
                f"Analysis subsample has only {len(df_sub)} rows after filtering "
                f"(ability_cutoff_quantile={self.ability_cutoff_quantile}, "
                f"labor_upper={self.labor_upper}). "
                "Try lowering ability_cutoff_quantile or increasing labor_upper."
            )

        # Step 3 & 4: Detrend by ability, compute residuals
        steps = tqdm(
            [('saving_ratio', 'saving_resid'),
             ('mu', 'multiplier_resid'),
             ('labor', 'labor_resid')],
            desc="Detrending", leave=False,
        )
        for col, resid_col in steps:
            steps.set_postfix(var=col)
            resid, trend_fn = self._detrend(df_sub, y_col=col, x_col='ability')
            df_sub[resid_col] = resid
            self._trend_fns[col] = trend_fn

        # Step 5: Z-score standardization
        for resid_col, z_col in [
            ('saving_resid', 'saving_z'),
            ('multiplier_resid', 'multiplier_z'),
            ('labor_resid', 'labor_z'),
        ]:
            vals = df_sub[resid_col].values
            std = vals.std()
            if std < 1e-8:
                df_sub[z_col] = 0.0
            else:
                df_sub[z_col] = (vals - vals.mean()) / std

        # Step 6: Feature matrix
        X = df_sub[['saving_z', 'multiplier_z', 'labor_z']].values

        # Step 7: GMM
        print("  Fitting GMM...")
        labels, probs, self._gmm = self._fit_gmm(X)
        df_sub['_cluster_label'] = labels

        # Step 8: Name regimes
        self._regime_map = self._name_regimes(df_sub)
        df_sub['regime'] = df_sub['_cluster_label'].map(self._regime_map)

        # Cluster prob = probability of belonging to regime_high
        high_cluster = [k for k, v in self._regime_map.items() if v == 'regime_high'][0]
        df_sub['cluster_prob'] = probs[:, high_cluster]

        # Step 9: Merge back to full panel
        enriched = df.copy()
        # Initialize new columns
        for col in ['saving_resid', 'multiplier_resid', 'labor_resid',
                    'saving_z', 'multiplier_z', 'labor_z', 'cluster_prob']:
            enriched[col] = np.nan
        enriched['regime'] = pd.NA

        merge_cols = ['saving_resid', 'multiplier_resid', 'labor_resid',
                      'saving_z', 'multiplier_z', 'labor_z',
                      'regime', 'cluster_prob']
        enriched.update(df_sub[merge_cols])

        # Clean up temp columns in df_sub
        return enriched.drop(columns=['_cluster_label'], errors='ignore')

    def compute_transitions(self, enriched_df: pd.DataFrame) -> Dict:
        """
        Compute regime paths, switch flags, transition matrix, and switch counts.

        Args:
            enriched_df: Output from fit(), with columns regime, agent_id, time.

        Returns:
            Dict with keys:
                enriched_df: DataFrame with regime_prev, switch_flag added
                transition_matrix: 2x2 np.ndarray (row=from, col=to)
                transition_labels: list of regime labels for matrix axes
                switch_counts: pd.Series indexed by agent_id
                time_in_regime: pd.DataFrame (agent_id, time_in_high, time_in_low)
        """
        df = enriched_df.copy()
        df = df.sort_values(['agent_id', 'time']).reset_index(drop=True)

        # Step 11: regime_prev and switch_flag
        df['regime_prev'] = df.groupby('agent_id')['regime'].shift(1)
        df['switch_flag'] = (
            df['regime'].notna() &
            df['regime_prev'].notna() &
            (df['regime'] != df['regime_prev'])
        ).astype(int)

        # Step 12: switch counts per agent
        switch_counts = df.groupby('agent_id')['switch_flag'].sum().rename('total_switches')

        # Time in each regime
        time_counts = (
            df[df['regime'].notna()]
            .groupby(['agent_id', 'regime'])
            .size()
            .unstack(fill_value=0)
        )
        for r in ['regime_high', 'regime_low']:
            if r not in time_counts.columns:
                time_counts[r] = 0
        time_in_regime = time_counts[['regime_high', 'regime_low']].rename(
            columns={'regime_high': 'time_in_high', 'regime_low': 'time_in_low'}
        )

        # Step 13: 2x2 transition matrix
        labels_used = ['regime_low', 'regime_high']
        trans = np.zeros((2, 2), dtype=np.float64)
        valid = df[df['regime'].notna() & df['regime_prev'].notna()]
        for i, r_from in enumerate(labels_used):
            for j, r_to in enumerate(labels_used):
                trans[i, j] = ((valid['regime_prev'] == r_from) &
                               (valid['regime'] == r_to)).sum()
        # Normalize rows to probabilities
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        trans = trans / row_sums

        return {
            'enriched_df': df,
            'transition_matrix': trans,
            'transition_labels': labels_used,
            'switch_counts': switch_counts,
            'time_in_regime': time_in_regime,
        }

    def compute_summary_stats(self, enriched_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-regime mean, std, and quantiles for key variables.

        Returns:
            Multi-index DataFrame: (statistic, variable) × regime
        """
        vars_of_interest = [
            v for v in ['saving_ratio', 'mu', 'labor', 'ability',
                        'consumption', 'savings', 'money_disposable',
                        'income_before_tax', 'wage', 'ret']
            if v in enriched_df.columns
        ]
        df = enriched_df[enriched_df['regime'].notna()].copy()

        rows = []
        for regime in ['regime_low', 'regime_high']:
            sub = df[df['regime'] == regime]
            for var in vars_of_interest:
                if var not in sub.columns:
                    continue
                vals = sub[var].dropna()
                rows.append({
                    'regime': regime,
                    'variable': var,
                    'mean': vals.mean(),
                    'std': vals.std(),
                    'q10': vals.quantile(0.10),
                    'q25': vals.quantile(0.25),
                    'q50': vals.quantile(0.50),
                    'q75': vals.quantile(0.75),
                    'q90': vals.quantile(0.90),
                    'n': len(vals),
                })

        summary = pd.DataFrame(rows)
        return summary

    def robustness_check(
        self,
        panel_df: pd.DataFrame,
        cutoff_quantiles: List[float] = (0.4, 0.5, 0.6),
        labor_uppers: List[float] = (0.90, 0.95),
        k_values: List[int] = (1, 2, 3),
    ) -> pd.DataFrame:
        """
        Compare BIC/AIC across K values and filter cutoffs (Step 16).

        Returns:
            DataFrame with columns: ability_cutoff_quantile, labor_upper, K, BIC, AIC, n_obs
        """
        from sklearn.mixture import GaussianMixture

        rows = []
        combos = [(c, lu) for c in cutoff_quantiles for lu in labor_uppers]
        for cutoff, lu in tqdm(combos, desc="Robustness check", leave=False):
            ability_cutoff = np.nanquantile(panel_df['ability'], cutoff)
            mask = (
                (panel_df['ability'] >= ability_cutoff) &
                (panel_df['labor'] < lu) &
                panel_df['saving_ratio'].notna() &
                panel_df['mu'].notna() &
                panel_df['labor'].notna()
            )
            df_sub = panel_df[mask].copy()

            if len(df_sub) < 6:
                continue

            # Detrend and z-score
            try:
                analyzer = RegimeAnalyzer(
                    ability_cutoff_quantile=cutoff,
                    labor_upper=lu,
                    detrend_method=self.detrend_method,
                    lowess_frac=self.lowess_frac,
                    random_state=self.random_state,
                )
                for col, _ in [('saving_ratio', 'sr'), ('mu', 'm'), ('labor', 'lb')]:
                    resid, _ = analyzer._detrend(df_sub, y_col=col, x_col='ability')
                    df_sub[f'_r_{col}'] = resid

                for col in ['_r_saving_ratio', '_r_mu', '_r_labor']:
                    vals = df_sub[col].values
                    std = vals.std()
                    df_sub[f'_z{col}'] = 0.0 if std < 1e-8 else (vals - vals.mean()) / std

                X = df_sub[['_z_r_saving_ratio', '_z_r_mu', '_z_r_labor']].values

                for k in k_values:
                    if len(df_sub) < 2 * k:
                        continue
                    gmm = GaussianMixture(
                        n_components=k, covariance_type='full',
                        random_state=self.random_state, n_init=3,
                    )
                    gmm.fit(X)
                    rows.append({
                        'ability_cutoff_quantile': cutoff,
                        'labor_upper': lu,
                        'K': k,
                        'BIC': gmm.bic(X),
                        'AIC': gmm.aic(X),
                        'n_obs': len(df_sub),
                    })
            except Exception as e:
                rows.append({
                    'ability_cutoff_quantile': cutoff,
                    'labor_upper': lu,
                    'K': k if 'k' in dir() else -1,
                    'BIC': np.nan,
                    'AIC': np.nan,
                    'n_obs': len(df_sub),
                    'error': str(e),
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detrend(
        self,
        df: pd.DataFrame,
        y_col: str,
        x_col: str = 'ability',
    ) -> Tuple[np.ndarray, Callable]:
        """
        Detrend y_col by x_col using LOWESS (or polynomial fallback).

        Returns:
            (residuals, trend_function)
            trend_function(x) -> predicted trend values at x.
        """
        x = df[x_col].values.astype(np.float64)
        y = df[y_col].values.astype(np.float64)
        valid = np.isfinite(x) & np.isfinite(y)

        if valid.sum() < 5:
            return np.zeros_like(y), lambda _x: np.zeros(len(np.atleast_1d(_x)))

        x_v, y_v = x[valid], y[valid]

        if self.detrend_method == 'lowess':
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                # return_sorted=False: returns 1-D smoothed values aligned to x_v.
                # Avoids the (n, 2) sorted array and interp1d complications.
                trend_at_valid = np.asarray(
                    lowess(y_v, x_v, frac=self.lowess_frac, return_sorted=False)
                ).ravel()

                # Build an interpolating trend_fn for future evaluation at arbitrary x.
                # Deduplicate x before passing to interp1d (required for monotone input).
                sort_idx = np.argsort(x_v)
                xs, ys_trend = x_v[sort_idx], trend_at_valid[sort_idx]
                _, uniq_idx = np.unique(xs, return_index=True)
                xs_u, ys_u = xs[uniq_idx], ys_trend[uniq_idx]

                from scipy.interpolate import interp1d
                trend_fn = interp1d(
                    xs_u, ys_u,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(float(ys_u[0]), float(ys_u[-1])),
                )

                # Trend at all positions (including invalid): use trend_fn on full x.
                trend = np.asarray(trend_fn(x)).ravel()

            except ImportError:
                # Fallback: polynomial degree 3
                coeffs = np.polyfit(x_v, y_v, deg=3)
                trend_fn = lambda _x, c=coeffs: np.polyval(c, np.asarray(_x).ravel())
                trend = trend_fn(x)
        else:
            # Polynomial fallback
            coeffs = np.polyfit(x_v, y_v, deg=3)
            trend_fn = lambda _x, c=coeffs: np.polyval(c, np.asarray(_x).ravel())
            trend = trend_fn(x)

        resid = np.where(valid, y - trend, np.nan)
        return resid.ravel(), trend_fn

    def _fit_gmm(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, object]:
        """
        Fit GaussianMixture to feature matrix X.

        Returns:
            (labels, probabilities, fitted_gmm)
            probabilities shape: (n_obs, n_components)
        """
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            random_state=self.random_state,
            n_init=5,
        )
        gmm.fit(X)
        labels = gmm.predict(X)
        probs = gmm.predict_proba(X)
        return labels, probs, gmm

    def _name_regimes(self, df_sub: pd.DataFrame) -> Dict[int, str]:
        """
        Assign regime names based on mean saving_resid per cluster.

        The cluster with higher mean saving_resid → 'regime_high'.
        The cluster with lower mean saving_resid → 'regime_low'.
        """
        if self.n_components != 2:
            # Generic naming for K != 2
            means = {
                k: df_sub.loc[df_sub['_cluster_label'] == k, 'saving_resid'].mean()
                for k in range(self.n_components)
            }
            sorted_k = sorted(means, key=lambda k: means[k])
            return {k: f'regime_{i}' for i, k in enumerate(sorted_k)}

        means = {
            k: df_sub.loc[df_sub['_cluster_label'] == k, 'saving_resid'].mean()
            for k in range(self.n_components)
        }
        high_k = max(means, key=lambda k: means[k])
        low_k = 1 - high_k
        return {high_k: 'regime_high', low_k: 'regime_low'}
