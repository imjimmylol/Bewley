# Plan: GMM-Based Regime Tracking System

## Context
The model exhibits **endogenous bifurcation** — at higher ability levels, saving_ratio, multiplier, and labor split into two distinct behavioral branches. This is not driven by the exogenous superstar type but emerges from the agent's policy. We need to formally identify, label, and track these regimes using GMM clustering on joint outcomes, following the 17-step procedure in `Cluster.md`.

## Architecture: 3 New Files + 2 Modifications

### New Files
1. **`src/cluster_analysis.py`** — Core clustering logic (PanelBuffer, RegimeAnalyzer, lightweight snapshot)
2. **`src/cluster_visualization.py`** — All regime-related plots (separate from the 2246-line visualization.py)
3. **`run_cluster_analysis.py`** — Standalone post-hoc analysis script

### Modified Files
4. **`src/train.py`** — Panel buffer integration + periodic lightweight clustering
5. **`src/plot_data_io.py`** — Panel data save/load helpers

---

## Step 1: `src/cluster_analysis.py`

### 1a. `PanelBuffer` class
- Ring buffer accumulating per-step agent data for **batch 0** (agents persist within a batch across steps)
- Pre-allocates numpy arrays `(max_steps, n_agents)` for each tracked variable
- Tracked vars: `ability, saving_ratio, mu, labor, money_disposable, consumption, savings, income_before_tax, wage, ret`
- Methods: `append(step, data_dict)`, `to_dataframe() -> pd.DataFrame`, `to_npz(path)`, `from_npz(path)`
- DataFrame output columns: `agent_id, time, ability, saving_ratio, mu, labor, ...`

### 1b. `RegimeAnalyzer` class — Full Cluster.md pipeline (steps 2-16)
- `__init__(ability_cutoff_quantile=0.5, labor_upper=0.95, n_components=2, detrend_method='lowess')`
- `fit(panel_df) -> enriched_df` — Filter → LOWESS detrend → residuals → z-score → GMM → label → merge back
  - Output adds columns: `saving_resid, multiplier_resid, labor_resid, saving_z, multiplier_z, labor_z, regime, cluster_prob`
- `_detrend(df, y_col, x_col='ability')` — Uses `statsmodels.nonparametric.lowess`
- `_fit_gmm(X)` — `sklearn.mixture.GaussianMixture(n_components=2)`
- `_name_regimes(df_sub)` — Assigns "regime_high"/"regime_low" based on mean saving_resid
- `compute_transitions(enriched_df)` — Steps 11-13: `regime_prev`, `switch_flag`, 2x2 transition matrix, switch counts per agent
- `compute_summary_stats(enriched_df)` — Step 15: mean/std/quantiles by regime
- `robustness_check(panel_df, cutoff_quantiles, labor_uppers, k_values)` — Step 16: BIC/AIC table

### 1c. `lightweight_cluster_snapshot()` function
- Fast cross-sectional GMM for inline training monitoring (no pandas, just numpy + sklearn)
- Input: flattened arrays of ability, saving_ratio, mu, labor
- Does: filter → z-score (skip detrending for speed) → GMM(K=2)
- Returns: `regime_fraction_high, regime_fraction_low, separation_score, bic, aic, labels`

---

## Step 2: `src/train.py` modifications

### 2a. Initialize panel buffer (after `initialize_env_state`, ~line 203)
```python
from src.cluster_analysis import PanelBuffer, lightweight_cluster_snapshot

panel_buffer = PanelBuffer(
    max_steps=getattr(config.training, 'panel_buffer_size', 200),
    n_agents=config.training.agents,
    tracked_vars=[...])
cluster_interval = getattr(config.training, 'cluster_interval', 5000)
```

### 2b. Collect panel data each step (after variable extraction, before `del` block)
Insert between line ~262 and line ~463:
```python
panel_buffer.append(step, {
    'ability': ability_t[0].detach().cpu().numpy(),
    'saving_ratio': savings_ratio_t[0].detach().cpu().numpy(),
    'mu': mu_t[0].detach().cpu().numpy(),
    'labor': labor_t[0].detach().cpu().numpy(),
    'money_disposable': money_disposable_t[0].detach().cpu().numpy(),
    'consumption': consumption_t[0].detach().cpu().numpy(),
    'savings': savings_t[0].detach().cpu().numpy(),
    'income_before_tax': ibt[0].detach().cpu().numpy(),
    'wage': wage_t[0].detach().cpu().numpy(),
    'ret': ret_t[0].detach().cpu().numpy(),
})
```
Note: `[0]` selects batch 0. All `.detach().cpu().numpy()` calls happen before `del`.

### 2c. Lightweight clustering at `cluster_interval` (inside plotting block)
```python
if step % cluster_interval == 0 and step > 0:
    cluster_result = lightweight_cluster_snapshot(
        ability=ability_t.detach().cpu().numpy().flatten(),
        saving_ratio=savings_ratio_t.detach().cpu().numpy().flatten(),
        mu=mu_t.detach().cpu().numpy().flatten(),
        labor=labor_t.detach().cpu().numpy().flatten())
    wandb.log({"regime/fraction_high": ..., "regime/separation_score": ..., ...}, step=step)
```

### 2d. Save panel buffer at `save_interval` (alongside model checkpoint, ~line 477)
```python
panel_dir = os.path.join(base_checkpoint_dir, "plot_data", "panel")
os.makedirs(panel_dir, exist_ok=True)
panel_buffer.to_npz(os.path.join(panel_dir, f"panel_step_{step}.npz"))
```

---

## Step 3: `src/cluster_visualization.py`

Six plotting functions, all with `save_path` and `log_to_wandb` options:
1. `plot_regime_scatter(enriched_df)` — 3 panels: saving_ratio, mu, labor vs ability, colored by regime
2. `plot_regime_paths(enriched_df, n_agents=5)` — Time series of regime + variables for representative agents
3. `plot_transition_matrix(matrix_2x2)` — Heatmap
4. `plot_switch_distribution(switch_counts)` — Histogram of switches per agent
5. `plot_regime_summary(summary_df)` — Bar/box comparing regime stats
6. `plot_robustness_bic(robustness_df)` — BIC/AIC across K and cutoffs

Color scheme: blue = `regime_low`, red/orange = `regime_high`, gray = `NA`.

---

## Step 4: `src/plot_data_io.py` modifications

- Add `save_panel_data(panel_buffer, step, exp_name, save_dir)` wrapper
- Add `load_panel_data(npz_path) -> PanelBuffer` wrapper
- Update `discover_runs()` to detect `panel/` subdirectory

---

## Step 5: `run_cluster_analysis.py`

Standalone script for detailed post-hoc analysis:
1. Load panel NPZ (or re-simulate from checkpoint)
2. `PanelBuffer.to_dataframe()`
3. `RegimeAnalyzer.fit()` → enriched panel
4. `compute_transitions()` → transition matrix, switch counts
5. `compute_summary_stats()` → regime comparison table
6. `robustness_check()` → BIC/AIC table
7. Generate all plots
8. Save enriched panel as CSV, print summary

Args: `--panel_npz` or `--checkpoint_dir --state_step`, `--ability_cutoff_quantile`, `--labor_upper`, `--n_components`, `--output_dir`

---

## Dependencies
- `scikit-learn` (GaussianMixture) — new
- `statsmodels` (lowess) — new  
- `pandas` (DataFrame ops) — likely already available

## Memory Impact
Ring buffer: 200 steps x 100 agents x 10 vars x 8 bytes = **1.6 MB** — negligible.

---

## Implementation Checklist

### Phase 1: Core Infrastructure
- [x] Create `src/cluster_analysis.py`
  - [x] Implement `PanelBuffer` class (ring buffer, append, to_dataframe, to_npz, from_npz)
  - [x] Implement `lightweight_cluster_snapshot()` function
  - [x] Implement `RegimeAnalyzer.__init__()` with configurable parameters
  - [x] Implement `RegimeAnalyzer._detrend()` using LOWESS
  - [x] Implement `RegimeAnalyzer._fit_gmm()` using sklearn GaussianMixture
  - [x] Implement `RegimeAnalyzer._name_regimes()` based on mean saving_resid
  - [x] Implement `RegimeAnalyzer.fit()` — full pipeline: filter → detrend → residuals → z-score → GMM → label → merge
  - [x] Implement `RegimeAnalyzer.compute_transitions()` — regime_prev, switch_flag, transition matrix, switch counts
  - [x] Implement `RegimeAnalyzer.compute_summary_stats()` — mean/std/quantiles by regime
  - [x] Implement `RegimeAnalyzer.robustness_check()` — BIC/AIC across K values and cutoffs

### Phase 2: Training Loop Integration
- [x] Modify `src/train.py` — import PanelBuffer and lightweight_cluster_snapshot
- [x] Add panel buffer initialization after `initialize_env_state`
- [x] Add `panel_buffer.append()` call each step (batch 0 data, before `del` block)
- [x] Add lightweight clustering at `cluster_interval` with wandb logging
- [x] Add panel buffer save at `save_interval`

### Phase 3: Data I/O
- [x] Modify `src/plot_data_io.py` — add `save_panel_data()` and `load_panel_data()` wrappers
- [x] Update `discover_runs()` to detect `panel/` subdirectory

### Phase 4: Visualization
- [x] Create `src/cluster_visualization.py`
  - [x] Implement `plot_regime_scatter()` — 3 panels colored by regime
  - [x] Implement `plot_regime_paths()` — representative agent time series
  - [x] Implement `plot_transition_matrix()` — 2x2 heatmap
  - [x] Implement `plot_switch_distribution()` — histogram of switch counts
  - [x] Implement `plot_regime_summary()` — bar/box comparing regime stats
  - [x] Implement `plot_robustness_bic()` — BIC/AIC comparison

### Phase 5: Standalone Analysis Script
- [x] Create `run_cluster_analysis.py` — argparse, load panel, run full analysis pipeline, generate all plots, save outputs

### Phase 6: Verification
- [ ] Run short training (`training_steps=100, cluster_interval=50`) → check wandb regime metrics
- [ ] Load saved panel NPZ → run `run_cluster_analysis.py` → verify plots and tables
- [ ] Check regime scatter plots visually match the bifurcation in original scatter plots
- [ ] Verify transition matrix rows sum to 1.0
- [ ] BIC comparison: K=2 should beat K=1 if bifurcation is real
