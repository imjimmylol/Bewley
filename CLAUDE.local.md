# claude.md — Decision Rule Plots (Vibe-Coding Guideline)

## 0 目標
用一套固定規格把模型的 **self-insurance / history dependence / borrowing constraint / superstar 機制**用圖「釘死」。

---

## 0.1 核心方法論：Synthetic Grid Evaluation

### 問題
直接畫模擬資料會導致所有變數自然相關（高 ability → 高 income → 高 savings → 高 consumption）。
這顯示的是 **均衡結果 (equilibrium outcomes)**，不是 **決策規則 (decision rules)**。

### 解法
要視覺化真正的 **決策規則（policy function）**，必須：
1. **固定 (Fix)** conditioning variables 在特定值
2. **變動 (Vary)** 只有 x 軸變數，系統性地建立 grid
3. **餵入合成狀態 (Synthetic states)** 到 policy network
4. **繪製** policy 的回應

這等同於畫 `c(m | v=v̄, a=ā)` — 消費作為 money 的函數，固定 ability 和 assets。

### 架構
```
Training Loop (已在運行)
        │
        ▼
┌───────────────────────────────┐
│   HistoricalRanges            │  ← 訓練過程中收集（不需額外模擬）
│   - min/max/percentiles       │
│   - 對象: m_t, a_t, v_t, s_t  │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│   PolicyEvaluator             │
│   - Option A: Single-agent    │  ← 忽略 aggregate effects
│   - 建立 synthetic grid       │
│   - Normalize → policy_net    │
│   - 回傳 (x, y, metadata)     │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│   Plotting Functions          │
│   - Hexbin（非 scatter）      │
│   - log1p(x) 用於右尾分佈     │
│   - 5 色階（quantiles）       │
│   - 參考線 (y=0, y=x)         │
│   - 不分面，用顏色區分        │
└───────────────────────────────┘
```

### 設計決策（已確認）
| 決策項目 | 選擇 |
|----------|------|
| Model input structure | **Option A**: Single-agent（忽略 aggregates）|
| Conditioning quantile levels | **5 levels**: q5, q25, q50, q75, q95 |
| Regime 視覺化方式 | **同圖內用顏色區分**（不分面）|
| 要實作的圖 | **全部** (A1-G2) |
| 資料收集方式 | **整合至 training loop**（無需額外模擬）|
| 繪圖類型 | **Hexbin**（密集資料）/ **Line plots**（grid evaluation）|
| x 軸轉換 | **log1p(x)** 用於 m_t, a_t, y_t |

---

## 0.2 變數角色定義

| 變數 | 符號 | 可為 x 軸 | 可為 y 軸 | 可為 conditioning |
|------|------|-----------|-----------|-------------------|
| Money disposable | `m_t` | ✓ | ✗ (state) | ✓ |
| Assets/Savings | `a_t` | ✓ | ✓ (`a_tp1`) | ✓ |
| Ability | `v_t` | ✓ | ✗ (exogenous) | ✓ |
| Regime | `s_t` | ✓ | ✗ (exogenous) | ✓ (discrete) |
| Consumption | `c_t` | ✗ | ✓ (decision) | ✗ |
| Savings ratio | `ζ_t` | ✗ | ✓ (decision) | ✗ |
| Labor | `l_t` | ✗ | ✓ (decision) | ✗ |
| Next savings | `a_tp1` | ✗ | ✓ (decision) | ✗ |
| Multiplier | `μ_t` | ✗ | ✓ (decision) | ✗ |

### Validation Rule
```python
def validate_plot_config(x_var, y_var, condition_vars):
    """
    Raises ValueError if:
    - x_var is in condition_vars
    - y_var is in condition_vars (if y_var is a state variable)
    """
    if x_var in condition_vars:
        raise ValueError(f"x_var '{x_var}' cannot be in condition_vars")
    if y_var in condition_vars:
        raise ValueError(f"y_var '{y_var}' cannot be in condition_vars")
```

---

## 1 變數約定
**State（x 軸候選）**
- `a_t`：期初資產（history）
- `m_t`：money-on-hand / disposable resources（當期資源）
- `y_t`：當期收入（可選）
- `v_t`：ability（連續或離散）
- `s_t`：regime（Normal / Superstar）

**Decision（y 軸候選）**
- `a_tp1`：`a_{t+1}`（下一期資產）
- `zeta_t`：`a_tp1 / m_t`（儲蓄率）
- `c_t`：消費
- `da_t`：`a_tp1 - a_t`（淨儲蓄）
- `mu_t`：借款限制 multiplier（若有）
- `euler_resid`：Euler residual（若有）
- `fb_resid`：FB / complementarity residual（若有）

---

## 2 通用繪圖規則（強制）
- 點數多：**hexbin / 2D histogram**（禁用純 scatter 黑成一坨）
- x 軸有 0 且右尾長：用 `log1p(x)`（優先用在 `m_t`, `a_t`, `y_t`）
- 參考線必加：
  - borrowing constraint：`y = 0`（用在 `a_tp1`, `zeta_t`, `mu_t`, `da_t`）
  - 消費貼現：`y = x`（用在 `c_t` vs `m_t`）
  - 資產動態：`y = x`（用在 `a_tp1` vs `a_t`）
- 顏色/分面優先級：`s_t`（regime） > `v_t`（ability） > `a_t` 分位數
- 主文圖：控制張數（4 張），其餘進 Appendix

---

## 3 必畫清單（Axes + 規格）

### A. Core decision rules（主力）
**A1 — Resources → Next assets**
- type: hexbin
- x: `log1p(m_t)`
- y: `a_tp1`
- ref: `y=0`
- color/facet: facet by `s_t` (N/S) 或 color by `v_t`

**A2 — Resources → Saving rate**
- type: hexbin
- x: `log1p(m_t)`
- y: `zeta_t = a_tp1 / m_t`
- ref: `y=0`
- color/facet: facet by `s_t` 或 color by `v_t`
- note: `m_t` 很小時先做 winsorize / 加 epsilon 防爆

**A3 — Resources → Consumption**
- type: hexbin
- x: `log1p(m_t)`
- y: `c_t`
- ref: `y=x`（用原始 `m_t` 對照時要一致；或畫 `c_t` vs `m_t` 的版本）
- color/facet: color by `a_t` quantile (q10/q50/q90) 或 facet by `s_t`

---

### B. History dependence（資產承載歷史）
**B1 — Assets today → Assets tomorrow**
- type: hexbin
- x: `log1p(a_t)`（或 `a_t`）
- y: `a_tp1`
- ref: `y=x`, `y=0`
- color/facet: facet by `s_t` 或 color by `v_t`

**B2 — Assets → Consumption (conditional)**
- type: hexbin
- x: `log1p(a_t)`
- y: `c_t`
- conditioning: 固定 `v_t`（分面）或 color by `m_t`

---

### C. Borrowing constraint fingerprints（被綁住在哪）
**C1 — Resources → Binding indicator**
- type: hexbin (or jittered scatter if small)
- x: `log1p(m_t)`
- y: `I_bind = 1[a_tp1 == 0]`
- color/facet: facet by `s_t` 或 color by `v_t`

**C2 — Resources → Multiplier (if available)**
- type: hexbin
- x: `log1p(m_t)`
- y: `mu_t`
- ref: `y=0`
- color/facet: facet by `s_t` 或 color by `v_t`

**C3 — Assets → Multiplier (if available)**
- type: hexbin
- x: `log1p(a_t)`
- y: `mu_t`
- ref: `y=0`

---

### D. Superstar / regime mechanism（最有戲的動態）
**D1 — Ability/Income → Net saving**
- type: hexbin
- x: `v_t`（或 `log1p(y_t)`）
- y: `da_t = a_tp1 - a_t`
- ref: `y=0`
- color: `s_t`

**D2 — Transition panels (event-style)**
- type: hexbin
- x: `log1p(a_t)`
- y: `da_t`
- ref: `y=0`
- facet: 4 panels by `(s_t -> s_tp1)` in {N→N, N→S, S→S, S→N}

---

### E. Consumption smoothing（不是只看 c(m)）
**E1 — Income → Consumption**
- type: hexbin
- x: `log1p(y_t)`（或 `y_t`）
- y: `c_t`
- color: `a_t` quantile

**E2 — Shock response (optional, if you can define shock)**
- define: `eps_tp1 = y_tp1 - E[y_tp1 | v_t, s_t]`
- type: hexbin
- x: `eps_tp1`
- y: `dc_tp1 = c_tp1 - c_t`  (or `da_t`)
- color: `a_t` quantile

---

### F. MPC signature（最能打 reviewer）
**F1 — Wealth → MPC**
- type: scatter/hexbin
- x: `log1p(a_t)`（或 `log1p(m_t)`）
- y: `mpc_t`
- note: `mpc_t` 用 local slope 或 finite-diff 估

---

### G. Solver/quality diagnostics（Appendix）
**G1 — Resources → Euler residual**
- type: hexbin
- x: `log1p(m_t)`
- y: `log1p(abs(euler_resid))`

**G2 — Resources → FB residual**
- type: hexbin
- x: `log1p(m_t)`
- y: `log1p(abs(fb_resid))`

---

## 4 主文 vs Appendix（固定排版）
**Main (4 figs)**
1. A1 `m_t → a_tp1`
2. A3 `m_t → c_t`
3. B1 `a_t → a_tp1` (with 45°)
4. D2 transition panels `(s_t→s_tp1): a_t → da_t`

**Appendix**
- A2, B2, C1–C3, D1, E1–E2, F1, G1–G2

---

## 5 匯出與命名規範
- 檔名：`fig_{ID}_{shortname}.pdf`
  例：`fig_A1_m_to_atp1.pdf`, `fig_D2_transition_da.pdf`
- 每張圖標題包含：
  - `state → decision`（例如 `m_t → a_{t+1}`）
  - regime/facet 說明（若有）
- 統一字體/大小、axis label 用模型符號（`m_t, a_{t+1}, c_t, ζ_t`）

---

## 6 Implementation Checklist

### Phase 1: Core Infrastructure

#### 1.1 Create `src/policy_evaluation.py`
- [ ] **`HistoricalRanges` dataclass**
  ```python
  @dataclass
  class HistoricalRanges:
      # For each variable: min, max, percentiles [5, 25, 50, 75, 95]
      m_t: dict  # {"min": float, "max": float, "percentiles": np.ndarray}
      a_t: dict
      v_t: dict
      s_t: dict  # {"values": [0, 1], "counts": [n_normal, n_superstar]}
      y_t: dict  # income_before_tax
  ```

- [ ] **`collect_ranges_from_step()` function**
  - Input: `temp_state: TemporaryState`, `existing_ranges: Optional[HistoricalRanges]`
  - Update running min/max/percentile estimates
  - Called every training step (lightweight update)

- [ ] **`PolicyEvaluator` class**
  ```python
  class PolicyEvaluator:
      def __init__(self, policy_net, normalizer, ranges: HistoricalRanges, device):
          ...

      def evaluate_on_grid(
          self,
          x_var: str,           # e.g., "m_t"
          condition_vars: dict, # e.g., {"v_t": "q50", "a_t": "q50", "s_t": False}
          n_points: int = 100
      ) -> dict:
          """
          Returns:
              {
                  "x_values": np.ndarray,      # grid points
                  "c_t": np.ndarray,           # consumption output
                  "zeta_t": np.ndarray,        # savings ratio output
                  "a_tp1": np.ndarray,         # next savings output
                  "l_t": np.ndarray,           # labor output
                  "mu_t": np.ndarray,          # multiplier output
                  "condition_values": dict,    # actual values used for conditioning
              }
          """
  ```

- [ ] **`validate_plot_config()` function**
  - Check x_var not in condition_vars
  - Check y_var not in condition_vars
  - Raise `ValueError` with clear message

- [ ] **`_build_synthetic_state()` helper**
  - Single-agent mode (Option A)
  - Create (1, 1) shaped tensors for policy evaluation
  - Handle normalization properly

#### 1.2 Update `src/visualization.py`

- [ ] **Keep existing functions** (backwards compatibility)
  - `prepare_data_for_plotting()`
  - `plot_decision_rules_scatter()`
  - `plot_binned_decision_rules()`
  - `plot_state_distributions()`

- [ ] **Add `plot_decision_rule()` master function**
  ```python
  def plot_decision_rule(
      evaluator: PolicyEvaluator,
      x_var: str,
      y_var: str,
      color_var: Optional[str] = None,  # Variable to use for color (5 quantiles)
      condition_vars: Optional[dict] = None,
      use_log1p_x: bool = True,
      ref_lines: Optional[List[str]] = None,  # ["y=0", "y=x"]
      save_path: Optional[str] = None,
      log_to_wandb: bool = False,
      step: Optional[int] = None
  ) -> plt.Figure:
  ```

- [ ] **Add helper functions**
  - `_apply_log1p_transform()`
  - `_add_reference_lines()`
  - `_get_quantile_colors()` — returns 5 colors for quantile levels
  - `_format_axis_label()` — converts "m_t" to "$m_t$" for LaTeX

- [ ] **Add convenience wrappers for each plot type**
  ```python
  def plot_A1_resources_to_assets(evaluator, **kwargs): ...
  def plot_A2_resources_to_saving_rate(evaluator, **kwargs): ...
  def plot_A3_resources_to_consumption(evaluator, **kwargs): ...
  def plot_B1_assets_to_assets(evaluator, **kwargs): ...
  def plot_B2_assets_to_consumption(evaluator, **kwargs): ...
  def plot_C1_binding_indicator(evaluator, **kwargs): ...
  def plot_C2_resources_to_multiplier(evaluator, **kwargs): ...
  def plot_C3_assets_to_multiplier(evaluator, **kwargs): ...
  def plot_D1_ability_to_net_saving(evaluator, **kwargs): ...
  def plot_D2_transition_panels(evaluator, **kwargs): ...  # Special: 4 subplots
  def plot_E1_income_to_consumption(evaluator, **kwargs): ...
  def plot_F1_wealth_to_mpc(evaluator, **kwargs): ...
  def plot_G1_euler_residual(evaluator, **kwargs): ...
  def plot_G2_fb_residual(evaluator, **kwargs): ...
  ```

- [ ] **Add `plot_all_decision_rules()` function**
  - Generates all plots (A1-G2)
  - Saves to organized directory structure
  - Optional wandb logging

#### 1.3 Update `src/train.py`

- [ ] **Import new modules**
  ```python
  from src.policy_evaluation import (
      HistoricalRanges,
      PolicyEvaluator,
      collect_ranges_from_step
  )
  ```

- [ ] **Initialize HistoricalRanges** before training loop

- [ ] **Call `collect_ranges_from_step()`** every training step
  - Lightweight operation (just update running stats)

- [ ] **Create PolicyEvaluator** at visualization intervals

- [ ] **Call new visualization functions** at save_interval
  ```python
  if step % save_interval == 0:
      evaluator = PolicyEvaluator(policy_net, normalizer, ranges, device)
      plot_all_decision_rules(evaluator, save_dir=..., step=step)
  ```

---

### Phase 2: Individual Plot Implementation

#### 2.1 Section A: Core Decision Rules
- [ ] A1: `log1p(m_t)` → `a_tp1`, color by `v_t`, ref: `y=0`
- [ ] A2: `log1p(m_t)` → `ζ_t`, color by `v_t`, ref: `y=0`
- [ ] A3: `log1p(m_t)` → `c_t`, color by `a_t`, ref: `y=x`

#### 2.2 Section B: History Dependence
- [ ] B1: `log1p(a_t)` → `a_tp1`, color by `v_t`, ref: `y=x`, `y=0`
- [ ] B2: `log1p(a_t)` → `c_t`, color by `m_t`

#### 2.3 Section C: Borrowing Constraint
- [ ] C1: `log1p(m_t)` → `I[a_tp1=0]`, color by `v_t`
- [ ] C2: `log1p(m_t)` → `μ_t`, color by `v_t`, ref: `y=0`
- [ ] C3: `log1p(a_t)` → `μ_t`, ref: `y=0`

#### 2.4 Section D: Superstar/Regime
- [ ] D1: `v_t` → `Δa_t`, color by `s_t`, ref: `y=0`
- [ ] D2: `log1p(a_t)` → `Δa_t`, 4 panels by `(s_t → s_tp1)`, ref: `y=0`

#### 2.5 Section E: Consumption Smoothing
- [ ] E1: `log1p(y_t)` → `c_t`, color by `a_t`
- [ ] E2: (Optional) Shock response plot

#### 2.6 Section F: MPC Signature
- [ ] F1: `log1p(m_t)` → `mpc_t` (finite-diff estimated)

#### 2.7 Section G: Diagnostics
- [ ] G1: `log1p(m_t)` → `log1p(|euler_resid|)`
- [ ] G2: `log1p(m_t)` → `log1p(|fb_resid|)`

---

### Phase 3: Testing & Integration

- [ ] **Unit tests for `PolicyEvaluator`**
  - Test synthetic state creation
  - Test normalization pipeline
  - Test validation rules

- [ ] **Integration test**
  - Run short training with visualization
  - Verify all plots generate correctly

- [ ] **Documentation**
  - Docstrings for all new functions
  - Usage examples in README or separate doc

---

### File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `src/policy_evaluation.py` | **CREATE** | HistoricalRanges, PolicyEvaluator, collect_ranges_from_step |
| `src/visualization.py` | **UPDATE** | Add plot_decision_rule, plot_A1-G2 wrappers |
| `src/train.py` | **UPDATE** | Integrate range collection and new visualizations |
| `CLAUDE.local.md` | **UPDATE** | This checklist (done) |

---

### Variable Mapping Reference

| CLAUDE.local.md Symbol | Code Variable | State Class |
|------------------------|---------------|-------------|
| `m_t` | `money_disposable` | TemporaryState |
| `a_t` | `savings` (input) | MainState |
| `a_tp1` | `savings` (output) | TemporaryState |
| `v_t` | `ability` | MainState/TemporaryState |
| `s_t` | `is_superstar` | MainState (is_superstar_vA/vB) |
| `c_t` | `consumption` | TemporaryState |
| `ζ_t` | `savings_ratio` | TemporaryState |
| `l_t` | `labor` | TemporaryState |
| `μ_t` | `mu` | TemporaryState |
| `y_t` | `income_before_tax` | TemporaryState |
| `Δa_t` | `savings - savings_input` | Computed |

---
