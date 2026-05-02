# Plan — `notebooks/granger.ipynb`

## Context

**Hypothesis to validate**: 在某些 batch 中，當有 agent 的 labor 上升 → 推高該 batch 的 aggregate effective labor → wage 下降 → batch 中其他 agents 為了平滑消費被迫提高 labor。

**Why this notebook is needed**:
1. `focal_trajectories.ipynb` 的 Granger 測試 (cell 11/12) 跑在 **cross-batch focal agents** 上 — 但每個 batch 有獨立的 equilibrium wage/return，cross-batch 比較沒意義（user 之前提過：「我這個比較弱，因為 focal agent 是 cross batch 而每個 batch 的 equlibrium wage 與 return 不一樣」）。
2. `replay_simulation.ipynb` 已產生乾淨的 `(2000, 32, 100, 13)` 資料 — 每個 batch 內所有 agents 共享 wage（已驗證 [environment.py:163-164](../src/environment.py#L163-L164)：wage 是 batch-level，`mean(dim=1)`）。
3. 我們需要 **batch-internal** 的 Granger 測試，並聚焦在「wage 跳動明顯」的 batch（mechanism 最容易看到的地方）。

**Intended outcome**: 一份視覺證據 + 統計檢定報告，顯示在波動明顯的 batch 內，「他人 labor → wage → 自己 labor」這條鏈成立。

---

## Validation Strategy (核心設計)

### Stage 0: 載入 replay simulation 資料
從 [`../simulated_run/{run_name}_from_step{S}/sim_data_{N}steps.npz`](../simulated_run/) 載入。
- Shape: `(T=2000, B=32, A=100, V=13)`
- Wage (idx 10) 在 batch 內所有 agents 相同（已用 [environment.py:176](../src/environment.py#L176) 驗證）→ 取 `data[:, b, 0, 10]` 作為 batch b 的 wage 序列。
- Labor (idx 12) 是 per-agent: `data[:, b, i, 12]`。

### Stage 1: 識別「wage 極端跳動」的 batches

由於 wage 由 K/L ratio 決定且為 batch-level，跳動主要來自 aggregate labor 變化。對每個 batch b，計算：

| 指標 | 公式 | 說明 |
|------|------|------|
| `wage_std_b` | `std(w_b[t])` after burn-in | 整體波動 |
| `max_drop_b` | `max_t (w_b[t] - w_b[t+k])` for k=1..5 | 最大短期下跌 |
| `event_count_b` | `sum( |Δw_b[t]| > λ * global_σ_w )` | 大幅事件數量 |

**排序指標 (已確認)**: `max_drop_b` = `max_{t,k∈[1,5]} (w_b[t] - w_b[t+k])`，與 hypothesis 「wage 下降」直接對齊。
**輔助列出**: `wage_std_b`、`event_count_b` 一併計算並印出，但只用 `max_drop_b` 排序。

**Top-K (已確認)**: K = 5。

視覺化所有 32 batches 的 wage trajectory，把 top-5 用紅色高亮，其餘用淡灰色 — 類似 [focal_trajectories.ipynb cell 5](../notebooks/focal_trajectories.ipynb) 用 `axvline + text` 標重點 agent 的視覺風格；額外標出 `max_drop` 的時間點。

### Stage 2: 對每個 flagged batch，逐 agent 跑 mediation chain

**Window (已確認)**: 全 2000 steps（不做 event-window 子分析）。

對 batch b 中每個 agent i，建構三條序列：
- **w_t**: wage 序列（batch level），`data[:, b, 0, 10]`
- **L_{-i,t}**: 其他 agents 的 aggregate effective labor，`(Σ_{j≠i} l_{j,t} * v_{j,t}) / (A-1)`（用 effective labor 而非 raw labor — 與 wage 公式一致）
- **l_{i,t}**: agent i 自己的 labor，`data[:, b, i, 12]`

#### Test 1: `L_{-i, t-k} → w_t` (應為負相關)
- 機制檢查：他人勞動推高 → wage 下降。
- 部分由生產函數機械決定，但因為 `L_{-i}` 排除了 i 的貢獻，這是 i 看到的「外生衝擊」。
- 跑 Granger 並檢查 OLS 係數的符號（重複利用 cell 11 / cell 12 的 helper）。

#### Test 2: `w_{t-k} → l_{i,t}` (應為負相關)
- **這是真正的行為檢定**：agent i 看到 wage 下降後是否提高 labor。
- 必須先把 `l_{i,t}` 對 `v_{i,t}, m_{i,t}` 做 residualize（重複利用 cell 12 的 `_residualize`）→ 控制 ability/wealth 對 labor 的直接影響。
- 即測試: `w_{t-k} → ε^l_{i,t}`。

#### Test 3: 完整鏈 `L_{-i,t-2} → l_{i,t}` (transitive)
- Granger（with one extra lag）：他人勞動經由 wage 影響我的勞動。
- 補強性測試 — 即使 Test 1 和 Test 2 都通過，這條 transitive Granger 確認鏈條本身。

### Stage 3: Stationarity preflight
重複利用 cell 13 的 `_run_adf` 和 `_run_kpss`。對每個 flagged batch 的 `w_t`、`L_{-i,t}`、`ε^l_{i,t}` 跑 ADF + KPSS。
- 若 non-stationary → 對該序列取一階差分，再跑 Granger（標註 in 報告）。

### Stage 4: 結果聚合
| 層級 | 統計 |
|------|------|
| Per (batch, agent) | 三個 Granger p-values + 係數符號 |
| Per batch | fraction of agents passing all 3 tests with correct signs |
| Across flagged batches | overall fraction; report best/worst batch |

成功的判定: p < 0.05 **且** 係數符號為負 (Test 1, 2) 或負 (Test 3)。

---

## Notebook 結構 (預計 11 cells)

| Cell ID | 類型 | 內容 |
|---------|------|------|
| `h-intro` | markdown | hypothesis、validation strategy 摘要、變數對照 |
| `c-imports` | code | numpy, pandas, statsmodels, matplotlib + 配置 plt.rcParams |
| `c-config` | code | `SIM_DATA_PATH`, `BURN_IN`, `TOP_K`, `GRANGER_MAX_LAG`, `JUMP_LAMBDA` |
| `c-load` | code | 載入 `sim_data_*.npz` + meta.pkl，印出 shape |
| `c-jumps` | code | 計算 `wage_std`, `max_drop`, `event_count` per batch；用 `max_drop` 排名；選 top-5 |
| `c-jumps-viz` | code | 32 batches wage trajectory grid；標出 top-5（fig: `fig_wage_extremes.png`） |
| `c-helpers` | code | 重複利用 / inline 版本的 `_residualize`, `_run_adf`, `_run_kpss`, Granger wrapper（複製自 cell 11–13） |
| `c-stationarity` | code | 對 flagged batches 跑 stationarity check，印 summary |
| `c-test1` | code | Test 1: `L_{-i,t-k} → w_t` for each flagged batch × agent；輸出 dataframe |
| `c-test2` | code | Test 2: `w_{t-k} → ε^l_{i,t}`；輸出 dataframe |
| `c-test3` | code | Test 3: transitive `L_{-i} → l_i`；合併三個結果；繪製 pass-rate heatmap (batch × test) |
| `c-summary` | markdown | 結論：哪些 batch 鏈成立、整體 pass rate、警語 |

---

## Critical Files

| File | Role | 動作 |
|------|------|------|
| [notebooks/granger.ipynb](../notebooks/granger.ipynb) | **新建** | 整個 notebook |
| [notebooks/focal_trajectories.ipynb](../notebooks/focal_trajectories.ipynb) | reference | 複製 cell 11, 12, 13 的 helper functions（`_residualize`, `_run_adf`, `_run_kpss`, Granger pattern） |
| [notebooks/replay_simulation.ipynb](../notebooks/replay_simulation.ipynb) | data source | 必須先跑過產生 `sim_data_2000steps.npz` |
| [src/environment.py](../src/environment.py#L140-L177) | reference | `_compute_market_equilibrium` 確認 wage 是 per-batch 而非 per-agent |

**No source-code changes required** — 完全 self-contained 在 notebook 中。

---

## Reusable Functions (重複利用)

| Function | Source | 用途 |
|----------|--------|------|
| `_residualize(y, controls)` | `focal_trajectories.ipynb` cell 12 | 把 `l_{i,t}` 對 `v_{i,t}, m_{i,t}` 做 OLS partial-out |
| `_run_adf(series)` | `focal_trajectories.ipynb` cell 13 | ADF stationarity test |
| `_run_kpss(series)` | `focal_trajectories.ipynb` cell 13 | KPSS stationarity test |
| `grangercausalitytests` | `statsmodels.tsa.stattools` | Granger F-test |

---

## Confirmed Design Decisions

| 決定項 | 選擇 |
|--------|------|
| Jump metric | `max_drop_b` over 1–5 step window |
| Top-K flagged batches | 5 |
| Granger window | full 2000 steps (no event-window subset) |

---

## Verification Plan

1. **Smoke test**: load `sim_data_2000steps.npz` → assert `data.shape == (2000, 32, 100, 13)`，wage 在 batch 內 invariant: `np.allclose(data[:, b, 0, 10], data[:, b, 5, 10])` for any b。
2. **Sanity check Test 1**: 因為 wage = f(K/L)，L_{-i,t} 與 w_t **同期**應有強負相關（Pearson r 應接近 -0.9 在 max_drop batches）。若不是，產生函數鏈斷了。
3. **Sanity check Stationarity**: 大概率 wage 序列會在某些 batch 不通過 KPSS（locally non-stationary）→ 用差分後重跑，確認結果不全部消失。
4. **End-to-end**: 跑完整個 notebook，至少在 1 個 flagged batch 中找到 ≥3 個 agents 滿足全鏈（Test 1+2+3 都通過）。如果連一個都找不到，hypothesis 可能在這個 checkpoint 不成立 → 報告負結果。
