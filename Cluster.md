# Regime Tracking Procedure

## 1. 準備資料
建立一個 panel dataframe，至少包含以下欄位：

- `agent_id`
- `time`
- `ability`
- `saving_ratio`
- `multiplier`
- `labor`

可選欄位：

- `wealth`
- `money`
- `tax_wedge`
- `binding_flag`

---

## 2. 選取分析樣本
先限制在 bifurcation 較明顯的樣本區域。

### 2.1 ability 篩選
設定一個 ability 門檻，例如：

- `ability >= ability_cutoff`

### 2.2 排除明顯 binding 區域
先保留 interior 區域，例如：

- `labor < 0.95`

得到分析子樣本：

- `df_sub`

---

## 3. 對每個 outcome 分別做 ability 去趨勢
對以下三個變數，各自用 `ability` 做一條平滑趨勢線：

- `saving_ratio`
- `multiplier`
- `labor`

可使用方法：

- LOWESS
- spline
- local regression

分別得到：

- `saving_trend(ability)`
- `multiplier_trend(ability)`
- `labor_trend(ability)`

---

## 4. 計算 residual
對每筆 observation 計算：

- `saving_resid = saving_ratio - saving_trend(ability)`
- `multiplier_resid = multiplier - multiplier_trend(ability)`
- `labor_resid = labor - labor_trend(ability)`

將 residual 存回 dataframe：

- `saving_resid`
- `multiplier_resid`
- `labor_resid`

---

## 5. 標準化 residual
對三個 residual 分別做 z-score standardization：

- `saving_z`
- `multiplier_z`
- `labor_z`

計算方式：

- `saving_z = (saving_resid - mean(saving_resid)) / std(saving_resid)`
- `multiplier_z = (multiplier_resid - mean(multiplier_resid)) / std(multiplier_resid)`
- `labor_z = (labor_resid - mean(labor_resid)) / std(labor_resid)`

---

## 6. 組成 clustering feature matrix
建立特徵矩陣：

- `X = [saving_z, multiplier_z, labor_z]`

每一列對應一筆 observation。

---

## 7. 用 GMM 做兩群分群
對 `X` 套用 Gaussian Mixture Model，設定：

- `n_components = 2`

模型輸出：

- `cluster_label`
- `cluster_prob_0`
- `cluster_prob_1`

將結果存回 dataframe：

- `cluster_label`
- `cluster_prob`

其中 `cluster_prob` 可存成屬於某一群的 posterior probability。

---

## 8. 指定 regime 名稱
計算每個 cluster 的平均值：

- 平均 `saving_resid`
- 平均 `multiplier_resid`
- 平均 `labor_resid`

根據平均特徵命名兩群，例如：

- `regime_low`
- `regime_high`

新增欄位：

- `regime`

---

## 9. 回填到原始 panel data
將 `df_sub` 中的以下欄位 merge 回原始 dataframe：

- `agent_id`
- `time`
- `regime`
- `cluster_prob`

如果某些 observation 未進入分析子樣本，`regime` 可設為：

- `NA`

---

## 10. 檢查 regime 分群結果
重新畫圖並用 `regime` 著色：

- `saving_ratio` vs `ability`
- `multiplier` vs `ability`
- `labor` vs `ability`

確認兩個 regime 是否對應原本肉眼觀察到的兩條分支。

---

## 11. 追蹤 agent 的 regime path
依照 `agent_id` 與 `time` 排序後，建立每個 agent 的 regime 序列：

- `R_{i,t}`

新增欄位：

- `regime_prev`
- `switch_flag = 1(regime != regime_prev)`

---

## 12. 計算每個 agent 的切換次數
對每個 `agent_id` 統計：

- `total_switches = sum(switch_flag)`

可另外計算：

- `time_in_regime_low`
- `time_in_regime_high`

---

## 13. 建立 transition matrix
根據相鄰兩期 regime：

- `R_t -> R_{t+1}`

統計轉移機率矩陣：

- `P(regime_{t+1} = j | regime_t = i)`

輸出 2x2 matrix：

- `low -> low`
- `low -> high`
- `high -> low`
- `high -> high`

---

## 14. 畫 representative agent path
挑選幾個 representative agents，畫出時間序列：

### 14.1 regime 序列
- x 軸：`time`
- y 軸：`regime`

### 14.2 疊加其他變數
可另外畫：

- `ability`
- `saving_ratio`
- `multiplier`
- `labor`
- `tax_wedge`
- `wealth`

---

## 15. 比較不同 regime 的平均特徵
對兩個 regime 分別計算平均值：

- `saving_ratio`
- `multiplier`
- `labor`
- `ability`
- `wealth`
- `tax_wedge`

另外可計算：

- standard deviation
- quantiles

---

## 16. 穩健性檢查
重複上述流程，替換不同設定：

### 16.1 不同 ability 子樣本
- 改變 `ability_cutoff`

### 16.2 不同 interior 限制
- `labor < 0.90`
- `labor < 0.95`

### 16.3 不同去趨勢方法
- LOWESS
- spline
- polynomial fit

### 16.4 不同群數比較
用 GMM 分別估：

- `K = 1`
- `K = 2`
- `K = 3`

比較：

- BIC
- AIC

---

## 17. 最終輸出
最後至少整理出以下結果：

### 資料欄位
- `agent_id`
- `time`
- `ability`
- `saving_ratio`
- `multiplier`
- `labor`
- `saving_resid`
- `multiplier_resid`
- `labor_resid`
- `saving_z`
- `multiplier_z`
- `labor_z`
- `regime`
- `cluster_prob`
- `switch_flag`

### 圖
- regime-colored scatter plots
- representative agent regime paths

### 表
- regime summary statistics
- transition matrix
- switch count summary
