# GEMINI.md

這份文件是給會幫我寫/重構程式的 AI 用的。請照這裡的結構產生程式碼，目標是：

1. 同一套程式可以「單一 config run」
2. 也可以被 wandb sweep 重複呼叫
3. 訓練過程的狀態要能分開存：weights / states / normalizer

---

## 1. 資料夾結構

```text
project/
    main.py                 # 唯一入口：讀 config → init wandb → 呼叫 src/train.py
    .gitignore

    checkpoints/
        {run_name}/         # run_name 優先用 wandb.run.name，沒有就用 config.exp_name
            states/         # 存整包環境/agent 的訓練狀態
            normalizer/     # 存 src/normalizer.py 的 running stats
            weights/        # 存模型權重

    src/
        __init__.py
        train.py            # 真正的訓練 loop，只吃一包 config
        normalizer.py       # RunningNormalizer class，提供 update/normalize/save/load
        models/             # 模型定義
        utils/
            configloader.py # 載入並合併 config 的工具

    config/
        default.yaml        # baseline 訓練 & 結構參數
        local.yaml          # (可選) 本機覆蓋
        model.yaml          # (可選) 模型細部設定

    sweepconfig/
        default_sweep.yaml  # wandb sweep 設定，program 指向 main.py
````

---

## 2. 檔案角色說明（一定要看）

### 2.1 `main.py`（入口）

* 功能：

  1. 解析 CLI 參數（例如 `--config config/default.yaml`）
  2. 用 `src/utils/configloader.py` 把多個 yaml 合併成一個 dict
     合併順序：`config/default.yaml → 其他指定的 yaml → CLI / sweep 參數`
  3. 呼叫 `wandb.init(project=..., config=merged_config)`
  4. 把最終的 `wandb.config` 傳進 `src.train.train(...)`
* 不做的事：

  * 不寫真正的訓練 loop
  * 不直接存模型
  * 不在這裡寫業務邏輯

### 2.2 `src/train.py`（訓練核心）

* 功能：

  * 定義一個 `train(config)` 函式
  * 用 `config` 裡的參數建 model / dataloader / optimizer
  * 執行訓練迴圈
  * 在需要時存：

    * `checkpoints/{run_name}/weights/...`
    * `checkpoints/{run_name}/states/...`
    * `checkpoints/{run_name}/normalizer/...`
* 規則：

  * 不要在這裡再呼叫 `wandb.init()`，因為入口已經做了
  * 所有超參數都從 `config` 拿，不能寫死

### 2.3 `src/utils/configloader.py`

* 功能：

  * 提供像 `load_config(list_of_paths) -> dict` 這樣的介面
  * 可以把多個 yaml 合併，後面的覆蓋前面的
* 目的：

  * 讓 `main.py` 保持乾淨，不要在那裡寫一堆 yaml 處理

### 2.4 `src/normalizer.py`

* 功能：

  * 定義 RunningNormalizer（或類似名稱）
  * 要能：

    * `update(batch)`
    * `normalize(x)`
    * `state_dict()` / `load_state_dict(...)`
  * 這樣才能被存到 `checkpoints/{run_name}/normalizer/` 裡，之後再載回來繼續訓練

### 2.5 `config/` 與 `sweepconfig/`

* `config/`：放「我手動要跑的一組參數」
* `sweepconfig/`：放「wandb 要反覆測的參數空間」
* AI 生成程式碼時，請預設 `main.py` 的 `program` 會被 sweep 呼叫

---

## 3. 單一 config run 的使用方式

```bash
python main.py --config config/default.yaml
```

執行流程：

1. `main.py` 讀取 config
2. init wandb → 這就是一個獨立的 run
3. 呼叫 `src.train.train(config)`
4. 產物會存到：`checkpoints/{run_name}/...`

你可以再做一個 `config/run_lr5e-5.yaml`，跑的時候改路徑就好。

---

## 4. wandb sweep 的使用方式

`sweepconfig/default_sweep.yaml` 示意：

```yaml
program: main.py
method: grid
metric:
  name: val_loss
  goal: minimize
parameters:
  lr:
    values: [1e-4, 5e-5, 2e-5]
  batch_size:
    values: [16, 32]
```

執行：

```bash
wandb sweep sweepconfig/default_sweep.yaml
wandb agent <entity>/<project>/<sweep_id>
```

agent 會一直呼叫 `main.py`，而 `main.py` 已經會 init wandb 並呼叫同一個 `src/train.py`，所以不需要額外寫一份「sweep 專用的訓練程式」。

---

## 5. checkpoints 命名規則

AI 在產生程式碼時請遵守下列優先順序決定 `{run_name}`：

1. 若有 `wandb.run.name` 就使用它
2. 否則使用 config 裡的 `exp_name`
3. 否則 fallback 成現在時間戳（例如 `2025-11-09_141500`）

這樣可以避免 sweep 的 run 互相覆蓋，也方便人眼識別。

---

## 6. AI 產碼時的注意事項

1. 只能在 `main.py` 呼叫一次 `wandb.init()`，其他檔案只能用已經存在的 `wandb.run` 或 `wandb.log(...)`。
2. 所有訓練參數必須從 `config` 取得，不能寫死在程式裡，這樣 sweep 才能覆蓋。
3. 新增的工具檔請放進 `src/` 或其子資料夾，不要放在 project 根目錄。
4. 存檔路徑請用組字串的方式根據 `{run_name}` 建立，不要硬寫成 `checkpoints/exp_xx/...`。


## 7. 環境與訓練的flow

```{yaml}
exogenous:
  - ability
  - is_superstar_vA
  - is_superstar_vB
  - tax_params   # becomes multi_agent if controlled by a policy agent

endogenous:
  - savings

multi_agent:
  - wage
  - ret

derived:
  - ibt
  - moneydisposable
  - consumption
  - policy_features  # engineered (not state)

history_dependent:
  - ability_history_A
  - ability_history_B

semi_exogenous:
  - []  # none currently (use beliefs here if you introduce partial observability)
```