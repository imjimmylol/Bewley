# README.md

# Bewley Model Training Framework

This project provides a structured framework for training models, specifically tailored for experiments like the Bewley model. It is designed to be flexible, allowing for both single runs with a specific configuration and automated hyperparameter sweeps using Weights & Biases (W&B).

The structure is based on the `GEMINI.md` guidelines.

## Project Structure

```text
Bewley/
├── main.py                 # Main entry point for all runs
├── .gitignore
├── README.md
│
├── checkpoints/            # Directory for saved model weights, states, etc.
│   └── {run_name}/
│       ├── states/
│       ├── normalizer/
│       └── weights/
│
├── config/
│   └── default.yaml        # Default parameters for a single run
│
├── sweepconfig/
│   └── default_sweep.yaml  # W&B sweep configuration
│
└── src/
    ├── __init__.py
    ├── train.py            # Core training loop
    ├── normalizer.py       # Normalizer class with save/load state
    ├── models/             # Model definitions
    ├── data/               # Data loading and processing
    └── utils/
        └── configloader.py # Utility for loading and merging YAML configs
```

## Prerequisites

1.  **Install dependencies**:
    You will need Python and `pip`. The core dependencies for this framework are `pyyaml` for config handling and `wandb` for experiment tracking.

    ```bash
    pip install pyyaml wandb
    ```

2.  **Login to Weights & Biases**:
    To use experiment tracking or sweeps, you need a W&B account.

    ```bash
    wandb login
    ```

## How to Run

There are two primary ways to run experiments:

### 1. Single Experiment Run

This mode is for running a single experiment with a defined set of parameters. You can use the default configuration or specify your own.

**To run with the default configuration:**

```bash
python main.py
```

This will use `config/default.yaml` by default.

**To run with one or more custom configurations:**

You can create new `.yaml` files in the `config/` directory. The configurations are merged in the order they are provided, with later files overriding earlier ones.

For example, create `config/fast_lr.yaml`:
```yaml
# config/fast_lr.yaml
training:
  learning_rate: 5e-4
```

Then run:
```bash
python main.py --configs config/default.yaml config/fast_lr.yaml
```
In this case, `learning_rate` will be `5e-4`, while other parameters are taken from `default.yaml`.

### 2. Hyperparameter Sweep with W&B

Sweeps allow you to automatically run multiple experiments with different hyperparameter combinations. This is useful for hyperparameter tuning and exploring the parameter space.

**Step 1: Create or modify a sweep configuration**

The sweep configuration is defined in `sweepconfig/default_sweep.yaml`. You can modify this file or create a new one. The configuration specifies:
- Which parameters to sweep over
- The sweep method (grid, random, or Bayesian optimization)
- The metric to optimize

Example sweep configuration:
```yaml
program: main.py
method: grid  # Options: grid, random, bayes
metric:
  name: loss/total
  goal: minimize

parameters:
  training:
    parameters:
      learning_rate:
        values: [1e-3, 5e-4, 1e-4]
      batch_size:
        values: [128, 256]
  bewley_model:
    parameters:
      beta:
        values: [0.97, 0.975, 0.98]
```

**Step 2: Initialize the sweep**

```bash
wandb sweep sweepconfig/default_sweep.yaml
```

This command will create a sweep on W&B and output a sweep ID. You'll see something like:
```
wandb: Creating sweep from: sweepconfig/default_sweep.yaml
wandb: Creating sweep with ID: neujkn8c
wandb: View sweep at: https://wandb.ai/zhinghe78-uccu/Bewley/sweeps/neujkn8c
wandb: Run sweep agent with: wandb agent zhinghe78-uccu/Bewley/neujkn8c
```

**Step 3: Run the sweep agent**

Copy the agent command from the previous step and run it:
```bash
wandb agent zhinghe78-uccu/Bewley/neujkn8c
```

The agent will automatically run experiments with different hyperparameter combinations. Each run will:
- Get a unique auto-generated name (e.g., "divine-sweep-1", "cosmic-sweep-2")
- Save checkpoints to separate directories: `checkpoints/<run_name>/`
- Log metrics to W&B for comparison

**Step 4 (Optional): Run multiple agents in parallel**

To speed up the sweep, you can run multiple agents in parallel. Simply open additional terminal windows and run the same agent command:

```bash
# Terminal 1
wandb agent your-entity/Bewley-Project-Example/abc123def

# Terminal 2
wandb agent your-entity/Bewley-Project-Example/abc123def

# Terminal 3
wandb agent your-entity/Bewley-Project-Example/abc123def
```

Each agent will pick up different parameter combinations from the sweep queue and run them in parallel.

**Viewing sweep results:**

Navigate to the W&B sweep URL (printed in Step 2) to view:
- Parallel coordinates plot showing parameter relationships
- Metrics comparison across all runs
- Best performing configurations ranked by your chosen metric

**Sweep methods:**

- **`grid`**: Exhaustively tries all combinations of parameters
- **`random`**: Randomly samples parameter combinations (add `count: N` to limit runs)
- **`bayes`**: Uses Bayesian optimization to intelligently search the parameter space

**Note on checkpoints during sweeps:**

During sweeps, the `exp_name` from config files is automatically ignored to prevent checkpoint conflicts. Each sweep run gets its own unique checkpoint directory based on the auto-generated run name.

## Visualizing Decision Rules

After training, you can visualize the learned decision rules (policy functions) from saved checkpoints. The `vis_dc_rle.py` script allows you to see how agent decisions (consumption, labor, savings) vary with different state variables.

### Basic Usage

```bash
python vis_dc_rle.py --checkpoint_dir checkpoints/bewley_default_run --step 2500
```

**Arguments:**
- `--checkpoint_dir`: Path to the checkpoint directory (e.g., `checkpoints/Tax_Exp_BaselineRun`)
- `--step`: Training step to load model weights from (e.g., `2500`)
- `--state_step`: (Optional) Training step to load state from (default: same as `--step`)
- `--config`: (Optional) Path to config file (default: `config/baseline.yaml`)

```bash
# Different step for env and weights
python vis_dc_rle.py --checkpoint_dir checkpoints/bewley_default_run --step 10000 --state_step 2500
```

### What the Script Does

The visualization script:

1. **Loads a checkpoint**: Loads the trained policy network, normalizer, and environment state from the specified checkpoint
2. **Creates a grid of states**: Varies one agent's characteristic (e.g., wealth from 0.1 to 10.0) while keeping everything else fixed
3. **Evaluates the policy**: Runs the policy network on each grid point to get decisions
4. **Plots decision rules**: Creates a 2x2 subplot showing:
   - Consumption policy
   - Labor supply policy
   - Savings policy
   - Savings ratio policy

### Output

The script generates:
- Console output showing policy evaluation results
- A plot saved as `decision_rules_agent{X}_batch{Y}.png`

**Example output:**
```
decision_rules_agent0_batch0.png
```

This plot shows how Agent 0 in World 0 makes decisions as their wealth (money_disposable) varies.

### Customization

The default behavior varies `money_disposable` for Agent 0 in Batch 0. You can customize this by modifying the `make_decision_rules_inputs()` call in `vis_dc_rle.py`:

```python
grid_result = make_decision_rules_inputs(
    state=initial_state,
    batch_idx=0,              # Which world to analyze
    agent_idx=0,              # Which agent to analyze
    vary_var="money_disposable",  # Variable to vary
    vary_range=(0.1, 10.0),   # Range to sweep
    n_points=50               # Number of grid points
)
```

**Variables you can vary:**
- `"money_disposable"` - Current wealth
- `"ability"` - Current productivity
- `"savings"` - Previous period savings

### Understanding the Plots

The decision rule plots help you understand:
- **Consumption smoothing**: How agents adjust consumption in response to wealth changes
- **Labor-leisure tradeoff**: How labor supply responds to productivity/wealth
- **Precautionary savings**: How agents save for future uncertainty
- **Policy heterogeneity**: Whether different agents have learned different strategies

## Tracking Regime Dynamics

The framework automatically tracks behavioral regime clustering (high-saving vs. low-saving agents) during and after training.

### During Training (Automatic)

Two things run at every `cluster_interval` steps (default: 5000):

1. **Lightweight GMM snapshot** — fast cross-sectional clustering on the current batch, logged to W&B as:
   - `regime/fraction_high`, `regime/fraction_low`
   - `regime/separation_score`, `regime/bic_k2`, `regime/aic_k2`

2. **Full `RegimeAnalyzer` pipeline** — runs in a background thread (does not block training) once the panel buffer has ≥50 steps of history. Results are saved to:
   ```
   checkpoints/{run_name}/plot_data/regime_analysis/step_{N}/
     enriched_panel.csv       # Panel with regime labels
     regime_summary.csv       # Per-regime mean/std/quantiles
     regime_scatter.png
     regime_paths.png
     transition_matrix.png
     switch_distribution.png
     regime_summary.png
   ```

Panel data (ring buffer, last `panel_buffer_size=200` steps) is saved at every checkpoint:
```
checkpoints/{run_name}/plot_data/panel/panel_step_{N}.npz
```

### After Training

Run the standalone script for a full analysis including robustness checks (BIC/AIC across K and filter settings):

```bash
python run_cluster_analysis.py \
  --panel_npz checkpoints/{run_name}/plot_data/panel/panel_step_{final_step}.npz \
  --output_dir checkpoints/{run_name}/plot_data/regime_analysis/final
```

Key flags:
| Flag | Default | Description |
|------|---------|-------------|
| `--ability_cutoff_quantile` | `0.5` | Lower → include more agents in analysis |
| `--labor_upper` | `0.95` | Upper bound on labor to filter constrained agents |
| `--n_components` | `2` | Number of GMM clusters |
| `--skip_robustness` | off | Skip BIC/AIC grid (faster) |
| `--no_plots` | off | Skip plot generation |

Check `robustness_bic.csv` to confirm K=2 has lower BIC than K=1 and K=3 before trusting the classification.

### Reconstructing Full Training History

The ring buffer only holds the last 200 steps per checkpoint, but NPZ files across checkpoints can be stitched together:

```python
import pandas as pd
from src.cluster_analysis import PanelBuffer

dfs = []
for step in [5000, 10000, 15000, ...]:
    buf = PanelBuffer.from_npz(f"checkpoints/{run_name}/plot_data/panel/panel_step_{step}.npz")
    dfs.append(buf.to_dataframe())

full_df = pd.concat(dfs).drop_duplicates(subset=['agent_id', 'time'])
# Then pass full_df to RegimeAnalyzer for the complete trajectory
```

---

## Comparing Runs

After training multiple configs, use `compare_runs.py` to overlay or grid decision rules and input-output plots from different runs side by side. Numerical plot data (`.npz`) is saved automatically during training under `checkpoints/{run_name}/plot_data/`.

### Basic Usage

```bash
# Compare two runs (overlay mode, median quantile line)
python compare_runs.py checkpoints/Bewley_Hetero_ability_low_uncer checkpoints/Bewley_Hetero_ability_high_uncer

# Specify a training step and which plots to compare
python compare_runs.py checkpoints/RunA checkpoints/RunB --step 20000 --plots A1 B1 H1

# Show all quantile lines instead of just the median
python compare_runs.py checkpoints/RunA checkpoints/RunB --color-level all

# Use grid layout (one subplot per run) instead of overlay
python compare_runs.py checkpoints/RunA checkpoints/RunB --layout grid

# Compare input-output plots
python compare_runs.py checkpoints/RunA checkpoints/RunB --type input_output

# Clip x-axis to the intersection of runs' ranges
python compare_runs.py checkpoints/RunA checkpoints/RunB --shared-xrange

# Custom output directory
python compare_runs.py checkpoints/RunA checkpoints/RunB --output-dir my_comparison/
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `run_dirs` | (required) | Two or more checkpoint directories to compare |
| `--step` | latest common | Training step to compare |
| `--plots` | all common | Plot IDs to generate (e.g., `A1 B1 A1-1 H1`) |
| `--type` | `grid` | Data type: `grid` (decision rules) or `input_output` |
| `--layout` | auto | `overlay` (<=3 runs) or `grid` (4+ runs) |
| `--color-level` | `q50` | Quantile level: `q5`/`q25`/`q50`/`q75`/`q95`/`all` |
| `--shared-xrange` | off | Clip x-axis to intersection of all runs' ranges |
| `--output-dir` | `comparison_plots` | Where to save output PDFs |

### Output

Comparison plots are saved as PDFs in the output directory:
```
comparison_plots/
  compare_A1_step_20000.pdf
  compare_B1_step_20000.pdf
  compare_H1_step_20000.pdf
  ...
```

