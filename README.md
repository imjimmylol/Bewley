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
- `--step`: Training step to load (e.g., `2500`)
- `--config`: (Optional) Path to config file (default: `config/baseline.yaml`)

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

