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

This mode uses Weights & Biases to automatically run multiple experiments to find the best hyperparameters.

**Step 1: Initialize the Sweep**

Tell W&B to create a new sweep based on your configuration file.

```bash
wandb sweep sweepconfig/default_sweep.yaml
```

W&B will output a command with a unique **sweep ID**. It will look like this:
`wandb agent <YOUR_ENTITY>/<YOUR_PROJECT>/<SWEEP_ID>`

**Step 2: Run the W&B Agent**

Copy the command from the previous step and run it in your terminal.

```bash
wandb agent <YOUR_ENTITY>/<YOUR_PROJECT>/<SWEEP_ID>
```

The agent will now start executing `main.py` repeatedly with different hyperparameter combinations defined in `default_sweep.yaml`. You can view the results in real-time on your W&B dashboard.
