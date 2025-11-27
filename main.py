# main.py
import argparse
import wandb
import os
from src.train import train
from src.utils.configloader import load_configs, dict_to_namespace, compute_derived_params

# Set WANDB_API_KEY from environment variables if you have it there
# os.environ["WANDB_API_KEY"] = "YOUR_API_KEY"

def main():
    """
    Main entry point for the training script.
    Handles config loading, wandb initialization, and starting the training.
    """
    # 1. Parse CLI arguments for config files
    parser = argparse.ArgumentParser(description="Run a training experiment based on GEMINI.md structure.")
    parser.add_argument(
        '--configs',
        nargs='+',
        default=['config/default.yaml'],
        help='Paths to one or more config files. They are merged in the given order.'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='Bewley-Project-Example',
        help='Wandb project name (default: Bewley-Project-Example)'
    )
    args, unknown = parser.parse_known_args()

    # 2. Load and merge configs from specified YAML files
    # This forms the base configuration.
    base_config = load_configs(args.configs)

    # 3. Initialize wandb
    # - `project`: Name of the project in wandb.
    # - `name`: Use exp_name from config if provided, BUT NOT in sweep mode
    #           (sweeps need unique auto-generated names for each run)
    # - `config`: The base config dictionary. Wandb will override this with
    #             parameters from a sweep agent if one is running.

    # Check if we're in sweep mode by looking for WANDB_SWEEP_ID env variable
    is_sweep = os.environ.get('WANDB_SWEEP_ID') is not None

    if is_sweep:
        # In sweep mode: let wandb auto-generate unique names
        run = wandb.init(project=args.project, config=base_config)
    else:
        # Normal mode: use exp_name from config if provided
        run_name = base_config.get('exp_name', None)
        run = wandb.init(project=args.project, name=run_name, config=base_config)
    
    # 4. Get the final configuration from wandb
    # `wandb.config` is a special object that holds the definitive parameters for this run,
    # including any overrides from a sweep.
    final_config_dict = dict(wandb.config)

    # 5. Re-run derived parameter computation
    # This is crucial because a sweep might change a parameter (e.g., rho_v)
    # that a derived parameter (e.g., v_min) depends on.
    final_config_dict = compute_derived_params(final_config_dict)
    
    # 6. Convert the final config dict to a namespace for easy access
    # (e.g., `config.training.learning_rate` instead of `config['training']['learning_rate']`)
    final_config_ns = dict_to_namespace(final_config_dict)

    # 7. Call the training function with the final config and the wandb run object
    print("--- Starting Training Run ---")
    train(config=final_config_ns, run=run)
    print("--- Training Run Finished ---")
    
    # 8. Finalize the wandb run
    run.finish()

if __name__ == "__main__":
    # Before running, make sure you have wandb installed (`pip install wandb`)
    # and are logged in (`wandb login`).
    main()
