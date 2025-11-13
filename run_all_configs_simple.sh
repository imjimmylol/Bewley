#!/bin/bash
# run_all_configs_simple.sh
# Simple script to run all configs sequentially without fancy logging

# Activate conda environment
source ~/.bash_profile
conda activate sml

# Run each config file
for config in config/*.yaml; do
    echo "Running: ${config}"
    python main.py --configs ${config}
    echo "Completed: ${config}"
    echo "---"
done

echo "All configs completed!"
