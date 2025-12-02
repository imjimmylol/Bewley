#!/bin/bash
# run_all_configs_simple.sh
# Simple script to run all configs sequentially without fancy logging
#
# Usage:
#   ./run_all_configs_simple.sh                    # Uses config/ directory (default)
#   ./run_all_configs_simple.sh config/1202        # Uses specific directory
#   ./run_all_configs_simple.sh config/experiments # Uses custom directory

# Get config directory from argument or use default
CONFIG_DIR="${1:-config}"

# Check if directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Directory '$CONFIG_DIR' does not exist"
    echo "Usage: $0 [config_directory]"
    echo "Example: $0 config/1202"
    exit 1
fi

# Count yaml files
YAML_COUNT=$(find "$CONFIG_DIR" -maxdepth 1 -name "*.yaml" | wc -l)
if [ "$YAML_COUNT" -eq 0 ]; then
    echo "Error: No .yaml files found in '$CONFIG_DIR'"
    exit 1
fi

echo "========================================================================"
echo "Running all configs from: $CONFIG_DIR"
echo "Found $YAML_COUNT config file(s)"
echo "========================================================================"
echo ""

# Activate conda environment
source ~/.bash_profile
conda activate sml

# Run each config file
CONFIG_NUM=0
for config in "$CONFIG_DIR"/*.yaml; do
    CONFIG_NUM=$((CONFIG_NUM + 1))
    echo "[$CONFIG_NUM/$YAML_COUNT] Running: ${config}"
    echo "------------------------------------------------------------------------"

    python main.py --configs "${config}"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Completed: ${config}"
    else
        echo "✗ Failed: ${config} (exit code: $EXIT_CODE)"
    fi

    echo "------------------------------------------------------------------------"
    echo ""
done

echo "========================================================================"
echo "All configs completed!"
echo "Total: $CONFIG_NUM config(s) processed from $CONFIG_DIR"
echo "========================================================================"
