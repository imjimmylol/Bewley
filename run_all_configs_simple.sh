#!/bin/bash
# run_all_configs_simple.sh
# Simple script to run all configs sequentially without fancy logging
#
# Usage:
#   ./run_all_configs_simple.sh
#   ./run_all_configs_simple.sh config/1202
#   ./run_all_configs_simple.sh --project myproj
#   ./run_all_configs_simple.sh config/1202 --project myproj
#   ./run_all_configs_simple.sh --project myproj config/1202

set -u

CONFIG_DIR="config"
PROJECT=""

usage() {
  echo "Usage: $0 [config_directory] [--project <name>]"
  echo "       $0 [--project <name>] [config_directory]"
  echo ""
  echo "Examples:"
  echo "  $0"
  echo "  $0 config/1202"
  echo "  $0 --project myproj"
  echo "  $0 config/experiments --project myproj"
}

# Parse args (supports: --project X, --project=X, -p X, and optional positional CONFIG_DIR)
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--project)
      if [[ $# -lt 2 ]]; then
        echo "Error: --project requires a value"
        usage
        exit 1
      fi
      PROJECT="$2"
      shift 2
      ;;
    --project=*)
      PROJECT="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Error: Unknown option: $1"
      usage
      exit 1
      ;;
    *)
      # positional: config directory (only allow one)
      if [[ "$CONFIG_DIR" != "config" ]]; then
        echo "Error: Multiple config directories provided: '$CONFIG_DIR' and '$1'"
        usage
        exit 1
      fi
      CONFIG_DIR="$1"
      shift
      ;;
  esac
done

# Check if directory exists
if [ ! -d "$CONFIG_DIR" ]; then
  echo "Error: Directory '$CONFIG_DIR' does not exist"
  usage
  exit 1
fi

# Count yaml files
YAML_COUNT=$(find "$CONFIG_DIR" -maxdepth 1 -name "*.yaml" -print | wc -l | tr -d ' ')
if [ "$YAML_COUNT" -eq 0 ]; then
  echo "Error: No .yaml files found in '$CONFIG_DIR'"
  exit 1
fi

echo "========================================================================"
echo "Running all configs from: $CONFIG_DIR"
echo "Found $YAML_COUNT config file(s)"
if [[ -n "$PROJECT" ]]; then
  echo "Project: $PROJECT"
fi
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

  if [[ -n "$PROJECT" ]]; then
    python main.py --configs "${config}" --project "${PROJECT}"
  else
    python main.py --configs "${config}"
  fi

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