# Batch Training Guide

This guide explains how to run training with multiple config files automatically.

## Available Scripts

Three shell scripts are provided for running all configs in the `config/` directory:

| Script | Mode | Use Case |
|--------|------|----------|
| `run_all_configs.sh` | Sequential with detailed logging | **Recommended** - Production runs with full logs |
| `run_all_configs_simple.sh` | Sequential minimal output | Quick testing |
| `run_all_configs_parallel.sh` | Parallel execution | Multiple GPUs or fast iteration |

---

## 1. Standard Sequential Run (Recommended)

**Script:** `run_all_configs.sh`

### Features:
- ✅ Runs one config at a time (safe for single GPU)
- ✅ Colored console output with progress tracking
- ✅ Saves individual log files for each config
- ✅ Shows success/failure summary at the end
- ✅ Asks for confirmation before starting

### Usage:

```bash
./run_all_configs.sh
```

### Example Output:

```
========================================
Running All Configs in config/ Directory
========================================

Activating conda environment: sml
✓ Conda environment activated

Found 3 config file(s):
  - config/default.yaml
  - config/savings.yaml
  - config/savings_income.yaml

Press Enter to start training, or Ctrl+C to cancel...

========================================
[1/3] Running: config/default.yaml
========================================
...training output...
✓ [1/3] SUCCESS: config/default.yaml

========================================
[2/3] Running: config/savings.yaml
========================================
...training output...
✓ [2/3] SUCCESS: config/savings.yaml

========================================
Batch Training Complete
========================================
Total configs: 3
Successful: 3
Failed: 0

Logs saved to: logs/batch_run_20251113_120000
```

### Log Files:

Each config run is logged separately:
```
logs/batch_run_20251113_120000/
├── default.log
├── savings.log
└── savings_income.log
```

---

## 2. Simple Sequential Run

**Script:** `run_all_configs_simple.sh`

### Features:
- Minimal output
- No log files
- Quick and simple
- Good for testing script changes

### Usage:

```bash
./run_all_configs_simple.sh
```

### Example Output:

```
Running: config/default.yaml
...training output...
Completed: config/default.yaml
---
Running: config/savings.yaml
...training output...
Completed: config/savings.yaml
---
All configs completed!
```

---

## 3. Parallel Run (Advanced)

**Script:** `run_all_configs_parallel.sh`

### ⚠️ WARNING:
- Runs **all configs simultaneously**
- Requires sufficient memory and compute resources
- Multiple processes will compete for GPU/CPU
- May cause out-of-memory errors if not careful

### Features:
- ✅ Fastest option (if you have resources)
- ✅ Uses GNU `parallel` if available (recommended)
- ✅ Fallback to `xargs` if `parallel` not installed
- ✅ Individual log files for each run
- ✅ Configurable max parallel jobs

### Usage:

**Default (run all at once):**
```bash
./run_all_configs_parallel.sh
```

**Limit to 2 parallel jobs:**
```bash
# Edit the script and uncomment/modify:
MAX_JOBS=2
```

### Installing GNU Parallel (Optional but Recommended):

```bash
# macOS
brew install parallel

# Linux
sudo apt-get install parallel
```

### Example Output:

```
=========================================
WARNING: Running all configs in PARALLEL
This may require significant resources!
=========================================
Logs will be saved to: logs/parallel_run_20251113_120000

Using GNU parallel...
Started: config/default.yaml
Started: config/savings.yaml
Started: config/savings_income.yaml
✓ Completed: config/default.yaml
✓ Completed: config/savings.yaml
✓ Completed: config/savings_income.yaml

=========================================
All parallel runs completed!
Check logs in: logs/parallel_run_20251113_120000
=========================================
```

---

## Configuration Files

Currently available configs:

```bash
$ ls config/
default.yaml           # Default configuration
savings.yaml           # Savings-focused variant
savings_income.yaml    # Savings + income tax variant
```

---

## Running Individual Configs

To run a single config file:

```bash
python main.py --configs config/default.yaml
```

To run with multiple configs (they merge in order):

```bash
python main.py --configs config/default.yaml config/savings.yaml
```

---

## Customizing the Scripts

### Change Conda Environment Name

Edit the script and modify:

```bash
CONDA_ENV="sml"  # Change to your environment name
```

### Filter Which Configs to Run

**Option 1: Temporarily move configs**
```bash
mkdir config/archive
mv config/savings.yaml config/archive/  # Only run default.yaml and savings_income.yaml
```

**Option 2: Edit script to use pattern**
```bash
# In the script, change:
CONFIG_FILES=($(ls ${CONFIG_DIR}/*.yaml))

# To:
CONFIG_FILES=($(ls ${CONFIG_DIR}/default*.yaml))  # Only configs starting with "default"
```

### Add Pre/Post Processing

Edit any script and add code before/after the main loop:

```bash
# Before training
echo "Starting batch at $(date)"
python prepare_data.py  # Example preprocessing

# ... main training loop ...

# After training
echo "Finished batch at $(date)"
python analyze_results.py  # Example post-processing
```

---

## Troubleshooting

### Script Won't Run

```bash
# Make sure it's executable
chmod +x run_all_configs.sh

# Check conda environment
conda env list
```

### Out of Memory (Parallel Script)

Reduce parallel jobs:
```bash
# Edit run_all_configs_parallel.sh
MAX_JOBS=1  # Run only 1 at a time (same as sequential)
```

Or use the sequential script instead:
```bash
./run_all_configs.sh
```

### Logs Not Saving

Check that the `logs/` directory is writable:
```bash
mkdir -p logs
ls -la logs/
```

### WandB Rate Limiting

If running many configs quickly, you may hit WandB rate limits. Add delays:

```bash
# In the script, after each python run:
sleep 10  # Wait 10 seconds between runs
```

---

## Tips & Best Practices

### 1. Test First

Run one config manually before batch processing:
```bash
python main.py --configs config/default.yaml
```

### 2. Check Logs Regularly

When running in background:
```bash
# Watch the latest log file
tail -f logs/batch_run_*/default.log
```

### 3. Run in Background

For long batch runs:
```bash
nohup ./run_all_configs.sh > batch_output.log 2>&1 &

# Check progress
tail -f batch_output.log
```

### 4. Use tmux/screen

For persistent sessions:
```bash
tmux new -s training
./run_all_configs.sh
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

### 5. Monitor Resources

```bash
# CPU/Memory
htop

# GPU (if using CUDA)
watch -n 1 nvidia-smi

# Disk space
df -h
```

---

## Example Workflow

**Complete workflow for running all experiments:**

```bash
# 1. Test single config
python main.py --configs config/default.yaml

# 2. If successful, run all configs
./run_all_configs.sh

# 3. Check logs
ls -lh logs/batch_run_*/

# 4. View specific log
less logs/batch_run_20251113_120000/savings.log

# 5. Analyze results in WandB
# Visit: https://wandb.ai/your-username/Bewley-Project-Example
```

---

## Summary

| Need | Use |
|------|-----|
| Production batch runs | `./run_all_configs.sh` |
| Quick testing | `./run_all_configs_simple.sh` |
| Multiple GPUs/fast iteration | `./run_all_configs_parallel.sh` |
| Single config | `python main.py --configs config/NAME.yaml` |

**Recommended:** Start with `run_all_configs.sh` for safe, logged, sequential execution.
