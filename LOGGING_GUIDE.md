# Training Logging Guide

## Overview

The training loop logs comprehensive metrics including losses, state statistics, market prices, and **normalized network inputs** for debugging.

## Logged Metrics

### 1. Losses

| Metric | Description | Expected Range |
|--------|-------------|----------------|
| `loss/total` | Total weighted loss | 0 to ∞ (decreasing) |
| `loss/fb` | Forward-Backward consistency | 0 to ∞ (→ 0) |
| `loss/aux_mu` | Euler equation loss | 0 to ∞ (→ 0) |
| `loss/labor_foc` | Labor FOC loss | 0 to ∞ (→ 0) |

**What to watch:**
- All losses should **decrease over time**
- If losses plateau, consider adjusting learning rate or loss weights
- If losses explode, check for NaN/Inf in gradients

### 2. State Statistics

| Metric | Description | Typical Range |
|--------|-------------|---------------|
| `state/consumption_mean` | Mean consumption across agents | > 0 |
| `state/labor_mean` | Mean labor supply | [0, 1] |
| `state/savings_mean` | Mean savings | > 0 |
| `state/ability_mean` | Mean ability | ~1.5 (v_bar) |

**What to watch:**
- Consumption and savings should be positive
- Labor should be in [0, 1] (enforced by sigmoid)
- Ability should stabilize around `v_bar` from config

### 3. Market Prices

| Metric | Description | Typical Range |
|--------|-------------|---------------|
| `market/wage` | Equilibrium wage rate | > 0 |
| `market/return` | Return to capital | ~0.04 (4%) |

**What to watch:**
- Wage should be reasonable given productivity
- Return should be close to `config.bewley_model.r`

### 4. **Normalized Network Inputs (Debugging)** ⭐

These metrics help you debug the normalization process and ensure the policy network receives well-scaled inputs.

#### Normalized Features

| Metric | Description | Target |
|--------|-------------|--------|
| `debug/normalized_money_mean` | Mean of normalized money fed to network | ~0 |
| `debug/normalized_money_std` | Std of normalized money | ~1 |
| `debug/normalized_ability_mean` | Mean of normalized ability fed to network | ~0 |
| `debug/normalized_ability_std` | Std of normalized ability | ~1 |

**Expected behavior:**
- After warmup period (~100-1000 steps), means should be **close to 0**
- After warmup period, stds should be **close to 1**
- These indicate the `RunningPerAgentWelford` normalizer is working correctly

#### Raw Features

| Metric | Description | Expected |
|--------|-------------|----------|
| `debug/raw_money_mean` | Mean of raw money before normalization | Variable |
| `debug/raw_money_std` | Std of raw money before normalization | Variable |
| `debug/raw_ability_mean` | Mean of raw ability before normalization | ~1.5 |
| `debug/raw_ability_std` | Std of raw ability before normalization | Variable |

**What to watch:**
- Raw ability mean should stabilize around `v_bar = 1.5`
- Raw money varies based on economic dynamics
- Large discrepancies between raw and normalized indicate normalization is working

## Example Output

### Console Output (every `display_step`)

```
Step 1000/100000
  Loss: 2.4531 (fb=0.1234, euler=1.8765, labor=0.4532)
  State Statistics:
    - Mean consumption: 1.245
    - Mean labor: 0.456
    - Mean savings: 3.123
    - Mean ability: 1.498
    - Market wage: 1.876
    - Market return: 0.0405
  Normalized Network Inputs (debugging):
    - Normalized money: mean=0.012, std=0.987
    - Normalized ability: mean=-0.003, std=1.012
    - Raw money: mean=4.567, std=3.234
    - Raw ability: mean=1.498, std=0.456
```

### Interpretation

**Good normalization (step 1000+):**
```
Normalized money: mean=0.012, std=0.987  ✅ Close to (0, 1)
Normalized ability: mean=-0.003, std=1.012  ✅ Close to (0, 1)
```

**Bad normalization (potential issues):**
```
Normalized money: mean=2.345, std=5.678  ❌ Not centered, high variance
Normalized ability: mean=-0.001, std=0.001  ❌ Collapsed variance
```

## WandB Logging

All metrics are logged to WandB (if enabled) under namespaces:

```
loss/
  - total
  - fb
  - aux_mu
  - labor_foc

state/
  - consumption_mean
  - labor_mean
  - savings_mean
  - ability_mean

market/
  - wage
  - return

debug/  ⭐ NEW
  - normalized_money_mean
  - normalized_money_std
  - normalized_ability_mean
  - normalized_ability_std
  - raw_money_mean
  - raw_money_std
  - raw_ability_mean
  - raw_ability_std
```

## Debugging Guide

### Issue: Losses not decreasing

**Check:**
1. Are normalized inputs well-scaled? (`debug/normalized_*` near (0, 1)?)
2. Are gradients flowing? (Enable gradient clipping logs)
3. Is learning rate appropriate?

**Actions:**
- If normalized inputs bad: Increase warmup steps before real training
- If gradients too large: Enable gradient clipping
- If gradients too small: Increase learning rate

### Issue: Normalized inputs have wrong scale

**Symptoms:**
```
debug/normalized_money_mean: 5.678  # Should be ~0
debug/normalized_money_std: 0.001   # Should be ~1
```

**Possible causes:**
1. Not enough warmup steps (normalizer needs time to accumulate stats)
2. `update_normalizer=False` during training (should be True)
3. Extreme outliers in data

**Actions:**
- Check `update_normalizer=True` in `env.step()`
- Add warmup phase (first 100-1000 steps without gradient updates)
- Check for NaN/Inf in raw data

### Issue: Raw ability diverging from v_bar

**Symptoms:**
```
debug/raw_ability_mean: 10.456  # Should be ~1.5
```

**Possible causes:**
1. Ability shocks not detached (gradient accumulation)
2. Wrong AR(1) parameters
3. Superstar dynamics dominating

**Actions:**
- Check `src/shocks.py` has `.detach()` calls (already fixed)
- Verify `config.shock.rho_v`, `v_bar` are correct
- Check superstar probability `p` and persistence `q`

## Feature Extraction Details

The normalized features are extracted from `env._prepare_features()` which:

1. **Normalizes** money and ability using `RunningPerAgentWelford`
2. **Builds inputs** via `src/utils/buildipnuts.py`
3. **Returns** feature tensor of shape `(B, A, 2A+2)`

### Feature Structure

```python
normalized_features: (B, A, 2A+2)
  [0:A]       → All agents' money (aggregate info)
  [A:2A]      → All agents' ability (aggregate info)
  [-2]        → This agent's money (individual)
  [-1]        → This agent's ability (individual)
```

The logging focuses on **individual features** (`[-2]` and `[-1]`) since these show per-agent normalization.

## Recommended Monitoring

### During Training

**Every 1000 steps:**
- Check `loss/total` is decreasing
- Check `debug/normalized_*_mean` near 0
- Check `debug/normalized_*_std` near 1

**Every 10000 steps:**
- Plot loss curves in WandB
- Check state statistics are reasonable
- Verify market prices make economic sense

### Post-Training

**Final metrics (last 1000 steps):**
- `loss/total` < 0.1 (ideally)
- `state/consumption_mean` > 0
- `state/labor_mean` ∈ [0.2, 0.8] (reasonable work hours)
- `market/return` ≈ `config.bewley_model.r`

## Performance Tips

The normalized feature extraction is done **with torch.no_grad()** to avoid:
- Building computational graph for debugging tensors
- Memory overhead from gradient tracking

This adds minimal overhead (~0.1ms per step) while providing valuable debugging info.
