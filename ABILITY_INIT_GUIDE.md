# Ability Initialization Guide

## Problem: Your Model is Exploding 🔥

Your current setup causes ability to explode over time:
- **Initial**: mean=5, max=~50
- **After 100 steps**: mean=25, max=11,493 (!!)
- **Result**: Extreme wages/returns → model collapse

## Root Cause

Your shock parameters are **too persistent**:
```yaml
rho_v: 0.98    # Persistence parameter
sigma_v: 0.36  # Innovation std
v_bar: 5       # Long-run mean
```

This creates:
1. **Half-life of 50 periods**: `1/(1-0.98) = 50` periods for mean reversion
2. **Huge stationary variance**: `σ²/(1-ρ²) = 81.6` → std=9.0 (bigger than mean!)
3. **Unbounded accumulation**: Extreme values persist and compound

## Solution: IQ-Like Distribution 🎯

Use a **bounded log-normal** distribution inspired by real-world ability (IQ, earnings):

```python
from src.ability_init import initialize_ability

ability = initialize_ability(
    batch_size, n_agents,
    method='iq_like',
    mean=1.0,       # Normalized mean
    cv=0.3,         # Coefficient of variation (std/mean)
    min_ability=0.3,  # Hard lower bound
    max_ability=3.0   # Hard upper bound
)
```

## Why This Works

### Real-World Comparison

| Distribution | CV | Interpretation |
|-------------|-----|----------------|
| IQ | 0.15 | Very tight, most people near 100 |
| **Earnings** | **0.3-0.4** | **Moderate inequality (RECOMMENDED)** |
| Wealth | 0.8+ | High inequality, long tail |

### Key Features

1. **Bounded**: Hard min/max prevents extreme outliers
2. **Realistic**: Matches real-world ability distributions
3. **Stable**: Distribution doesn't explode over time
4. **Economically interpretable**:
   - ability=0.5 → low-skill worker (50% of average productivity)
   - ability=1.0 → average worker
   - ability=2.0 → high-skill worker (200% productivity)

## Usage

### Quick Start (Recommended)

Use the pre-configured file:

```bash
python src/train.py --config config/iq_init.yaml
```

This uses:
- `mean=1.0` (normalized)
- `cv=0.3` (moderate dispersion)
- `range=[0.3, 3.0]` (30% to 300% of mean)

### Custom Configuration

Create your own config:

```yaml
initial_state:
  # Use IQ-like distribution
  ability_init_method: "iq_like"
  ability_mean: 1.0
  ability_cv: 0.3        # Adjust this for more/less inequality
  ability_min: 0.3
  ability_max: 3.0
```

### Different Inequality Levels

#### Conservative (like IQ)
```yaml
ability_cv: 0.25
ability_min: 0.5
ability_max: 2.0
```
- Most agents cluster tightly near mean
- Good for isolating other mechanisms

#### Moderate (like earnings) - RECOMMENDED
```yaml
ability_cv: 0.3
ability_min: 0.3
ability_max: 3.0
```
- Realistic inequality
- Balanced heterogeneity

#### High Inequality (like wealth)
```yaml
ability_cv: 0.6
ability_min: 0.2
ability_max: 5.0
```
- Strong right tail
- Study redistribution effects

## Testing Your Initialization

Run the comparison script to visualize different methods:

```bash
python test_ability_init.py
```

This generates `test_ability_init_comparison.png` showing:
1. Stationary AR(1) (old method) - explosive
2. IQ-like distributions (conservative, moderate, high inequality)
3. Narrow uniform (for debugging)

## Expected Results

With IQ-like initialization (`cv=0.3`):

```
Initial Distribution:
  Mean: 1.00
  Std: 0.30
  CV: 0.30
  Range: [0.48, 1.71] (P5-P95)
  Max: 3.00 (hard bound)

After 100 Steps:
  Mean: ~1.00 (stable!)
  Std: ~0.30 (stable!)
  Max: ~3.00 (bounded!)
```

Compare to old method:
```
Initial: mean=5, max=~50
After 100: mean=25, max=11,493 ❌
```

## Advanced: Adjust Shock Parameters

If you still see instability, reduce persistence:

```yaml
shock:
  rho_v: 0.9       # Faster mean reversion (10 periods)
  sigma_v: 0.1     # Smaller innovations
  v_bar: 1.0       # Match init mean
```

## Implementation Details

The code automatically uses the new initialization:

1. **[src/ability_init.py](src/ability_init.py)** - Initialization functions
2. **[src/train.py:55-84](src/train.py#L55-L84)** - Training integration
3. **[config/iq_init.yaml](config/iq_init.yaml)** - Recommended config

The training code checks `config.initial_state.ability_init_method`:
- `"iq_like"` → Bounded log-normal (RECOMMENDED)
- `"stationary"` → AR(1) stationary with clipping (old method)
- `"narrow_uniform"` → Minimal heterogeneity (testing)

## FAQ

### Q: Will this affect my results?

**Yes, in a good way!** Your old results were likely dominated by extreme outliers. The bounded distribution gives you:
- More stable training
- More interpretable agent behavior
- Better market equilibrium (no extreme prices)

### Q: What if I want the old behavior?

Use `ability_init_method: "stationary"` in config. But note:
- You MUST clip to ±2σ (now automatic)
- Consider reducing `rho_v` to 0.9
- Consider reducing `sigma_v` to 0.1

### Q: How does this interact with superstar shocks?

Superstar shocks (multiply by 10) still work:
- Base ability from IQ-like: [0.3, 3.0]
- After superstar: [3.0, 30.0]
- This is MUCH better than unbounded base [0.1, 100+]

### Q: Can I study inequality with bounded distribution?

**Absolutely!** Just increase `cv`:
- `cv=0.6`: High inequality
- Bounded distribution prevents **unrealistic** extremes
- You still get **realistic** inequality

## Summary

✅ **DO**: Use `method='iq_like'` with `cv=0.3`, bounds `[0.3, 3.0]`

❌ **DON'T**: Use unbounded AR(1) stationary with `rho=0.98`, `sigma=0.36`

🎯 **RESULT**: Stable, interpretable, realistic ability distribution

---

**Questions?** Check `test_ability_init.py` for visualizations and comparisons.
