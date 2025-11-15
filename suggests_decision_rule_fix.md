# Suggestions for Fixing Flat Decision Rules

**Problem**: Savings policy shows minimal variation across wealth levels (nearly linear), suggesting the model hasn't learned realistic precautionary savings behavior.

## Diagnosis Summary

After analyzing your code, I've identified several potential causes ranked by likelihood:

---

## 🔴 Critical Issues (Fix These First)

### 1. Normalizer: Initial Values Bias + High Momentum

**Location**: `src/normalizer.py:226-227`

**Problem**:
```python
alpha = 1.0 - self.momentum  # alpha = 0.01 when momentum=0.99
s.mean = self.momentum * s.mean + alpha * mean_b  # 99% old, 1% new
```

- **First batch determines everything**: The very first batch completely initializes the normalizer statistics
- **Extremely slow adaptation**: With `momentum=0.99`, it takes ~460 steps for the first batch's influence to decay to 1%
- **Early-training bias persists**: If agents start with atypical wealth (e.g., all initialized at 1.0), the normalizer "remembers" this for hundreds of steps

**Impact on Your Model**:
1. Early in training, policy is random → wealth distribution is narrow/atypical
2. Normalizer learns `mean_wealth ≈ X`, `std_wealth ≈ Y` from this bad distribution
3. Even as policy improves and wealth distribution changes, normalizer adapts very slowly
4. Network sees distorted inputs for most of training → learns poor policy

**Evidence to Check**:
- Load your normalizer checkpoint and inspect statistics for `moneydisposalbe` (note: typo in code!)
- Check if `mean` and `std` values seem reasonable for your wealth distribution
- Test: Would values in range [0.1, 10.0] get clipped after normalization?

### 2. Normalizer: Count Saturation

**Location**: `src/normalizer.py:231`

```python
s.count = torch.clamp(s.count + 1, max=1000.0)
```

**Problem**:
- After 1000 training steps, `count` stops growing
- Variance calculation uses count: `var = M2 / (count - 1)`
- Combined with EMA on `M2`, normalizer effectively **freezes** after 1000 steps
- Your model trained for 10,000+ steps → last 9,000 steps had frozen normalizer statistics!

**Impact**: Even if wealth distribution shifts dramatically in late training, normalizer can't adapt.

### 3. Insufficient Training Steps

**Your Config**: `config/baseline.yaml` shows only **100 training steps**

**Problem**:
- 100 steps is far too short for convergence
- Normalizer only sees 100 batches → statistics not representative
- Policy network has barely started learning

**Check**:
- Which config did you use for the checkpoint you visualized?
- `config/default.yaml` has 10,000 steps (better, but still might be insufficient)
- Look at your wandb loss curves - are they still decreasing at the end?

---

## 🟡 Moderate Issues

### 4. FB Loss May Suppress Savings Variation

**Location**: `src/calloss.py:174-177`

```python
r1 = savings_ratio
r2 = (1-mu)
return torch.mean((r1+r2-torch.sqrt(r1**2+r2**2))**2)
```

**Problem**:
This enforces complementary slackness: `mu * savings_ratio = 0`

The loss is minimized when:
- **Either** `savings_ratio ≈ 0` (at borrowing constraint, mu > 0)
- **Or** `mu ≈ 1` (interior solution, savings_ratio can be anything)

**Potential Issue**:
- If the network learns to set `mu ≈ constant` across all states, this forces `savings_ratio ≈ constant`
- This could explain why you see flat savings behavior
- The loss doesn't directly incentivize varying savings with wealth

### 5. Product-Based Euler Loss

**Location**: `src/calloss.py:217`

```python
return torch.mean(eulerloss_A * eulerloss_B)
```

**Problem**:
- Takes **product** of Euler residuals from two future branches
- If `eulerloss_A` and `eulerloss_B` have opposite signs, they partially cancel
- Loss can be small even if both branches have large errors (if errors have opposite signs)

**Better Alternative**:
```python
return torch.mean(eulerloss_A**2) + torch.mean(eulerloss_B**2)
```

This penalizes errors in both branches independently.

### 6. Weak Shock Uncertainty

**Your Config**:
```yaml
shock:
  rho_v: 0.95        # Very persistent
  sigma_v: 0.2       # Moderate volatility
```

**Problem**:
- With `rho_v=0.95`, shocks are very persistent → agents can predict future income well
- `sigma_v=0.2` might not create enough uncertainty to motivate precautionary savings
- Less uncertainty → weaker precautionary savings motive → flatter savings policy

**Test**: Temporarily increase to `sigma_v: 0.4` and see if savings behavior becomes more responsive.

---

## 🟢 Minor Issues / Worth Checking

### 7. Typo in Normalizer Variable Name

**Location**: `src/environment.py:77`

```python
moneydisposable_normalized = self.normalizer.transform("moneydisposalbe", ...)
```

Note the typo: `"moneydisposalbe"` instead of `"moneydisposable"`

**Impact**: Minor, but means normalizer stats are saved under misspelled key. Might cause confusion when inspecting.

### 8. No Explicit Loss Weighting

**Location**: `src/calloss.py:48-50`

All loss weights default to 1.0:
```python
self.weight_fb = getattr(config.training, 'weight_fb', 1.0)
self.weight_aux_mu = getattr(config.training, 'weight_aux_mu', 1.0)
self.weight_labor = getattr(config.training, 'weight_labor', 1.0)
```

**Problem**: Different losses might have very different scales. Equal weights might cause one loss to dominate.

**Check**: Look at wandb logs for `loss/fb`, `loss/aux_mu`, `loss/labor` - are they similar magnitude?

---

## 🔧 Recommended Fixes (Priority Order)

### Fix 1: Reduce Normalizer Momentum (CRITICAL)

**File**: `src/normalizer.py:23`

**Change**:
```python
def __init__(self, batch_dim: int = 0, agent_dim: int = 1, eps: float = 1e-4,
             min_std: float = 0.01, momentum: float = 0.95, clip_range: float = 10.0):  # Changed from 0.99 to 0.95
```

**Why**: Allows normalizer to adapt faster to changing wealth distribution during training.

**Better Yet**: Make it configurable in your config file:
```python
momentum: float = 0.95  # or whatever your config specifies
```

### Fix 2: Remove or Increase Count Cap

**File**: `src/normalizer.py:231`

**Option A (Remove cap)**:
```python
s.count = s.count + 1  # Remove clamp
```

**Option B (Increase cap)**:
```python
s.count = torch.clamp(s.count + 1, max=10000.0)  # Much higher cap
```

**Why**: Prevents normalizer from freezing in long training runs.

### Fix 3: Increase Training Steps

**File**: Your config (e.g., `config/baseline.yaml`)

**Change**:
```yaml
training:
  training_steps: 50000  # Increase from 100 to 50,000
  save_interval: 2500     # Save more frequently
```

**Why**: Gives both policy and normalizer time to converge.

### Fix 4: Add Normalizer Warmup (Advanced)

**File**: `src/train.py`

**Idea**: Run environment for N steps with random policy first, just to gather normalizer statistics, then start training.

```python
# Before training loop
print("Warming up normalizer...")
for _ in range(1000):  # Warmup steps
    with torch.no_grad():
        main_state, _, _, _ = env.step(
            main_state=main_state,
            policy_net=policy_net,
            update_normalizer=True,
            deterministic=False
        )
print("Normalizer warmup complete!")

# Now start actual training...
```

**Why**: Ensures normalizer sees diverse wealth states before policy training begins.

### Fix 5: Change Euler Loss to Sum of Squares

**File**: `src/calloss.py:217`

**Change**:
```python
# Old (product):
# return torch.mean(eulerloss_A * eulerloss_B)

# New (sum of squares):
return 0.5 * (torch.mean(eulerloss_A**2) + torch.mean(eulerloss_B**2))
```

**Why**: Penalizes errors in both branches independently, prevents cancellation.

### Fix 6: Increase Shock Volatility (Test)

**File**: Your config

**Change**:
```yaml
shock:
  sigma_v: 0.4  # Double from 0.2
```

**Why**: Creates stronger precautionary savings motive. Try this to see if savings become more responsive.

### Fix 7: Add Loss Balancing

**File**: Your config

**Add**:
```yaml
training:
  weight_fb: 1.0
  weight_aux_mu: 1.0
  weight_labor: 1.0
```

Then check wandb logs and adjust. If one loss is 100x larger, reduce its weight proportionally.

---

## 📊 Diagnostic Steps Before Fixing

Before making changes, gather evidence:

### Step 1: Check Which Checkpoint You Used

```bash
# What step did you visualize?
ls -lh checkpoints/*/weights/

# What config was used for training?
cat wandb/latest-run/files/config.yaml  # or check wandb UI
```

### Step 2: Inspect Normalizer Statistics

Create a simple script to check normalizer stats:

```python
import torch
from src.normalizer import RunningPerAgentWelford

norm = RunningPerAgentWelford.from_file('checkpoints/bewley_default_run/normalizer/norm_step_10000.pt')

print(f"Momentum: {norm.momentum}")
print(f"Clip range: ±{norm.clip_range}")

for name, stats in norm._stats.items():
    print(f"\n{name}:")
    print(f"  Count: {stats.count[0,0].item():.1f}")

    var = stats.M2 / torch.clamp(stats.count - 1, min=1.0)
    std = torch.sqrt(var + norm.eps)
    std = torch.clamp(std, min=norm.min_std)

    mean = stats.mean[0,0].item()
    std_val = std[0,0].item()

    print(f"  Mean: {mean:.4f}, Std: {std_val:.4f}")

    # Test visualization range
    for val in [0.1, 1.0, 5.0, 10.0]:
        normalized = (val - mean) / std_val
        print(f"  {val} → {normalized:.3f}" + (" CLIPPED!" if abs(normalized) > norm.clip_range else ""))
```

**What to look for**:
- Is count near 1000? (normalizer frozen)
- Are mean/std reasonable for your wealth distribution?
- Do test values [0.1, 10.0] get clipped? (network never saw these ranges)

### Step 3: Check Training Convergence

In wandb:
- Plot `loss/total` - still decreasing?
- Plot `actions/savings_ratio_mean` - stable or still changing?
- Plot `state/money_disposable_mean` - what's the typical wealth level?

### Step 4: Visualize Multiple Checkpoints

```bash
# Compare early vs late training
python vis_dc_rle.py --checkpoint_dir checkpoints/bewley_default_run --step 2500
python vis_dc_rle.py --checkpoint_dir checkpoints/bewley_default_run --step 10000

# Does behavior improve over training?
```

---

## 🎯 Recommended Action Plan

**Day 1: Diagnosis**
1. ✅ Check which checkpoint and config you used
2. ✅ Inspect normalizer statistics (run script above)
3. ✅ Review wandb loss curves
4. ✅ Visualize checkpoints at steps 2500, 5000, 7500, 10000

**Day 2: Quick Fixes**
1. Fix normalizer momentum → 0.95 (or 0.90)
2. Remove count cap
3. Increase training_steps → 50,000
4. Train new model with these changes

**Day 3: Monitor**
1. Watch wandb - is loss decreasing smoothly?
2. Visualize decision rules every 5000 steps
3. Check if savings policy becomes more responsive

**Day 4: Advanced Fixes (if still flat)**
1. Change Euler loss to sum of squares
2. Add normalizer warmup
3. Increase shock volatility
4. Tune loss weights

---

## 🔬 Expected Behavior After Fixes

A well-trained Bewley model should show:

### Savings Ratio vs Wealth:
- **Low wealth (0-2)**: High savings ratio (~15-20%) - precautionary motive strong
- **Medium wealth (2-5)**: Declining savings ratio - wealth effect dominates
- **High wealth (5+)**: Low, stable savings ratio (~5-10%) - self-insurance complete

### Consumption Policy:
- Should be **concave** (not linear!) - consumption smoothing
- Marginal propensity to consume should decrease with wealth

### Labor Supply:
- Slight **negative** wealth effect (rich work less)
- Strong ability effect (high productivity → work more)

### Your Current Plot Shows:
- ❌ Nearly linear consumption (no concavity)
- ❌ Flat savings ratio (barely changes with wealth)
- ✅ Labor has some wealth effect (drops at high wealth) - good sign!

This suggests the model has learned *something* (labor FOC) but not intertemporal optimization (savings).

---

## 📚 Additional Resources

### Understanding Normalizer Momentum

With momentum α:
- After 1 step: influence of initial value = α¹ = 0.99 (99%)
- After 100 steps: influence = α¹⁰⁰ = 0.99¹⁰⁰ = 37%
- After 500 steps: influence = α⁵⁰⁰ = 0.99⁵⁰⁰ = 0.7%

**Rule of thumb**: Need ~(log 0.01 / log α) steps to decay to 1%
- α=0.99: 459 steps
- α=0.95: 90 steps
- α=0.90: 44 steps

### Bewley Model Economics

Key insight: **Buffer stock savings**
- Poor agents save to build buffer against income shocks
- Rich agents dissave or save less (buffer already built)
- This creates heterogeneous savings rates

Your model isn't showing this → suggests it hasn't learned the precautionary motive.

---

## ❓ Questions to Answer

1. **Which checkpoint step did you visualize?**
   - If step < 5000, model might just be undertrained

2. **What does wandb show for `state/money_disposable_mean`?**
   - If mean wealth ≈ 2-3, then visualizing [0.1, 10.0] might be extrapolating far from training distribution

3. **Are losses still decreasing or have they plateaued?**
   - Decreasing → just train longer
   - Plateaued → model converged to suboptimal policy, need to fix loss/normalizer

4. **What are the actual normalizer statistics?**
   - Run the inspection script to see mean/std for `moneydisposalbe`

---

## 🚀 Next Steps

1. **Answer the questions above** (gather diagnostic info)
2. **Make the critical fixes** (momentum, count cap, training steps)
3. **Retrain** and monitor decision rules evolution
4. **Report back** with new visualizations and normalizer stats

Good luck! The fact that labor supply shows some reasonable behavior suggests your setup is mostly correct - likely just needs normalizer fixes and more training.
