# Normalization Stability Analysis

## Current Design

Your `RunningPerAgentWelford` normalizer:

```python
# Transform (line 47):
std_b = torch.sqrt(torch.clamp(var, min=0.0) + self.eps)  # eps=1e-6
y = (x_ba - mean_b) / std_b

# Update (line 110-112): Welford algorithm
new_mean = s.mean + delta * (m / (n + m))
new_M2 = s.M2 + M2_b + (delta ** 2) * (n * m / (n + m))
s.count = n + m  # ⚠️ Accumulates forever!
```

## 🔴 Critical Issues

### Issue 1: **Early Training Instability**

**Problem:**
- First step: `count = batch_size` (e.g., 256)
- Variance estimate is unstable with few samples
- Small variance → large std when sqrt → **extreme normalized values**

**Example:**
```python
Step 1:
  Raw money: mean=5.0, var=0.001 (low variance in initialization)
  std = sqrt(0.001 + 1e-6) ≈ 0.0316
  Normalized = (5.0 - 5.0) / 0.0316 = 0  # OK for mean

  But for outlier:
  Raw money = 6.0
  Normalized = (6.0 - 5.0) / 0.0316 ≈ 31.6  # ⚠️ HUGE!
```

**Impact:** Network receives values ~30-100x larger than expected → gradient explosion

---

### Issue 2: **Accumulation Without Decay** ⚠️ SEVERE

**Problem:**
- `count` grows forever: count[step] = count[step-1] + batch_size
- After 10,000 steps: count ≈ 10,000 × 256 = 2,560,000
- New batch weight: `m / (n+m)` ≈ 256 / 2,560,000 = **0.0001**
- Normalizer becomes "frozen" and **cannot adapt** to distribution shifts

**Example:**
```python
Step 1:
  Update weight = 256 / 256 = 1.0  ✓ Full update

Step 100:
  Update weight = 256 / 25,600 = 0.01  ⚠️ Only 1% influence

Step 10,000:
  Update weight = 256 / 2,560,000 = 0.0001  ❌ Effectively frozen!
```

**Impact:**
- Early bad estimates persist forever
- Cannot track distribution changes during training
- Policy evolves but normalizer doesn't → **distribution mismatch**

---

### Issue 3: **eps Too Small**

**Problem:**
- `eps = 1e-6` is very small
- If variance ≈ 1e-5, then std = sqrt(1e-5 + 1e-6) ≈ 0.00316
- Division by 0.003 amplifies values by 300x

**Comparison:**
```python
eps = 1e-6:  Dangerous, allows std as low as ~0.001
eps = 1e-5:  Safer, std >= ~0.003
eps = 1e-4:  Safe, std >= ~0.01
eps = 1e-3:  Very safe, std >= ~0.03
```

---

### Issue 4: **No Variance Floor**

**Problem:**
- Only `eps` is added to variance, no hard floor on std
- If all agents have similar values initially, var → 0
- Normalization amplifies tiny differences into huge signals

**Example:**
```python
# All agents initialized similarly
money = [5.01, 5.02, 4.99, 5.00, ...]
var ≈ 0.0001  # Very small!
std = sqrt(0.0001 + 1e-6) ≈ 0.01
normalized = (5.01 - 5.0) / 0.01 = 1.0  # Reasonable

# But with noise:
money = [5.01 + noise(0.0001)]
Small noise gets amplified 100x!
```

---

### Issue 5: **Feedback Loop Risk** 🔥 CRITICAL

**The Vicious Cycle:**

```
1. Bad normalization (early steps)
   ↓
2. Network receives extreme values (±30 to ±100)
   ↓
3. Gradients explode
   ↓
4. Policy parameters diverge
   ↓
5. Agent actions become extreme (all save 0% or 100%)
   ↓
6. State distribution shifts dramatically
   ↓
7. Normalizer (now frozen) can't adapt
   ↓
8. Even worse normalization
   ↓
9. TRAINING DIVERGES 💥
```

---

## 📊 Evidence to Check

Run your training and watch for:

1. **Early explosion** (steps 1-100):
   ```
   debug/normalized_money_mean: 0.1  ✓ OK
   debug/normalized_money_std: 15.7  ❌ EXPLODING!
   ```

2. **Frozen statistics** (steps 1000+):
   ```
   Step 1000: raw_money_mean = 5.0, normalized_money_mean = 0.1
   Step 2000: raw_money_mean = 8.0, normalized_money_mean = 0.1  ❌ Didn't change!
   ```

3. **Loss explosion**:
   ```
   Step 1: loss = 2.4
   Step 10: loss = 156.8  ❌ DIVERGING
   ```

4. **Extreme actions**:
   ```
   savings_ratio: all 0.999 or all 0.001  ❌ No diversity
   ```

---

## ✅ Recommended Fixes

### Fix 1: **Add Exponential Moving Average (EMA)** 🏆 BEST FIX

Replace infinite accumulation with decay:

```python
class RunningPerAgentWelford:
    def __init__(self, batch_dim=0, agent_dim=1, eps=1e-4, momentum=0.99):
        self.eps = eps
        self.momentum = momentum  # ← NEW: decay factor
        # ...

    def _update_stats(self, s, x):
        m, mean_b, M2_b = self._reduce_over_batch(x)
        mean_b = mean_b.detach()
        M2_b = M2_b.detach()

        # OLD (accumulation):
        # n = s.count
        # tot = n + m
        # new_mean = s.mean + delta * (m / tot)

        # NEW (EMA):
        if s.count.sum() == 0:  # First update
            s.mean = mean_b
            s.M2 = M2_b
            s.count = torch.ones_like(s.count) * m
        else:
            # Exponential moving average
            alpha = 1.0 - self.momentum  # e.g., 0.01 for momentum=0.99
            s.mean = self.momentum * s.mean + alpha * mean_b
            s.M2 = self.momentum * s.M2 + alpha * M2_b
            # Keep count for variance estimation but don't grow forever
            s.count = torch.clamp(s.count + 1, max=1000)  # Cap at 1000
```

**Benefits:**
- ✅ Adapts throughout training
- ✅ No freezing after many steps
- ✅ Stable weight for each update (alpha ≈ 0.01)

---

### Fix 2: **Increase eps and Add Variance Floor**

```python
class RunningPerAgentWelford:
    def __init__(self, ..., eps=1e-4, min_std=0.01):  # ← Increase eps
        self.eps = eps
        self.min_std = min_std  # ← NEW: hard floor on std

    def transform(self, name, x, update=True):
        # ...
        std_b = torch.sqrt(torch.clamp(var, min=0.0) + self.eps)
        std_b = torch.clamp(std_b, min=self.min_std)  # ← NEW: enforce floor
        y = (x_ba - mean_b) / std_b
        # ...
```

**Benefits:**
- ✅ Prevents division by tiny numbers
- ✅ Caps normalized values at reasonable range
- ✅ More stable early training

---

### Fix 3: **Add Warmup Period**

Don't update policy in first N steps, only collect statistics:

```python
# In train.py:
WARMUP_STEPS = 100

for step in range(1, total_steps + 1):
    main_state, temp_state, ... = env.step(...)

    if step <= WARMUP_STEPS:
        # Only update normalizer, don't train
        continue

    # Normal training from here
    losses = loss_calc.compute_all_losses(...)
    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()
```

**Benefits:**
- ✅ Normalizer has good statistics before training starts
- ✅ Network doesn't see extreme values early on
- ✅ More stable initialization

---

### Fix 4: **Clip Normalized Values**

Add safety clipping after normalization:

```python
def transform(self, name, x, update=True):
    # ... normalization ...
    y = (x_ba - mean_b) / std_b

    # ← NEW: Clip to reasonable range
    y = torch.clamp(y, min=-10.0, max=10.0)

    y = self._move_from_ba(y, x)
    return y
```

**Benefits:**
- ✅ Hard guarantee on normalized value range
- ✅ Prevents extreme inputs to network
- ✅ Simple safety net

---

## 🎯 Recommended Strategy

**Implement ALL of these (in order):**

1. **Short term** (immediate):
   - ✅ Increase `eps` from 1e-6 to **1e-4**
   - ✅ Add `min_std=0.01` floor
   - ✅ Add output clipping `(-10, 10)`

2. **Medium term** (before full training):
   - ✅ Add **warmup period** (100-1000 steps)
   - ✅ Implement **EMA** instead of infinite accumulation

3. **Long term** (for production):
   - ✅ Add normalization monitoring
   - ✅ Save/load normalizer checkpoints
   - ✅ Add adaptive eps based on observed variance

---

## 📈 Expected Improvement

**Before fixes:**
```
Step 1: normalized_std = 25.6  ❌ EXPLODING
Step 10: loss = 156.8          ❌ DIVERGING
Step 100: normalized_std = 0.1 ❌ COLLAPSING
```

**After fixes:**
```
Step 1: normalized_std = 1.2   ✓ Reasonable (warmup)
Step 10: normalized_std = 1.0  ✓ Converging to target
Step 100: normalized_std = 1.0 ✓ Stable
Step 10000: normalized_std = 1.0 ✓ Still adapting (EMA)
```

---

## 🔬 Testing the Fixes

After implementing fixes, check:

1. **Normalized stats stay bounded:**
   ```python
   assert -3 < normalized_money_mean < 3
   assert 0.5 < normalized_money_std < 2.0
   ```

2. **Loss doesn't explode:**
   ```python
   assert loss < 100  # Should be < 10 ideally
   ```

3. **Statistics adapt:**
   ```python
   # Change distribution intentionally
   main_state.moneydisposable *= 2
   # After a few steps, normalized mean should still be ~0
   ```

---

## Summary

**Your current normalizer CAN cause training to diverge because:**
1. ❌ Early instability (small variance → huge normalized values)
2. ❌ Late freezing (can't adapt after 1000+ steps)
3. ❌ No safety bounds (can produce arbitrarily large values)

**Priority fixes:**
1. 🔥 **Increase eps** to 1e-4 (immediate)
2. 🔥 **Add min_std floor** of 0.01 (immediate)
3. 🔥 **Add output clipping** to (-10, 10) (immediate)
4. ⭐ **Implement EMA** with momentum=0.99 (important)
5. ⭐ **Add warmup period** of 100 steps (important)

This is a **real risk** and likely contributing to any training instability you're seeing!
