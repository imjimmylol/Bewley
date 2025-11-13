# Memory Leak Fixes

## Problem
After ~6 minutes of training, memory usage increases dramatically due to gradient accumulation.

## Root Causes

### 1. **Ability Transitions Not Detached** (`src/shocks.py:136-139`)
**Issue**: Ability shocks are exogenous random variables that should NOT have gradients, but `transition_ability()` was returning tensors with computational graphs attached.

**Impact**: Every ability transition kept the entire computational graph from previous steps alive.

**Fix**:
```python
# After computing ability_tp1 and is_superstar_tp1
ability_tp1 = ability_tp1.detach()
is_superstar_tp1 = is_superstar_tp1.detach()
```

---

### 2. **Ability History Accumulating Gradients** (`src/shocks.py:163-177`)
**Issue**: The `update_ability_history()` function concatenates ability tensors across time without detaching, creating a computational graph that links back to ALL previous steps.

**Impact**: After thousands of steps, the history buffer contains a massive computational graph consuming gigabytes of memory.

**Fix**:
```python
# Detach new ability before adding to history
ability_new_detached = ability_new.detach()

# Detach history when concatenating
history_extended = torch.cat([ability_history.detach(), ability_new_expanded], dim=0)
```

---

### 3. **Normalizer Statistics Accumulating Gradients** (`src/normalizer.py:100-103`)
**Issue**: The normalizer's running statistics (mean, M2) were being updated with tensors that had gradients attached.

**Impact**: Normalizer statistics accumulate gradients from every single input feature across all time steps.

**Fix**:
```python
m, mean_b, M2_b = self._reduce_over_batch(x)
mean_b = mean_b.detach()  # Statistics should not have gradients
M2_b = M2_b.detach()
```

---

### 4. **Temporary Variables Holding References** (`src/train.py:205-210`)
**Issue**: Temporary state variables (temp_state, parallel states, outcomes) held references to computational graphs even after the backward pass.

**Impact**: Python's garbage collector couldn't free these tensors until the next iteration, causing memory to accumulate.

**Fix**:
```python
# Extract scalar values first
loss_total_val = total_loss.item()
consumption_mean_val = consumption_t.mean().item()
# ... etc

# Then delete tensor references
del temp_state, parallel_A, parallel_B, outcomes_A, outcomes_B
del consumption_t, labor_t, savings_ratio_t, mu_t, wage_t, ret_t, money_disposable_t
del consumption_A_tp1, consumption_B_tp1
del fb_loss, euler_loss, labor_foc_loss, total_loss

# Use scalar values for logging
print(f"Loss: {loss_total_val:.4f}")
```

---

## Why These Fixes Work

### Gradient Flow Principle
In your training loop, gradients should ONLY flow through:
1. **Policy network outputs** (savings_ratio, labor, mu)
2. **Computed economic variables** (consumption, market prices, taxes)

Gradients should NOT flow through:
1. **Exogenous shocks** (ability transitions, superstar status)
2. **Historical records** (ability_history)
3. **Normalizer statistics** (running mean/variance)
4. **Previous time steps** (detached at commit)

### Memory Accumulation Pattern
Without detaching:
```
Step 1: ability[1] → (gradient graph of size N)
Step 2: ability[2] → links to ability[1] → (graph size 2N)
Step 3: ability[3] → links to ability[2] → links to ability[1] → (graph size 3N)
...
Step 1000: ability[1000] → ... → ability[1] → (graph size 1000N) ⚠️ OOM!
```

With detaching:
```
Step 1: ability[1] (detached) → (no graph)
Step 2: ability[2] (detached) → (no graph)
Step 3: ability[3] (detached) → (no graph)
...
Step 1000: ability[1000] (detached) → (no graph) ✅ Memory stable!
```

---

## Testing the Fixes

### Before Fixes
- Memory grows linearly with training steps
- After 6 minutes (~thousands of steps), memory usage explodes
- Training eventually crashes with OOM

### After Fixes
- Memory usage stabilizes after warmup period
- Can train for hours without memory growth
- Only policy network gradients are tracked

### Verification Commands
```bash
# Monitor memory during training
python test_training_demo.py

# Check for gradient leaks (optional)
import torch
torch.cuda.memory_summary()  # If using CUDA
```

---

## Summary of Changes

| File | Lines | Change |
|------|-------|--------|
| `src/shocks.py` | 136-139 | Added `.detach()` to ability transitions |
| `src/shocks.py` | 163-177 | Added `.detach()` to history updates |
| `src/normalizer.py` | 100-103 | Added `.detach()` to normalizer statistics |
| `src/train.py` | 190-210 | Added explicit tensor deletion and scalar extraction |

---

## Best Practices Going Forward

1. **Always detach exogenous variables**: Random shocks, external data
2. **Always detach historical records**: Past states should not propagate gradients
3. **Always detach normalizer statistics**: Running stats are descriptive, not learnable
4. **Extract scalars before logging**: Use `.item()` to avoid holding tensor references
5. **Explicitly delete large temporary variables**: Help Python's GC free memory faster

---

## If Memory Issues Persist

If you still see memory growth after these fixes:

1. **Check for other history buffers**:
   ```python
   # Look for lists/tensors accumulating over time
   grep -r "\.append\|\.extend" src/
   ```

2. **Profile memory usage**:
   ```python
   import tracemalloc
   tracemalloc.start()
   # ... training code ...
   snapshot = tracemalloc.take_snapshot()
   top_stats = snapshot.statistics('lineno')
   for stat in top_stats[:10]:
       print(stat)
   ```

3. **Monitor computational graph size**:
   ```python
   def count_graph_nodes(tensor):
       if not tensor.requires_grad or tensor.grad_fn is None:
           return 0
       count = 1
       for next_fn, _ in tensor.grad_fn.next_functions:
           if next_fn is not None:
               count += count_graph_nodes(next_fn)
       return count

   print(f"Graph size: {count_graph_nodes(some_tensor)}")
   ```

4. **Enable gradient checkpointing** (for very large models):
   ```python
   from torch.utils.checkpoint import checkpoint
   # Wrap expensive forward passes
   ```
