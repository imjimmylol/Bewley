# MPS (Apple Silicon) Compatibility Fix

## Problem

When running on Apple Silicon (M1/M2/M3 Macs) with MPS backend:
```
TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.
```

## Root Cause

**MPS (Metal Performance Shaders) backend does NOT support float64 (double precision)**.

The normalizer was creating count tensors with `dtype=torch.float64`, which is incompatible with MPS devices.

## Error Trace

```python
File "src/normalizer.py", line 91, in _ensure_stats
    count = torch.zeros((A, *feat_shape), device=device, dtype=torch.float64)
TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64.
```

## Fixes Applied

### Fix 1: Normalizer Count Tensor (`src/normalizer.py:91-92`)

**Before:**
```python
count = torch.zeros((A, *feat_shape), device=device, dtype=torch.float64)
```

**After:**
```python
# Use float32 instead of float64 for MPS compatibility
count = torch.zeros((A, *feat_shape), device=device, dtype=torch.float32)
```

### Fix 2: Batch Size Tensor (`src/normalizer.py:143-144`)

**Before:**
```python
m = torch.tensor(float(B), device=x.device, dtype=torch.float64)
```

**After:**
```python
# Use float32 instead of float64 for MPS compatibility
m = torch.tensor(float(B), device=x.device, dtype=torch.float32)
```

## Impact

### Precision
- **Count values**: Changed from float64 to float32
- **Impact**: Negligible - count values are integers stored as floats
- **Maximum count**: float32 can represent integers up to 16,777,216 exactly (2^24)
- **Batch size**: Typical values (256-1024) are well within float32 range

### Numerical Stability
For the Welford algorithm used in the normalizer:
- **Mean**: Still computed in float32 (no change)
- **M2 (variance accumulator)**: Still computed in float32 (no change)
- **Count**: Now float32 instead of float64 (minimal impact)

The loss of precision from float64 → float32 for counts is insignificant for typical training scenarios.

## Device Compatibility Matrix

| Device | float32 | float64 | Status |
|--------|---------|---------|--------|
| CPU | ✅ | ✅ | Fully supported |
| CUDA | ✅ | ✅ | Fully supported |
| MPS (Apple Silicon) | ✅ | ❌ | **float64 NOT supported** |

## Verification

### Test on MPS
```python
device = torch.device("mps")

# Should work now
count = torch.zeros(10, device=device, dtype=torch.float32)  # ✅
m = torch.tensor(256.0, device=device, dtype=torch.float32)  # ✅

# Would fail on MPS
# count = torch.zeros(10, device=device, dtype=torch.float64)  # ❌
```

### Run Training
```bash
python main.py  # Will automatically use MPS if available
```

Should now see:
```
Using device: mps
✓ EconomyEnv initialized
  - Device: mps
  - Batch size: 256
  - Number of agents: 87
Training:   0%|█ ...
```

## Alternative: Force CPU on MPS Systems

If you prefer to use float64 precision and avoid MPS, modify `src/train.py:106-110`:

**Before:**
```python
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
```

**After:**
```python
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")  # Skip MPS, use CPU for float64 support
)
```

However, **this is NOT recommended** as:
1. MPS is much faster than CPU on Apple Silicon
2. float32 is sufficient for this application
3. Modern ML training uses float32 or even float16

## Best Practices for Cross-Platform Compatibility

### 1. Default to float32
```python
# ✅ Good - works everywhere
tensor = torch.zeros(10, device=device, dtype=torch.float32)

# ❌ Bad - breaks on MPS
tensor = torch.zeros(10, device=device, dtype=torch.float64)
```

### 2. Let PyTorch infer dtype from input
```python
# ✅ Good - inherits dtype from input_tensor
device, dtype = input_tensor.device, input_tensor.dtype
new_tensor = torch.zeros(10, device=device, dtype=dtype)
```

### 3. Use automatic mixed precision (AMP)
For CUDA, you can use AMP for even faster training:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Note: MPS doesn't support AMP yet (as of PyTorch 2.x).

### 4. Check device capabilities
```python
if device.type == 'mps':
    # MPS limitations
    # - No float64 support
    # - No automatic mixed precision
    # - Some operations not yet implemented
    pass
elif device.type == 'cuda':
    # CUDA fully featured
    pass
```

## Related Changes

This fix is part of the comprehensive device consistency updates:
- See `DEVICE_FIXES.md` for full device placement fixes
- See `MEMORY_LEAK_FIXES.md` for gradient accumulation fixes

## References

- [PyTorch MPS Backend Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [MPS Supported Operations](https://github.com/pytorch/pytorch/issues/77764)
- [Apple Silicon ML Performance](https://developer.apple.com/metal/pytorch/)
