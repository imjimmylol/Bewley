# Device Consistency Fixes

## Problem
Tensors created on different devices (CPU vs CUDA/MPS) cause runtime errors like:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!
```

## Issues Found and Fixed

### ⚠️ CRITICAL Issues

#### 1. **Policy Network Not Moved to Device** (`src/train.py:118`)
**Issue**: Policy network created on CPU but environment/state on target device.

**Before:**
```python
policy_net = FiLMResNet2In(
    state_dim=2*config.training.agents+2,
    cond_dim=5,
    output_dim=3
)
```

**After:**
```python
policy_net = FiLMResNet2In(
    state_dim=2*config.training.agents+2,
    cond_dim=5,
    output_dim=3
).to(device)  # ✅ FIXED
```

**Impact**: Every forward pass would fail with device mismatch error.

---

#### 2. **Initial State Created on CPU** (`src/train.py:41, 61-67`)
**Issue**: All initial state tensors created without device specification, defaulting to CPU.

**Before:**
```python
tax_params = torch.tensor(list(tax_params_values.values()), dtype=torch.float32)
# ...
moneydisposable = torch.tensor(moneydisposable, dtype=torch.float32)
savings = torch.tensor(savings, dtype=torch.float32)
ability = torch.tensor(ability, dtype=torch.float32)
is_superstar_vA = torch.tensor(is_superstar_vA, dtype=torch.bool)
is_superstar_vB = torch.tensor(is_superstar_vB, dtype=torch.bool)
```

**After:**
```python
tax_params = torch.tensor(list(tax_params_values.values()), dtype=torch.float32, device=device)  # ✅
# ...
moneydisposable = torch.tensor(moneydisposable, dtype=torch.float32, device=device)  # ✅
savings = torch.tensor(savings, dtype=torch.float32, device=device)  # ✅
ability = torch.tensor(ability, dtype=torch.float32, device=device)  # ✅
is_superstar_vA = torch.tensor(is_superstar_vA, dtype=torch.bool, device=device)  # ✅
is_superstar_vB = torch.tensor(is_superstar_vB, dtype=torch.bool, device=device)  # ✅
```

**Impact**: First environment step would fail with device mismatch.

---

### ⚠️ MINOR Issues

#### 3. **Random Branch Selection on CPU** (`src/environment.py:468`)
**Issue**: Random number generated on CPU for branch selection.

**Before:**
```python
return "A" if torch.rand(1).item() < 0.5 else "B"
```

**After:**
```python
return "A" if torch.rand(1, device=self.device).item() < 0.5 else "B"  # ✅ FIXED
```

**Impact**: Minor - `.item()` converts to Python scalar immediately, but generates warning about cross-device operation.

---

#### 4. **Test File Policy Network** (`test_environment.py:274`)
**Issue**: Test file also had network on CPU.

**Before:**
```python
net = FiLMResNet2In(
    state_dim=2*config.training.agents+2,
    cond_dim=5,
    output_dim=3
)
```

**After:**
```python
net = FiLMResNet2In(
    state_dim=2*config.training.agents+2,
    cond_dim=5,
    output_dim=3
).to(device)  # ✅ FIXED
```

---

## ✅ Already Correct

### 1. **build_inputs utility** (`src/utils/buildipnuts.py:10-12`)
Uses `torch.as_tensor(..., device=device)` which automatically handles device placement.

### 2. **Normalizer statistics** (`src/normalizer.py:91-93`)
Creates tensors on correct device by extracting from input:
```python
device, dtype = x.device, x.dtype
count = torch.zeros((A, *feat_shape), device=device, dtype=torch.float64)
```

### 3. **Shock transitions** (`src/shocks.py:50, 112, 120`)
Uses device from input tensors:
```python
device = ability_t.device
rand = torch.rand(B, A, device=device)
log_v_bar = torch.log(torch.tensor(v_bar, device=device))
eps = torch.randn(B, A, device=device) * sigma_v
```

### 4. **Market equilibrium** (`src/environment.py:154-165`)
Only performs operations on existing tensors, no new tensor creation.

### 5. **Income/tax computation** (`src/environment.py:192-197`)
Only performs operations on existing tensors, no new tensor creation.

---

## Verification

### Quick Test
Run on GPU to verify all tensors are on the same device:
```bash
python test_training_demo.py
```

If using CUDA, change line 35 to:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Expected Behavior

**Before fixes:**
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!
```

**After fixes:**
```
✓ Demo completed successfully!
  All tensors on: cuda:0
```

### Device Check Code
Add this to verify all tensors are on the same device:
```python
def check_device_consistency(main_state, policy_net, device):
    """Verify all tensors are on the same device."""
    errors = []

    # Check state tensors
    if main_state.moneydisposable.device != device:
        errors.append(f"moneydisposable on {main_state.moneydisposable.device}")
    if main_state.savings.device != device:
        errors.append(f"savings on {main_state.savings.device}")
    if main_state.ability.device != device:
        errors.append(f"ability on {main_state.ability.device}")
    if main_state.tax_params.device != device:
        errors.append(f"tax_params on {main_state.tax_params.device}")

    # Check model parameters
    for name, param in policy_net.named_parameters():
        if param.device != device:
            errors.append(f"Parameter {name} on {param.device}")

    if errors:
        print("❌ Device mismatch found:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print(f"✅ All tensors on {device}")
        return True
```

---

## Summary of Changes

| File | Line | Change | Priority |
|------|------|--------|----------|
| `src/train.py` | 41 | Added `device=device` to tax_params | 🔴 Critical |
| `src/train.py` | 61-67 | Added `device=device` to all state tensors | 🔴 Critical |
| `src/train.py` | 118 | Added `.to(device)` to policy_net | 🔴 Critical |
| `src/environment.py` | 468 | Added `device=self.device` to torch.rand | 🟡 Minor |
| `test_environment.py` | 274 | Added `.to(device)` to test network | 🟡 Minor |

---

## Best Practices

1. **Always specify device for tensor creation**:
   ```python
   # ❌ BAD
   x = torch.zeros(10)

   # ✅ GOOD
   x = torch.zeros(10, device=device)
   ```

2. **Move models to device immediately after creation**:
   ```python
   # ❌ BAD
   model = MyModel()
   # ... later ...
   model.to(device)

   # ✅ GOOD
   model = MyModel().to(device)
   ```

3. **Extract device from existing tensors**:
   ```python
   # ✅ GOOD - derive device from input
   device = input_tensor.device
   new_tensor = torch.randn(10, device=device)
   ```

4. **Use torch.as_tensor for flexibility**:
   ```python
   # ✅ GOOD - automatically moves to target device
   x = torch.as_tensor(numpy_array, device=device)
   ```

5. **Verify device in debugging**:
   ```python
   print(f"tensor device: {tensor.device}")
   assert tensor.device == expected_device, f"Device mismatch!"
   ```

---

## If Device Errors Persist

If you still see device errors after these fixes:

1. **Check intermediate tensors**:
   ```python
   # Add checks after each major operation
   assert wage.device == device, f"wage on wrong device: {wage.device}"
   assert consumption.device == device, f"consumption on wrong device: {consumption.device}"
   ```

2. **Enable anomaly detection**:
   ```python
   torch.autograd.set_detect_anomaly(True)
   ```

3. **Print device info**:
   ```python
   def print_tensor_devices(obj, prefix=""):
       if isinstance(obj, torch.Tensor):
           print(f"{prefix}: {obj.device}")
       elif hasattr(obj, '__dict__'):
           for key, value in obj.__dict__.items():
               print_tensor_devices(value, f"{prefix}.{key}")

   print_tensor_devices(main_state, "main_state")
   ```
