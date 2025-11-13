# Code Instructions and Documentation

This folder contains documentation files created during the Claude Code session to address various implementation issues and provide guidance.

## 📚 Documentation Files

### Bug Fixes & Issues

| File | Description | Created |
|------|-------------|---------|
| `MEMORY_LEAK_FIXES.md` | Identified and fixed 4 critical memory leaks causing OOM crashes | Session 1 |
| `DEVICE_FIXES.md` | Fixed device consistency issues (CPU/CUDA/MPS mismatches) | Session 1 |
| `MPS_COMPATIBILITY_FIX.md` | Fixed float64 incompatibility with Apple Silicon MPS backend | Session 1 |
| `LABOR_COLLAPSE_FIX.md` | **Critical**: Fixed sign error in labor FOC loss causing labor collapse | Session 2 |

### Feature Documentation

| File | Description |
|------|-------------|
| `LOSS_DESIGN.md` | Complete loss calculation module design and implementation |
| `LOGGING_GUIDE.md` | Training metrics and normalized input logging guide |
| `BATCH_TRAINING_GUIDE.md` | Guide for running all config files with shell scripts |
| `TAX_RETURN_DEBUGGING.md` | **NEW**: Systematic testing for tax distortion & return clipping issues |
| `CONSUMPTION_SAVINGS_ISSUE_CHECKLIST.md` | Diagnostic checklist for consumption/savings imbalances |

### Stability Improvements

| File | Description |
|------|-------------|
| `NORMALIZATION_ISSUES.md` | Analysis of 5 critical normalizer stability issues |
| `NORMALIZER_FIXES_APPLIED.md` | Implementation details of normalizer stability fixes |
| `STABILITY_FIXES_SUMMARY.md` | Quick reference summary of all stability fixes |

---

## 🔴 Critical Issues Fixed

### 1. Labor Collapse (Most Important!)
- **File**: `LABOR_COLLAPSE_FIX.md`
- **Issue**: Labor went to 0, causing economic model to collapse
- **Root Cause**: Sign error in labor FOC loss (negative cons_term)
- **Fix**: Removed extra negative sign in `src/calloss.py:261`
- **Impact**: Training now stable, labor stays in healthy range

### 2. Memory Leaks
- **File**: `MEMORY_LEAK_FIXES.md`
- **Issue**: Memory usage exploded after 6 minutes
- **Root Causes**: 4 sources of gradient accumulation
- **Fix**: Added `.detach()` calls in shocks, normalizer, and history
- **Impact**: Memory usage now constant

### 3. Normalizer Instability
- **Files**: `NORMALIZATION_ISSUES.md`, `NORMALIZER_FIXES_APPLIED.md`
- **Issue**: Normalizer could freeze or cause gradient explosion
- **Root Causes**:
  - eps too small (1e-6)
  - Infinite count accumulation
  - No variance floor or output clipping
- **Fixes**:
  - Increased eps to 1e-4
  - Added min_std=0.01 floor
  - Added output clipping [-10, +10]
  - Implemented EMA with momentum=0.99
- **Impact**: Stable normalization throughout training

---

## 📖 Reading Order (Recommended)

If you're new to this codebase or troubleshooting issues, read in this order:

1. **Start Here**: `LABOR_COLLAPSE_FIX.md` - Understand the main bug that was fixed
2. **Logging**: `LOGGING_GUIDE.md` - Learn what metrics to monitor
3. **Stability**: `STABILITY_FIXES_SUMMARY.md` - Quick overview of all fixes
4. **Deep Dive**: `NORMALIZATION_ISSUES.md` - Detailed analysis of normalizer issues
5. **Batch Training**: `BATCH_TRAINING_GUIDE.md` - How to run experiments

### For Debugging:

- Memory issues? → `MEMORY_LEAK_FIXES.md`
- Device errors? → `DEVICE_FIXES.md`
- MPS errors? → `MPS_COMPATIBILITY_FIX.md`
- Training diverges? → `LABOR_COLLAPSE_FIX.md` + `NORMALIZATION_ISSUES.md`
- Under-saving (consumption >> savings)? → `TAX_RETURN_DEBUGGING.md`
- Over-saving (savings >> consumption)? → `CONSUMPTION_SAVINGS_ISSUE_CHECKLIST.md`

---

## 🛠️ Implementation Status

| Component | Status | Files Modified |
|-----------|--------|----------------|
| Memory leaks | ✅ Fixed | `src/shocks.py`, `src/normalizer.py`, `src/train.py` |
| Device consistency | ✅ Fixed | `src/train.py`, `src/environment.py`, `test_environment.py` |
| MPS compatibility | ✅ Fixed | `src/normalizer.py` |
| Labor FOC sign error | ✅ Fixed | `src/calloss.py` |
| Normalizer stability | ✅ Fixed | `src/normalizer.py` |
| Market equilibrium bounds | ✅ Fixed | `src/environment.py` |
| Policy output bounds | ✅ Fixed | `src/environment.py` |
| Budget constraint logging | ✅ Added | `src/train.py` |

---

## 🔍 Key Metrics to Monitor

Based on the documentation in this folder, watch these during training:

### Critical Indicators (should be stable):
```
state/labor_mean: [0.3, 0.7]  ✓
market/wage: [0.5, 3.0]       ✓
market/return: [0.03, 0.15]   ✓
state/budget_ratio: ~1.0      ✓
```

### Normalization (should be near standard normal):
```
debug/normalized_money_mean: ~0.0      ✓
debug/normalized_money_std: ~1.0       ✓
debug/normalized_ability_mean: ~0.0    ✓
debug/normalized_ability_std: ~1.0     ✓
```

### Loss (should decrease smoothly):
```
loss/total: decreasing        ✓
loss/fb: → 0                  ✓
loss/aux_mu: → 0              ✓
loss/labor_foc: → 0           ✓
```

---

## 📝 Notes for Future Development

### If Training Diverges:

1. Check `state/labor_mean` - if it goes below 0.01, labor bounds may be failing
2. Check `market/return` - if it exceeds 0.5, market equilibrium clipping may be failing
3. Check `debug/normalized_*_std` - if > 5.0, normalizer may be unstable
4. Check `state/budget_ratio` - if not ~1.0, budget constraint is violated

### If Memory Usage Grows:

1. Verify all `.detach()` calls are in place (see `MEMORY_LEAK_FIXES.md`)
2. Check that temporary variables are deleted in training loop
3. Ensure normalizer stats are detached in `_update_stats()`

### If Labor Collapses Again:

1. Verify labor FOC sign is correct: `cons_term = c^(-θ)` (NO negative!)
2. Check policy output bounds: labor ∈ [0.01, 0.99]
3. Check market equilibrium clipping: wage ∈ [0.1, 10.0]

---

## 🎯 Summary

All critical issues identified during the Claude Code session have been fixed:
- ✅ Memory leaks resolved
- ✅ Device consistency ensured
- ✅ Normalizer stability improved
- ✅ Labor collapse bug fixed
- ✅ Market equilibrium protected with bounds
- ✅ Comprehensive logging added

**Training should now be stable and reproducible!**

For questions or issues, refer to the specific documentation file for each component.
