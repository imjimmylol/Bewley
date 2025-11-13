# Consumption/Savings Ratio Issue - Investigation Checklist

**Problem:** Savings ratio = 99%, Consumption ratio = 1%

**Date Created:** 2025-01-13

---

## 📊 Step 1: Collect Diagnostic Metrics

Run training and fill in these values from the logs:

### At Step 1 (Initial State)
```
[ ] market/return: ___________
[ ] market/wage: ___________
[ ] state/consumption_mean: ___________
[ ] state/savings_mean: ___________
[ ] state/labor_mean: ___________
[ ] state/money_disposable_mean: ___________
[ ] state/budget_ratio: ___________
[ ] loss/total: ___________
[ ] loss/fb: ___________
[ ] loss/aux_mu: ___________
[ ] loss/labor_foc: ___________
```

### At Step 100 (After Learning)
```
[ ] market/return: ___________
[ ] market/wage: ___________
[ ] state/consumption_mean: ___________
[ ] state/savings_mean: ___________
[ ] state/labor_mean: ___________
[ ] state/money_disposable_mean: ___________
[ ] state/budget_ratio: ___________
[ ] loss/total: ___________
[ ] loss/fb: ___________
[ ] loss/aux_mu: ___________
[ ] loss/labor_foc: ___________
```

### At Step 500 (Final)
```
[ ] market/return: ___________
[ ] market/wage: ___________
[ ] state/consumption_mean: ___________
[ ] state/savings_mean: ___________
[ ] state/labor_mean: ___________
[ ] state/money_disposable_mean: ___________
[ ] state/budget_ratio: ___________
[ ] loss/total: ___________
[ ] loss/fb: ___________
[ ] loss/aux_mu: ___________
[ ] loss/labor_foc: ___________
```

---

## 🔍 Step 2: Verify Code Version

Confirm all previous fixes are applied:

### Memory & Device Fixes
- [ ] Memory leak fixes applied (src/shocks.py has `.detach()` calls)
- [ ] Device consistency fixes applied (all tensors on same device)
- [ ] MPS compatibility fix applied (normalizer uses float32)

### Labor Collapse Fix (CRITICAL)
- [ ] Labor FOC sign fix applied (src/calloss.py line 261)
  - [ ] Check: `cons_term = self._safe_pow(consumption, -self.theta) / (1.0 + self.taxparams.tax_saving)`
  - [ ] Should NOT have negative sign before `self._safe_pow`
  - [ ] If it has negative, this is the bug - remove it!

### Policy Bounds
- [ ] Min labor bounds applied (src/environment.py line 119)
  - [ ] Check: `labor_t0 = labor_t0 * 0.98 + 0.01`
- [ ] Min savings bounds applied (src/environment.py line 121)
  - [ ] Check: `savings_t1 = savings_t1 * 0.98 + 0.01`

### Market Equilibrium Bounds
- [ ] Market equilibrium clipping applied (src/environment.py lines 165-180)
  - [ ] Check: `labor_eff_agg = torch.clamp(labor_eff_agg, min=0.01)`
  - [ ] Check: `ret = torch.clamp(ret, min=0.0, max=0.5)`
  - [ ] Check: `wage = torch.clamp(wage, min=0.1, max=10.0)`

### Budget Constraint Logging
- [ ] Budget verification added (src/train.py lines 239-244)
  - [ ] Check for: `budget_diff_val = money_disposable_t_mean_val - sav_plus_cons_val`

---

## 🚨 Step 3: Investigate Issue #1 - Return Clipping Too High

**Location:** `src/environment.py` line 178

### Symptoms to Check:
- [ ] Is `market/return` close to 0.5 in logs?
- [ ] Is `market/return` much higher than target `config.bewley_model.r = 0.04`?
- [ ] Did savings ratio increase as training progressed (1 → 100 → 500)?

### Economic Analysis:
```
Config values:
- beta (β): 0.975
- target r: 0.04 (4%)

Current clip: max=0.5 (50%)
→ β(1+r) = 0.975 * 1.5 = 1.4625 >> 1  ❌

For steady state, need: β(1+r) ≈ 1
→ r ≈ 1/β - 1 = 1/0.975 - 1 = 0.026 (2.6%)
```

- [ ] Calculate β(1+r) with actual return value: `___________`
- [ ] Is β(1+r) > 1? (If yes, this explains 99% savings)

### Root Cause:
- [ ] If `market/return` is high (>0.1), return clip is the problem
- [ ] High return → β(1+r) > 1 → Future consumption more valuable
- [ ] Policy learns to save everything now, consume later

### Proposed Fix:
**Option A: Lower Return Clip to 0.10**
```python
# Line 178
ret = torch.clamp(ret, min=0.0, max=0.10)  # Max 10% return
```

**Option B: Lower Return Clip to 0.05**
```python
# Line 178
ret = torch.clamp(ret, min=0.0, max=0.05)  # Max 5% return
```

**Option C: Match Target Exactly**
```python
# Line 178
ret = torch.clamp(ret, min=0.0, max=0.08)  # Max 8% return (2x target for headroom)
```

- [ ] Decision: Which option? `___________`
- [ ] Rationale: `___________`

---

## 🚨 Step 4: Investigate Issue #2 - Savings Tax Formula

**Location:** `src/environment.py` lines 130-131

### Current Formula:
```python
at = abt - ((1-self.config.tax_params.tax_saving)/(1-self.config.tax_params.saving_tax_elasticity))
```

With `tax_saving=0.1, saving_tax_elasticity=0.5`:
```
at = savings - (0.9 / 0.5)
   = savings - 1.8
```

### Problems to Check:
- [ ] Does tax depend on savings amount? (It shouldn't with this formula)
- [ ] For `savings = 1.0`: tax = `1.0 - 1.8 = -0.8` (negative!)
- [ ] For `savings = 3.0`: tax = `3.0 - 1.8 = 1.2`
- [ ] Is this creating wrong incentives?

### Verify Understanding:
- [ ] What is the intended economic tax formula?
  - [ ] My intention: `___________`
- [ ] Is this from a specific paper/reference?
  - [ ] Source: `___________`
- [ ] Should tax be:
  - [ ] Proportional: `tax = τ * savings`?
  - [ ] Progressive: `tax = savings - (1-τ) * savings^(1-ε)`?
  - [ ] Something else: `___________`?

### Check Logs:
- [ ] What is `state/savings_mean`? `___________`
- [ ] Is it close to 1.8 (the breakeven point)?
- [ ] Are agents avoiding savings near 1.8?

### Proposed Fix Options:

**Option A: Simple Proportional Tax**
```python
# Line 130
at = self.config.tax_params.tax_saving * abt
```

**Option B: Progressive Tax**
```python
# Line 130
tau_s = self.config.tax_params.tax_saving
eps_s = self.config.tax_params.saving_tax_elasticity
at = abt - (1 - tau_s) * torch.pow(abt, 1 - eps_s)
```

**Option C: Keep Current (Must Justify)**
```python
# Current formula - explain why:
# ___________________________________________
```

- [ ] Decision: Which option? `___________`
- [ ] Rationale: `___________`

---

## 🚨 Step 5: Investigate Issue #3 - Income Tax Formula

**Location:** `src/environment.py` lines 127-128

### Current Formula:
```python
it = ibt - (1 - self.config.tax_params.tax_income) * \
    (ibt**(1-self.config.tax_params.income_tax_elasticity)/(1-self.config.tax_params.income_tax_elasticity))
```

With `tax_income=0.2, income_tax_elasticity=0.5`:
```
it = ibt - 0.8 * (ibt^0.5 / 0.5)
   = ibt - 1.6 * √ibt
```

### Problems to Check:
- [ ] For `ibt = 1.0`: tax = `1 - 1.6 = -0.6` (negative tax!)
- [ ] For `ibt = 4.0`: tax = `4 - 3.2 = 0.8` (20% effective rate)
- [ ] For `ibt = 100.0`: tax = `100 - 16 = 84` (84% effective rate!)
- [ ] Are these rates reasonable for your model?

### Verify Understanding:
- [ ] What is the intended income tax structure?
  - [ ] My intention: `___________`
- [ ] Is this from a specific paper/reference?
  - [ ] Source: `___________`

### Check Logs:
- [ ] What is typical `income_before_tax` value? (from temp_state)
- [ ] At step 100: `___________`
- [ ] Is it in the "negative tax" range (<2.56)?

### Proposed Fix Options:

**Option A: Simple Proportional Tax**
```python
# Line 127
it = self.config.tax_params.tax_income * ibt
```

**Option B: Progressive with Floor**
```python
# Line 127
tau_i = self.config.tax_params.tax_income
eps_i = self.config.tax_params.income_tax_elasticity
it = torch.clamp(ibt - (1 - tau_i) * (ibt**(1-eps_i)/(1-eps_i)), min=0.0)
```

**Option C: Standard Progressive**
```python
# Line 127
tau_i = self.config.tax_params.tax_income
eps_i = self.config.tax_params.income_tax_elasticity
threshold = 2.0  # Exempt first $2 of income
it = tau_i * torch.pow(torch.clamp(ibt - threshold, min=0.0), 1 + eps_i)
```

- [ ] Decision: Which option? `___________`
- [ ] Rationale: `___________`

---

## 🚨 Step 6: Investigate Issue #4 - Time Index in Parallel States

**Location:** `src/environment.py` lines 410-429

### Trace Through Manually:

**MainState at time t:**
```
[ ] MainState.savings = ___________ (call this S[t])
[ ] MainState.moneydisposable = ___________ (call this M[t])
```

**After create_temporary_state:**
```
[ ] temp_state.savings = ___________ (should be S[t+1] = M[t] * savings_ratio)
[ ] temp_state.money_disposable = ___________ (should be M[t])
```

**After transition_to_parallel:**
```
[ ] parallel_A.savings = ___________ (should be S[t+1])
[ ] parallel_A.moneydisposable = ___________ (should be M[t])
```

**After compute_parallel_outcomes:**
```
[ ] updated_parallel_A.savings = ___________ (what time period is this?)
[ ] updated_parallel_A.moneydisposable = ___________ (what time period is this?)
```

### Check for Time Mismatch:

**In `compute_parallel_outcomes` (line 414-418):**
```python
income_tax_outcomes = self._compute_income_tax(
    wage=wage,
    labor=actions["labor"],
    ability=parallel_state.ability,
    savings=parallel_state.savings,  # This is S[t+1]
    ret_lagged=parallel_state.ret    # This is r[t]
)
```

**This computes:**
- [ ] `ibt[t+1] = wage[t+1] * labor[t+1] * ability[t+1] + (1-δ+r[t]) * S[t+1]`
- [ ] `money_disposable[t+1]` from `ibt[t+1]`

**Then (line 417):**
```python
consumption = income_tax_outcomes["money_disposable"] * (1.0 - actions["savings_ratio"])
savings = income_tax_outcomes["money_disposable"] * actions["savings_ratio"]
```

**This computes:**
- [ ] `savings = M[t+1] * savings_ratio[t+1]`
- [ ] By definition, this is `S[t+2]`!

### Verify:
- [ ] Are we storing the correct time period in `updated_parallel.savings`?
- [ ] When this gets committed to MainState, what time do we think it is?
- [ ] Compare to `environment_flow_diagram.md` lines 60-65

### Proposed Fix Options:

**Option A: Don't Re-compute Savings in Parallel**
```python
# Line 417: Use existing savings
updated_parallel = ParallelState(
    moneydisposable=income_tax_outcomes["money_disposable"],
    savings=parallel_state.savings,  # Keep S[t+1], don't compute S[t+2]
    ...
)
```

**Option B: Clarify This is Intentional**
- [ ] Document that parallel outcomes compute t+2 for Euler equation
- [ ] Verify this matches your economic model intention

**Option C: Investigate Further**
- [ ] Need to understand the intended data flow better
- [ ] Draw a diagram of time indices

- [ ] Decision: Which option? `___________`
- [ ] Rationale: `___________`

---

## 🚨 Step 7: Investigate Issue #5 - No Penalty for Extreme Savings

**Location:** Loss functions don't constrain savings ratio directly

### Check Loss Behavior:

**At 50% savings ratio:**
- [ ] loss/fb: `___________`
- [ ] loss/aux_mu: `___________`
- [ ] loss/labor_foc: `___________`

**At 99% savings ratio:**
- [ ] loss/fb: `___________`
- [ ] loss/aux_mu: `___________`
- [ ] loss/labor_foc: `___________`

### Questions:
- [ ] Do losses go UP when savings → 99%? (They should if it's wrong)
- [ ] Or do losses go DOWN? (This means 99% satisfies FOCs!)
- [ ] Is 99% savings rate economically reasonable in your model?

### Economic Context:
- [ ] What is typical steady-state savings rate in Bewley models?
  - [ ] Literature says: `___________`
- [ ] What is expected savings rate for your parameters?
  - [ ] My expectation: `___________`

### Proposed Fix Options:

**Option A: Add Regularization Loss**
```python
# In LossCalculator.compute_all_losses:
loss_reg = torch.mean((savings_ratio_t - 0.5) ** 2)
total_loss = ... + 0.1 * loss_reg
```

**Option B: Add Extreme Value Penalty**
```python
# Penalize only if > 90%
extreme_savings = torch.clamp(savings_ratio_t - 0.9, min=0.0)
loss_extreme = torch.mean(extreme_savings ** 2)
total_loss = ... + 1.0 * loss_extreme
```

**Option C: No Additional Loss (Let Economic Forces Work)**
- [ ] Fix Issues 1-4 first
- [ ] See if problem resolves naturally

- [ ] Decision: Which option? `___________`
- [ ] Rationale: `___________`

---

## 🚨 Step 8: Investigate Issue #6 - Policy Bounds Too Permissive

**Location:** `src/environment.py` lines 119-121

### Current Bounds:
```python
labor_t0 = labor_t0 * 0.98 + 0.01      # Range: [0.01, 0.99]
savings_t1 = savings_t1 * 0.98 + 0.01  # Range: [0.01, 0.99]
```

### Analysis:
- [ ] Is 99% savings ratio too high for economic realism?
- [ ] What is reasonable range for your model?
  - [ ] Literature: 5%-20% for households
  - [ ] My model: `___________`

### Check Logs:
- [ ] What percentage of agents have savings_ratio > 0.9? `___________`
- [ ] What percentage of agents have savings_ratio < 0.1? `___________`
- [ ] Is there a concentration at the bounds (0.01 or 0.99)?

### Proposed Fix Options:

**Option A: Tighten Bounds**
```python
# Line 119
labor_t0 = labor_t0 * 0.7 + 0.15      # Range: [0.15, 0.85]
# Line 121
savings_t1 = savings_t1 * 0.7 + 0.15  # Range: [0.15, 0.85]
```

**Option B: Asymmetric Bounds (Lower Max, Keep Min)**
```python
# Line 119
labor_t0 = labor_t0 * 0.88 + 0.01      # Range: [0.01, 0.89]
# Line 121
savings_t1 = savings_t1 * 0.78 + 0.01  # Range: [0.01, 0.79]
```

**Option C: Keep Current Bounds**
- [ ] Rationale: `___________`

- [ ] Decision: Which option? `___________`
- [ ] Rationale: `___________`

---

## 📋 Step 9: Prioritize and Execute Fixes

### Priority Order:

**1. [ ] HIGHEST PRIORITY: Issue #1 (Return Clipping)**
   - If `market/return` is near 0.5, this is the smoking gun
   - Fix this first before anything else
   - Expected impact: Major reduction in savings ratio

**2. [ ] HIGH PRIORITY: Issues #2 & #3 (Tax Formulas)**
   - If tax formulas are broken, they distort all incentives
   - Fix these second
   - Expected impact: Better consumption/savings balance

**3. [ ] MEDIUM PRIORITY: Issue #4 (Time Indices)**
   - If there's a time mismatch, it causes accumulating errors
   - Fix this third
   - Expected impact: Correct state transitions

**4. [ ] LOW PRIORITY: Issues #5 & #6 (Regularization & Bounds)**
   - Only needed if Issues 1-3 don't fix the problem
   - Try these last
   - Expected impact: Fine-tuning

### Execution Plan:

**Round 1: Fix Issue #1**
- [ ] Modify return clip (select option from Step 3)
- [ ] Re-run training for 500 steps
- [ ] Record metrics
- [ ] Did savings ratio improve? Yes/No: `___________`

**Round 2: Fix Issues #2 & #3**
- [ ] Modify tax formulas (select options from Steps 4 & 5)
- [ ] Re-run training for 500 steps
- [ ] Record metrics
- [ ] Did savings ratio improve further? Yes/No: `___________`

**Round 3: Fix Issue #4 (if needed)**
- [ ] Modify parallel state computation (select option from Step 6)
- [ ] Re-run training for 500 steps
- [ ] Record metrics
- [ ] Did savings ratio improve? Yes/No: `___________`

**Round 4: Add Regularization (if needed)**
- [ ] Add loss penalty or tighten bounds (select options from Steps 7 & 8)
- [ ] Re-run training for 500 steps
- [ ] Record metrics
- [ ] Final savings ratio: `___________`

---

## 📊 Step 10: Results Summary

### Before Fixes:
```
Savings ratio: 99%
Consumption ratio: 1%
market/return: ___________
loss/total: ___________
```

### After Round 1 (Return Clip Fix):
```
Savings ratio: ___________%
Consumption ratio: ___________%
market/return: ___________
loss/total: ___________
Improvement: Yes/No: ___________
```

### After Round 2 (Tax Formula Fixes):
```
Savings ratio: ___________%
Consumption ratio: ___________%
market/return: ___________
loss/total: ___________
Improvement: Yes/No: ___________
```

### After Round 3 (Time Index Fix):
```
Savings ratio: ___________%
Consumption ratio: ___________%
market/return: ___________
loss/total: ___________
Improvement: Yes/No: ___________
```

### After Round 4 (Regularization):
```
Savings ratio: ___________%
Consumption ratio: ___________%
market/return: ___________
loss/total: ___________
Final result: Success/Needs more work: ___________
```

---

## 📝 Notes & Observations

**Unexpected behaviors noticed:**
```
(Write notes here)
```

**Questions for Claude:**
```
(List questions here)
```

**Other issues discovered:**
```
(List any other problems found)
```

---

## ✅ Final Checklist

- [ ] All metrics collected and recorded
- [ ] All issues investigated
- [ ] Fix priorities determined
- [ ] Fixes implemented in order
- [ ] Training re-run after each fix
- [ ] Results compared and validated
- [ ] Problem resolved or escalated
- [ ] Documentation updated
- [ ] Code changes committed

**Status:** In Progress / Resolved / Needs More Investigation

**Final Resolution:**
```
(Describe what fixed the problem, or what still needs work)
```
