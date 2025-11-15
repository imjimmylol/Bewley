# Tax Distortion & Return Clipping Debugging Guide

**Problem:** Consumption/Savings ratio is approximately 60:1 (consumption = 3.0, savings = 0.05)

**Root Cause Hypotheses:**
1. Tax formulas are creating distortions that suppress saving
2. Return clipping at 0.5 is preventing market equilibrium
3. Both issues interact to create under-saving

**Date Created:** 2025-01-13

---

## 🔬 Testing Strategy Overview

We will run **4 controlled experiments** to isolate the effects of:
- Tax formula distortions
- Return clipping constraints
- Their interaction

Each experiment changes ONE or TWO variables while holding others constant.

---

## 📊 Experiment 0: BASELINE (Current State)

**Purpose:** Establish current behavior as reference point

### Configuration:
```yaml
tax_params:
  tax_consumption: 0.065
  tax_income: 0.2
  tax_saving: 0.1
  income_tax_elasticity: 0.5
  saving_tax_elasticity: 0.5

# In src/environment.py line 178:
ret = torch.clamp(ret, min=0.0, max=0.5)  # Current clip

# In src/environment.py lines 127-131:
# Current (potentially broken) tax formulas
it = ibt - (1 - tax_income) * (ibt^(1-eps_i) / (1-eps_i))
at = abt - ((1-tax_saving) / (1-eps_s))
```

### Run Command:
```bash
python main.py --configs config/baseline.yaml
```

### Metrics to Record:
```
Step 1:
  market/return: ___________
  market/wage: ___________
  state/labor_mean: ___________
  state/consumption_mean: ___________
  state/savings_mean: ___________
  state/money_disposable_mean: ___________
  state/budget_ratio: ___________
  loss/total: ___________

Step 100:
  market/return: ___________
  market/wage: ___________
  state/labor_mean: ___________
  state/consumption_mean: ___________
  state/savings_mean: ___________
  state/money_disposable_mean: ___________
  state/budget_ratio: ___________
  loss/total: ___________

Step 500:
  market/return: ___________
  market/wage: ___________
  state/labor_mean: ___________
  state/consumption_mean: ___________
  state/savings_mean: ___________
  state/money_disposable_mean: ___________
  state/budget_ratio: ___________
  loss/total: ___________
```

### Key Observations:
```
- Is market/return close to 0.5 (hitting clip)? Yes/No: ___________
- Is state/labor_mean healthy (0.3-0.7)? Yes/No: ___________
- Is state/budget_ratio ≈ 1.0? Yes/No: ___________
- Consumption/Savings ratio: ___________
```

---

## 🧪 Experiment 1: Remove ALL Taxes

**Purpose:** Isolate whether tax distortions are causing under-saving

**Hypothesis:** If taxes are the problem, removing them should dramatically increase savings.

### Step 1: Create New Config File

Create `config/no_tax.yaml`:
```yaml
# config/no_tax.yaml
# Experiment 1: Zero taxes to test tax distortion hypothesis

tax_params:
  tax_consumption: 0.0     # was 0.065
  tax_income: 0.0          # was 0.2
  tax_saving: 0.0          # was 0.1
  income_tax_elasticity: 0.5
  saving_tax_elasticity: 0.5
```

### Step 2: Run Training

```bash
python main.py --configs config/default.yaml config/no_tax.yaml
```

The second config will override tax parameters from default.

### Step 3: Record Metrics

```
Experiment 1: NO TAXES
Tax rates: all = 0.0
Return clip: 0.5 (unchanged)
Tax formulas: N/A (no taxes applied)

Step 1:
  market/return: ___________
  state/consumption_mean: ___________
  state/savings_mean: ___________
  state/labor_mean: ___________
  loss/total: ___________

Step 100:
  market/return: ___________
  state/consumption_mean: ___________
  state/savings_mean: ___________
  state/labor_mean: ___________
  loss/total: ___________

Step 500:
  market/return: ___________
  state/consumption_mean: ___________
  state/savings_mean: ___________
  state/labor_mean: ___________
  loss/total: ___________
```

### Analysis:
```
Compare to Baseline (Exp 0):
  Savings improvement: _________ (absolute difference)
  Savings improvement: _________% (percentage)

Did savings improve significantly (>50%)? Yes/No: ___________

If YES → Taxes are a major problem
If NO → Taxes are not the main issue, continue to Exp 2
```

---

## 🧪 Experiment 2: Fix Tax Formulas (Proportional)

**Purpose:** Test with economically correct tax formulas

**Hypothesis:** Current tax formulas have bugs. Fixing them should improve savings.

### Step 1: Modify Tax Function

In `src/environment.py`, find the `_taxfunc` method (lines 125-132).

**Current code:**
```python
def _taxfunc(self, ibt, abt) -> Tuple[Tensor, Tensor]:

    it = ibt - (1 - self.config.tax_params.tax_income) * \
        (ibt**(1-self.config.tax_params.income_tax_elasticity)/(1-self.config.tax_params.income_tax_elasticity))

    at = abt - ((1-self.config.tax_params.tax_saving)/(1-self.config.tax_params.saving_tax_elasticity))

    return it, at
```

**Replace with (TEMPORARY - for testing only):**
```python
def _taxfunc(self, ibt, abt) -> Tuple[Tensor, Tensor]:
    """
    TEMPORARY: Simple proportional taxes for Experiment 2

    Original formulas appear to have issues:
    - Income tax: it = ibt - 1.6*√ibt (can be negative for low income)
    - Savings tax: at = savings - 1.8 (doesn't depend on amount!)

    Testing with simple proportional taxes.
    """
    # Simple proportional taxes
    it = self.config.tax_params.tax_income * ibt
    at = self.config.tax_params.tax_saving * abt

    return it, at
```

### Step 2: Run Training

```bash
python main.py --configs config/default.yaml
```

This uses original tax rates (0.2 and 0.1) but with corrected formulas.

### Step 3: Record Metrics

```
Experiment 2: FIXED TAX FORMULAS
Tax rates: income=0.2, saving=0.1 (original)
Tax formulas: Proportional (FIXED)
Return clip: 0.5 (unchanged)

Step 1:
  market/return: ___________
  state/consumption_mean: ___________
  state/savings_mean: ___________
  state/labor_mean: ___________
  loss/total: ___________

Step 100:
  market/return: ___________
  state/consumption_mean: ___________
  state/savings_mean: ___________
  state/labor_mean: ___________
  loss/total: ___________

Step 500:
  market/return: ___________
  state/consumption_mean: ___________
  state/savings_mean: ___________
  state/labor_mean: ___________
  loss/total: ___________
```

### Analysis:
```
Compare to Baseline (Exp 0):
  Savings improvement: _________ (absolute)
  Savings improvement: _________% (percentage)

Compare to No Taxes (Exp 1):
  Savings in Exp 2 vs Exp 1: _________
  Are they similar? Yes/No: ___________

If Exp 2 ≈ Exp 1 → Original tax formulas were very broken
If Exp 2 < Exp 1 but > Exp 0 → Tax formulas had moderate issues
If Exp 2 ≈ Exp 0 → Tax formulas aren't the problem
```

### Step 4: Revert Changes

**IMPORTANT:** After recording results, revert the tax function to original:
```bash
git checkout src/environment.py
# Or manually restore the original code
```

---

## 🧪 Experiment 3: Remove Return Clip

**Purpose:** Test if return clipping is constraining equilibrium

**Hypothesis:** Market needs returns >0.5 to incentivize saving, but clip prevents this.

### Step 1: Modify Return Clip

**Keep the fixed tax formulas from Experiment 2** (proportional taxes).

In `src/environment.py`, find line 178:
```python
ret = torch.clamp(ret, min=0.0, max=0.5)  # Max 50% annual return
```

**Replace with:**
```python
ret = torch.clamp(ret, min=0.0, max=2.0)  # Effectively unconstrained for testing
```

### Step 2: Run Training

```bash
python main.py --configs config/default.yaml
```

### Step 3: Record Metrics

```
Experiment 3: NO RETURN CLIP + FIXED TAXES
Tax formulas: Proportional (from Exp 2)
Return clip: 2.0 (effectively removed)

Step 1:
  market/return: ___________  ← KEY METRIC!
  state/consumption_mean: ___________
  state/savings_mean: ___________
  state/labor_mean: ___________
  loss/total: ___________

Step 100:
  market/return: ___________  ← Does it exceed 0.5?
  state/consumption_mean: ___________
  state/savings_mean: ___________
  state/labor_mean: ___________
  loss/total: ___________

Step 500:
  market/return: ___________  ← What is natural equilibrium?
  state/consumption_mean: ___________
  state/savings_mean: ___________
  state/labor_mean: ___________
  loss/total: ___________
```

### Analysis:
```
Did market/return exceed 0.5?
  At step 100: Yes/No: ___________
  At step 500: Yes/No: ___________
  Maximum observed: ___________

If YES → Return clip was binding! Market needs high returns.
If NO → Return clip wasn't the problem.

Compare to Exp 2:
  Savings improvement: _________
  Did removing clip help? Yes/No: ___________

Calculate β(1+r):
  With r = _________ (from step 500)
  β(1+r) = 0.975 * (1 + _______) = _________

Is β(1+r) ≈ 1.0? Yes/No: ___________
If YES → This is the natural steady state
```

### Step 4: Revert Changes

**IMPORTANT:** Revert both changes:
```bash
git checkout src/environment.py
# Or manually restore both tax function and return clip
```

---

## 🧪 Experiment 4: Realistic Return Clip + Fixed Taxes

**Purpose:** Test if a lower, more economically realistic return clip works with fixed taxes

**Hypothesis:** With β=0.975, steady state needs r≈0.026 (2.6%). Max clip should be ~0.10.

### Step 1: Modify Code

**Keep fixed tax formulas from Experiment 2** (proportional).

In `src/environment.py`, line 178:
```python
ret = torch.clamp(ret, min=0.0, max=0.10)  # Max 10% annual return
```

### Step 2: Run Training

```bash
python main.py --configs config/default.yaml
```

### Step 3: Record Metrics

```
Experiment 4: LOW RETURN CLIP + FIXED TAXES
Tax formulas: Proportional (from Exp 2)
Return clip: 0.10 (realistic)

Step 1:
  market/return: ___________
  state/consumption_mean: ___________
  state/savings_mean: ___________
  state/labor_mean: ___________
  loss/total: ___________

Step 100:
  market/return: ___________  ← Is it hitting 0.10 clip?
  state/consumption_mean: ___________
  state/savings_mean: ___________
  state/labor_mean: ___________
  loss/total: ___________

Step 500:
  market/return: ___________
  state/consumption_mean: ___________
  state/savings_mean: ___________
  state/labor_mean: ___________
  loss/total: ___________
```

### Analysis:
```
Is return hitting the 0.10 clip?
  At step 100: Yes/No: ___________
  At step 500: Yes/No: ___________

If YES → Even 0.10 is too low, market needs higher returns
If NO → This clip is appropriate

Compare to Exp 2 (clip=0.5):
  Savings in Exp 4: _________
  Savings in Exp 2: _________
  Difference: _________

Compare to Exp 3 (no clip):
  Did Exp 4 get similar results? Yes/No: ___________

Calculate β(1+r):
  With r = _________ (from step 500)
  β(1+r) = 0.975 * (1 + _______) = _________

Is β(1+r) ≈ 1.0? Yes/No: ___________
```

### Step 4: Revert Changes

```bash
git checkout src/environment.py
```

---

## 📊 Summary Table

After completing all experiments, fill in this comparison table:

| Metric | Exp 0<br>(Baseline) | Exp 1<br>(No Taxes) | Exp 2<br>(Fixed Taxes) | Exp 3<br>(No Clip) | Exp 4<br>(Low Clip) |
|--------|---------------------|---------------------|------------------------|--------------------|--------------------|
| **Tax Formula** | Broken | N/A | Proportional | Proportional | Proportional |
| **Return Clip** | 0.5 | 0.5 | 0.5 | 2.0 | 0.10 |
| **market/return** | | | | | |
| **market/wage** | | | | | |
| **labor_mean** | | | | | |
| **consumption** | | | | | |
| **savings** | | | | | |
| **budget_ratio** | | | | | |
| **loss/total** | | | | | |
| **β(1+r)** | | | | | |

---

## 🔍 Diagnostic Questions

### Q1: Are Tax Formulas the Primary Issue?

```
Evidence from Exp 1 (No Taxes):
  Savings improvement vs baseline: __________%

Evidence from Exp 2 (Fixed Taxes):
  Savings improvement vs baseline: __________%
  Similarity to Exp 1: __________%

Conclusion:
[ ] YES - Tax formulas are broken (Exp 1 and Exp 2 show big improvement)
[ ] PARTIAL - Tax formulas have issues (moderate improvement)
[ ] NO - Tax formulas are not the problem (little improvement)

Detailed reasoning:
_________________________________________________________________
_________________________________________________________________
```

### Q2: Is Return Clipping Constraining Equilibrium?

```
Evidence from Exp 3 (No Clip):
  Did return exceed 0.5? Yes/No: ___________
  If yes, maximum return reached: ___________

Evidence from Exp 4 (Low Clip):
  Did return hit 0.10 ceiling? Yes/No: ___________

Comparison of savings:
  Exp 2 (clip=0.5): _________
  Exp 3 (clip=2.0): _________
  Exp 4 (clip=0.10): _________

Conclusion:
[ ] YES - Return clip of 0.5 is too high, allows over-saving
[ ] YES - Need high returns (>0.5) for equilibrium
[ ] NO - Return stays well below clip, not constraining
[ ] PARTIAL - Clip matters but not dominant issue

Detailed reasoning:
_________________________________________________________________
_________________________________________________________________
```

### Q3: What is the Natural Equilibrium Return?

```
From Exp 3 (unconstrained):
  Equilibrium return: _________
  β(1+r) = _________

Is this consistent with steady state (β(1+r) ≈ 1)?
  Required r ≈ 0.026 (2.6%)
  Actual r ≈ _________
  Difference: _________

Interpretation:
[ ] Market naturally converges to r ≈ 0.026 (steady state)
[ ] Market wants higher returns (r > 0.05)
[ ] Market settles at lower returns (r < 0.02)

Detailed reasoning:
_________________________________________________________________
_________________________________________________________________
```

### Q4: What is the Interaction Between Tax & Return?

```
Examine this pattern:
  Exp 0 (broken tax, clip 0.5): savings = _________
  Exp 1 (no tax, clip 0.5): savings = _________
  Exp 2 (fixed tax, clip 0.5): savings = _________
  Exp 3 (fixed tax, no clip): savings = _________

Is there an interaction effect?
  (Exp 3 - Exp 2) = _________ (pure clip effect)
  (Exp 2 - Exp 0) = _________ (pure tax formula effect)
  (Exp 3 - Exp 0) = _________ (combined effect)

Interaction = Combined - (Tax effect + Clip effect) = _________

Conclusion:
[ ] Independent - Effects are additive
[ ] Synergistic - Combined effect > sum of parts
[ ] Tax-dominant - Tax formula is the main issue
[ ] Clip-dominant - Return clip is the main issue

Detailed reasoning:
_________________________________________________________________
_________________________________________________________________
```

---

## 🎯 Interpretation Guide

After completing experiments, identify which scenario best matches your results:

### **Scenario A: Tax Formulas are Broken**

**Evidence Pattern:**
- Exp 0: Low savings (baseline)
- Exp 1: High savings (no taxes)
- Exp 2: High savings (fixed taxes)
- Exp 3: Similar to Exp 2 (clip not binding)
- Return never exceeds 0.5 in any experiment

**Interpretation:**
Current tax formulas create severe distortions:
- Income tax: `tax = ibt - 1.6√ibt` can be negative
- Savings tax: `tax = savings - 1.8` doesn't depend on amount

These broken formulas suppress saving regardless of returns.

**Recommended Fix:**
```python
# In src/environment.py _taxfunc:
def _taxfunc(self, ibt, abt) -> Tuple[Tensor, Tensor]:
    # Simple proportional taxes
    it = self.config.tax_params.tax_income * ibt
    at = self.config.tax_params.tax_saving * abt
    return it, at
```

**Next Steps:**
1. Implement proportional tax formula
2. Re-run training
3. Verify savings improve
4. Consider more sophisticated progressive tax later

---

### **Scenario B: Return Clip is Constraining**

**Evidence Pattern:**
- Exp 0: Low savings (baseline)
- Exp 1: Moderate improvement (some help from removing taxes)
- Exp 2: Similar to Exp 1 (tax fix helps a bit)
- Exp 3: Major improvement, return exceeds 0.5
- Savings increase dramatically when clip is removed

**Interpretation:**
Market equilibrium requires returns >0.5 to incentivize saving with β=0.975.
Current clip prevents market from reaching equilibrium.

**Recommended Fix Options:**

**Option A: Raise Return Clip**
```python
# In src/environment.py line 178:
ret = torch.clamp(ret, min=0.0, max=1.0)  # Higher ceiling
```

**Option B: Increase Discount Factor**
```yaml
# In config/default.yaml:
bewley_model:
  beta: 0.99  # was 0.975 (more patient agents)
```

With β=0.99, required r ≈ 0.01 (1%), easier to achieve.

**Option C: Remove Clip Entirely (Risky)**
```python
# Only use min clip, no max
ret = torch.clamp(ret, min=0.0)
```

**Next Steps:**
1. Try Option B first (safest)
2. If that doesn't work, try Option A
3. Monitor `market/return` to ensure it stays reasonable

---

### **Scenario C: Both Tax and Clip are Problems**

**Evidence Pattern:**
- Exp 0: Low savings (baseline)
- Exp 1: Moderate improvement (taxes help)
- Exp 2: Moderate improvement (tax formula fix helps)
- Exp 3: Large improvement (removing clip adds more)
- Exp 3 savings > Exp 2 savings significantly

**Interpretation:**
Tax distortions increase the returns needed for saving.
Return clip prevents market from providing those high returns.
Both issues compound each other.

**Recommended Fix:**
Implement BOTH fixes:

1. **Fix Tax Formulas:**
```python
def _taxfunc(self, ibt, abt) -> Tuple[Tensor, Tensor]:
    it = self.config.tax_params.tax_income * ibt
    at = self.config.tax_params.tax_saving * abt
    return it, at
```

2. **Adjust Return Clip (choose one):**

**Conservative (try first):**
```python
ret = torch.clamp(ret, min=0.0, max=0.15)  # Moderate ceiling
```

**Moderate:**
```python
ret = torch.clamp(ret, min=0.0, max=0.30)  # Higher ceiling
```

**Aggressive:**
```yaml
# In config - increase beta instead
beta: 0.99  # Make agents more patient
```

**Next Steps:**
1. Implement tax formula fix
2. Test with current return clip (0.5)
3. If still under-saving, lower clip to 0.15 or 0.10
4. Monitor and iterate

---

### **Scenario D: Neither is the Main Problem**

**Evidence Pattern:**
- All experiments show similar low savings
- Labor is very low (<0.1) in all experiments
- Return is very low (<0.02) in all experiments

**Interpretation:**
The real problem is elsewhere:
- Labor collapse not fully fixed
- Initial conditions are far from steady state
- Different structural issue

**Recommended Actions:**
1. **Check Labor FOC Sign:**
   ```python
   # In src/calloss.py line 261, verify:
   cons_term = self._safe_pow(consumption, -self.theta) / (1.0 + self.taxparams.tax_saving)
   # Should NOT have negative sign!
   ```

2. **Check Labor Values:**
   ```
   If state/labor_mean < 0.1:
     → Labor collapse still happening
     → Check policy output bounds
     → Check labor FOC formula
   ```

3. **Check Initial Conditions:**
   ```python
   # In src/train.py lines 45-46
   # May need different initialization
   ```

**Next Steps:**
1. Investigate labor collapse issue
2. Review labor FOC formula carefully
3. Check if min labor bounds (0.01) are too weak
4. Consider different initialization strategy

---

## 📝 Implementation Checklist

### Before Starting:
- [ ] Backup current code
  ```bash
  git add .
  git commit -m "Before tax/return experiments"
  ```
- [ ] Create experiment log directory
  ```bash
  mkdir -p experiments/tax_return
  ```
- [ ] Prepare results spreadsheet or text file

### For Each Experiment:
- [ ] Read experiment instructions carefully
- [ ] Make ONLY the specified changes
- [ ] Document what you changed
- [ ] Run for at least 500 steps
- [ ] Record ALL requested metrics
- [ ] Save console output to log file
  ```bash
  python main.py --configs ... | tee experiments/tax_return/exp_N.log
  ```
- [ ] Take WandB screenshots if available
- [ ] Revert changes before next experiment
  ```bash
  git checkout src/environment.py
  # Or manually restore original code
  ```

### After All Experiments:
- [ ] Fill in summary table
- [ ] Answer all diagnostic questions
- [ ] Identify matching scenario
- [ ] Document observations and anomalies
- [ ] Share results for further analysis

---

## 🚀 Quick Start

**Fastest path to diagnosis:**

### Step 1: Run Exp 1 (No Taxes) - 10 minutes

This immediately tests if taxes are the issue:

```bash
# Create config/no_tax.yaml (see Experiment 1)
python main.py --configs config/default.yaml config/no_tax.yaml
```

**If savings improve dramatically** → Taxes are the problem, skip to Exp 2
**If savings stay low** → Taxes not the issue, run Exp 2 anyway for comparison

### Step 2: Run Exp 2 (Fixed Taxes) - 10 minutes

This tests if tax FORMULAS specifically are broken:

```bash
# Modify src/environment.py _taxfunc (see Experiment 2)
python main.py --configs config/default.yaml
# Revert changes after
```

**Compare Exp 1 vs Exp 2:**
- Similar results → Tax formulas are broken
- Different results → Tax rates matter more than formula

### Step 3: Decide Next Steps

**If Exp 1 or Exp 2 fixed it:**
- You found the problem! Implement the fix permanently
- Skip Exp 3 and 4 (clip is not the issue)

**If Exp 1 and Exp 2 didn't help:**
- Run Exp 3 to test return clip
- Run Exp 4 to test realistic clip

This staged approach saves time by prioritizing likely causes.

---

## 🔬 Advanced Analysis

### Calculate Required Return for Steady State

Given your parameters:
```
β = 0.975 (discount factor)
θ = 1.0 (CRRA coefficient)

Euler equation at steady state:
  c[t]^(-θ) = β(1+r) * c[t+1]^(-θ)

If c[t] = c[t+1] (steady state):
  1 = β(1+r)
  1+r = 1/β
  r = 1/0.975 - 1
  r ≈ 0.0256 (2.56%)
```

**Check your experiments:**
- Is equilibrium return close to 2.56%?
- If yes → Model is finding steady state ✓
- If r << 2.56% → Agents under-save (consume too much)
- If r >> 2.56% → Agents over-save (consume too little)

### Tax Wedge Analysis

With proportional taxes:
```
After-tax return = (1 - τ_s) * r

For steady state with taxes:
  1 = β(1 + (1-τ_s)*r)
  r = (1/β - 1) / (1-τ_s)

With τ_s = 0.1:
  r = 0.0256 / 0.9
  r ≈ 0.0284 (2.84%)
```

So taxes increase the required return by ~10%.

### Capital-Labor Ratio Analysis

From Cobb-Douglas:
```
r = A * α * (K/L)^(α-1)

With α = 0.33:
  r = A * 0.33 * (K/L)^(-0.67)

For target r = 0.03:
  (K/L) = (0.03 / (A * 0.33))^(-1/0.67)

With A = 1.0:
  (K/L) ≈ 2.6
```

**Check in your experiments:**
```
K/L ratio = state/savings_mean / (state/labor_mean * state/ability_mean)

Calculate:
  K/L = _________ / (_________ * _________)
  K/L = _________

Expected: K/L ≈ 2-3 for balanced economy
If K/L >> 3: Too much capital, return too low
If K/L << 2: Too little capital, return too high
```

---

## 📞 Support

If after completing all experiments you're still unclear on the root cause, document:

1. **All experiment results** (summary table filled)
2. **All diagnostic question answers**
3. **Any anomalies or unexpected behaviors**
4. **Console logs** from at least Exp 0, 1, 2, 3

Share these findings for further analysis.

---

## ✅ Success Criteria

After implementing fixes, training should show:

```
Target Metrics (Steady State):
  market/return: 0.025 - 0.035 (2.5% - 3.5%)
  β(1+r): 0.95 - 1.05 (near 1.0)
  state/labor_mean: 0.3 - 0.7 (healthy participation)
  state/consumption_mean: 1.5 - 3.0 (reasonable)
  state/savings_mean: 0.3 - 1.0 (reasonable saving rate)
  Consumption/Savings ratio: 2 - 5 (balanced, not 60!)
  state/budget_ratio: 0.99 - 1.01 (accounting correct)
  loss/total: <1.0 (decreasing)
```

If you achieve these targets, the problem is solved! 🎉

---

**End of Debugging Guide**

**Last Updated:** 2025-01-13
**Version:** 1.0
