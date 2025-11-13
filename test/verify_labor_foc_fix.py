# verify_labor_foc_fix.py
"""Verify the labor FOC sign fix."""

import torch

# Simulate Step 1 values
labor = 0.542
consumption = 2.32
wage = 1.28
ability = 1.5
gamma = 2.0
theta = 1.0
tax_saving = 0.0
tax_income = 0.3
ibt_elasticity = 0.5
ibt = 5.0

print("=" * 70)
print("Labor FOC Sign Fix Verification")
print("=" * 70)

# OLD FORMULA (with negative cons_term)
print("\n🔴 OLD FORMULA (WRONG):")
labor_term = -(labor ** gamma)
cons_term_old = -(consumption ** (-theta)) / (1.0 + tax_saving)  # NEGATIVE
ibt_term = ibt ** (-ibt_elasticity)
tax_factor = (1.0 - tax_income) * ibt_term
prod_term = wage * ability * tax_factor

loss_foc_old = labor_term + cons_term_old * prod_term

print(f"  labor_term = -labor^γ = -{labor}^{gamma} = {labor_term:.4f}")
print(f"  cons_term = -c^(-θ)/(1+τ_s) = -{consumption}^(-{theta}) / {1.0+tax_saving} = {cons_term_old:.4f}")
print(f"  prod_term = w*v*tax_factor = {wage}*{ability}*{tax_factor:.3f} = {prod_term:.4f}")
print(f"  loss_foc = {labor_term:.4f} + ({cons_term_old:.4f}) * {prod_term:.4f}")
print(f"           = {labor_term:.4f} + {cons_term_old * prod_term:.4f}")
print(f"           = {loss_foc_old:.4f}")
print(f"  loss = loss_foc^2 = {loss_foc_old**2:.4f}")

# Gradient (analytical)
grad_labor_term = -gamma * (labor ** (gamma - 1))
grad_old = 2 * loss_foc_old * grad_labor_term
print(f"\n  ∂(labor_term)/∂labor = -γ*labor^(γ-1) = {grad_labor_term:.4f}")
print(f"  ∂(loss)/∂labor = 2 * loss_foc * ∂(labor_term)/∂labor")
print(f"                 = 2 * {loss_foc_old:.4f} * {grad_labor_term:.4f}")
print(f"                 = {grad_old:.4f}")

if grad_old > 0:
    print(f"  ❌ POSITIVE gradient → policy DECREASES labor!")
else:
    print(f"  ✓ NEGATIVE gradient → policy INCREASES labor")

# NEW FORMULA (with positive cons_term)
print("\n" + "=" * 70)
print("✅ NEW FORMULA (CORRECT):")
cons_term_new = (consumption ** (-theta)) / (1.0 + tax_saving)  # POSITIVE

loss_foc_new = labor_term + cons_term_new * prod_term

print(f"  labor_term = -labor^γ = -{labor}^{gamma} = {labor_term:.4f}")
print(f"  cons_term = c^(-θ)/(1+τ_s) = {consumption}^(-{theta}) / {1.0+tax_saving} = {cons_term_new:.4f}")
print(f"  prod_term = w*v*tax_factor = {wage}*{ability}*{tax_factor:.3f} = {prod_term:.4f}")
print(f"  loss_foc = {labor_term:.4f} + ({cons_term_new:.4f}) * {prod_term:.4f}")
print(f"           = {labor_term:.4f} + {cons_term_new * prod_term:.4f}")
print(f"           = {loss_foc_new:.4f}")
print(f"  loss = loss_foc^2 = {loss_foc_new**2:.4f}")

# Gradient (analytical)
grad_new = 2 * loss_foc_new * grad_labor_term
print(f"\n  ∂(labor_term)/∂labor = -γ*labor^(γ-1) = {grad_labor_term:.4f}")
print(f"  ∂(loss)/∂labor = 2 * loss_foc * ∂(labor_term)/∂labor")
print(f"                 = 2 * {loss_foc_new:.4f} * {grad_labor_term:.4f}")
print(f"                 = {grad_new:.4f}")

if grad_new > 0:
    print(f"  ❌ POSITIVE gradient → policy DECREASES labor!")
else:
    print(f"  ✅ NEGATIVE gradient → policy INCREASES labor!")

# Compare
print("\n" + "=" * 70)
print("COMPARISON:")
print("=" * 70)
print(f"  OLD: loss_foc = {loss_foc_old:.4f}, gradient = {grad_old:+.4f}")
print(f"  NEW: loss_foc = {loss_foc_new:.4f}, gradient = {grad_new:+.4f}")
print(f"\n  Gradient sign FLIPPED: {grad_old > 0} → {grad_new < 0}")

if grad_old > 0 and grad_new < 0:
    print("\n✅ FIX CONFIRMED: Labor FOC now has correct sign!")
    print("   Policy will now INCREASE labor instead of decreasing it.")
else:
    print("\n❌ Something is still wrong...")

# Economic interpretation
print("\n" + "=" * 70)
print("ECONOMIC INTERPRETATION:")
print("=" * 70)
print(f"  Current labor = {labor:.3f}")
print(f"  Optimal FOC should be: labor^γ = c^(-θ) * wage * (1-tax)")
print(f"    LHS: labor^γ = {labor**gamma:.4f}")
print(f"    RHS: c^(-θ) * wage * (1-tax) = {cons_term_new:.4f} * {prod_term:.4f} = {cons_term_new * prod_term:.4f}")
print(f"  Difference: {labor**gamma:.4f} - {cons_term_new * prod_term:.4f} = {labor**gamma - cons_term_new * prod_term:.4f}")

if abs(labor**gamma - cons_term_new * prod_term) < 0.01:
    print(f"  ✓ Close to equilibrium!")
elif labor**gamma < cons_term_new * prod_term:
    print(f"  → Labor is TOO LOW (should increase)")
else:
    print(f"  → Labor is TOO HIGH (should decrease)")
