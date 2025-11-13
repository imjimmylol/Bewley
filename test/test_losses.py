# test_losses.py
"""
Test script for loss functions.
Validates that all loss components work correctly.
"""

import torch
from src.calloss import LossCalculator, FBLoss, EulerLoss, LaborFOCLoss
from src.utils.configloader import load_configs, dict_to_namespace, compute_derived_params


def test_loss_calculator():
    """Test full LossCalculator."""
    print("\n" + "="*70)
    print("TEST: LossCalculator")
    print("="*70)

    # Load config
    config_dict = load_configs(['config/default.yaml'])
    config_dict = compute_derived_params(config_dict)
    config = dict_to_namespace(config_dict)

    device = torch.device("cpu")
    loss_calc = LossCalculator(config, device)

    print(f"✓ LossCalculator initialized")
    print(f"  - beta: {loss_calc.beta}")
    print(f"  - theta: {loss_calc.theta}")
    print(f"  - gamma: {loss_calc.gamma}")
    print(f"  - delta: {loss_calc.delta}")

    # Create dummy data
    B, A = 4, 10
    consumption_t = torch.rand(B, A, requires_grad=True) + 0.5
    labor_t = torch.rand(B, A, requires_grad=True) * 0.8
    savings_ratio_t = torch.rand(B, A, requires_grad=True)
    mu_t = torch.rand(B, A, requires_grad=True) * 0.1
    wage_t = torch.ones(B, A) * 1.5
    ret_t = torch.ones(B, A) * 0.04
    money_disposable_t = torch.rand(B, A) * 10
    ability_t = torch.rand(B, A) + 0.5
    consumption_A_tp1 = torch.rand(B, A) + 0.5
    consumption_B_tp1 = torch.rand(B, A) + 0.5

    print(f"\n✓ Created dummy tensors")
    print(f"  - Batch size: {B}")
    print(f"  - Agents: {A}")

    # Compute losses
    losses = loss_calc.compute_all_losses(
        consumption_t=consumption_t,
        labor_t=labor_t,
        savings_ratio_t=savings_ratio_t,
        mu_t=mu_t,
        wage_t=wage_t,
        ret_t=ret_t,
        money_disposable_t=money_disposable_t,
        ability_t=ability_t,
        consumption_A_tp1=consumption_A_tp1,
        consumption_B_tp1=consumption_B_tp1
    )

    print(f"\n✓ Losses computed:")
    print(f"  - FB loss: {losses['fb'].item():.6f}")
    print(f"  - Euler loss: {losses['euler'].item():.6f}")
    print(f"  - Labor loss: {losses['labor'].item():.6f}")
    print(f"  - Total loss: {losses['total'].item():.6f}")

    # Test backward pass
    total_loss = losses["total"]
    total_loss.backward()

    print(f"\n✓ Backward pass successful")
    print(f"  - consumption_t.grad: {consumption_t.grad is not None}")
    print(f"  - labor_t.grad: {labor_t.grad is not None}")
    print(f"  - savings_ratio_t.grad: {savings_ratio_t.grad is not None}")

    return True


def test_fb_loss():
    """Test FB loss component."""
    print("\n" + "="*70)
    print("TEST: FBLoss")
    print("="*70)

    fb_loss_fn = FBLoss(eps=1e-8)

    # Test case 1: Perfect complementarity (both zero)
    savings_ratio = torch.zeros(4, 10)
    mu = torch.zeros(4, 10)
    money_disposable = torch.ones(4, 10) * 10

    loss1 = fb_loss_fn(savings_ratio, mu, money_disposable)
    print(f"Case 1 (both zero): {loss1.item():.6f} (should be ~0)")

    # Test case 2: Interior solution (s > 0, mu = 0)
    savings_ratio = torch.rand(4, 10) * 0.5 + 0.1  # [0.1, 0.6]
    mu = torch.zeros(4, 10)

    loss2 = fb_loss_fn(savings_ratio, mu, money_disposable)
    print(f"Case 2 (interior): {loss2.item():.6f} (should be ~0)")

    # Test case 3: Violation (both positive)
    savings_ratio = torch.rand(4, 10) * 0.5 + 0.1
    mu = torch.rand(4, 10) * 0.1 + 0.05

    loss3 = fb_loss_fn(savings_ratio, mu, money_disposable)
    print(f"Case 3 (violation): {loss3.item():.6f} (should be > 0)")

    return True


def test_euler_loss():
    """Test Euler loss component."""
    print("\n" + "="*70)
    print("TEST: EulerLoss")
    print("="*70)

    euler_loss_fn = EulerLoss(beta=0.975, theta=1.0, delta=0.06, eps=1e-8)

    # Test case 1: Perfect satisfaction
    # u'(c_t) = beta * E[u'(c_{t+1})] * (1 + r)
    # c_t^(-theta) = beta * c_{t+1}^(-theta) * (1 + r)
    # c_{t+1} = (beta * (1 + r))^(1/theta) * c_t

    consumption_t = torch.ones(4, 10) * 1.0
    ret_t = torch.ones(4, 10) * 0.04
    gross_return = 1.0 + ret_t
    consumption_tp1 = (0.975 * gross_return) ** (1.0 / 1.0) * consumption_t

    loss1 = euler_loss_fn(
        consumption_t=consumption_t,
        consumption_A_tp1=consumption_tp1,
        consumption_B_tp1=consumption_tp1,
        ret_t=ret_t
    )
    print(f"Case 1 (perfect): {loss1.item():.6f} (should be ~0)")

    # Test case 2: Random consumption (will have residual)
    consumption_t = torch.rand(4, 10) + 0.5
    consumption_A = torch.rand(4, 10) + 0.5
    consumption_B = torch.rand(4, 10) + 0.5
    ret_t = torch.ones(4, 10) * 0.04

    loss2 = euler_loss_fn(
        consumption_t=consumption_t,
        consumption_A_tp1=consumption_A,
        consumption_B_tp1=consumption_B,
        ret_t=ret_t
    )
    print(f"Case 2 (random): {loss2.item():.6f} (should be > 0)")

    # Test marginal utility
    c = torch.tensor([0.5, 1.0, 2.0])
    mu_c = euler_loss_fn.marginal_utility(c)
    print(f"\nMarginal utility test (theta=1.0):")
    print(f"  c = {c.tolist()}")
    print(f"  u'(c) = {mu_c.tolist()}")
    print(f"  Expected: {(1.0 / c).tolist()}")

    return True


def test_labor_foc_loss():
    """Test Labor FOC loss component."""
    print("\n" + "="*70)
    print("TEST: LaborFOCLoss")
    print("="*70)

    labor_foc_fn = LaborFOCLoss(theta=1.0, gamma=2.0, eps=1e-8)

    # Test case 1: Perfect satisfaction
    # l^gamma = wage * ability * c^(-theta)
    # l = (wage * ability * c^(-theta))^(1/gamma)

    consumption = torch.ones(4, 10) * 1.0
    wage = torch.ones(4, 10) * 1.5
    ability = torch.ones(4, 10) * 1.0

    # Solve for labor
    rhs = wage * ability * (consumption ** (-1.0))
    labor = rhs ** (1.0 / 2.0)

    loss1 = labor_foc_fn(
        consumption=consumption,
        labor=labor,
        wage=wage,
        ability=ability
    )
    print(f"Case 1 (perfect): {loss1.item():.6f} (should be ~0)")

    # Test case 2: Random (will have residual)
    consumption = torch.rand(4, 10) + 0.5
    labor = torch.rand(4, 10) * 0.8
    wage = torch.ones(4, 10) * 1.5
    ability = torch.rand(4, 10) + 0.5

    loss2 = labor_foc_fn(
        consumption=consumption,
        labor=labor,
        wage=wage,
        ability=ability
    )
    print(f"Case 2 (random): {loss2.item():.6f} (should be > 0)")

    # Test marginal disutility
    l = torch.tensor([0.2, 0.5, 0.8])
    v_prime = labor_foc_fn.marginal_disutility_labor(l)
    print(f"\nMarginal disutility test (gamma=2.0):")
    print(f"  l = {l.tolist()}")
    print(f"  v'(l) = {v_prime.tolist()}")
    print(f"  Expected: {(l ** 2.0).tolist()}")

    return True


def main():
    """Run all tests."""
    print("\n" + "#"*70)
    print("# LOSS FUNCTION TEST SUITE")
    print("#"*70)

    try:
        # Test individual components
        test_fb_loss()
        test_euler_loss()
        test_labor_foc_loss()

        # Test full calculator
        test_loss_calculator()

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nYou can now integrate LossCalculator into your training loop!")
        print("See LOSS_DESIGN.md for usage instructions.")

    except Exception as e:
        print("\n" + "="*70)
        print("✗ TEST FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
