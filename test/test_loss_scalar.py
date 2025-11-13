# test_loss_scalar.py
"""Quick test to verify all losses return scalars."""

import torch
from src.calloss import LossCalculator
from src.utils.configloader import load_configs, dict_to_namespace, compute_derived_params

# Load config
config_dict = load_configs(['config/default.yaml'])
config_dict = compute_derived_params(config_dict)
config = dict_to_namespace(config_dict)

device = torch.device("cpu")
loss_calc = LossCalculator(config, device)

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
ibt = torch.rand(B, A) * 5 + 1
consumption_A_tp1 = torch.rand(B, A) + 0.5
consumption_B_tp1 = torch.rand(B, A) + 0.5
ibt_A_tp1 = torch.rand(B, A) * 5 + 1
ibt_B_tp1 = torch.rand(B, A) * 5 + 1

print("Testing loss calculator...")
print(f"Input shapes: consumption_t = {consumption_t.shape}")

# Compute losses
losses = loss_calc.compute_all_losses(
    consumption_t=consumption_t,
    labor_t=labor_t,
    ibt=ibt,
    savings_ratio_t=savings_ratio_t,
    mu_t=mu_t,
    wage_t=wage_t,
    ret_t=ret_t,
    money_disposable_t=money_disposable_t,
    ability_t=ability_t,
    consumption_A_tp1=consumption_A_tp1,
    consumption_B_tp1=consumption_B_tp1,
    ibt_A_tp1=ibt_A_tp1,
    ibt_B_tp1=ibt_B_tp1
)

print("\nLoss shapes:")
for key, value in losses.items():
    print(f"  {key:20s}: shape={value.shape}, is_scalar={value.ndim == 0}, value={value.item():.6f}")

# Test backward
print("\nTesting backward pass...")
total_loss = losses["total"]
total_loss.backward()

print("✓ Backward pass successful!")
print(f"✓ All losses are scalars: {all(v.ndim == 0 for v in losses.values())}")

if all(v.ndim == 0 for v in losses.values()):
    print("\n✅ ALL TESTS PASSED - All losses return scalars!")
else:
    print("\n❌ FAILED - Some losses are not scalars!")
    for key, value in losses.items():
        if value.ndim != 0:
            print(f"  ❌ {key} has shape {value.shape}")
