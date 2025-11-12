# Loss Design for Bewley Model

## Overview

The loss calculation module (`src/calloss.py`) implements three economic losses for training the policy network to satisfy equilibrium conditions in the Bewley heterogeneous agent model.

## Architecture

```
LossCalculator (orchestrator)
├── FBLoss (Forward-Backward consistency)
├── EulerLoss (Intertemporal consumption smoothing)
└── LaborFOCLoss (Labor-leisure optimality)
```

## Three Loss Components

### 1. Forward-Backward (FB) Loss

**Economic Interpretation**: Ensures the savings decision is consistent with the Lagrange multiplier from the budget constraint.

**Equation**: KKT complementary slackness condition
```
μ[t] * s[t] = 0
```

where:
- `μ[t]`: Lagrange multiplier (≥ 0)
- `s[t]`: Savings ratio ∈ [0, 1]

**Condition**:
- If `s[t] > 0` (interior solution), then `μ[t] = 0` (no binding constraint)
- If `s[t] = 0` (corner solution), then `μ[t] ≥ 0` (constraint binds)

**Loss**: Mean squared error of complementarity product
```python
loss_fb = E[(μ * s)²]
```

### 2. Euler Equation Loss

**Economic Interpretation**: Ensures optimal consumption smoothing over time.

**Equation**: Intertemporal first-order condition
```
u'(c[t]) = β * E[u'(c[t+1]) * (1 + r[t])]
```

where:
- `u(c) = c^(1-θ) / (1-θ)`: CRRA utility
- `u'(c) = c^(-θ)`: Marginal utility
- `β`: Discount factor
- `r[t]`: Net return to capital
- `E[·]`: Expectation over two branches A and B

**Loss**: Mean squared Euler equation residual
```python
lhs = c[t]^(-θ)
rhs = β * 0.5 * (c_A[t+1]^(-θ) + c_B[t+1]^(-θ)) * (1 + r[t])
loss_euler = E[(lhs - rhs)²]
```

**Special Case**: If `θ = 1` (log utility), use `u'(c) = 1/c`

### 3. Labor FOC Loss

**Economic Interpretation**: Ensures optimal labor-leisure tradeoff.

**Equation**: Intratemporal first-order condition
```
v'(l[t]) = wage[t] * ability[t] * u'(c[t])
```

where:
- `v(l) = l^(1+γ) / (1+γ)`: Disutility of labor
- `v'(l) = l^γ`: Marginal disutility
- `γ`: Inverse Frisch elasticity of labor supply

**Loss**: Mean squared FOC residual
```python
lhs = l[t]^γ
rhs = wage[t] * ability[t] * c[t]^(-θ)
loss_labor = E[(lhs - rhs)²]
```

## Parameters

Loaded from `config.bewley_model`:

| Parameter | Symbol | Typical Value | Description |
|-----------|--------|---------------|-------------|
| `beta` | β | 0.975 | Discount factor |
| `theta` | θ | 1.0 | CRRA coefficient (risk aversion) |
| `gamma` | γ | 2.0 | Inverse Frisch elasticity |
| `delta` | δ | 0.06 | Depreciation rate |

Loss weights from `config.training` (optional):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weight_fb` | 1.0 | Weight for FB loss |
| `weight_euler` | 1.0 | Weight for Euler loss |
| `weight_labor` | 1.0 | Weight for Labor loss |

## Usage

### Basic Usage

```python
from src.calloss import LossCalculator

# Initialize (once at start of training)
loss_calc = LossCalculator(config, device)

# In training loop
losses = loss_calc.compute_all_losses(
    # Current period (t)
    consumption_t=temp_state.consumption,
    labor_t=temp_state.labor,
    savings_ratio_t=temp_state.savings_ratio,
    mu_t=temp_state.mu,
    wage_t=temp_state.wage,
    ret_t=temp_state.ret,
    money_disposable_t=temp_state.money_disposable,
    ability_t=main_state.ability,
    # Next period (t+1)
    consumption_A_tp1=outcomes_A["consumption"],
    consumption_B_tp1=outcomes_B["consumption"]
)

# Get total loss for backprop
total_loss = losses["total"]
total_loss.backward()
```

### Integration with Training Loop

Replace placeholder losses in `src/train.py`:

**Before:**
```python
# Placeholder losses
fb_loss = torch.tensor(0.0, device=device, requires_grad=True)
euler_loss = torch.tensor(0.0, device=device, requires_grad=True)
labor_foc_loss = torch.tensor(0.0, device=device, requires_grad=True)
total_loss = fb_loss + euler_loss + labor_foc_loss
```

**After:**
```python
# Initialize loss calculator (once, before training loop)
loss_calc = LossCalculator(config, device)

# In training loop (after env.step)
losses = loss_calc.compute_all_losses(
    consumption_t=consumption_t,
    labor_t=labor_t,
    savings_ratio_t=savings_ratio_t,
    mu_t=mu_t,
    wage_t=wage_t,
    ret_t=ret_t,
    money_disposable_t=money_disposable_t,
    ability_t=main_state.ability,
    consumption_A_tp1=consumption_A_tp1,
    consumption_B_tp1=consumption_B_tp1
)

# Extract losses
fb_loss = losses["fb"]
euler_loss = losses["euler"]
labor_foc_loss = losses["labor"]
total_loss = losses["total"]
```

### With Custom Weights

Add to `config/default.yaml`:

```yaml
training:
  agents: 87
  learning_rate: 1e-3
  # ... existing params ...

  # Loss weights (optional)
  weight_fb: 1.0
  weight_euler: 10.0      # Emphasize Euler equation
  weight_labor: 1.0
```

### Individual Loss Components

You can also use individual losses directly:

```python
from src.calloss import EulerLoss

euler_loss_fn = EulerLoss(beta=0.975, theta=1.0, delta=0.06)
euler_loss = euler_loss_fn(
    consumption_t=c_t,
    consumption_A_tp1=c_A,
    consumption_B_tp1=c_B,
    ret_t=r_t
)
```

## Configuration Options

### Add to config/default.yaml

```yaml
bewley_model:
  theta: 1.0         # CRRA (required)
  beta: 0.975        # Discount factor (required)
  gamma: 2.0         # Inverse Frisch elasticity (required)
  delta: 0.06        # Depreciation rate (required)

training:
  # ... existing params ...

  # Optional: loss weights
  weight_fb: 1.0
  weight_euler: 1.0
  weight_labor: 1.0
```

## Return Values

`compute_all_losses()` returns a dictionary:

```python
{
    "fb": Tensor,              # Unweighted FB loss
    "euler": Tensor,           # Unweighted Euler loss
    "labor": Tensor,           # Unweighted Labor loss
    "fb_weighted": Tensor,     # Weighted FB loss
    "euler_weighted": Tensor,  # Weighted Euler loss
    "labor_weighted": Tensor,  # Weighted Labor loss
    "total": Tensor            # Sum of weighted losses
}
```

## Numerical Stability

All loss functions include:
- **Clamping**: Consumption and labor clamped to `[eps, ∞)` where `eps=1e-8`
- **Special cases**: Log utility when `θ ≈ 1`
- **Gradient stability**: All operations are differentiable and numerically stable

## Debugging

### Check Individual Losses

```python
losses = loss_calc.compute_all_losses(...)

print(f"FB loss: {losses['fb'].item():.6f}")
print(f"Euler loss: {losses['euler'].item():.6f}")
print(f"Labor loss: {losses['labor'].item():.6f}")
print(f"Total loss: {losses['total'].item():.6f}")
```

### Validate Economic Conditions

```python
# Check if Euler equation is satisfied
from src.calloss import EulerLoss

euler = EulerLoss(beta=0.975, theta=1.0, delta=0.06)

lhs = euler.marginal_utility(consumption_t)
marginal_u_A = euler.marginal_utility(consumption_A_tp1)
marginal_u_B = euler.marginal_utility(consumption_B_tp1)
rhs = 0.975 * 0.5 * (marginal_u_A + marginal_u_B) * (1 + ret_t)

print(f"LHS mean: {lhs.mean():.6f}")
print(f"RHS mean: {rhs.mean():.6f}")
print(f"Ratio LHS/RHS: {(lhs/rhs).mean():.6f}")  # Should be ~1.0 at equilibrium
```

## Customization

### Modify FB Loss

The FB loss has a **PLACEHOLDER** implementation. You should customize it based on your model:

```python
class FBLoss:
    def __call__(self, savings_ratio, mu, money_disposable):
        # OPTION 1: Complementary slackness (current)
        loss = (mu * savings_ratio).pow(2).mean()

        # OPTION 2: Only penalize interior violations
        # interior_mask = (savings_ratio > 0.01)
        # loss = (mu[interior_mask] ** 2).mean()

        # OPTION 3: Budget residual
        # expected_savings = compute_optimal_savings(money_disposable, ...)
        # loss = (savings_ratio - expected_savings).pow(2).mean()

        return loss
```

### Add New Loss

```python
class SomeNewLoss:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def __call__(self, x, y):
        # Compute loss
        return loss

# Add to LossCalculator
class LossCalculator:
    def __init__(self, config, device):
        # ... existing code ...
        self.new_loss_fn = SomeNewLoss(param1=..., param2=...)

    def compute_all_losses(self, ...):
        # ... existing code ...
        new_loss = self.new_loss_fn(x, y)
        total_loss += self.weight_new * new_loss

        return {..., "new": new_loss}
```

## Testing

Test individual components:

```python
# Test with dummy data
import torch
from src.calloss import LossCalculator
from src.utils.configloader import load_configs, dict_to_namespace, compute_derived_params

config_dict = load_configs(['config/default.yaml'])
config_dict = compute_derived_params(config_dict)
config = dict_to_namespace(config_dict)

device = torch.device("cpu")
loss_calc = LossCalculator(config, device)

# Create dummy tensors
B, A = 4, 10
consumption_t = torch.rand(B, A) + 0.5
labor_t = torch.rand(B, A)
savings_ratio_t = torch.rand(B, A)
mu_t = torch.rand(B, A) * 0.1
wage_t = torch.ones(B, A) * 1.5
ret_t = torch.ones(B, A) * 0.04
money_disposable_t = torch.rand(B, A) * 10
ability_t = torch.rand(B, A) + 0.5
consumption_A_tp1 = torch.rand(B, A) + 0.5
consumption_B_tp1 = torch.rand(B, A) + 0.5

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

print("Losses:", {k: v.item() for k, v in losses.items()})
```

## References

- Bewley, T. (1986). "Stationary Monetary Equilibrium with a Continuum of Independently Fluctuating Consumers"
- Aiyagari, S. R. (1994). "Uninsured Idiosyncratic Risk and Aggregate Saving"
- Huggett, M. (1993). "The risk-free rate in heterogeneous-agent incomplete-insurance economies"
