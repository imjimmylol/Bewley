# Bewley Environment Usage Guide

## Overview

The Bewley environment is now modularized into four main components:

```
src/
├── shocks.py         # Ability transitions (AR(1) + superstar dynamics)
├── market.py         # Market equilibrium (wage, return)
├── income_tax.py     # Income, taxes, consumption calculations
└── environment.py    # Main orchestrator (ties everything together)
```

## Design Principles

### 1. **Stateless Environment**
- Environment doesn't store state internally
- Caller (training loop) manages `MainState`
- Pure functional style: state_in → state_out

### 2. **Two-Branch Architecture**
- Supports dual-branch execution for Euler equation learning
- Branch A and Branch B use independent random shocks
- Each branch maintains separate ability history and superstar status

### 3. **Normalizer Integration**
- Features are normalized before policy network
- Uses `RunningPerAgentWelford` for per-agent statistics
- Automatically updates during training steps

### 4. **Clear Gradient Flow**
- Only policy network outputs are differentiable
- Market prices and taxes are computed functions (gradients flow through)
- Shock transitions are detached (purely exogenous)

---

## Key Classes and Functions

### **BewleyEnvironment**

Main orchestrator class that coordinates all components.

```python
from src.environment import BewleyEnvironment
from src.normalizer import RunningPerAgentWelford

# Initialize
normalizer = RunningPerAgentWelford(batch_dim=0, agent_dim=1)
env = BewleyEnvironment(config, normalizer, device="cuda")
```

### **Core Methods**

#### 1. `step()` - Single Branch Step
```python
new_state = env.step(
    state=main_state,
    policy_net=policy_network,
    branch="A",  # or "B"
    deterministic=False,
    update_normalizer=True
)
```

Returns a `ParallelState` representing the next period.

#### 2. `dual_step()` - Two-Branch Step (for Euler Loss)
```python
state_A, state_B = env.dual_step(
    state=main_state,
    policy_net=policy_network,
    deterministic=False,
    update_normalizer=True
)
```

Returns two independent `ParallelState` objects with different ability shocks.

#### 3. `rollout()` - Multi-Step Trajectory
```python
final_state, trajectory = env.rollout(
    initial_state=main_state,
    policy_net=policy_network,
    n_steps=100,
    commit_strategy="random"  # or "alternating", "A", "B"
)
```

Useful for validation and logging.

#### 4. `validate_step()` - Diagnostic Step
```python
diagnostics = env.validate_step(
    state=main_state,
    policy_net=policy_network
)
# Returns: {savings_mean, ability_mean, ret_mean, ...}
```

---

## Policy Network Interface

### **Input Features**

Your policy network receives a dictionary with these keys:

```python
features = {
    # Individual state (B, A)
    "individual_money": ...,      # My disposable money (normalized)
    "individual_ability": ...,    # My ability (normalized)
    "individual_savings": ...,    # My savings (normalized)

    # Aggregate state (B, A) - same for all agents in batch
    "agg_mean_money": ...,        # Average money across agents
    "agg_mean_ability": ...,      # Average ability across agents
    "agg_mean_savings": ...,      # Average savings across agents

    # Tax environment (B, A)
    "tax_consumption": ...,       # Consumption tax rate
    "tax_income": ...,            # Income tax rate
    "tax_saving": ...,            # Savings tax rate
}
```

### **Expected Outputs**

Your policy network should return:

**Option 1: Dictionary**
```python
{
    "savings_ratio": torch.Tensor,  # (B, A) - fraction to save, in [0, 1]
    "labor": torch.Tensor,          # (B, A) - labor supply, in [0, 1]
    "mu": torch.Tensor              # (B, A) - optional Lagrange multiplier
}
```

**Option 2: Tuple**
```python
(savings_ratio, mu, labor)  # All (B, A) tensors
```

### **Example Policy Network**

```python
import torch
import torch.nn as nn

class BewleyPolicyNetwork(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()

        # Input: 9 features per agent
        self.network = nn.Sequential(
            nn.Linear(9, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3)  # Output: savings_ratio, mu, labor
        )

    def forward(self, features):
        # Stack features: (B, A, 9)
        x = torch.stack([
            features["individual_money"],
            features["individual_ability"],
            features["individual_savings"],
            features["agg_mean_money"],
            features["agg_mean_ability"],
            features["agg_mean_savings"],
            features["tax_consumption"],
            features["tax_income"],
            features["tax_saving"],
        ], dim=-1)  # (B, A, 9)

        # Forward pass
        out = self.network(x)  # (B, A, 3)

        # Apply activation functions
        savings_ratio = torch.sigmoid(out[..., 0])  # [0, 1]
        mu = torch.softplus(out[..., 1])            # [0, ∞)
        labor = torch.sigmoid(out[..., 2])          # [0, 1]

        return {
            "savings_ratio": savings_ratio,
            "mu": mu,
            "labor": labor
        }
```

---

## Training Loop Integration

### **Basic Training Loop**

```python
from src.train import initialize_env_state
from src.environment import BewleyEnvironment
from src.normalizer import RunningPerAgentWelford
from src.market import compute_market_equilibrium
from src.income_tax import full_tax_and_income_pipeline

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
normalizer = RunningPerAgentWelford(batch_dim=0, agent_dim=1)
env = BewleyEnvironment(config, normalizer, device)

# Initialize state
main_state = initialize_env_state(config, device)

# Initialize policy network
policy_net = BewleyPolicyNetwork().to(device)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=config.training.learning_rate)

# Training loop
for step in range(config.training.training_steps):
    # ==== STEP 1: Agents observe current state and act ====
    # Prepare features from current state
    features = env._prepare_policy_features(
        state=main_state,
        ability_current=main_state.ability,
        update_normalizer=True
    )

    # Policy network makes decisions based on current state
    policy_output = policy_net(features)
    savings_ratio = policy_output["savings_ratio"]  # (B, A)
    labor = policy_output["labor"]  # (B, A)
    mu = policy_output.get("mu", None)  # (B, A)

    # ==== STEP 2: Calculate market prices at time t ====
    # Market equilibrium based on current decisions
    wage_t, ret_t = compute_market_equilibrium(
        savings=main_state.savings,
        labor=labor,
        ability=main_state.ability,
        config=config,
        broadcast=True
    )

    # Compute income, taxes, consumption at time t
    (
        income_before_tax_t,
        money_disposable_t,
        consumption_t,
        savings_next_t,
        income_tax_t,
        savings_tax_t
    ) = full_tax_and_income_pipeline(
        wage=wage_t,
        labor=labor,
        ability=main_state.ability,
        savings=main_state.savings,
        ret_lagged=main_state.ret,  # ret[t-1]
        savings_ratio=savings_ratio,
        config=config
    )

    # ==== STEP 3: Store current period outcomes (realized values at time t) ====
    current_outcomes = {
        "consumption": consumption_t,
        "labor": labor,
        "savings_ratio": savings_ratio,
        "mu": mu,
        "wage": wage_t,
        "ret": ret_t,
        "income_before_tax": income_before_tax_t,
        "money_disposable": money_disposable_t,
    }

    # ==== STEP 4: Dual-step to get next period states (t+1) ====
    # This transitions ability and computes two future branches
    # Returns both states AND outcomes (consumption, labor, etc.)
    (state_A, outcomes_A), (state_B, outcomes_B) = env.dual_step(main_state, policy_net)

    # Extract consumption for t+1 from outcomes (already computed inside dual_step)
    consumption_A_tp1 = outcomes_A["consumption"]
    consumption_B_tp1 = outcomes_B["consumption"]

    # ==== STEP 5: Compute losses ====
    # FB loss: Forward-backward consistency at time t
    fb_loss = compute_fb_loss(
        savings_ratio=savings_ratio,
        mu=mu,
        money_disposable=money_disposable_t
    )

    # Euler loss: Compare consumption[t] with expected consumption[t+1]
    euler_loss = compute_euler_loss(
        consumption_t=consumption_t,
        consumption_A_tp1=consumption_A_tp1,
        consumption_B_tp1=consumption_B_tp1,
        ret_t=ret_t,
        config=config
    )

    # Labor FOC loss: First-order condition for labor
    labor_foc_loss = compute_labor_foc_loss(
        consumption=consumption_t,
        labor=labor,
        wage=wage_t,
        ability=main_state.ability,
        config=config
    )

    total_loss = fb_loss + euler_loss + labor_foc_loss

    # ==== STEP 6: Backward pass ====
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # ==== STEP 7: Commit one branch to become realized path ====
    branch = "A" if torch.rand(1).item() < 0.5 else "B"
    chosen = state_A if branch == "A" else state_B
    main_state.commit(chosen, branch=branch, detach=True)

    # ==== STEP 8: Logging ====
    if step % 100 == 0:
        print(f"Step {step}: Loss={total_loss.item():.4f}")
        print(f"  Mean consumption: {consumption_t.mean():.3f}")
        print(f"  Mean savings: {main_state.savings.mean():.3f}")
        print(f"  Market wage: {wage_t.mean():.3f}, ret: {ret_t.mean():.3f}")
```

---

## Variable Timing and Dependencies

### **Critical Timing Details**

From `environment_flow_diagram.md`:

```
Period t-1: ret[t-1] determined
    ↓
Period t:   ret[t-1] affects capital income
            New ret[t] determined by current aggregates
    ↓
Period t+1: ret[t] will affect capital income
```

### **State Transitions**

```
MainState[t] → dual_step() → ParallelState_A[t+1]
                          → ParallelState_B[t+1]

# Choose one to commit:
MainState[t+1] ← commit(ParallelState_A, "A")
```

### **Key Variables**

| Variable | Timing | Source |
|----------|--------|--------|
| `ability[t+1]` | Next period | Exogenous shock transition |
| `savings[t+1]` | Next period | Policy decision (savings_ratio) |
| `ret[t]` | Current period | Market equilibrium (will be lagged next period) |
| `wage[t]` | Current period | Market equilibrium (used immediately) |
| `consumption[t]` | Current period | Budget constraint |

---

## Advanced Usage

### **Validation Without Updating Normalizer**

```python
# Turn off normalizer updates during validation
with torch.no_grad():
    state_A, state_B = env.dual_step(
        main_state,
        policy_net,
        deterministic=True,        # No random shocks
        update_normalizer=False    # Don't update statistics
    )
```

### **Custom Commit Strategy**

```python
# Commit based on which branch has lower loss
state_A, state_B = env.dual_step(main_state, policy_net)

loss_A = compute_loss(state_A)
loss_B = compute_loss(state_B)

if loss_A < loss_B:
    main_state.commit(state_A, "A", detach=True)
else:
    main_state.commit(state_B, "B", detach=True)
```

### **Extracting Trajectories**

```python
# Generate a long trajectory for analysis
final_state, trajectory = env.rollout(
    initial_state=main_state,
    policy_net=policy_net,
    n_steps=1000,
    deterministic=False,
    commit_strategy="random"
)

# Extract time series
savings_series = torch.stack([state.savings for state in trajectory])  # (1000, B, A)
ability_series = torch.stack([state.ability for state in trajectory])  # (1000, B, A)
```

---

## Troubleshooting

### **NaN Values**

If you see NaN values:

1. **Check normalization**: Early steps may have unstable statistics
   - Solution: Start with `update_normalizer=False` for first few steps

2. **Division by zero**: Low savings or labor can cause issues
   - Solution: All functions use `clamp(min=1e-8)` for stability

3. **Exploding values**: Ability shocks might be too large
   - Solution: Check `v_min`, `v_max` bounds in config

### **Gradient Issues**

If gradients are not flowing:

1. **Check policy output ranges**: Ensure `savings_ratio` and `labor` are in [0, 1]
2. **Verify loss computation**: Make sure losses depend on policy outputs
3. **Check detachment**: Only shock transitions should be detached

### **Memory Issues**

For large batch sizes or many agents:

1. **Reduce batch size**: Lower `config.training.batch_size`
2. **Reduce history length**: Lower `config.training.history_length`
3. **Use gradient checkpointing**: Implement in policy network
4. **Clear cache**: Call `torch.cuda.empty_cache()` periodically

---

## Next Steps

1. **Implement loss functions** (FB loss, Euler loss, Labor FOC)
2. **Integrate with wandb** for logging
3. **Add checkpointing** for saving/loading environment state
4. **Create visualization tools** for analyzing agent behavior

See `environment_flow_diagram.md` for detailed variable flow documentation.
