# Environment Design Decisions

This document explains the careful design choices made in the Bewley environment implementation.

---

## 1. Modular Architecture

### Decision: Split into 4 focused modules

**Modules:**
- `shocks.py` - Ability transitions
- `market.py` - Market equilibrium
- `income_tax.py` - Income and tax calculations
- `environment.py` - Orchestrator

**Rationale:**
- **Testability**: Each module can be unit-tested independently
- **Clarity**: Matches sections in `environment_flow_diagram.md`
- **Reusability**: Market clearing can be used in other contexts
- **Maintainability**: Easy to modify one component without affecting others

**Alternative considered:**
- Monolithic environment class
- Rejected because it would be harder to test and understand

---

## 2. Stateless Environment

### Decision: Environment doesn't store state

**Implementation:**
```python
# Caller manages state
main_state = initialize_env_state(config, device)

# Environment is stateless
state_new = env.step(main_state, policy_net, branch="A")
```

**Rationale:**
- **Explicit state management**: No hidden mutations
- **Easier debugging**: State transitions are visible
- **Gradient clarity**: Clear what's differentiable vs detached
- **Functional style**: Pure functions (input → output)

**Alternative considered:**
- Environment stores internal state
- Rejected because it would hide state transitions and make gradient flow unclear

---

## 3. ParallelState Return Type

### Decision: `step()` returns `ParallelState`, not `MainState`

**Rationale:**
- **Matches existing design**: Your `commit()` method already expects ParallelState
- **Clean branching**: Natural separation of Branch A and Branch B
- **Explicit commit**: Caller decides when to commit to MainState
- **Supports dual-branch learning**: Both branches coexist before commitment

**Flow:**
```python
MainState[t] → step(branch="A") → ParallelState_A[t+1]
             → step(branch="B") → ParallelState_B[t+1]

# Caller chooses one:
MainState[t+1] ← commit(ParallelState_A, "A")
```

---

## 4. Feature Normalization Strategy

### Decision: Normalize inside environment, before policy network

**Features normalized:**
- Individual state: money, ability, savings
- Aggregate state: mean money, mean ability, mean savings
- Tax parameters: NOT normalized (they're already bounded and meaningful)

**Rationale:**
- **Consistent preprocessing**: Normalization always happens, can't be forgotten
- **Agent-specific statistics**: Each agent dimension has separate running stats
- **Training stability**: Prevents explosion from large ability shocks
- **Update control**: Can disable updates during validation

**Alternative considered:**
- Normalize inside policy network
- Rejected because it couples normalization with policy architecture

---

## 5. Timing of Return (ret)

### Decision: Store ret[t] in state, use ret[t-1] for income calculation

**Implementation:**
```python
# Current state has ret[t-1]
ibt = wage[t] * labor[t] * ability[t] + (1 - δ + ret[t-1]) * savings[t]

# Compute new ret[t] from current aggregates
ret[t] = market_equilibrium(K[t], L[t])

# Store ret[t] for next period
new_state.ret = ret[t]  # Will become ret[t-1] next period
```

**Rationale:**
- **Matches economic timing**: Capital income uses lagged return
- **Consistent with diagram**: See `environment_flow_diagram.md` line 48
- **Prevents circular dependency**: ret[t] depends on aggregates, which depend on decisions

**Critical from diagram:**
```
Period t:   ret[t-1] affects capital income
            New ret[t] determined by current aggregates
Period t+1: ret[t] will affect capital income
```

---

## 6. Policy Network Interface

### Decision: Pass dictionary of named features

**Interface:**
```python
features = {
    "individual_money": ...,
    "individual_ability": ...,
    "agg_mean_money": ...,
    # ... 9 total features
}

output = policy_net(features)
```

**Rationale:**
- **Self-documenting**: Clear what each feature represents
- **Flexible architecture**: Policy network can choose which features to use
- **Easy to extend**: Can add new features without breaking interface
- **Type safety**: Can use TypedDict in the future

**Alternative considered:**
- Concatenate all features into single tensor
- Rejected because it loses semantic meaning

---

## 7. Dual-Step for Euler Loss

### Decision: `dual_step()` returns two independent branches

**Implementation:**
```python
state_A, state_B = env.dual_step(main_state, policy_net)

# Different ability shocks, but same policy
assert not torch.equal(state_A.ability, state_B.ability)  # Different
assert torch.equal(state_A.savings, state_B.savings)      # Same policy output
```

**Rationale:**
- **Euler equation learning**: Need two future states for expectation
- **Independent shocks**: Branch A and B have different ability realizations
- **Same policy**: Both use same policy network (for consistency)
- **Efficient**: Computed in one forward pass (features computed once)

**Economic interpretation:**
Euler equation: `u'(c[t]) = β E[u'(c[t+1]) * (1 + r[t+1])]`
- Need to approximate expectation with two samples

---

## 8. Gradient Flow

### Decision: Explicit about what's differentiable

**Differentiable:**
- Policy network outputs (savings_ratio, labor, mu)
- Market prices (wage, ret) - computed from differentiable aggregates
- Income and tax calculations - computed from differentiable variables
- Consumption and savings - derived from policy outputs

**Non-differentiable (detached):**
- Ability shocks - purely exogenous
- Superstar transitions - random process
- When committing to MainState - `detach=True` breaks gradient flow

**Rationale:**
- **Clear gradient path**: Policy → decisions → prices → income → loss
- **No gradients through randomness**: Shocks are exogenous
- **Prevents gradient explosion**: Detaching at commit prevents accumulation across steps

---

## 9. Tax Function Design

### Decision: Progressive tax with elasticity parameter

**Formula:**
```python
effective_rate = base_rate * (income / reference)^elasticity
tax = effective_rate * income
```

**Rationale:**
- **Flexible progressivity**: elasticity=1 (proportional), >1 (progressive), <1 (regressive)
- **Normalized**: Uses reference income to prevent scale issues
- **Differentiable**: Smooth function, gradients flow cleanly
- **Economically meaningful**: Matches real-world progressive tax structures

**Parameters from config:**
- `tax_income`: Base income tax rate (0.2)
- `income_tax_elasticity`: Progressivity (0.5)
- `tax_saving`: Base savings tax rate (0.1)
- `saving_tax_elasticity`: Progressivity (0.5)

---

## 10. Broadcast vs Per-Agent Prices

### Decision: Compute aggregate prices, then broadcast to agents

**Implementation:**
```python
# Compute at aggregate level (B,)
wage, ret = compute_market_prices(...)  # Returns (B,)

# Broadcast to agents (B, A)
wage_broadcast = wage.unsqueeze(1).expand(-1, n_agents)
```

**Rationale:**
- **Economic reality**: All agents face same market prices
- **Computational efficiency**: Aggregate calculation is faster
- **Clear structure**: Separates aggregate from individual level
- **Flexible**: Can choose to broadcast or not

---

## 11. History Management

### Decision: Rolling window with optional initialization

**Implementation:**
```python
if ability_history is None:
    # Initialize: repeat current ability
    history = ability.unsqueeze(0).repeat(max_length, 1, 1)
else:
    # Append and keep last max_length
    history = torch.cat([history, ability.unsqueeze(0)], dim=0)[-max_length:]
```

**Rationale:**
- **Memory efficient**: Fixed maximum size
- **Initialization**: Gracefully handles None (first step)
- **Path dependence**: Preserves recent history for learning
- **Shape (L, B, A)**: Time first, consistent with sequence models

---

## 12. Validation Mode

### Decision: Separate deterministic mode for validation

**Implementation:**
```python
diagnostics = env.validate_step(
    state=main_state,
    policy_net=policy_net
)
```

**Features:**
- Deterministic shocks (no randomness)
- No normalizer updates
- Returns diagnostic metrics
- Checks branch consistency

**Rationale:**
- **Reproducible validation**: Same input → same output
- **Fair comparison**: No random variation in metrics
- **Preserves normalizer**: Don't corrupt training statistics with validation data
- **Debugging**: Branch consistency check catches bugs

---

## 13. Rollout Function

### Decision: Provide multi-step rollout for trajectory generation

**Use cases:**
- Validation: Generate long trajectories without training
- Logging: Extract time series for visualization
- Analysis: Study agent behavior over time

**Commit strategies:**
- `"random"`: Random choice (realistic)
- `"alternating"`: A, B, A, B (balanced)
- `"A"` or `"B"`: Always same (for debugging)

**Rationale:**
- **Convenience**: Common operation, don't repeat code
- **Flexibility**: Multiple commit strategies for different purposes
- **Clean interface**: Single function for trajectory generation

---

## 14. Error Handling

### Decision: Defensive programming with clamps and fallbacks

**Examples:**
```python
# Prevent division by zero
capital_labor_ratio = K_agg / (L_eff + 1e-8)

# Ensure valid ranges
savings_ratio = torch.clamp(savings_ratio, min=0.0, max=1.0)
labor = torch.clamp(labor, min=0.0, max=1.0)

# Non-negative income
money_disposable = torch.clamp(money_disposable, min=0.0)
```

**Rationale:**
- **Numerical stability**: Prevents NaN and Inf
- **Economic constraints**: Enforces valid ranges (savings_ratio ∈ [0,1])
- **Graceful degradation**: Clipping is better than crashing

---

## Summary

These design decisions prioritize:

1. **Clarity**: Code matches economic concepts and flow diagram
2. **Testability**: Each component can be tested independently
3. **Flexibility**: Easy to modify and extend
4. **Stability**: Numerical safeguards prevent crashes
5. **Efficiency**: Compute only what's needed, when needed

The architecture separates concerns cleanly:
- **Exogenous**: Shocks (no gradients)
- **Endogenous**: Policy decisions (gradients flow)
- **Equilibrium**: Market prices (computed from aggregates)
- **Derived**: Income, taxes, consumption (functions of above)

This makes it clear where gradients flow and what the policy is learning to optimize.
