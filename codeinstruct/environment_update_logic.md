# Environment Update Logic Pseudocode

## Overview
This pseudocode describes the update logic for a multi-agent economic environment with heterogeneous agents, 
stochastic productivity shocks, and endogenous price formation.

## State Variables Update Flow

```python
class BewleyEnvironment:
    """
    State at time t:
    - moneydisposable[t]: Available resources for each agent
    - savings[t]: Capital/savings from previous period
    - ability[t]: Productivity of each agent
    - is_superstar_vA[t], is_superstar_vB[t]: Superstar status flags
    - tax_params: Tax policy parameters (fixed)
    - ret[t]: Return on savings from previous period
    - ability_history_A, ability_history_B: Historical ability trajectories
    """
    
    def step(state_t, model):
        """
        Main update logic: state[t] -> state[t+1]
        """
        
        # ============================================
        # PHASE 1: Policy Network Decision (t=0)
        # ============================================
        
        # 1.1 Build neural network inputs
        features = concat([
            aggregate_state_info,     # (B, A, 2A) - all agents' money & ability
            agent_own_money[t],       # (B, A, 1) - self money
            agent_own_ability[t]      # (B, A, 1) - self ability
        ])  # Total: (B, A, 2A+2)
        
        conditions = expand(tax_params, dim=agents)  # (B, A, 5)
        
        # 1.2 Get policy outputs
        model_outputs = neural_network(features, conditions)
        savings_ratio[t+1] = sigmoid(model_outputs[0])    # ∈ [0,1]
        mu[t] = softplus(model_outputs[1])                # > 0 (multiplier)
        labor[t] = sigmoid(model_outputs[2])              # ∈ [0,1]
        
        # ============================================
        # PHASE 2: Market Equilibrium (Prices)
        # ============================================
        
        # 2.1 Calculate aggregate quantities
        K_agg = mean(savings[t], across_agents)
        L_eff_agg = mean(labor[t] * ability[t], across_agents)
        
        # 2.2 Compute equilibrium prices (multi-agent influenced)
        wage[t] = A * (1 - α) * (K_agg / L_eff_agg)^α
        ret_new[t] = A * α * (K_agg / L_eff_agg)^(α-1)
        
        # ============================================
        # PHASE 3: Income and Resources (Derived)
        # ============================================
        
        # 3.1 Calculate before-tax income
        labor_income = wage[t] * labor[t] * ability[t]
        capital_income = (1 - δ + ret[t-1]) * savings[t]  # Uses PREVIOUS period's return
        ibt[t] = labor_income + capital_income
        
        # 3.2 Apply tax function
        income_tax, savings_tax = taxfunc(ibt[t], savings[t], tax_params)
        
        # 3.3 Calculate disposable money (derived)
        money_disposable[t] = (ibt[t] - income_tax) + (savings[t] - savings_tax)
        
        # ============================================
        # PHASE 4: Agent Decisions (Endogenous)
        # ============================================
        
        # 4.1 Transform NN outputs to actual quantities
        consumption[t] = money_disposable[t] * (1 - savings_ratio[t+1])
        savings[t+1] = money_disposable[t] * savings_ratio[t+1]  # ENDOGENOUS update
        
        # ============================================
        # PHASE 5: Stochastic Transitions (Exogenous)
        # ============================================
        
        # 5.1 Transition ability (two branches for auxiliary loss)
        for branch in [A, B]:
            # 5.1.1 AR(1) process with bounds
            log_ability_shock = ρ_v * log(ability[t]) + σ_v * ε  # ε ~ N(0,1)
            ability_new = exp(log_ability_shock)
            ability_new = clip(ability_new, v_min, v_max)
            
            # 5.1.2 Superstar dynamics
            if not is_superstar[t]:
                if random() < p:  # Become superstar
                    is_superstar[t+1] = True
                    ability[t+1] = v_bar  # Jump to high productivity
                else:
                    is_superstar[t+1] = False
                    ability[t+1] = ability_new
            else:  # Currently superstar
                if random() < q:  # Remain superstar
                    is_superstar[t+1] = True
                    ability[t+1] = v_bar
                else:  # Lose superstar status
                    is_superstar[t+1] = False
                    ability[t+1] = ability_new
        
        # 5.2 Update ability history (history-dependent)
        ability_history = append(ability_history, ability[t+1])
        if len(ability_history) > MAX_HISTORY_LEN:
            ability_history = ability_history[-MAX_HISTORY_LEN:]
        
        # ============================================
        # PHASE 6: State Update Summary
        # ============================================
        
        state[t+1] = {
            # Exogenous (independent of actions)
            'ability': ability[t+1],              # Stochastic transition
            'is_superstar_vA': is_superstar_A[t+1],
            'is_superstar_vB': is_superstar_B[t+1],
            'tax_params': tax_params,             # Fixed
            
            # Endogenous (action-dependent)
            'savings': savings[t+1],              # From NN decision
            
            # Multi-agent influenced
            'wage': wage[t],                      # From market equilibrium
            'ret': ret_new[t],                    # From market equilibrium
            
            # Derived (computed from other variables)
            'moneydisposable': money_disposable[t],
            'consumption': consumption[t],
            'ibt': ibt[t],
            
            # History-dependent
            'ability_history_A': ability_history_A,
            'ability_history_B': ability_history_B
        }
        
        return state[t+1]
```

## Detailed Variable Dependencies

### 1. Exogenous Variables
```python
# Ability transition (stochastic, not affected by actions)
ability[t+1] = f(ability[t], ε_shock, is_superstar[t], p, q, v_bar)
    where: ε ~ N(0, σ_v²)
           transition follows AR(1) with superstar jumps

# Superstar status (probabilistic transitions)
is_superstar[t+1] = f(is_superstar[t], p, q, random())

# Tax parameters (constant)
tax_params[t+1] = tax_params[t]
```

### 2. Endogenous Variables
```python
# Savings (directly controlled by agent through NN)
savings[t+1] = money_disposable[t] * NN_savings_ratio(state[t])
```

### 3. Multi-Agent Influenced Variables
```python
# Wages and returns (general equilibrium)
wage[t] = f(mean(savings[t]), mean(labor[t] * ability[t]))
ret[t] = f(mean(savings[t]), mean(labor[t] * ability[t]))
```

### 4. Derived Variables
```python
# Income before tax
ibt[t] = wage[t] * labor[t] * ability[t] + (1 - δ + ret[t-1]) * savings[t]

# Money disposable (after taxes)
money_disposable[t] = tax_adjusted(ibt[t], savings[t])

# Consumption
consumption[t] = money_disposable[t] * (1 - NN_savings_ratio(state[t]))
```

### 5. History-Dependent Variables
```python
# Ability histories (rolling window)
ability_history[t+1] = concatenate(ability_history[t], ability[t+1])[-MAX_LEN:]
```

## Key Timing Notes

1. **Return on savings lag**: `ret[t-1]` affects income at time `t`
   - Current period savings earn next period's return
   
2. **Price formation**: Wages and returns determined by aggregate behavior
   - All agents move simultaneously (not sequential)
   
3. **Branching for auxiliary loss**: 
   - Two ability transition paths (A, B) for counterfactual reasoning
   - Same initial state, different shocks
   
4. **Neural network timing**:
   - Observes state[t]
   - Outputs decisions for t (labor) and t+1 (savings ratio)

## Loss Computation Dependencies

```python
# Forward-backward consistency loss
fb_loss = f(savings_ratio[t+1], mu[t])

# Auxiliary Euler equation loss (uses both branches)
aux_loss = f(
    consumption[t],        # From branch at t
    consumption_A[t+1],    # From branch A at t+1
    consumption_B[t+1],    # From branch B at t+1
    savings_A[t+2],        # Next period savings, branch A
    savings_B[t+2],        # Next period savings, branch B
    mu[t],                 # Multiplier at t
    ibt_A[t+1],           # Income branch A
    ibt_B[t+1],           # Income branch B
    ret[t]                # Return at t
)

# Labor first-order condition loss
labor_foc_loss = f(
    labor[t],
    consumption[t],
    ibt[t],
    wage[t],
    ability[t]
)
```

## Environment Class Structure Suggestion

```python
class BewleyEnvironment:
    def __init__(self, config):
        self.agents = config.agents
        self.tax_params = config.tax_params
        self.shock_params = config.shock_params
        
    def reset(self):
        """Initialize all state variables"""
        return initial_state
        
    def step(self, model):
        """Single environment step with NN policy"""
        # 1. Get NN decisions
        # 2. Update prices
        # 3. Calculate income/resources
        # 4. Update savings/consumption
        # 5. Transition abilities
        # 6. Return new state
        
    def compute_losses(self, trajectory):
        """Calculate training losses from trajectory"""
        # FB loss, Auxiliary loss, Labor FOC loss
```
