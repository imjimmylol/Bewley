# Environment Update Flow Diagram

## Visual Flow of Variable Updates

```
┌─────────────────────────────────────────────────────────────────┐
│                         TIME t                                   │
├─────────────────────────────────────────────────────────────────┤
│ Initial State:                                                   │
│ • moneydisposable[t]                                            │
│ • savings[t]                                                    │
│ • ability[t]                                                    │
│ • ret[t-1] (from previous period)                              │
│ • tax_params (fixed)                                            │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    NEURAL NETWORK POLICY                         │
├─────────────────────────────────────────────────────────────────┤
│ Input Features:                                                  │
│ • Aggregate info (all agents' money & ability)                  │
│ • Individual state (own money & ability)                        │
│ • Tax parameters (conditions)                                   │
│                                                                  │
│ Outputs:                                                         │
│ • savings_ratio[t+1] → sigmoid                                  │
│ • mu[t] (multiplier) → softplus                                 │
│ • labor[t] → sigmoid                                            │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MARKET EQUILIBRIUM                            │
├─────────────────────────────────────────────────────────────────┤
│ Aggregates:                                                      │
│ • K_agg = mean(savings[t])                                      │
│ • L_eff = mean(labor[t] * ability[t])                          │
│                                                                  │
│ Prices (Multi-agent influenced):                                │
│ • wage[t] = A(1-α)(K/L)^α                                      │
│ • ret[t] = Aα(K/L)^(α-1)                                       │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    INCOME CALCULATION                            │
├─────────────────────────────────────────────────────────────────┤
│ Before Tax Income:                                               │
│ • ibt[t] = wage[t]*labor[t]*ability[t]                         │
│           + (1-δ+ret[t-1])*savings[t]                          │
│             ↑                                                    │
│          Uses PREVIOUS period return!                           │
│                                                                  │
│ After Tax (Derived):                                            │
│ • Apply tax function → income_tax, savings_tax                  │
│ • money_disposable[t] = (ibt-income_tax)+(savings-savings_tax) │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT DECISIONS                               │
├─────────────────────────────────────────────────────────────────┤
│ Endogenous Updates:                                              │
│ • consumption[t] = money_disposable[t] * (1 - savings_ratio)    │
│ • savings[t+1] = money_disposable[t] * savings_ratio           │
│                  ↑                                               │
│              MAIN ENDOGENOUS VARIABLE                           │
└─────────────────────────────────────────────────────────────────┘
                                ↓
        ┌───────────────────────┴───────────────────────┐
        ↓                                               ↓
┌──────────────────────┐                    ┌──────────────────────┐
│    BRANCH A          │                    │    BRANCH B          │
├──────────────────────┤                    ├──────────────────────┤
│ Ability Transition:  │                    │ Ability Transition:  │
│ • AR(1) shock        │                    │ • AR(1) shock        │
│ • Superstar dynamics │                    │ • Superstar dynamics │
│   - p: become star   │                    │   - p: become star   │
│   - q: remain star   │                    │   - q: remain star   │
│ • Bounded [v_min,    │                    │ • Bounded [v_min,    │
│           v_max]     │                    │           v_max]     │
└──────────────────────┘                    └──────────────────────┘
        ↓                                               ↓
        └───────────────────────┬───────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      STATE at TIME t+1                           │
├─────────────────────────────────────────────────────────────────┤
│ Updated Variables:                                               │
│ • ability[t+1] (exogenous transition)                           │
│ • savings[t+1] (endogenous from NN)                            │
│ • wage[t], ret[t] (multi-agent equilibrium)                    │
│ • money_disposable[t], consumption[t], ibt[t] (derived)        │
│ • ability_history updated (append new ability)                  │
└─────────────────────────────────────────────────────────────────┘
```

## Variable Update Summary Table

| Variable | Type | Update Mechanism | Depends On |
|----------|------|------------------|------------|
| **ability** | Exogenous | AR(1) + superstar shocks | ability[t], random shock |
| **is_superstar** | Exogenous | Probabilistic transition | is_superstar[t], p, q |
| **tax_params** | Exogenous | Fixed/constant | - |
| **savings** | Endogenous | NN output × money_disposable | state[t], NN policy |
| **wage** | Multi-agent | Market equilibrium | aggregate K, L×ability |
| **ret** | Multi-agent | Market equilibrium | aggregate K, L×ability |
| **ibt** | Derived | Income formula | wage, labor, ability, savings, ret[t-1] |
| **money_disposable** | Derived | Tax adjustment | ibt, savings, tax_params |
| **consumption** | Derived | Budget constraint | money_disposable, savings_ratio |
| **labor** | Policy output | NN decision | state[t] |
| **ability_history** | History-dependent | Rolling window | previous history, new ability |

## Critical Timing Details

### 1. Return Lag Structure
```
Period t-1: ret[t-1] determined
    ↓
Period t:   ret[t-1] affects capital income
            New ret[t] determined by current aggregates
    ↓  
Period t+1: ret[t] will affect capital income
```

### 2. Savings Flow
```
savings[t] → affects prices at t → generates income at t 
    ↓
money_disposable[t] × savings_ratio = savings[t+1]
```

### 3. Two-Branch Structure for Learning
```
State[t] → Branch A: ability_A[t+1] → State_A[t+1]
         ↘ Branch B: ability_B[t+1] → State_B[t+1]
         
Used for auxiliary loss (Euler equation consistency)
```

## Loss Functions Data Flow

### FB Loss (Forward-Backward Consistency)
```
savings_ratio[t+1] ←→ mu[t]
Tests: Complementary slackness condition
```

### Auxiliary Loss (Euler Equation)
```
consumption[t] → consumption_A[t+1], consumption_B[t+1]
              ↓
         Euler equation residuals
              ↓
         Mean squared error
```

### Labor FOC Loss
```
labor[t] → marginal disutility
consumption[t] → marginal utility  
wage[t] × ability[t] → marginal product
              ↓
        FOC residual = 0
```

## Notes for Environment Design

1. **State Representation**: Keep exogenous and endogenous variables separate for cleaner updates
2. **Batch Processing**: All operations are batched (B batches × A agents)
3. **Gradient Flow**: Only savings is truly endogenous to policy; prices are equilibrium outcomes
4. **Stochasticity**: Main source is ability shocks; can be disabled for validation
5. **History Management**: Ability histories need careful memory management (rolling window)
