# Taxonomy of State Variables in Reinforcement Learning Environments

## Overview

State variables in RL environments can be categorized based on their causal relationships to agent actions and other variables. Understanding these categories helps with model design, sample efficiency, credit assignment, and sim-to-real transfer.

---

## 1. Exogenous Variables

**Definition:** Completely independent of agent actions; follow their own autonomous dynamics.

**Characteristics:**
- Evolve according to: `s_{t+1} ~ P(s_t)` with no dependence on actions
- External to the agent's control
- May follow deterministic or stochastic processes

**Examples:**
- Weather patterns (temperature, precipitation)
- Time of day / seasonal cycles
- External market forces (stock prices in a trading simulation)
- Natural environmental changes
- Background noise or disturbances

**Implications:**
- Require separate modeling from action-dependent dynamics
- May need historical data for accurate prediction
- Often domain-specific and harder to generalize

---

## 2. Endogenous Variables (Action-Dependent)

**Definition:** Directly affected by agent actions; the core controllable aspects of the environment.

**Characteristics:**
- Update as: `s_{t+1} ~ P(s_t, a_t)`
- Respond directly to agent's decisions
- Primary focus of control and optimization

**Examples:**
- Robot position and velocity
- Inventory levels
- Account balance
- Object states manipulated by the agent
- Energy levels or resource consumption

**Implications:**
- Central to reward shaping and credit assignment
- Agent can learn causal relationships through exploration
- Most amenable to policy optimization

---

## 3. Semi-Exogenous / Partially Observable Exogenous Variables

**Definition:** Variables that are exogenous (not affected by actions) but whose observation or knowledge is endogenous (revealed through interaction).

**Characteristics:**
- Underlying state follows: `s_{t+1} ~ P(s_t)` (exogenous)
- Agent's belief/observation updated through actions: `b_{t+1} = f(b_t, a_t, o_t)`
- Information-gathering actions reduce uncertainty

**Examples:**
- Hidden opponent strategies (learned through play)
- Unknown environment parameters (friction coefficients, physics constants)
- Latent market conditions (revealed through trades)
- Sensor calibration errors (discovered through testing)

**Implications:**
- Exploration-exploitation tradeoff is critical
- May require active learning or information-seeking behaviors
- Often modeled with belief states or information states

---

## 4. Derived / Computed Variables

**Definition:** Deterministic functions of other state variables; no independent dynamics.

**Characteristics:**
- Update as: `s_{t+1} = f(s'_t, a_t)` for some deterministic function f
- Redundant information (could be computed on-demand)
- Often included for computational convenience or learning efficiency

**Examples:**
- Relative positions (distance between agent and goal)
- Ratios and percentages (health percentage, fuel efficiency)
- Aggregated statistics (moving averages, cumulative rewards)
- Normalized or scaled versions of raw variables
- Feature engineering outputs

**Implications:**
- Can improve learning by providing more interpretable features
- May reduce sample complexity by highlighting relevant relationships
- Should be carefully chosen to avoid redundancy or misleading signals

---

## 5. Multi-Agent Influenced Variables

**Definition:** Variables affected by other agents' actions in multi-agent settings.

**Characteristics:**
- Exogenous from ego-agent's perspective: `s_{t+1} ~ P(s_t, a^{ego}_t, a^{other}_t)`
- Endogenous to the overall system
- Introduce strategic complexity and non-stationarity

**Examples:**
- Traffic conditions (influenced by other drivers)
- Market prices (affected by other traders)
- Shared resources (contested by multiple agents)
- Social signals (reputation, cooperation levels)
- Team-based objectives

**Implications:**
- Require game-theoretic considerations
- May exhibit emergent behaviors
- Opponent modeling can be beneficial
- Environment becomes non-stationary from single agent's view

---

## 6. History-Dependent Variables

**Definition:** Variables that depend on sequences of past states/actions, not just the immediate previous state (violate Markov property).

**Characteristics:**
- Update as: `s_{t+1} = f(s_t, s_{t-1}, ..., s_0, a_t, a_{t-1}, ..., a_0)`
- Capture long-term effects or memory
- May require recurrent architectures or extended state representations

**Examples:**
- Cumulative damage or wear-and-tear
- Reputation and trust (built over many interactions)
- Learning progress or skill acquisition
- Fatigue or adaptation effects
- Path-dependent outcomes (e.g., hysteresis)

**Implications:**
- Standard MDPs may be insufficient; need POMDPs or history-based models
- Temporal credit assignment becomes more challenging
- May benefit from recurrent neural networks (RNN, LSTM, Transformers)
- Require careful consideration of temporal abstractions

---

## Design Considerations

### Model Architecture
- **Separate networks**: Consider separate dynamics models for exogenous vs. endogenous variables
- **Modular design**: Group variables by type for better interpretability and transfer

### Sample Efficiency
- **Exogenous variables**: May benefit from offline datasets or auxiliary prediction tasks
- **Endogenous variables**: Require active exploration and on-policy data

### Credit Assignment
- **Clear causality**: Endogenous variables provide clearer credit assignment signals
- **Confounding**: Mixed variable types can create spurious correlations

### Sim-to-Real Transfer
- **Domain gap**: Exogenous variables often differ most between simulation and reality
- **Invariances**: Endogenous dynamics may be more transferable if physics is accurate

### Exploration Strategy
- **Information gathering**: Semi-exogenous variables may require exploration bonuses
- **Causal discovery**: Actions designed to test causal relationships

---

## Practical Tips

1. **Document your variables**: Explicitly categorize each state variable in your environment
2. **Validate independence**: Test whether "exogenous" variables truly don't depend on actions
3. **Consider temporal scales**: Some variables may appear exogenous at short timescales but endogenous over longer horizons
4. **Hybrid approaches**: Many real variables have aspects of multiple categories
5. **Measurement**: Distinguish between true state and observed state (partial observability)

---

## References & Further Reading

- Sutton & Barto - "Reinforcement Learning: An Introduction" (MDPs and state representations)
- Pearl - "Causality" (causal modeling and interventions)
- Kaelbling et al. - "Planning and Acting in Partially Observable Stochastic Domains" (POMDPs)
- Multi-agent RL literature for game-theoretic extensions

# The Mapping for variables 
```yaml
exogenous:
  - ability
  - is_superstar_vA
  - is_superstar_vB
  - tax_params   # becomes multi_agent if controlled by a policy agent

endogenous:
  - savings

multi_agent:
  - wage
  - ret

derived:
  - ibt
  - moneydisposable
  - consumption
  - policy_features  # engineered (not state)

history_dependent:
  - ability_history_A
  - ability_history_B

semi_exogenous:
  - []  # none currently (use beliefs here if you introduce partial observability)
```

# The Update logic for the environment 
