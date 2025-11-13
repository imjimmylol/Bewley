# src/environment.py
"""
EconomyEnv: Bewley model environment orchestrator.

Workflow:
1. Agents observe MainState[t] and perform actions
2. Create TemporaryState (realized outcomes at time t)
3. Apply two-path dependent shocks → ParallelState_A, ParallelState_B
4. Agents act separately on each parallel state
5. Choose one branch → Commit to MainState[t+1]
"""

from typing import Tuple, Literal, Optional, Dict
import torch
from torch import Tensor
import torch.nn.functional as F
from src.utils.buildipnuts import build_inputs
from src.env_state import MainState, TemporaryState, ParallelState, make_parallel
from src.shocks import transition_ability_with_history


class EconomyEnv:
    """
    Bewley model economy environment.

    This environment manages the full economic simulation workflow:
    - Agent decision-making based on current state
    - Market equilibrium computation
    - Two-branch parallel state transitions
    - State commitment and updates
    """

    def __init__(self, config, normalizer=None, device="cpu"):
        """
        Initialize the economy environment.

        Args:
            config: Configuration namespace with all model parameters
            normalizer: Optional RunningPerAgentWelford for feature normalization
            device: Device to place tensors on ("cpu" or "cuda")
        """
        self.config = config
        self.normalizer = normalizer
        self.device = device

        # Extract key dimensions
        self.n_agents = config.training.agents
        self.batch_size = config.training.batch_size
        self.history_length = getattr(config.training, 'history_length', 50)

    # ========================================================================
    # STEP 1: Create TemporaryState from MainState
    # ========================================================================

    def _prepare_features(
        self,
        state: MainState,
        update_normalizer: bool = True
    ) -> Dict[str, Tensor]:
        """
        Prepare normalized features for policy network input.

        Args:
            state: Current MainState
            update_normalizer: Whether to update normalizer statistics

        Returns:
            features: Dictionary of normalized features for policy network
        """
        # TODO: Implement feature preparation
        # - Extract individual and aggregate features from state
        moneydisposable = state.moneydisposable
        ability = state.ability
        
        # - Normalize using self.normalizer if available
        ability_normalized = self.normalizer.transform("ability", ability, update=update_normalizer)
        moneydisposable_normalized = self.normalizer.transform("moneydisposalbe", moneydisposable, update=update_normalizer)

        # - Return dict of features
        model_inputs = build_inputs(moneydisposable=moneydisposable_normalized,
                     ability=ability_normalized,
                     tax_params=state.tax_params,
                     device=self.device)
        
        return model_inputs
        # raise NotImplementedError("Feature preparation not yet implemented")

    def _compute_agent_actions(
        self,
        state: MainState | ParallelState,
        policy_net,
        update_normalizer: bool = True
    ) -> Dict[str, Tensor]:
        """
        Compute agent actions based on current MainState.

        Args:
            state: Current MainState
            policy_net: Policy network
            update_normalizer: Whether to update normalizer statistics

        Returns:
            actions: Dictionary containing:
                - savings_ratio: (B, A)
                - labor: (B, A)
                - mu: (B, A) or None
        """
        # TODO: Implement agent action computation
        # 1. Prepare features from state
        features, condition = self._prepare_features(state, update_normalizer)
        # 2. Call policy_net(features)
        acts = [torch.sigmoid, lambda x: F.softplus(x) + 1e-6, torch.sigmoid]
        out = policy_net(features, condition)
        savings_t1, mu_t0, labor_t0 = [acts[i](out[..., i]) for i in range(out.shape[-1])]
        savings_t1, mu_t0, labor_t0 = savings_t1.squeeze(-1), mu_t0.squeeze(-1), labor_t0.squeeze(-1)

        # CRITICAL: Enforce minimum labor and savings to prevent collapse
        # Labor in [0.01, 0.99] instead of [0, 1]
        labor_t0 = labor_t0 * 0.98 + 0.01
        # Savings ratio in [0.01, 0.99] instead of [0, 1]
        savings_t1 = savings_t1 * 0.98 + 0.01

        # 3. Return actions dict
        return {
            "savings_ratio": savings_t1,
            "mu": mu_t0,
            "labor": labor_t0,
        }
        # raise NotImplementedError("Agent action computation not yet implemented")

    def _taxfunc(self, ibt, abt) -> Tuple[Tensor, Tensor]:

        it = ibt - (1 - self.config.tax_params.tax_income) * \
            (ibt**(1-self.config.tax_params.income_tax_elasticity)/(1-self.config.tax_params.income_tax_elasticity))

        at = abt - ((1-self.config.tax_params.tax_saving)/(1-self.config.tax_params.saving_tax_elasticity))

        return it, at

    def _compute_market_equilibrium(
        self,
        savings: Tensor,
        labor: Tensor,
        ability: Tensor,
        A: float,
        alpha: float
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute market equilibrium prices.

        Args:
            savings: Current savings (B, A)
            labor: Labor supply (B, A)
            ability: Ability (B, A)

        Returns:
            wage: Market wage (B, A) or (B,)
            ret: Return to capital (B, A) or (B,)
        """
        savings_agg = savings.mean(dim=1, keepdim=True)
        labor_eff_agg = (labor * ability).mean(dim=1, keepdim=True)

        # CRITICAL: Enforce strong floors to prevent collapse
        # With labor >= 0.01 from policy bounds, this ensures labor_eff >= 0.01 * v_bar
        labor_eff_agg = torch.clamp(labor_eff_agg, min=0.01)
        savings_agg = torch.clamp(savings_agg, min=0.01)

        # Capital-labor ratio with bounds to prevent extreme prices
        ratio = savings_agg / labor_eff_agg
        ratio = torch.clamp(ratio, min=0.1, max=10.0)  # Bounded K/L ratio

        # Compute prices from Cobb-Douglas production
        wage = A * (1 - alpha) * (ratio ** alpha)
        ret = A * alpha * (ratio ** (alpha - 1))

        # CRITICAL: Clip prices to economically reasonable ranges
        # Prevents gradient explosion from extreme return values in Euler equation
        ret = torch.clamp(ret, min=0.0, max=0.5)  # Max 50% annual return
        wage = torch.clamp(wage, min=0.1, max=10.0)  # Reasonable wage range

        return wage, ret

    def _compute_income_tax(
        self,
        wage: Tensor,
        labor: Tensor,
        ability: Tensor,
        savings: Tensor,
        ret_lagged: Tensor
    ) -> Dict[str, Tensor]:
        """
        Compute income, taxes, and disposable money.

        Args:
            wage: Market wage (B, A)
            labor: Labor supply (B, A)
            ability: Ability (B, A)
            savings: Current savings (B, A)
            ret_lagged: Lagged return ret[t-1]

        Returns:
            outcomes: Dictionary containing:
                - income_before_tax: (B, A)
                - money_disposable: (B, A)
        """
        # TODO: Implement income and tax computation
        # - Compute income before tax (labor + capital income)
        ability = torch.clamp(ability, min=0.1)
        ibt = wage * labor * ability + (1 - self.config.bewley_model.delta + ret_lagged) * savings  # before-tax income
        # - Apply tax functions
        it, at = self._taxfunc(ibt=ibt, abt=savings)
        # - Compute disposable money
        money_disposable = (ibt-it) + (savings-at) 

        return {
            "money_disposable":money_disposable, 
            "income_before_tax":ibt,
            "income_tax":it,
            "savings_tax":at
        }
        # raise NotImplementedError("Income and tax computation not yet implemented")

    def create_temporary_state(
        self,
        main_state: MainState,
        policy_net,
        update_normalizer: bool = True
    ) -> TemporaryState:
        """
        STEP 1: Create TemporaryState from MainState.

        This computes all realized outcomes at time t after agents act
        on the current MainState.

        Args:
            main_state: Current MainState at time t
            policy_net: Policy network for agent decisions
            update_normalizer: Whether to update normalizer statistics

        Returns:
            temp_state: TemporaryState with all realized outcomes at time t
        """
        # 1. Compute agent actions (savings_ratio, labor, mu)
        actions = self._compute_agent_actions(main_state, policy_net, update_normalizer)

        # 2. Compute market equilibrium (wage, ret)
        wage, ret = self._compute_market_equilibrium(
            savings=main_state.savings,
            labor=actions["labor"],
            ability=main_state.ability,
            A=self.config.bewley_model.A,
            alpha=self.config.bewley_model.alpha
        )
        # 3. Compute income and taxes
        income_tax_outcomes = self._compute_income_tax(
            wage=wage,
            labor=actions["labor"],
            ability=main_state.ability,
            savings=main_state.savings,
            ret_lagged=main_state.ret  # This is ret[t-1]
        )
        # 4. Compute consumption
        consumption = income_tax_outcomes["money_disposable"] * (1.0 - actions["savings_ratio"])
        savings = income_tax_outcomes["money_disposable"] * actions["savings_ratio"]
        
        # 5. Package into TemporaryState
        temp_state = TemporaryState(
            # Current state (before shocks)
            savings=savings, 
            ability=main_state.ability,

            # Agent decisions
            consumption=consumption,
            labor=actions["labor"],
            savings_ratio=actions["savings_ratio"],
            mu=actions.get("mu", None),

            # Market outcomes
            wage=wage,
            ret=ret,

            # Income/tax
            income_before_tax=income_tax_outcomes["income_before_tax"],
            money_disposable=income_tax_outcomes["money_disposable"],
            income_tax=income_tax_outcomes["income_tax"],
            savings_tax=income_tax_outcomes["savings_tax"],

            # Tax parameters
            tax_params=main_state.tax_params,

            # Branch memory (carry forward)
            is_superstar_vA=main_state.is_superstar_vA,
            is_superstar_vB=main_state.is_superstar_vB,
            ability_history_vA=main_state.ability_history_vA,
            ability_history_vB=main_state.ability_history_vB,
        )

        return temp_state

    # ========================================================================
    # STEP 2: Transition from TemporaryState to ParallelState
    # ========================================================================

    def _transition_ability(
        self,
        ability_t: Tensor,
        is_superstar_t: Tensor,
        ability_history_t: Optional[Tensor],
        branch: Literal["A", "B"],
        deterministic: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Transition ability using AR(1) + superstar dynamics.

        Args:
            ability_t: Current ability (B, A)
            is_superstar_t: Current superstar status (B, A)
            ability_history_t: Current ability history (L, B, A) or None
            branch: Which branch ("A" or "B")
            deterministic: If True, no random shocks

        Returns:
            ability_tp1: Next period ability (B, A)
            is_superstar_tp1: Next period superstar status (B, A)
            ability_history_tp1: Updated ability history (L, B, A)
        """
        # Use the transition_ability_with_history function from shocks module
        ability_tp1, is_superstar_tp1, ability_history_tp1 = transition_ability_with_history(
            ability_t=ability_t,
            is_superstar_t=is_superstar_t,
            ability_history_t=ability_history_t,
            config=self.config,
            history_length=self.history_length,
            deterministic=deterministic
        )

        return ability_tp1, is_superstar_tp1, ability_history_tp1

    def transition_to_parallel(
        self,
        temp_state: TemporaryState,
        branch: Literal["A", "B"],
        deterministic: bool = False
    ) -> ParallelState:
        """
        STEP 2: Transition from TemporaryState to ParallelState.

        Apply path-dependent shock to create one branch.

        Args:
            temp_state: TemporaryState at time t
            branch: Which branch to create ("A" or "B")
            deterministic: If True, no random shocks

        Returns:
            parallel_state: ParallelState for the specified branch
        """
        # Get branch-specific memory
        is_superstar_t = temp_state.is_superstar_vA if branch == "A" else temp_state.is_superstar_vB
        ability_history_t = temp_state.ability_history_vA if branch == "A" else temp_state.ability_history_vB

        # Transition ability
        ability_tp1, is_superstar_tp1, ability_history_tp1 = self._transition_ability(
            ability_t=temp_state.ability,
            is_superstar_t=is_superstar_t,
            ability_history_t=ability_history_t,
            branch=branch,
            deterministic=deterministic
        )

        # Compute next period savings: savings[t+1] = money_disposable[t] * savings_ratio
        # savings_tp1 = temp_state.money_disposable * temp_state.savings_ratio 

        # Create ParallelState using make_parallel factory
        # Note: We need to create a "dummy" MainState to use make_parallel
        # Or we can construct ParallelState directly
        parallel_state = ParallelState(
            moneydisposable=temp_state.money_disposable,  # Carry forward
            savings=temp_state.savings,  # Updated savings for t+1
            ability=ability_tp1,  # Transitioned ability
            ret=temp_state.ret,  # Current ret becomes ret[t-1] next period
            tax_params=temp_state.tax_params,
            is_superstar=is_superstar_tp1,
            ability_history=ability_history_tp1
        )

        return parallel_state

    # ========================================================================
    # STEP 3: Compute outcomes for ParallelState
    # ========================================================================

    def compute_parallel_outcomes(
        self,
        parallel_state: ParallelState,
        policy_net,
        update_normalizer: bool = False
    ) -> Tuple[ParallelState, Dict]:
        """
        STEP 3: Compute outcomes for a ParallelState.

        Agents act on the parallel state and we compute all variables.

        Args:
            parallel_state: ParallelState to evaluate
            policy_net: Policy network
            update_normalizer: Whether to update normalizer (typically False for parallel)

        Returns:
            outcomes: Dictionary of computed outcomes
        """
        # TODO: Implement parallel state outcome computation
        # Similar to create_temporary_state but for ParallelState
        # 1. Prepare features from parallel_state
        # 2. Get actions from policy_net
        actions = self._compute_agent_actions(parallel_state, policy_net, update_normalizer=update_normalizer)
        # 3. Compute market equilibrium
        wage, ret = self._compute_market_equilibrium(
            savings=parallel_state.savings,
            labor=actions["labor"],
            ability=parallel_state.ability,
            A=self.config.bewley_model.A,
            alpha=self.config.bewley_model.alpha
        )
        # 4. Compute income, taxes, consumption
        income_tax_outcomes = self._compute_income_tax(
            wage=wage,
            labor=actions["labor"],
            ability=parallel_state.ability,
            savings=parallel_state.savings,
            ret_lagged=parallel_state.ret  # This is ret[t-1]
        )
        consumption = income_tax_outcomes["money_disposable"] * (1.0 - actions["savings_ratio"])
        savings = income_tax_outcomes["money_disposable"] * actions["savings_ratio"]

        # No ibt in ParallelState, however it is thus more clean for commit 
        updated_parallel = ParallelState(
            moneydisposable=income_tax_outcomes["money_disposable"],
            savings=savings, 
            ability=parallel_state.ability,
            ret=ret,
            tax_params=parallel_state.tax_params,
            is_superstar=parallel_state.is_superstar,
            ability_history=parallel_state.ability_history
        )

        outcomes = {
            "consumption": consumption,
            "labor": actions["labor"],
            "wage": wage,
            "ret": ret,
            "savings_ratio": actions["savings_ratio"],
            "mu": actions["mu"],
            "income_before_tax": income_tax_outcomes["income_before_tax"]
        }
        return updated_parallel, outcomes

        # raise NotImplementedError("Parallel state outcomes not yet implemented")

    # ========================================================================
    # STEP 4: Choose branch and commit to MainState
    # ========================================================================

    def choose_branch(
        self,
        parallel_A: ParallelState,
        parallel_B: ParallelState,
        outcomes_A: Dict[str, Tensor],
        outcomes_B: Dict[str, Tensor],
        strategy: Literal["random", "A", "B"] = "random"
    ) -> Literal["A", "B"]:
        """
        STEP 4a: Choose which branch to commit.

        Args:
            parallel_A: ParallelState for branch A
            parallel_B: ParallelState for branch B
            outcomes_A: Outcomes for branch A
            outcomes_B: Outcomes for branch B
            strategy: How to choose ("random", "A", "B")

        Returns:
            chosen_branch: "A" or "B"
        """
        if strategy == "random":
            return "A" if torch.rand(1, device=self.device).item() < 0.5 else "B"
        elif strategy == "A":
            return "A"
        elif strategy == "B":
            return "B"
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def commit_to_main(
        self,
        main_state: MainState,
        chosen_parallel: ParallelState,
        branch: Literal["A", "B"]
    ) -> MainState:
        """
        STEP 4b: Commit chosen ParallelState to MainState.

        Args:
            main_state: Current MainState (will be mutated in-place)
            chosen_parallel: The chosen ParallelState
            branch: Which branch was chosen

        Returns:
            main_state: Updated MainState (same object, modified in-place)
        """
        main_state.commit(chosen_parallel, branch=branch, detach=True)
        return main_state

    # ========================================================================
    # MAIN STEP FUNCTION
    # ========================================================================

    def step(
        self,
        main_state: MainState,
        policy_net,
        *,
        deterministic: bool = False,
        update_normalizer: bool = True,
        commit_strategy: Literal["random", "A", "B"] = "random"
    ) -> Tuple[MainState, TemporaryState, Tuple[ParallelState, Dict], Tuple[ParallelState, Dict]]:
        """
        Execute one full environment step.

        Workflow:
        1. Agents observe MainState[t] and act
        2. Create TemporaryState (realized outcomes at time t)
        3. Apply two-path shocks → ParallelState_A, ParallelState_B
        4. Agents act on each parallel state
        5. Choose one branch and commit to MainState[t+1]

        Args:
            main_state: Current MainState at time t (will be modified in-place)
            policy_net: Policy network for agent decisions
            deterministic: If True, no random shocks
            update_normalizer: If True, update normalizer statistics
            commit_strategy: How to choose branch ("random", "A", "B")

        Returns:
            main_state: Updated MainState at time t+1 (same object, modified)
            temp_state: TemporaryState (realized outcomes at time t)
            (parallel_A, outcomes_A): Branch A state and outcomes
            (parallel_B, outcomes_B): Branch B state and outcomes
        """
        # STEP 1 & 2: Create TemporaryState
        temp_state = self.create_temporary_state(
            main_state=main_state,
            policy_net=policy_net,
            update_normalizer=update_normalizer
        )
        # print(f"!!!!!!!!!!! MD in temporaray State {temp_state.money_disposable.mean()}")
        # STEP 3: Transition to two parallel branches
        parallel_A = self.transition_to_parallel(
            temp_state=temp_state,
            branch="A",
            deterministic=deterministic
        )
        # print(f"!!!!!!!!!!! MD in temporaray State transition 2 Parallel {parallel_A.moneydisposable.mean()}")
        parallel_B = self.transition_to_parallel(
            temp_state=temp_state,
            branch="B",
            deterministic=deterministic
        )

        # STEP 4: Compute outcomes for both branches (returns updated parallel states)
        parallel_A, outcomes_A = self.compute_parallel_outcomes(
            parallel_state=parallel_A,
            policy_net=policy_net,
            update_normalizer=False  # Don't update normalizer for parallel branches
        )
        # print(f"!!!!!!!!!!! MD in Parallel after action {parallel_A.moneydisposable.mean()}")
        parallel_B, outcomes_B = self.compute_parallel_outcomes(
            parallel_state=parallel_B,
            policy_net=policy_net,
            update_normalizer=False
        )

        # STEP 5: Choose and commit one branch
        chosen_branch = self.choose_branch(
            parallel_A, parallel_B, outcomes_A, outcomes_B, strategy=commit_strategy
        )

        chosen_parallel = parallel_A if chosen_branch == "A" else parallel_B
        self.commit_to_main(main_state, chosen_parallel, chosen_branch)

        return main_state, temp_state, (parallel_A, outcomes_A), (parallel_B, outcomes_B)

    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================

    def rollout(
        self,
        main_state: MainState,
        policy_net,
        n_steps: int,
        *,
        deterministic: bool = False,
        update_normalizer: bool = True,
        commit_strategy: Literal["random", "A", "B"] = "random"
    ) -> Tuple[MainState, list]:
        """
        Execute multiple steps and collect trajectory.

        Args:
            main_state: Initial MainState
            policy_net: Policy network
            n_steps: Number of steps to execute
            deterministic: If True, no random shocks
            update_normalizer: If True, update normalizer statistics
            commit_strategy: How to choose branch

        Returns:
            main_state: Final MainState after n_steps
            trajectory: List of (temp_state, parallel_A, parallel_B, outcomes_A, outcomes_B)
        """
        trajectory = []

        for _ in range(n_steps):
            main_state, temp_state, branch_A, branch_B = self.step(
                main_state=main_state,
                policy_net=policy_net,
                deterministic=deterministic,
                update_normalizer=update_normalizer,
                commit_strategy=commit_strategy
            )

            trajectory.append((temp_state, branch_A, branch_B))

        return main_state, trajectory
