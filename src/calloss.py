# src/calloss.py
"""
Loss functions for Bewley model training.

Implements three economic losses:
1. Forward-Backward (FB) consistency
2. Euler equation (intertemporal)
3. Labor first-order condition (FOC)
"""

import torch
from torch import Tensor
from typing import Dict, Optional
from argparse import Namespace


class LossCalculator:
    """
    Main loss calculator that orchestrates all loss components.

    Usage:
        loss_calc = LossCalculator(config, device)
        losses = loss_calc.compute_all_losses(
            consumption_t=consumption_t,
            labor_t=labor_t,
            ...
        )
        total_loss = losses["total"]
    """

    def __init__(self, config, device: str = "cpu"):
        """
        Initialize loss calculator with model parameters.

        Args:
            config: Configuration namespace with bewley_model parameters
            device: Device for tensor operations
        """
        self.device = device

        # Extract economic parameters from config
        self.beta = config.bewley_model.beta          # Discount factor (e.g., 0.975)
        self.theta = config.bewley_model.theta        # CRRA coefficient (e.g., 1.0)
        self.gamma = config.bewley_model.gamma        # Inverse Frisch elasticity (e.g., 2.0)
        self.delta = config.bewley_model.delta        # Depreciation rate (e.g., 0.06)
        self.tax_params = config.tax_params
        # Loss weights (can be made configurable)
        self.weight_fb = getattr(config.training, 'weight_fb', 1.0)
        self.weight_aux_mu = getattr(config.training, 'weight_aux_mu', 1.0)
        self.weight_labor = getattr(config.training, 'weight_labor', 1.0)

        # Numerical stability
        self.eps = 1e-8

        # Initialize individual loss components
        self.fb_loss_fn = FBLoss(eps=self.eps)
        self.aux_loss_mu_fn = AuxLossMu(taxparams=self.tax_params, beta=self.beta, theta=self.theta, eps=self.eps)
        self.labor_loss_fn = LaborFOCLoss(taxparams=self.tax_params, theta=self.theta, gamma=self.gamma, eps=self.eps)

    def compute_all_losses(
        self,
        # Current period (t)
        consumption_t: Tensor,
        labor_t: Tensor,
        ibt: Tensor,
        savings_ratio_t: Tensor,
        mu_t: Tensor,
        wage_t: Tensor,
        ret_t: Tensor,
        money_disposable_t: Tensor,
        ability_t: Tensor,
        # Next period (t+1) - two branches
        consumption_A_tp1: Tensor,
        consumption_B_tp1: Tensor,
        ibt_A_tp1: Tensor,
        ibt_B_tp1: Tensor,
        # Optional: masks for valid agents
        mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Compute all losses.

        Args:
            All tensors are (B, A) where B=batch, A=agents

        Returns:
            Dictionary with keys:
                - "fb": Forward-Backward loss
                - "euler": Euler equation loss
                - "labor": Labor FOC loss
                - "total": Weighted sum of all losses
        """
        # 1. Forward-Backward consistency loss
        fb_loss = self.fb_loss_fn(
            savings_ratio=savings_ratio_t,
            mu=mu_t,
        )

        # 2. Auxilary loss of multiplier
        aux_loss_mu = self.aux_loss_mu_fn(
            c0=consumption_t,
            mu0=mu_t,
            ret0=ret_t,
            savings_tp=savings_ratio_t*money_disposable_t,
            c1_A=consumption_A_tp1,
            c1_B=consumption_B_tp1,
            ibt_A=ibt_A_tp1,
            ibt_B=ibt_B_tp1
        )

        # 3. Labor FOC loss
        labor_loss = self.labor_loss_fn(
            consumption=consumption_t,
            ibt=ibt,
            wage=wage_t,
            labor=labor_t,
            ability=ability_t
        )

        # Apply weights
        fb_loss_weighted = self.weight_fb * fb_loss
        aux_mu_loss_weighted = self.weight_aux_mu * aux_loss_mu
        labor_loss_weighted = self.weight_labor * labor_loss

        # Total loss
        total_loss = fb_loss_weighted + aux_mu_loss_weighted + labor_loss_weighted

        return {
            "fb": fb_loss,
            "aux_mu": aux_loss_mu,
            "labor": labor_loss,
            "fb_weighted": fb_loss_weighted,
            "aux_mu_loss_weighted": aux_mu_loss_weighted,
            "labor_weighted": labor_loss_weighted,
            "total": total_loss
        }


# ============================================================================
# INDIVIDUAL LOSS COMPONENTS
# ============================================================================

class FBLoss:
    """
    Forward-Backward consistency loss.

    TODO: Specify exact formula based on your model's FOC.

    Possible formulations:
    1. KKT complementary slackness: mu * savings_ratio = 0
    2. Budget constraint residual
    3. First-order condition matching
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(
        self,
        savings_ratio: Tensor,
        mu: Tensor,
    ) -> Tensor:
        """
        Compute FB loss.

        Args:
            savings_ratio: (B, A) - fraction saved
            mu: (B, A) - Lagrange multiplier
            money_disposable: (B, A) - disposable money

        Returns:
            loss: scalar
        """
        r1 = savings_ratio
        r2 = (1-mu)

        return torch.mean((r1+r2-torch.sqrt(r1**2+r2**2))**2)

class AuxLossMu:
    def __init__(self, taxparams: Namespace, theta: float, beta: float = 1.0,
                 eps: float = 1e-8):
        self.taxparams = taxparams
        self.theta = theta
        self.beta = beta
        self.eps = eps

    def _safe_pow(self, base, exp):
        base_safe = torch.clamp(base, min=self.eps)
        return torch.exp(exp * torch.log(base_safe))

    def _safe_ratio(self, num, denom):
        denom_safe = torch.clamp(denom, min=self.eps)
        return num / denom_safe

    def eulerloss(self, c0, c1, mu0, ret0, ibt_parallel, savings_tp):
        cons_ratio = self._safe_ratio(c1, c0)
        inc_term = self._safe_pow(ibt_parallel, self.taxparams.income_tax_elasticity)
        save_term = self._safe_pow(savings_tp, self.taxparams.income_tax_elasticity)

        term1 = self.beta * self._safe_pow(cons_ratio, -self.theta)
        term2 = ret0 * (1-self.taxparams.tax_income) * inc_term
        term3 = 1-(1-self.taxparams.tax_saving) * save_term
        
        eulerloss = mu0-term1*(term2+term3)
        return eulerloss

    def __call__(self, c0, mu0, ret0, savings_tp, c1_A, c1_B, ibt_A, ibt_B):

        eulerloss_A = self.eulerloss(c0=c0, c1=c1_A, mu0=mu0, ret0=ret0, ibt_parallel=ibt_A,
                                     savings_tp=savings_tp)
        eulerloss_B = self.eulerloss(c0=c0, c1=c1_B, mu0=mu0, ret0=ret0, ibt_parallel=ibt_B,
                                     savings_tp=savings_tp)
        # --- stabilize before mean ---
        eulerloss_A = torch.nan_to_num(eulerloss_A, nan=0.0, posinf=1e6, neginf=-1e6)
        eulerloss_B = torch.nan_to_num(eulerloss_B, nan=0.0, posinf=1e6, neginf=-1e6)

        return torch.mean(eulerloss_A * eulerloss_B)

class LaborFOCLoss:
    """
    Numerically stable labor first-order-condition loss.

    F.O.C. form (target = 0):
        -labor**γ + (consumption**(-θ)/(1+τ_s)) * [wage*ability*(1-(1-τ_i) * ibt**(-ε_i))]

    Robust version keeps equilibrium point identical but clamps extreme values.
    """

    def __init__(self, taxparams: Namespace, theta: float, gamma: float, eps: float = 1e-8, base: float = 1, clip_val: float=1e4, scale: float=1):
        self.taxparams = taxparams
        self.theta = theta
        self.gamma = gamma
        self.eps = eps
        self.base = base
        self.clip_val = clip_val
        self.scale = scale

    def _safe_pow(self, base, exp):
        base = torch.clamp(torch.abs(base / self.scale), min=self.eps, max=self.clip_val)
        return torch.exp(exp * torch.log(base + self.eps))

    def __call__(
        self,
        consumption: Tensor,
        ibt: Tensor,
        wage: Tensor,
        ability: Tensor,
        labor: Tensor,
    ) -> Tensor:
        """
        Compute labor FOC loss.

        Args:
            consumption: (B, A)
            labor: (B, A)
            wage: (B, A)
            ability: (B, A)

        Returns:
            loss: scalar
        """
        labor_term = -self._safe_pow(labor, self.gamma)
        cons_term = self._safe_pow(consumption, -self.theta) / (1.0 + self.taxparams.tax_saving)  # FIXED: Removed negative sign
        ibt_term = self._safe_pow(ibt, -self.taxparams.income_tax_elasticity)
        tax_factor = (1.0 - self.taxparams.tax_income) * ibt_term
        prod_term = torch.clamp(wage * ability * tax_factor, min=-self.clip_val, max=self.clip_val)
        loss_foc = labor_term + cons_term * prod_term
        # --- clean up NaN / Inf ---
        loss_foc = torch.nan_to_num(loss_foc, nan=0.0, posinf=self.clip_val, neginf=-self.clip_val)

        # CRITICAL: Return scalar by taking mean
        return torch.mean(loss_foc ** 2)

