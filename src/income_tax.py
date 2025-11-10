# src/income_tax.py
"""
Income and tax calculations for Bewley model.

Implements:
1. Income before tax (labor + capital income)
2. Tax functions (income tax and savings tax with elasticity)
3. Money disposable calculation
4. Consumption from budget constraint
"""

from typing import Tuple, Optional
import torch
from torch import Tensor


def compute_income_before_tax(
    wage: Tensor,
    labor: Tensor,
    ability: Tensor,
    savings: Tensor,
    ret_lagged: Tensor,
    delta: float
) -> Tensor:
    """
    Compute total income before tax.

    Income has two components:
    1. Labor income: wage[t] * labor[t] * ability[t]
    2. Capital income: (1 - δ + ret[t-1]) * savings[t]

    Note: Capital income uses LAGGED return ret[t-1] because
    savings[t] earns the return determined in previous period.

    Args:
        wage: Wage rate (B, A) - can be broadcast from (B,)
        labor: Labor supply (B, A)
        ability: Ability/productivity (B, A)
        savings: Current capital holdings (B, A)
        ret_lagged: Return from previous period ret[t-1] (B, A) or scalar
        delta: Depreciation rate

    Returns:
        ibt: Income before tax (B, A)
    """
    # Labor income
    labor_income = wage * labor * ability

    # Capital income with gross return = 1 - δ + ret[t-1]
    gross_return = 1.0 - delta + ret_lagged
    capital_income = gross_return * savings

    # Total income before tax
    ibt = labor_income + capital_income

    return ibt


def compute_income_tax(
    income_before_tax: Tensor,
    tax_rate: float,
    elasticity: float,
    *,
    reference_income: Optional[float] = None
) -> Tensor:
    """
    Compute progressive income tax.

    Tax function: tax = tax_rate * (income / reference_income)^elasticity * income

    - elasticity = 1.0: proportional tax
    - elasticity > 1.0: progressive tax (tax rate increases with income)
    - elasticity < 1.0: regressive tax (tax rate decreases with income)

    Args:
        income_before_tax: Income before tax (B, A)
        tax_rate: Base tax rate (e.g., 0.2 for 20%)
        elasticity: Tax elasticity parameter (controls progressivity)
        reference_income: Reference income for normalization (default: mean of ibt)

    Returns:
        income_tax: Tax amount (B, A)
    """
    # Normalize income by reference level
    if reference_income is None:
        reference_income = income_before_tax.mean().item()

    # Avoid division by zero
    reference_income = max(reference_income, 1e-6)

    # Normalized income
    income_normalized = income_before_tax / reference_income

    # Progressive tax function
    # tax = base_rate * (income/ref)^elasticity * income
    effective_rate = tax_rate * (income_normalized ** elasticity)
    income_tax = effective_rate * income_before_tax

    # Ensure non-negative tax
    income_tax = torch.clamp(income_tax, min=0.0)

    return income_tax


def compute_savings_tax(
    savings: Tensor,
    tax_rate: float,
    elasticity: float,
    *,
    reference_savings: Optional[float] = None
) -> Tensor:
    """
    Compute progressive savings/wealth tax.

    Tax function: tax = tax_rate * (savings / reference_savings)^elasticity * savings

    Args:
        savings: Current savings/wealth (B, A)
        tax_rate: Base tax rate (e.g., 0.1 for 10%)
        elasticity: Tax elasticity parameter (controls progressivity)
        reference_savings: Reference savings for normalization (default: mean of savings)

    Returns:
        savings_tax: Tax amount (B, A)
    """
    # Normalize savings by reference level
    if reference_savings is None:
        reference_savings = savings.mean().item()

    # Avoid division by zero
    reference_savings = max(reference_savings, 1e-6)

    # Normalized savings
    savings_normalized = savings / reference_savings

    # Progressive tax function
    effective_rate = tax_rate * (savings_normalized ** elasticity)
    savings_tax = effective_rate * savings

    # Ensure non-negative tax
    savings_tax = torch.clamp(savings_tax, min=0.0)

    return savings_tax


def apply_taxes(
    income_before_tax: Tensor,
    savings: Tensor,
    tax_income_rate: float,
    tax_savings_rate: float,
    income_tax_elasticity: float,
    savings_tax_elasticity: float
) -> Tuple[Tensor, Tensor]:
    """
    Apply both income and savings taxes.

    Args:
        income_before_tax: Income before tax (B, A)
        savings: Current savings (B, A)
        tax_income_rate: Income tax rate
        tax_savings_rate: Savings tax rate
        income_tax_elasticity: Income tax elasticity
        savings_tax_elasticity: Savings tax elasticity

    Returns:
        income_tax: Income tax amount (B, A)
        savings_tax: Savings tax amount (B, A)
    """
    income_tax = compute_income_tax(
        income_before_tax,
        tax_income_rate,
        income_tax_elasticity
    )

    savings_tax = compute_savings_tax(
        savings,
        tax_savings_rate,
        savings_tax_elasticity
    )

    return income_tax, savings_tax


def compute_money_disposable(
    income_before_tax: Tensor,
    savings: Tensor,
    income_tax: Tensor,
    savings_tax: Tensor
) -> Tensor:
    """
    Compute disposable money after taxes.

    Formula: money_disposable = (ibt - income_tax) + (savings - savings_tax)

    This represents the total resources available for consumption and saving.

    Args:
        income_before_tax: Income before tax (B, A)
        savings: Current savings (B, A)
        income_tax: Income tax paid (B, A)
        savings_tax: Savings tax paid (B, A)

    Returns:
        money_disposable: Disposable money (B, A)
    """
    after_income_tax = income_before_tax - income_tax
    after_savings_tax = savings - savings_tax

    money_disposable = after_income_tax + after_savings_tax

    # Ensure non-negative (agents can't have negative disposable income)
    money_disposable = torch.clamp(money_disposable, min=0.0)

    return money_disposable


def compute_consumption(
    money_disposable: Tensor,
    savings_ratio: Tensor
) -> Tensor:
    """
    Compute consumption from budget constraint.

    Budget constraint:
    - consumption[t] = money_disposable[t] * (1 - savings_ratio)
    - savings[t+1] = money_disposable[t] * savings_ratio

    Args:
        money_disposable: Disposable money (B, A)
        savings_ratio: Fraction saved (B, A), should be in [0, 1]

    Returns:
        consumption: Consumption amount (B, A)
    """
    # Ensure savings_ratio is in valid range [0, 1]
    savings_ratio_clipped = torch.clamp(savings_ratio, min=0.0, max=1.0)

    # Consumption is what's not saved
    consumption = money_disposable * (1.0 - savings_ratio_clipped)

    return consumption


def compute_savings_next(
    money_disposable: Tensor,
    savings_ratio: Tensor
) -> Tensor:
    """
    Compute next period savings from budget constraint.

    Budget constraint: savings[t+1] = money_disposable[t] * savings_ratio

    Args:
        money_disposable: Disposable money (B, A)
        savings_ratio: Fraction saved (B, A), should be in [0, 1]

    Returns:
        savings_next: Next period savings (B, A)
    """
    # Ensure savings_ratio is in valid range [0, 1]
    savings_ratio_clipped = torch.clamp(savings_ratio, min=0.0, max=1.0)

    savings_next = money_disposable * savings_ratio_clipped

    return savings_next


def full_tax_and_income_pipeline(
    wage: Tensor,
    labor: Tensor,
    ability: Tensor,
    savings: Tensor,
    ret_lagged: Tensor,
    savings_ratio: Tensor,
    config
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Complete pipeline: from wages/labor to consumption and next savings.

    This is a convenience function that chains all steps together.

    Args:
        wage: Wage rate (B, A)
        labor: Labor supply (B, A)
        ability: Ability (B, A)
        savings: Current savings (B, A)
        ret_lagged: Lagged return ret[t-1] (B, A) or scalar
        savings_ratio: Savings ratio from policy (B, A)
        config: Config namespace with tax_params and milf_inputs

    Returns:
        income_before_tax: (B, A)
        money_disposable: (B, A)
        consumption: (B, A)
        savings_next: (B, A)
        income_tax: (B, A)
        savings_tax: (B, A)
    """
    # Get delta from config
    if hasattr(config.bewley_model, 'delta'):
        delta = config.bewley_model.delta
    elif hasattr(config, 'milf_inputs') and hasattr(config.milf_inputs, 'delta'):
        delta = config.milf_inputs.delta
    else:
        raise ValueError("Delta (depreciation rate) not found in config")

    # 1. Compute income before tax
    income_before_tax = compute_income_before_tax(
        wage, labor, ability, savings, ret_lagged, delta
    )

    # 2. Apply taxes
    income_tax, savings_tax = apply_taxes(
        income_before_tax,
        savings,
        config.tax_params.tax_income,
        config.tax_params.tax_saving,
        config.tax_params.income_tax_elasticity,
        config.tax_params.saving_tax_elasticity
    )

    # 3. Compute disposable money
    money_disposable = compute_money_disposable(
        income_before_tax, savings, income_tax, savings_tax
    )

    # 4. Compute consumption and next savings
    consumption = compute_consumption(money_disposable, savings_ratio)
    savings_next = compute_savings_next(money_disposable, savings_ratio)

    return income_before_tax, money_disposable, consumption, savings_next, income_tax, savings_tax
