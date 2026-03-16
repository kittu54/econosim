"""Abstract policy interfaces for all agent types.

Each policy exposes a stable act(...) interface that takes agent state
and macro context, and returns a typed action dataclass.

Design principles:
- Policies are stateless functions of observable state
- Actions are typed dataclasses (not raw dicts)
- The simulator calls policy.act() and applies the returned action
- Policies can be swapped without changing engine logic
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


# --- Action types ---

@dataclass
class FirmAction:
    """Actions a firm can take each period."""
    vacancies: int = 0
    price_adjustment: float = 1.0  # multiplicative factor on current price
    wage_adjustment: float = 1.0   # multiplicative factor on current wage
    loan_request: float = 0.0      # amount to borrow (0 = don't borrow)


@dataclass
class HouseholdAction:
    """Actions a household can take each period."""
    consumption_fraction: float = 0.8  # fraction of budget to spend
    labor_participation: bool = True
    reservation_wage_adjustment: float = 1.0


@dataclass
class BankAction:
    """Actions the bank can take each period."""
    base_rate_adjustment: float = 0.0    # additive change to base rate
    capital_target_adjustment: float = 0.0  # additive change to capital adequacy target
    risk_premium_adjustment: float = 0.0


@dataclass
class GovernmentAction:
    """Actions the government can take each period."""
    tax_rate: float = 0.2
    transfer_per_unemployed: float = 50.0
    spending_per_period: float = 2000.0


# --- State containers for policy inputs ---

@dataclass
class MacroState:
    """Aggregate macroeconomic state visible to all agents."""
    period: int = 0
    gdp: float = 0.0
    gdp_growth: float = 0.0
    inflation_rate: float = 0.0
    unemployment_rate: float = 0.0
    avg_price: float = 10.0
    avg_wage: float = 100.0
    total_credit: float = 0.0
    credit_growth: float = 0.0
    bank_capital_ratio: float = 0.0
    lending_rate: float = 0.007


@dataclass
class FirmState:
    """Observable state of a single firm."""
    deposits: float = 0.0
    inventory: float = 0.0
    price: float = 10.0
    posted_wage: float = 100.0
    workers_count: int = 0
    revenue: float = 0.0
    prev_revenue: float = 0.0
    wage_bill: float = 0.0
    units_sold: float = 0.0
    prev_units_sold: float = 0.0
    total_debt: float = 0.0
    equity: float = 0.0
    labor_productivity: float = 5.0
    target_inventory_ratio: float = 0.2


@dataclass
class HouseholdState:
    """Observable state of a single household."""
    deposits: float = 0.0
    employed: bool = False
    wage_income: float = 0.0
    consumption_spending: float = 0.0
    total_debt: float = 0.0
    consumption_propensity: float = 0.8
    wealth_propensity: float = 0.2
    reservation_wage: float = 50.0


@dataclass
class BankState:
    """Observable state of the bank."""
    total_loans: float = 0.0
    total_deposits: float = 0.0
    equity: float = 0.0
    capital_ratio: float = 0.0
    lending_rate: float = 0.007
    interest_income: float = 0.0
    default_losses: float = 0.0
    active_loans_count: int = 0
    base_interest_rate: float = 0.005
    risk_premium: float = 0.002


@dataclass
class GovernmentState:
    """Observable state of the government."""
    deposits: float = 0.0
    tax_revenue: float = 0.0
    transfers_paid: float = 0.0
    goods_spending: float = 0.0
    budget_balance: float = 0.0
    income_tax_rate: float = 0.2
    transfer_per_unemployed: float = 50.0
    spending_per_period: float = 2000.0
    cumulative_debt: float = 0.0


# --- Abstract policy interfaces ---

class FirmPolicy(ABC):
    """Policy interface for firm agent decisions."""

    @abstractmethod
    def act(self, firm_state: FirmState, macro_state: MacroState) -> FirmAction:
        """Decide firm actions given current state."""
        ...

    def name(self) -> str:
        return self.__class__.__name__


class HouseholdPolicy(ABC):
    """Policy interface for household agent decisions."""

    @abstractmethod
    def act(self, hh_state: HouseholdState, macro_state: MacroState) -> HouseholdAction:
        """Decide household actions given current state."""
        ...

    def name(self) -> str:
        return self.__class__.__name__


class BankPolicy(ABC):
    """Policy interface for bank decisions."""

    @abstractmethod
    def act(self, bank_state: BankState, macro_state: MacroState) -> BankAction:
        """Decide bank actions given current state."""
        ...

    def name(self) -> str:
        return self.__class__.__name__


class GovernmentPolicy(ABC):
    """Policy interface for government decisions."""

    @abstractmethod
    def act(self, govt_state: GovernmentState, macro_state: MacroState) -> GovernmentAction:
        """Decide government actions given current state."""
        ...

    def name(self) -> str:
        return self.__class__.__name__
