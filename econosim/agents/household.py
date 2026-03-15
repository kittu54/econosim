"""
Household agent: supplies labor, consumes goods, saves in deposits,
optionally borrows from the bank.

Rule-based decision logic for MVP:
- Participates in labor market if labor_participation flag is set
- Consumes a fraction of disposable income (consumption propensity)
- Saves the remainder as deposits
- Accepts/rejects wage offers based on reservation wage
"""

from __future__ import annotations

from typing import Any

from econosim.core.accounting import AccountType, Ledger
from econosim.agents.base import BaseAgent


class Household(BaseAgent):
    """A single household agent."""

    def __init__(
        self,
        agent_id: str,
        ledger: Ledger,
        initial_deposits: float = 1000.0,
        consumption_propensity: float = 0.8,
        wealth_propensity: float = 0.2,
        reservation_wage: float = 50.0,
        labor_participation: bool = True,
    ) -> None:
        self.consumption_propensity = consumption_propensity
        self.wealth_propensity = wealth_propensity
        self.reservation_wage = reservation_wage
        self.labor_participation = labor_participation
        self._initial_deposits = initial_deposits

        # Period state (reset each period)
        self.employed: bool = False
        self.employer_id: str | None = None
        self.wage_income: float = 0.0
        self.consumption_spending: float = 0.0
        self.taxes_paid: float = 0.0
        self.transfers_received: float = 0.0

        super().__init__(agent_id=agent_id, agent_type="household", ledger=ledger)

    def _setup_accounts(self) -> None:
        self.balance_sheet.add_account("deposits", AccountType.ASSET, self._initial_deposits)
        self.balance_sheet.add_account("loans_payable", AccountType.LIABILITY, 0.0)
        self.balance_sheet.add_account("equity", AccountType.EQUITY, self._initial_deposits)

    def reset_period_state(self) -> None:
        """Reset per-period flow variables."""
        self.wage_income = 0.0
        self.consumption_spending = 0.0
        self.taxes_paid = 0.0
        self.transfers_received = 0.0

    @property
    def disposable_income(self) -> float:
        return self.wage_income - self.taxes_paid + self.transfers_received

    def desired_consumption(self) -> float:
        """Buffer-stock consumption rule (standard SFC model):
        C = α1 * disposable_income + α2 * wealth

        α1 (consumption_propensity) controls income-based spending.
        α2 (wealth_propensity) ensures spending continues even when
        income drops, preventing deflationary death spirals.
        Cannot spend more than available deposits.
        """
        income_part = self.consumption_propensity * max(0.0, self.disposable_income)
        wealth_part = self.wealth_propensity * max(0.0, self.deposits)
        desired = income_part + wealth_part
        return min(desired, max(0.0, self.deposits))

    def wants_to_work(self) -> bool:
        return self.labor_participation and not self.employed

    def accept_wage(self, wage: float) -> bool:
        return wage >= self.reservation_wage

    @property
    def total_debt(self) -> float:
        return self.balance_sheet.get_account("loans_payable").balance

    def get_observation(self) -> dict[str, Any]:
        obs = super().get_observation()
        obs.update({
            "employed": self.employed,
            "wage_income": self.wage_income,
            "consumption_spending": self.consumption_spending,
            "disposable_income": self.disposable_income,
            "total_debt": self.total_debt,
            "consumption_propensity": self.consumption_propensity,
        })
        return obs
