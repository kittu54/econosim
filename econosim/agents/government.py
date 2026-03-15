"""
Government agent: collects taxes, pays transfers, purchases goods from firms.

Rule-based decision logic for MVP:
- Flat income tax on wage income
- Fixed transfer to unemployed households
- Fixed spending on goods from firms
- Currency issuer: creates money to fund deficit spending (SFC/MMT approach)
- Government equity can go negative (national debt)
"""

from __future__ import annotations

from typing import Any

from econosim.core.accounting import AccountType, Ledger, round_money
from econosim.agents.base import BaseAgent


class Government(BaseAgent):
    """The single government agent in the MVP economy."""

    def __init__(
        self,
        agent_id: str,
        ledger: Ledger,
        income_tax_rate: float = 0.2,
        transfer_per_unemployed: float = 50.0,
        spending_per_period: float = 500.0,
        initial_deposits: float = 100000.0,
    ) -> None:
        self.income_tax_rate = income_tax_rate
        self.transfer_per_unemployed = transfer_per_unemployed
        self.spending_per_period = spending_per_period
        self._initial_deposits = initial_deposits

        # Period state
        self.tax_revenue: float = 0.0
        self.transfers_paid: float = 0.0
        self.goods_spending: float = 0.0
        self.money_created: float = 0.0  # this period
        self.cumulative_money_created: float = 0.0  # all time

        super().__init__(agent_id=agent_id, agent_type="government", ledger=ledger)

    def _setup_accounts(self) -> None:
        self.balance_sheet.add_account("deposits", AccountType.ASSET, self._initial_deposits)
        self.balance_sheet.add_account("equity", AccountType.EQUITY, self._initial_deposits)

    def reset_period_state(self) -> None:
        self.tax_revenue = 0.0
        self.transfers_paid = 0.0
        self.goods_spending = 0.0
        self.money_created = 0.0

    def ensure_solvency(self, required: float, period: int) -> None:
        """Ensure government has enough deposits to cover required spending.

        As a currency issuer, the government creates money by crediting
        its own deposits and equity. This is the standard SFC/MMT mechanism:
        government spending injects money into the economy, taxes remove it.
        """
        shortfall = required - self.deposits
        if shortfall > 0:
            amount = round_money(shortfall)
            # Create money: increase both deposits (asset) and equity
            self.ledger.post(
                period=period,
                debit_owner=self.agent_id,
                debit_account="deposits",
                credit_owner=self.agent_id,
                credit_account="equity",
                amount=amount,
                description="sovereign money creation",
            )
            self.money_created = round_money(self.money_created + amount)
            self.cumulative_money_created = round_money(
                self.cumulative_money_created + amount
            )

    def compute_tax(self, wage_income: float) -> float:
        """Compute income tax on a household's wage income."""
        return round_money(wage_income * self.income_tax_rate)

    def collect_tax(self, period: int, household_id: str, amount: float) -> None:
        """Collect tax from a household via deposit transfer."""
        if amount <= 0:
            return
        self.ledger.transfer_deposits(
            period=period,
            from_id=household_id,
            to_id=self.agent_id,
            amount=amount,
            description=f"income tax from {household_id}",
        )
        self.tax_revenue = round_money(self.tax_revenue + amount)

    def pay_transfer(self, period: int, household_id: str, amount: float) -> None:
        """Pay unemployment transfer to a household."""
        if amount <= 0:
            return
        self.ledger.transfer_deposits(
            period=period,
            from_id=self.agent_id,
            to_id=household_id,
            amount=amount,
            description=f"transfer to {household_id}",
        )
        self.transfers_paid = round_money(self.transfers_paid + amount)

    def purchase_goods(self, period: int, firm_id: str, amount: float) -> None:
        """Government purchases goods from a firm (demand injection)."""
        if amount <= 0:
            return
        self.ledger.transfer_deposits(
            period=period,
            from_id=self.agent_id,
            to_id=firm_id,
            amount=amount,
            description=f"govt purchase from {firm_id}",
        )
        self.goods_spending = round_money(self.goods_spending + amount)

    @property
    def budget_balance(self) -> float:
        return round_money(self.tax_revenue - self.transfers_paid - self.goods_spending)

    def get_observation(self) -> dict[str, Any]:
        obs = super().get_observation()
        obs.update({
            "tax_revenue": self.tax_revenue,
            "transfers_paid": self.transfers_paid,
            "goods_spending": self.goods_spending,
            "budget_balance": self.budget_balance,
            "income_tax_rate": self.income_tax_rate,
            "money_created": self.money_created,
            "cumulative_money_created": self.cumulative_money_created,
        })
        return obs
