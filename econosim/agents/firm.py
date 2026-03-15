"""
Firm agent: hires labor, produces goods, sets prices, sells to households,
optionally borrows from the bank.

Rule-based decision logic for MVP:
- Posts wages and vacancies based on expected demand
- Produces goods using hired labor (linear production function)
- Adjusts price based on inventory relative to target
- Adjusts wage based on vacancy fill rate
"""

from __future__ import annotations

from typing import Any

from econosim.core.accounting import AccountType, Ledger, round_money
from econosim.core.goods import Inventory
from econosim.agents.base import BaseAgent


class Firm(BaseAgent):
    """A single firm agent."""

    def __init__(
        self,
        agent_id: str,
        ledger: Ledger,
        initial_deposits: float = 10000.0,
        initial_inventory: float = 100.0,
        initial_price: float = 10.0,
        initial_wage: float = 100.0,
        labor_productivity: float = 5.0,
        target_inventory_ratio: float = 0.2,
        price_adjustment_speed: float = 0.05,
        wage_adjustment_speed: float = 0.03,
        max_leverage: float = 3.0,
    ) -> None:
        self.price = initial_price
        self.posted_wage = initial_wage
        self.labor_productivity = labor_productivity
        self.target_inventory_ratio = target_inventory_ratio
        self.price_adjustment_speed = price_adjustment_speed
        self.wage_adjustment_speed = wage_adjustment_speed
        self.max_leverage = max_leverage
        self._initial_deposits = initial_deposits
        self._initial_inventory = initial_inventory

        # Period state
        self.workers: list[str] = []
        self.vacancies: int = 0
        self.vacancies_filled: int = 0
        self.production: float = 0.0
        self.revenue: float = 0.0
        self.prev_revenue: float = 0.0
        self.wage_bill: float = 0.0
        self.units_sold: float = 0.0
        self.prev_units_sold: float = 0.0

        super().__init__(agent_id=agent_id, agent_type="firm", ledger=ledger)

        self.inventory = Inventory(
            owner_id=agent_id,
            quantity=initial_inventory,
            unit_cost=initial_price * 0.5,
        )

    def _setup_accounts(self) -> None:
        self.balance_sheet.add_account("deposits", AccountType.ASSET, self._initial_deposits)
        self.balance_sheet.add_account("inventory_asset", AccountType.ASSET, 0.0)
        self.balance_sheet.add_account("loans_payable", AccountType.LIABILITY, 0.0)
        self.balance_sheet.add_account("equity", AccountType.EQUITY, self._initial_deposits)

    def reset_period_state(self) -> None:
        self.vacancies_filled = 0
        self.production = 0.0
        self.prev_revenue = self.revenue
        self.revenue = 0.0
        self.wage_bill = 0.0
        self.prev_units_sold = self.units_sold
        self.units_sold = 0.0

    # ── Decision rules ──────────────────────────────────────────

    def decide_vacancies(self) -> int:
        """Decide how many workers to hire this period.

        Uses both units-sold and revenue signals to estimate demand.
        Revenue-based estimation captures government service contracts
        and other non-goods income that should still drive hiring.
        Always tries to hire at least 1 worker if the firm can afford it.
        """
        # Units-based demand estimate
        expected_sales = max(self.prev_units_sold, 1.0)
        # Revenue-based demand estimate (converts revenue to equivalent units)
        revenue_units = self.prev_revenue / max(self.price, 0.01)
        # Use the larger signal
        demand_estimate = max(expected_sales, revenue_units)

        target_inv = demand_estimate * self.target_inventory_ratio
        production_needed = max(0.0, demand_estimate + target_inv - self.inventory.quantity)
        workers_needed = int(production_needed / max(self.labor_productivity, 0.01)) + 1
        # Don't hire more than we can afford (rough check)
        affordable = int(self.deposits / max(self.posted_wage, 1.0))
        # Minimum hiring floor: always try to hire at least 1 if affordable
        min_hire = 1 if affordable >= 1 else 0
        self.vacancies = max(min_hire, min(workers_needed, affordable))
        return self.vacancies

    def adjust_price(self) -> float:
        """Adjust price based on inventory-to-sales ratio.

        If inventory is above target → lower price.
        If inventory is below target AND there were actual sales → raise price.
        If there are no sales and no inventory, hold price (dead market ≠ excess demand).
        """
        expected_sales = max(self.prev_units_sold, 1.0)
        target_inv = expected_sales * self.target_inventory_ratio
        inv_ratio = self.inventory.quantity / max(target_inv, 1.0)

        if inv_ratio > 1.2:
            self.price = round_money(self.price * (1 - self.price_adjustment_speed))
        elif inv_ratio < 0.8 and self.prev_units_sold > 0.1:
            # Only raise prices if there was actual demand in previous period
            self.price = round_money(self.price * (1 + self.price_adjustment_speed))

        self.price = max(0.01, self.price)
        return self.price

    def adjust_wage(self) -> float:
        """Adjust wage based on vacancy fill rate."""
        if self.vacancies > 0:
            fill_rate = self.vacancies_filled / self.vacancies
            if fill_rate < 0.5:
                self.posted_wage = round_money(
                    self.posted_wage * (1 + self.wage_adjustment_speed)
                )
            elif fill_rate > 0.9 and self.vacancies_filled > 0:
                self.posted_wage = round_money(
                    self.posted_wage * (1 - self.wage_adjustment_speed * 0.5)
                )
        self.posted_wage = max(1.0, self.posted_wage)
        return self.posted_wage

    def produce(self) -> float:
        """Produce goods based on number of workers hired.

        Linear production function: output = workers * labor_productivity.
        Cost of production = wage bill (already paid).
        """
        num_workers = len(self.workers)
        output = round_money(num_workers * self.labor_productivity)
        self.production = output
        if output > 0:
            self.inventory.produce(output, self.wage_bill)
        return output

    @property
    def total_debt(self) -> float:
        return self.balance_sheet.get_account("loans_payable").balance

    @property
    def equity_value(self) -> float:
        return self.balance_sheet.get_account("equity").balance

    def can_borrow(self, amount: float) -> bool:
        """Check if firm can take on more debt within leverage constraint."""
        equity = max(self.equity_value, 1.0)
        return (self.total_debt + amount) / equity <= self.max_leverage

    def get_observation(self) -> dict[str, Any]:
        obs = super().get_observation()
        obs.update({
            "price": self.price,
            "posted_wage": self.posted_wage,
            "inventory": self.inventory.quantity,
            "workers": len(self.workers),
            "vacancies": self.vacancies,
            "production": self.production,
            "revenue": self.revenue,
            "wage_bill": self.wage_bill,
            "units_sold": self.units_sold,
            "total_debt": self.total_debt,
        })
        return obs
