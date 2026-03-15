"""
Goods market: households purchase consumption goods from firms.

MVP matching mechanism:
- Households decide desired consumption spending
- Firms offer goods at their posted price
- Proportional rationing if demand exceeds supply
- Transactions settled via ledger
"""

from __future__ import annotations

import numpy as np

from econosim.core.accounting import Ledger, round_money
from econosim.agents.household import Household
from econosim.agents.firm import Firm


class GoodsMarket:
    """Clears the goods market each period."""

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        # Period stats
        self.total_demand: float = 0.0
        self.total_supply: float = 0.0
        self.total_transacted: float = 0.0
        self.total_units_sold: float = 0.0
        self.average_price: float = 0.0

    def clear(
        self,
        households: list[Household],
        firms: list[Firm],
        period: int,
        rng: np.random.Generator,
    ) -> None:
        """Run goods market clearing for one period.

        Steps:
        1. Each household decides desired consumption spending
        2. Each firm offers inventory at posted price
        3. Proportional rationing across firms if needed
        4. Settle transactions via ledger
        """
        self.total_demand = 0.0
        self.total_supply = 0.0
        self.total_transacted = 0.0
        self.total_units_sold = 0.0

        # Collect supply: each firm offers its inventory at its price
        firm_supply: dict[str, tuple[Firm, float, float]] = {}  # firm_id -> (firm, qty, price)
        for firm in firms:
            qty = firm.inventory.quantity
            if qty > 0:
                firm_supply[firm.agent_id] = (firm, qty, firm.price)
                self.total_supply += qty

        if not firm_supply:
            return

        # Compute average price for CPI
        total_value = sum(qty * price for _, qty, price in firm_supply.values())
        self.average_price = round_money(total_value / self.total_supply) if self.total_supply > 0 else 0.0

        # Collect demand: each household decides spending
        buyer_demand: list[tuple[Household, float]] = []
        for hh in households:
            desired_spending = hh.desired_consumption()
            if desired_spending > 0.01:
                buyer_demand.append((hh, desired_spending))
                self.total_demand += desired_spending

        if not buyer_demand:
            return

        # Shuffle buyers for fairness
        rng.shuffle(buyer_demand)

        # Simple market clearing: each buyer visits firms in random order
        firm_ids = list(firm_supply.keys())

        for hh, budget in buyer_demand:
            remaining_budget = min(budget, hh.deposits)
            if remaining_budget < 0.01:
                continue

            # Shuffle firm order for this buyer
            visit_order = list(firm_ids)
            rng.shuffle(visit_order)

            for fid in visit_order:
                if remaining_budget < 0.01:
                    break
                firm, avail_qty, price = firm_supply[fid]
                if avail_qty < 0.01 or price < 0.01:
                    continue

                # How many units can buyer afford?
                affordable_qty = remaining_budget / price
                # How many units available?
                purchase_qty = min(affordable_qty, avail_qty)
                purchase_amount = round_money(purchase_qty * price)

                if purchase_amount < 0.01:
                    continue

                # Settle transaction
                self.ledger.transfer_deposits(
                    period=period,
                    from_id=hh.agent_id,
                    to_id=firm.agent_id,
                    amount=purchase_amount,
                    description=f"goods {hh.agent_id}->{firm.agent_id}",
                )

                # Update firm inventory
                firm.inventory.sell(purchase_qty)
                firm.revenue = round_money(firm.revenue + purchase_amount)
                firm.units_sold = round_money(firm.units_sold + purchase_qty)

                # Update household
                hh.consumption_spending = round_money(hh.consumption_spending + purchase_amount)

                # Update available supply tracking
                new_avail = round_money(avail_qty - purchase_qty)
                firm_supply[fid] = (firm, new_avail, price)

                remaining_budget = round_money(remaining_budget - purchase_amount)
                self.total_transacted = round_money(self.total_transacted + purchase_amount)
                self.total_units_sold = round_money(self.total_units_sold + purchase_qty)
