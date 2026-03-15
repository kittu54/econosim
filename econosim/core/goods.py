"""
Goods and inventory tracking.

In the MVP there is a single consumption good. This module tracks
production, inventory levels, and goods conservation.
"""

from __future__ import annotations

from dataclasses import dataclass

from econosim.core.accounting import round_money


@dataclass
class Inventory:
    """Tracks goods inventory for a firm."""

    owner_id: str
    quantity: float = 0.0
    unit_cost: float = 0.0  # weighted-average cost per unit

    def produce(self, quantity: float, total_cost: float) -> None:
        """Add produced goods to inventory, updating weighted-average cost."""
        if quantity < 0:
            raise ValueError(f"Cannot produce negative quantity: {quantity}")
        if quantity == 0:
            return
        old_value = self.quantity * self.unit_cost
        new_value = old_value + total_cost
        self.quantity = round_money(self.quantity + quantity)
        self.unit_cost = round_money(new_value / self.quantity) if self.quantity > 0 else 0.0

    def sell(self, quantity: float) -> float:
        """Remove goods from inventory. Returns the cost of goods sold."""
        if quantity < 0:
            raise ValueError(f"Cannot sell negative quantity: {quantity}")
        quantity = min(quantity, self.quantity)
        cogs = round_money(quantity * self.unit_cost)
        self.quantity = round_money(self.quantity - quantity)
        if self.quantity < 0.01:
            self.quantity = 0.0
            self.unit_cost = 0.0
        return cogs

    @property
    def total_value(self) -> float:
        return round_money(self.quantity * self.unit_cost)

    def __repr__(self) -> str:
        return f"Inventory({self.owner_id}: qty={self.quantity:.2f}, cost={self.unit_cost:.2f})"
