"""Tests for goods inventory tracking."""

import pytest

from econosim.core.goods import Inventory


class TestInventory:
    def test_produce_increases_quantity(self):
        inv = Inventory("firm_0")
        inv.produce(100.0, 500.0)
        assert inv.quantity == 100.0
        assert inv.unit_cost == 5.0

    def test_produce_weighted_average_cost(self):
        inv = Inventory("firm_0")
        inv.produce(100.0, 500.0)  # cost=5.0/unit
        inv.produce(100.0, 1000.0)  # cost=10.0/unit
        assert inv.quantity == 200.0
        assert inv.unit_cost == 7.5  # (500+1000)/200

    def test_sell_decreases_quantity(self):
        inv = Inventory("firm_0")
        inv.produce(100.0, 500.0)
        cogs = inv.sell(40.0)
        assert inv.quantity == 60.0
        assert cogs == 200.0  # 40 * 5.0

    def test_sell_more_than_available_clips(self):
        inv = Inventory("firm_0")
        inv.produce(10.0, 50.0)
        cogs = inv.sell(20.0)
        assert inv.quantity == 0.0
        assert cogs == 50.0

    def test_negative_produce_raises(self):
        inv = Inventory("firm_0")
        with pytest.raises(ValueError):
            inv.produce(-10.0, 50.0)

    def test_negative_sell_raises(self):
        inv = Inventory("firm_0")
        with pytest.raises(ValueError):
            inv.sell(-5.0)

    def test_total_value(self):
        inv = Inventory("firm_0")
        inv.produce(50.0, 250.0)
        assert inv.total_value == 250.0
