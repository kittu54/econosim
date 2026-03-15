"""Tests for the Goods market module."""

import pytest
import numpy as np

from econosim.core.accounting import Ledger
from econosim.agents.household import Household
from econosim.agents.firm import Firm
from econosim.markets.goods import GoodsMarket


@pytest.fixture
def ledger():
    return Ledger()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def households(ledger):
    return [
        Household(f"hh_{i:04d}", ledger, initial_deposits=1000.0,
                  consumption_propensity=0.8, wealth_propensity=0.4)
        for i in range(5)
    ]


@pytest.fixture
def firms(ledger):
    return [
        Firm(f"firm_{i:03d}", ledger, initial_deposits=15000.0,
             initial_inventory=100.0, initial_price=10.0)
        for i in range(2)
    ]


class TestGoodsMarketClearing:
    def test_goods_are_sold(self, ledger, households, firms, rng):
        market = GoodsMarket(ledger)
        market.clear(households, firms, period=0, rng=rng)
        assert market.total_transacted > 0
        assert market.total_units_sold > 0

    def test_inventory_decreases(self, ledger, households, firms, rng):
        total_before = sum(f.inventory.quantity for f in firms)
        market = GoodsMarket(ledger)
        market.clear(households, firms, period=0, rng=rng)
        total_after = sum(f.inventory.quantity for f in firms)
        assert total_after < total_before

    def test_household_deposits_decrease(self, ledger, households, firms, rng):
        total_before = sum(hh.deposits for hh in households)
        market = GoodsMarket(ledger)
        market.clear(households, firms, period=0, rng=rng)
        total_after = sum(hh.deposits for hh in households)
        assert total_after < total_before

    def test_firm_revenue_increases(self, ledger, households, firms, rng):
        market = GoodsMarket(ledger)
        market.clear(households, firms, period=0, rng=rng)
        total_revenue = sum(f.revenue for f in firms)
        assert total_revenue > 0

    def test_balance_sheets_balanced(self, ledger, households, firms, rng):
        market = GoodsMarket(ledger)
        market.clear(households, firms, period=0, rng=rng)
        for hh in households:
            assert hh.balance_sheet.check_balanced()
        for f in firms:
            assert f.balance_sheet.check_balanced()

    def test_no_inventory_no_sales(self, ledger, households, rng):
        firms = [Firm("firm_001", Ledger(), initial_inventory=0.0)]
        # Use a fresh ledger for this specific test since we need empty inventory
        l2 = Ledger()
        hhs = [Household("hh_0001", l2, initial_deposits=1000.0)]
        fms = [Firm("firm_001", l2, initial_inventory=0.0)]
        market = GoodsMarket(l2)
        market.clear(hhs, fms, period=0, rng=rng)
        assert market.total_units_sold == 0.0

    def test_no_demand_no_sales(self, ledger, firms, rng):
        hhs = [Household("hh_0001", ledger, initial_deposits=0.0)]
        market = GoodsMarket(ledger)
        market.clear(hhs, firms, period=0, rng=rng)
        assert market.total_transacted == 0.0
