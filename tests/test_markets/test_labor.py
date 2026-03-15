"""Tests for the Labor market module."""

import pytest
import numpy as np

from econosim.core.accounting import Ledger
from econosim.agents.household import Household
from econosim.agents.firm import Firm
from econosim.markets.labor import LaborMarket


@pytest.fixture
def ledger():
    return Ledger()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def households(ledger):
    return [
        Household(f"hh_{i:04d}", ledger, initial_deposits=1000.0, reservation_wage=50.0)
        for i in range(10)
    ]


@pytest.fixture
def firms(ledger):
    firms = []
    for i in range(2):
        f = Firm(f"firm_{i:03d}", ledger, initial_deposits=15000.0, initial_wage=60.0)
        f.prev_units_sold = 20.0
        f.prev_revenue = 200.0
        firms.append(f)
    return firms


class TestLaborMarketClearing:
    def test_matches_workers_to_firms(self, ledger, households, firms, rng):
        market = LaborMarket(ledger)
        market.clear(households, firms, period=0, rng=rng)
        assert market.total_matches > 0

    def test_employed_households_marked(self, ledger, households, firms, rng):
        market = LaborMarket(ledger)
        market.clear(households, firms, period=0, rng=rng)
        employed = [hh for hh in households if hh.employed]
        assert len(employed) == market.total_matches

    def test_wages_paid(self, ledger, households, firms, rng):
        market = LaborMarket(ledger)
        market.clear(households, firms, period=0, rng=rng)
        total_wages = sum(hh.wage_income for hh in households)
        assert total_wages > 0

    def test_firm_workers_list_populated(self, ledger, households, firms, rng):
        market = LaborMarket(ledger)
        market.clear(households, firms, period=0, rng=rng)
        total_workers = sum(len(f.workers) for f in firms)
        assert total_workers == market.total_matches

    def test_balance_sheets_balanced_after_clearing(self, ledger, households, firms, rng):
        market = LaborMarket(ledger)
        market.clear(households, firms, period=0, rng=rng)
        for hh in households:
            assert hh.balance_sheet.check_balanced()
        for f in firms:
            assert f.balance_sheet.check_balanced()

    def test_wage_below_reservation_no_match(self, ledger, rng):
        hhs = [Household("hh_0001", ledger, reservation_wage=100.0)]
        firms = [Firm("firm_001", ledger, initial_wage=50.0, initial_deposits=15000.0)]
        firms[0].prev_units_sold = 20.0
        firms[0].prev_revenue = 200.0
        market = LaborMarket(ledger)
        market.clear(hhs, firms, period=0, rng=rng)
        assert market.total_matches == 0

    def test_no_seekers_no_matches(self, ledger, firms, rng):
        hhs = [Household("hh_0001", ledger, labor_participation=False)]
        market = LaborMarket(ledger)
        market.clear(hhs, firms, period=0, rng=rng)
        assert market.total_matches == 0


class TestUnemploymentRate:
    def test_full_employment(self, ledger, rng):
        hhs = [Household(f"hh_{i:04d}", ledger, reservation_wage=10.0) for i in range(3)]
        firms = [Firm("firm_001", ledger, initial_deposits=50000.0, initial_wage=60.0)]
        firms[0].prev_units_sold = 100.0
        firms[0].prev_revenue = 1000.0
        market = LaborMarket(ledger)
        market.clear(hhs, firms, period=0, rng=rng)
        assert market.unemployment_rate < 0.5  # at least some employment

    def test_zero_seekers_zero_rate(self):
        market = LaborMarket(Ledger())
        market.total_seekers = 0
        market.total_matches = 0
        assert market.unemployment_rate == 0.0
