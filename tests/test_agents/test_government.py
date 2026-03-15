"""Tests for the Government agent."""

import pytest

from econosim.core.accounting import Ledger
from econosim.agents.government import Government
from econosim.agents.household import Household


@pytest.fixture
def ledger():
    return Ledger()


@pytest.fixture
def govt(ledger):
    return Government(
        agent_id="govt_0",
        ledger=ledger,
        income_tax_rate=0.2,
        transfer_per_unemployed=50.0,
        spending_per_period=2000.0,
        initial_deposits=100000.0,
    )


@pytest.fixture
def household(ledger):
    return Household("hh_0001", ledger, initial_deposits=5000.0)


class TestGovernmentInit:
    def test_initial_deposits(self, govt):
        assert govt.deposits == 100000.0

    def test_balance_sheet_balanced(self, govt):
        assert govt.balance_sheet.check_balanced()

    def test_initial_equity(self, govt):
        assert govt.balance_sheet.total_equity == pytest.approx(100000.0)


class TestGovernmentTax:
    def test_compute_tax(self, govt):
        assert govt.compute_tax(100.0) == pytest.approx(20.0)

    def test_collect_tax(self, govt, household):
        old_govt = govt.deposits
        old_hh = household.deposits
        govt.collect_tax(period=0, household_id="hh_0001", amount=100.0)
        assert govt.deposits == pytest.approx(old_govt + 100.0)
        assert household.deposits == pytest.approx(old_hh - 100.0)
        assert govt.tax_revenue == pytest.approx(100.0)

    def test_collect_zero_tax(self, govt, household):
        old_govt = govt.deposits
        govt.collect_tax(period=0, household_id="hh_0001", amount=0.0)
        assert govt.deposits == old_govt


class TestGovernmentTransfers:
    def test_pay_transfer(self, govt, household):
        old_hh = household.deposits
        govt.pay_transfer(period=0, household_id="hh_0001", amount=50.0)
        assert household.deposits == pytest.approx(old_hh + 50.0)
        assert govt.transfers_paid == pytest.approx(50.0)


class TestSovereignMoney:
    def test_ensure_solvency_no_creation(self, govt):
        govt.ensure_solvency(1000.0, period=0)
        assert govt.money_created == 0.0

    def test_ensure_solvency_creates_money(self, govt):
        govt.ensure_solvency(200000.0, period=0)
        assert govt.money_created > 0.0
        assert govt.deposits >= 200000.0
        assert govt.balance_sheet.check_balanced()

    def test_cumulative_tracking(self, govt):
        govt.ensure_solvency(200000.0, period=0)
        first = govt.cumulative_money_created
        govt.reset_period_state()
        govt.ensure_solvency(300000.0, period=1)
        assert govt.cumulative_money_created > first


class TestGovernmentBudget:
    def test_budget_balance(self, govt, household):
        govt.collect_tax(0, "hh_0001", 200.0)
        govt.pay_transfer(0, "hh_0001", 50.0)
        assert govt.budget_balance == pytest.approx(200.0 - 50.0 - 0.0)


class TestGovernmentObservation:
    def test_observation_keys(self, govt):
        obs = govt.get_observation()
        assert "tax_revenue" in obs
        assert "budget_balance" in obs
        assert "income_tax_rate" in obs
        assert "money_created" in obs
