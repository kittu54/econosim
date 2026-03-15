"""Tests for the Firm agent."""

import pytest

from econosim.core.accounting import Ledger
from econosim.agents.firm import Firm


@pytest.fixture
def ledger():
    return Ledger()


@pytest.fixture
def firm(ledger):
    return Firm(
        agent_id="firm_001",
        ledger=ledger,
        initial_deposits=15000.0,
        initial_inventory=100.0,
        initial_price=10.0,
        initial_wage=60.0,
        labor_productivity=8.0,
    )


class TestFirmInit:
    def test_initial_deposits(self, firm):
        assert firm.deposits == 15000.0

    def test_initial_inventory(self, firm):
        assert firm.inventory.quantity == 100.0

    def test_initial_price(self, firm):
        assert firm.price == 10.0

    def test_balance_sheet_balanced(self, firm):
        assert firm.balance_sheet.check_balanced()

    def test_inventory_asset_on_balance_sheet(self, firm):
        inv_account = firm.balance_sheet.get_account("inventory_asset")
        assert inv_account.balance == pytest.approx(firm.inventory.total_value, abs=0.01)

    def test_equity_includes_inventory(self, firm):
        expected_equity = 15000.0 + firm.inventory.total_value
        assert firm.equity_value == pytest.approx(expected_equity, abs=0.01)


class TestFirmDecisions:
    def test_decide_vacancies_positive(self, firm):
        firm.prev_units_sold = 50.0
        firm.prev_revenue = 500.0
        vacancies = firm.decide_vacancies()
        assert vacancies >= 1

    def test_decide_vacancies_with_zero_demand(self, firm):
        firm.prev_units_sold = 0.0
        firm.prev_revenue = 0.0
        vacancies = firm.decide_vacancies()
        # Should still hire at least 1 if affordable
        assert vacancies >= 0

    def test_adjust_price_lowers_on_excess_inventory(self, firm):
        firm.prev_units_sold = 10.0
        firm.inventory.quantity = 100.0  # very high relative to sales
        old_price = firm.price
        firm.adjust_price()
        assert firm.price < old_price

    def test_adjust_price_raises_on_low_inventory(self, firm):
        firm.prev_units_sold = 50.0
        firm.inventory.quantity = 1.0  # very low relative to sales
        old_price = firm.price
        firm.adjust_price()
        assert firm.price > old_price

    def test_adjust_price_holds_when_no_sales(self, firm):
        firm.prev_units_sold = 0.0
        firm.inventory.quantity = 0.0
        old_price = firm.price
        firm.adjust_price()
        assert firm.price == old_price

    def test_adjust_wage_raises_on_low_fill_rate(self, firm):
        firm.vacancies = 10
        firm.vacancies_filled = 2  # 20% fill rate
        old_wage = firm.posted_wage
        firm.adjust_wage()
        assert firm.posted_wage > old_wage

    def test_adjust_wage_lowers_on_high_fill_rate(self, firm):
        firm.vacancies = 10
        firm.vacancies_filled = 10  # 100% fill rate
        old_wage = firm.posted_wage
        firm.adjust_wage()
        assert firm.posted_wage < old_wage


class TestFirmProduction:
    def test_produce_with_workers(self, firm):
        firm.workers = ["hh_0001", "hh_0002"]
        firm.wage_bill = 120.0
        old_qty = firm.inventory.quantity
        output = firm.produce()
        assert output == pytest.approx(16.0)  # 2 * 8.0
        assert firm.inventory.quantity == pytest.approx(old_qty + 16.0)

    def test_produce_no_workers(self, firm):
        firm.workers = []
        output = firm.produce()
        assert output == 0.0

    def test_produce_syncs_inventory_asset(self, firm):
        firm.workers = ["hh_0001"]
        firm.wage_bill = 60.0
        firm.produce()
        inv_account = firm.balance_sheet.get_account("inventory_asset")
        assert inv_account.balance == pytest.approx(firm.inventory.total_value, abs=0.01)
        assert firm.balance_sheet.check_balanced()


class TestFirmPeriodState:
    def test_reset_preserves_prev_values(self, firm):
        firm.units_sold = 50.0
        firm.revenue = 500.0
        firm.reset_period_state()
        assert firm.prev_units_sold == 50.0
        assert firm.prev_revenue == 500.0
        assert firm.units_sold == 0.0
        assert firm.revenue == 0.0


class TestFirmBorrowing:
    def test_can_borrow_within_leverage(self, firm):
        assert firm.can_borrow(1000.0) is True

    def test_cannot_borrow_excessive(self, firm):
        # Max leverage = 3.0, equity ~15500, so max debt = 46500
        # But first, set up some existing debt
        firm.balance_sheet.get_account("loans_payable").credit(50000.0)
        assert firm.can_borrow(10000.0) is False


class TestFirmObservation:
    def test_observation_keys(self, firm):
        obs = firm.get_observation()
        assert "price" in obs
        assert "posted_wage" in obs
        assert "inventory" in obs
        assert "revenue" in obs
        assert "total_debt" in obs
