"""Tests for the Credit market module."""

import pytest

from econosim.core.accounting import Ledger
from econosim.agents.firm import Firm
from econosim.agents.bank import Bank
from econosim.markets.credit import CreditMarket


@pytest.fixture
def ledger():
    return Ledger()


@pytest.fixture
def bank(ledger):
    return Bank("bank_0", ledger, initial_reserves=20000.0)


@pytest.fixture
def firms(ledger):
    firms = []
    for i in range(3):
        f = Firm(f"firm_{i:03d}", ledger, initial_deposits=10.0, initial_wage=60.0)
        f.vacancies = 3  # Set vacancies so expected_wage_bill = 60 * 3 = 180 > 10
        firms.append(f)
    return firms


class TestCreditMarketClearing:
    def test_firms_with_shortfall_apply(self, ledger, bank, firms):
        # Firms have only 10 deposits but need 180+ for wages (vacancies=3)
        for f in firms:
            f.prev_units_sold = 20.0
            f.prev_revenue = 200.0
        market = CreditMarket(ledger)
        market.clear(firms, bank, period=0)
        assert market.applications > 0

    def test_loans_issued(self, ledger, bank, firms):
        for f in firms:
            f.prev_units_sold = 20.0
            f.prev_revenue = 200.0
        market = CreditMarket(ledger)
        market.clear(firms, bank, period=0)
        assert market.total_issued > 0

    def test_no_shortfall_no_application(self, ledger, bank):
        # Firms have plenty of deposits
        rich_firms = [
            Firm(f"firm_{i:03d}", ledger, initial_deposits=100000.0, initial_wage=60.0)
            for i in range(2)
        ]
        for f in rich_firms:
            f.prev_units_sold = 20.0
            f.prev_revenue = 200.0
        market = CreditMarket(ledger)
        market.clear(rich_firms, bank, period=0)
        assert market.applications == 0

    def test_balance_sheets_balanced_after_lending(self, ledger, bank, firms):
        for f in firms:
            f.prev_units_sold = 20.0
            f.prev_revenue = 200.0
        market = CreditMarket(ledger)
        market.clear(firms, bank, period=0)
        assert bank.balance_sheet.check_balanced()
        for f in firms:
            assert f.balance_sheet.check_balanced()
