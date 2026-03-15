"""Tests for the Bank agent."""

import pytest

from econosim.core.accounting import Ledger
from econosim.agents.bank import Bank
from econosim.agents.firm import Firm


@pytest.fixture
def ledger():
    return Ledger()


@pytest.fixture
def bank(ledger):
    return Bank(
        agent_id="bank_0",
        ledger=ledger,
        initial_equity=50000.0,
        initial_reserves=20000.0,
        base_interest_rate=0.005,
        risk_premium=0.002,
        capital_adequacy_ratio=0.08,
        default_threshold_periods=3,
        loan_term_periods=12,
    )


@pytest.fixture
def firm(ledger):
    return Firm("firm_001", ledger, initial_deposits=15000.0)


class TestBankInit:
    def test_balance_sheet_balanced(self, bank):
        assert bank.balance_sheet.check_balanced()

    def test_initial_equity(self, bank):
        assert bank.equity_value == 20000.0  # equity = reserves

    def test_lending_rate(self, bank):
        assert bank.lending_rate == pytest.approx(0.007)


class TestBankLending:
    def test_can_lend_within_capacity(self, bank):
        assert bank.can_lend(1000.0) is True

    def test_approve_loan(self, bank, firm):
        contract = bank.approve_loan(
            borrower_id="firm_001",
            requested_amount=5000.0,
            borrower_deposits=15000.0,
            borrower_debt=0.0,
            period=0,
        )
        assert contract is not None
        assert contract.principal == 5000.0
        assert bank.loans_issued_this_period == pytest.approx(5000.0)

    def test_reject_loan_capital_inadequacy(self, bank, firm):
        # Issue huge loan first to exhaust capital
        bank.approve_loan("firm_001", 200000.0, 15000.0, 0.0, 0)
        result = bank.approve_loan("firm_001", 100000.0, 15000.0, 200000.0, 0)
        assert result is None

    def test_reject_zero_amount(self, bank):
        result = bank.approve_loan("firm_001", 0.0, 15000.0, 0.0, 0)
        assert result is None


class TestBankDefaults:
    def test_default_after_threshold(self, bank, firm):
        contract = bank.approve_loan("firm_001", 1000.0, 15000.0, 0.0, 0)
        # Simulate missed payments
        for _ in range(3):
            contract.periods_delinquent += 1
        defaulted = bank.process_defaults(period=3)
        assert len(defaulted) == 1

    def test_no_default_before_threshold(self, bank, firm):
        contract = bank.approve_loan("firm_001", 1000.0, 15000.0, 0.0, 0)
        contract.periods_delinquent = 1
        defaulted = bank.process_defaults(period=1)
        assert len(defaulted) == 0


class TestBankObservation:
    def test_observation_keys(self, bank):
        obs = bank.get_observation()
        assert "total_loans" in obs
        assert "equity" in obs
        assert "capital_ratio" in obs
        assert "lending_rate" in obs
