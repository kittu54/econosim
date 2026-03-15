"""Tests for core accounting primitives: Account, BalanceSheet, Ledger, Transaction."""

import pytest

from econosim.core.accounting import (
    Account,
    AccountType,
    BalanceSheet,
    Ledger,
    Transaction,
    round_money,
)


class TestAccount:
    def test_asset_debit_increases(self):
        a = Account("cash", AccountType.ASSET, 100.0)
        a.debit(50.0)
        assert a.balance == 150.0

    def test_asset_credit_decreases(self):
        a = Account("cash", AccountType.ASSET, 100.0)
        a.credit(30.0)
        assert a.balance == 70.0

    def test_liability_credit_increases(self):
        a = Account("deposits", AccountType.LIABILITY, 100.0)
        a.credit(50.0)
        assert a.balance == 150.0

    def test_liability_debit_decreases(self):
        a = Account("deposits", AccountType.LIABILITY, 100.0)
        a.debit(30.0)
        assert a.balance == 70.0

    def test_equity_credit_increases(self):
        a = Account("equity", AccountType.EQUITY, 100.0)
        a.credit(25.0)
        assert a.balance == 125.0

    def test_negative_debit_raises(self):
        a = Account("cash", AccountType.ASSET, 100.0)
        with pytest.raises(ValueError):
            a.debit(-10.0)

    def test_negative_credit_raises(self):
        a = Account("cash", AccountType.ASSET, 100.0)
        with pytest.raises(ValueError):
            a.credit(-10.0)

    def test_rounding(self):
        a = Account("cash", AccountType.ASSET, 0.0)
        a.debit(1.005)
        assert a.balance == 1.0  # rounds to 2 decimal places


class TestBalanceSheet:
    def test_add_and_get_account(self):
        bs = BalanceSheet("test")
        bs.add_account("cash", AccountType.ASSET, 100.0)
        assert bs.get_account("cash").balance == 100.0

    def test_duplicate_account_raises(self):
        bs = BalanceSheet("test")
        bs.add_account("cash", AccountType.ASSET, 100.0)
        with pytest.raises(ValueError):
            bs.add_account("cash", AccountType.ASSET, 50.0)

    def test_missing_account_raises(self):
        bs = BalanceSheet("test")
        with pytest.raises(KeyError):
            bs.get_account("nonexistent")

    def test_total_assets(self):
        bs = BalanceSheet("test")
        bs.add_account("cash", AccountType.ASSET, 100.0)
        bs.add_account("inventory", AccountType.ASSET, 50.0)
        assert bs.total_assets == 150.0

    def test_total_liabilities(self):
        bs = BalanceSheet("test")
        bs.add_account("debt", AccountType.LIABILITY, 80.0)
        assert bs.total_liabilities == 80.0

    def test_net_worth(self):
        bs = BalanceSheet("test")
        bs.add_account("cash", AccountType.ASSET, 200.0)
        bs.add_account("debt", AccountType.LIABILITY, 50.0)
        assert bs.net_worth == 150.0

    def test_balanced_identity(self):
        bs = BalanceSheet("test")
        bs.add_account("cash", AccountType.ASSET, 200.0)
        bs.add_account("debt", AccountType.LIABILITY, 50.0)
        bs.add_account("equity", AccountType.EQUITY, 150.0)
        assert bs.check_balanced()

    def test_unbalanced_detected(self):
        bs = BalanceSheet("test")
        bs.add_account("cash", AccountType.ASSET, 200.0)
        bs.add_account("debt", AccountType.LIABILITY, 50.0)
        bs.add_account("equity", AccountType.EQUITY, 100.0)  # should be 150
        assert not bs.check_balanced()


class TestLedger:
    def _make_two_entity_ledger(self):
        ledger = Ledger()
        bs_a = BalanceSheet("A")
        bs_a.add_account("deposits", AccountType.ASSET, 100.0)
        bs_a.add_account("equity", AccountType.EQUITY, 100.0)
        bs_b = BalanceSheet("B")
        bs_b.add_account("deposits", AccountType.ASSET, 50.0)
        bs_b.add_account("equity", AccountType.EQUITY, 50.0)
        ledger.register_balance_sheet(bs_a)
        ledger.register_balance_sheet(bs_b)
        return ledger, bs_a, bs_b

    def test_transfer_deposits(self):
        ledger, bs_a, bs_b = self._make_two_entity_ledger()
        txs = ledger.transfer_deposits(
            period=0, from_id="A", to_id="B", amount=30.0, description="test"
        )
        assert len(txs) == 2
        assert bs_a.get_account("deposits").balance == 70.0
        assert bs_b.get_account("deposits").balance == 80.0
        # Equity adjusted: A-L=E preserved
        assert bs_a.get_account("equity").balance == 70.0
        assert bs_b.get_account("equity").balance == 80.0
        assert bs_a.check_balanced()
        assert bs_b.check_balanced()

    def test_zero_amount_returns_empty(self):
        ledger, _, _ = self._make_two_entity_ledger()
        txs = ledger.transfer_deposits(
            period=0, from_id="A", to_id="B", amount=0.0, description="noop"
        )
        assert txs == []

    def test_negative_amount_raises(self):
        ledger, _, _ = self._make_two_entity_ledger()
        with pytest.raises(ValueError):
            ledger.post(
                period=0,
                debit_owner="B",
                debit_account="deposits",
                credit_owner="A",
                credit_account="deposits",
                amount=-10.0,
                description="bad",
            )

    def test_transaction_recorded(self):
        ledger, _, _ = self._make_two_entity_ledger()
        ledger.transfer_deposits(
            period=0, from_id="A", to_id="B", amount=25.0, description="test"
        )
        # 2 transactions: sender-side and receiver-side
        assert len(ledger.transactions) == 2
        assert ledger.transactions[0].amount == 25.0
        assert ledger.transactions[1].amount == 25.0

    def test_deposit_transfer_conserves_total(self):
        ledger, bs_a, bs_b = self._make_two_entity_ledger()
        total_before = bs_a.get_account("deposits").balance + bs_b.get_account("deposits").balance
        ledger.transfer_deposits(
            period=0, from_id="A", to_id="B", amount=30.0, description="test"
        )
        total_after = bs_a.get_account("deposits").balance + bs_b.get_account("deposits").balance
        assert abs(total_before - total_after) < 0.01


class TestLoanAccounting:
    def _make_bank_borrower_ledger(self):
        ledger = Ledger()
        bank_bs = BalanceSheet("bank")
        bank_bs.add_account("reserves", AccountType.ASSET, 1000.0)
        bank_bs.add_account("loans", AccountType.ASSET, 0.0)
        bank_bs.add_account("deposits", AccountType.LIABILITY, 0.0)
        bank_bs.add_account("equity", AccountType.EQUITY, 1000.0)
        ledger.register_balance_sheet(bank_bs)

        borrower_bs = BalanceSheet("borrower")
        borrower_bs.add_account("deposits", AccountType.ASSET, 0.0)
        borrower_bs.add_account("loans_payable", AccountType.LIABILITY, 0.0)
        borrower_bs.add_account("equity", AccountType.EQUITY, 0.0)
        ledger.register_balance_sheet(borrower_bs)

        return ledger, bank_bs, borrower_bs

    def test_loan_issuance_creates_money(self):
        ledger, bank_bs, borrower_bs = self._make_bank_borrower_ledger()
        txs = ledger.issue_loan(period=0, bank_id="bank", borrower_id="borrower", amount=500.0)
        assert len(txs) == 2
        # Bank: loans asset up, deposits liability up
        assert bank_bs.get_account("loans").balance == 500.0
        assert bank_bs.get_account("deposits").balance == 500.0
        # Borrower: deposits up, loans_payable up
        assert borrower_bs.get_account("deposits").balance == 500.0
        assert borrower_bs.get_account("loans_payable").balance == 500.0

    def test_loan_issuance_preserves_balance(self):
        ledger, bank_bs, borrower_bs = self._make_bank_borrower_ledger()
        ledger.issue_loan(period=0, bank_id="bank", borrower_id="borrower", amount=500.0)
        assert bank_bs.check_balanced()
        assert borrower_bs.check_balanced()

    def test_loan_repayment_destroys_money(self):
        ledger, bank_bs, borrower_bs = self._make_bank_borrower_ledger()
        ledger.issue_loan(period=0, bank_id="bank", borrower_id="borrower", amount=500.0)
        ledger.repay_loan(period=1, bank_id="bank", borrower_id="borrower", amount=200.0)

        assert bank_bs.get_account("loans").balance == 300.0
        assert bank_bs.get_account("deposits").balance == 300.0
        assert borrower_bs.get_account("deposits").balance == 300.0
        assert borrower_bs.get_account("loans_payable").balance == 300.0

    def test_loan_repayment_preserves_balance(self):
        ledger, bank_bs, borrower_bs = self._make_bank_borrower_ledger()
        ledger.issue_loan(period=0, bank_id="bank", borrower_id="borrower", amount=500.0)
        ledger.repay_loan(period=1, bank_id="bank", borrower_id="borrower", amount=200.0)
        assert bank_bs.check_balanced()
        assert borrower_bs.check_balanced()

    def test_write_off_transfers_loss_to_equity(self):
        ledger, bank_bs, borrower_bs = self._make_bank_borrower_ledger()
        ledger.issue_loan(period=0, bank_id="bank", borrower_id="borrower", amount=500.0)
        ledger.write_off_loan(period=1, bank_id="bank", borrower_id="borrower", amount=500.0)

        # Bank absorbs loss
        assert bank_bs.get_account("loans").balance == 0.0
        assert bank_bs.get_account("equity").balance == 500.0  # was 1000, lost 500
        # Borrower debt forgiven
        assert borrower_bs.get_account("loans_payable").balance == 0.0
        assert borrower_bs.get_account("equity").balance == 500.0  # gained 500
