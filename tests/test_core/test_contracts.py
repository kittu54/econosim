"""Tests for loan contracts and loan book."""

import pytest

from econosim.core.contracts import LoanBook, LoanContract, LoanStatus


class TestLoanContract:
    def test_initial_balance_equals_principal(self):
        loan = LoanContract(
            loan_id="L001", bank_id="bank", borrower_id="firm",
            principal=1000.0, interest_rate=0.01, term_periods=12,
            origination_period=0,
        )
        assert loan.remaining_balance == 1000.0

    def test_periodic_payment_positive(self):
        loan = LoanContract(
            loan_id="L001", bank_id="bank", borrower_id="firm",
            principal=1000.0, interest_rate=0.01, term_periods=12,
            origination_period=0,
        )
        assert loan.periodic_payment > 0

    def test_zero_rate_payment(self):
        loan = LoanContract(
            loan_id="L001", bank_id="bank", borrower_id="firm",
            principal=1200.0, interest_rate=0.0, term_periods=12,
            origination_period=0,
        )
        assert loan.periodic_payment == 100.0

    def test_payment_reduces_balance(self):
        loan = LoanContract(
            loan_id="L001", bank_id="bank", borrower_id="firm",
            principal=1000.0, interest_rate=0.01, term_periods=12,
            origination_period=0,
        )
        payment = loan.periodic_payment
        interest_paid, principal_paid = loan.record_payment(payment)
        assert interest_paid > 0
        assert principal_paid > 0
        assert loan.remaining_balance < 1000.0

    def test_full_repayment_marks_paid_off(self):
        loan = LoanContract(
            loan_id="L001", bank_id="bank", borrower_id="firm",
            principal=100.0, interest_rate=0.0, term_periods=10,
            origination_period=0,
        )
        for _ in range(10):
            loan.record_payment(loan.periodic_payment)
        assert loan.status == LoanStatus.PAID_OFF
        assert loan.remaining_balance == 0.0

    def test_missed_payment_increments_delinquency(self):
        loan = LoanContract(
            loan_id="L001", bank_id="bank", borrower_id="firm",
            principal=1000.0, interest_rate=0.01, term_periods=12,
            origination_period=0,
        )
        loan.record_payment(0.0)
        assert loan.periods_delinquent == 1

    def test_mark_default(self):
        loan = LoanContract(
            loan_id="L001", bank_id="bank", borrower_id="firm",
            principal=1000.0, interest_rate=0.01, term_periods=12,
            origination_period=0,
        )
        loan.mark_default()
        assert loan.status == LoanStatus.DEFAULTED
        assert not loan.is_active


class TestLoanBook:
    def test_create_loan(self):
        book = LoanBook()
        loan = book.create_loan("bank", "firm", 1000.0, 0.01, 12, 0)
        assert loan.loan_id == "loan_0000"
        assert loan.principal == 1000.0

    def test_active_loans(self):
        book = LoanBook()
        book.create_loan("bank", "firm_a", 1000.0, 0.01, 12, 0)
        book.create_loan("bank", "firm_b", 500.0, 0.01, 12, 0)
        assert len(book.active_loans()) == 2

    def test_total_outstanding(self):
        book = LoanBook()
        book.create_loan("bank", "firm_a", 1000.0, 0.01, 12, 0)
        book.create_loan("bank", "firm_b", 500.0, 0.01, 12, 0)
        assert book.total_outstanding() == 1500.0

    def test_loans_for_borrower(self):
        book = LoanBook()
        book.create_loan("bank", "firm_a", 1000.0, 0.01, 12, 0)
        book.create_loan("bank", "firm_b", 500.0, 0.01, 12, 0)
        book.create_loan("bank", "firm_a", 200.0, 0.01, 6, 0)
        assert len(book.loans_for_borrower("firm_a")) == 2
        assert len(book.loans_for_borrower("firm_b")) == 1
