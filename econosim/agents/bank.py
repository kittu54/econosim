"""
Bank agent: accepts deposits (liability), issues loans (asset),
manages reserves, enforces capital adequacy, handles defaults.

Rule-based decision logic for MVP:
- Approves loans if capital adequacy ratio is met and borrower passes checks
- Charges base rate + risk premium
- Tracks delinquency and triggers default after threshold
- Absorbs losses from defaults through equity
"""

from __future__ import annotations

from typing import Any

from econosim.core.accounting import AccountType, Ledger, round_money
from econosim.core.contracts import LoanBook, LoanContract, LoanStatus
from econosim.agents.base import BaseAgent


class Bank(BaseAgent):
    """The single bank in the MVP economy."""

    def __init__(
        self,
        agent_id: str,
        ledger: Ledger,
        initial_equity: float = 50000.0,
        initial_reserves: float = 20000.0,
        base_interest_rate: float = 0.005,
        risk_premium: float = 0.002,
        capital_adequacy_ratio: float = 0.08,
        max_loan_to_value: float = 0.8,
        default_threshold_periods: int = 3,
        loan_term_periods: int = 12,
    ) -> None:
        self.base_interest_rate = base_interest_rate
        self.risk_premium = risk_premium
        self.capital_adequacy_ratio = capital_adequacy_ratio
        self.max_loan_to_value = max_loan_to_value
        self.default_threshold_periods = default_threshold_periods
        self.loan_term_periods = loan_term_periods
        self._initial_equity = initial_equity
        self._initial_reserves = initial_reserves

        self.loan_book = LoanBook()

        # Period state
        self.interest_income: float = 0.0
        self.default_losses: float = 0.0
        self.loans_issued_this_period: float = 0.0

        super().__init__(agent_id=agent_id, agent_type="bank", ledger=ledger)

    def _setup_accounts(self) -> None:
        self.balance_sheet.add_account("reserves", AccountType.ASSET, self._initial_reserves)
        self.balance_sheet.add_account("loans", AccountType.ASSET, 0.0)
        self.balance_sheet.add_account("deposits", AccountType.LIABILITY, 0.0)
        self.balance_sheet.add_account("equity", AccountType.EQUITY, self._initial_reserves)

    def reset_period_state(self) -> None:
        self.interest_income = 0.0
        self.default_losses = 0.0
        self.loans_issued_this_period = 0.0

    @property
    def lending_rate(self) -> float:
        return self.base_interest_rate + self.risk_premium

    @property
    def total_loans(self) -> float:
        return self.balance_sheet.get_account("loans").balance

    @property
    def total_deposits_liability(self) -> float:
        return self.balance_sheet.get_account("deposits").balance

    @property
    def equity_value(self) -> float:
        return self.balance_sheet.get_account("equity").balance

    @property
    def capital_ratio(self) -> float:
        loans = max(self.total_loans, 1.0)
        return self.equity_value / loans

    def can_lend(self, amount: float) -> bool:
        """Check if bank can issue a new loan while maintaining capital adequacy."""
        projected_loans = self.total_loans + amount
        if projected_loans <= 0:
            return True
        projected_ratio = self.equity_value / projected_loans
        return projected_ratio >= self.capital_adequacy_ratio

    def approve_loan(
        self,
        borrower_id: str,
        requested_amount: float,
        borrower_deposits: float,
        borrower_debt: float,
        period: int,
    ) -> LoanContract | None:
        """Evaluate and potentially approve a loan application.

        Returns LoanContract if approved, None if rejected.
        """
        if requested_amount <= 0:
            return None

        if not self.can_lend(requested_amount):
            return None

        # Simple creditworthiness: debt-to-income proxy
        if borrower_debt > 0 and borrower_deposits < borrower_debt * 0.1:
            return None

        contract = self.loan_book.create_loan(
            bank_id=self.agent_id,
            borrower_id=borrower_id,
            principal=requested_amount,
            interest_rate=self.lending_rate,
            term_periods=self.loan_term_periods,
            origination_period=period,
        )

        # Accounting: create money via loan issuance
        self.ledger.issue_loan(
            period=period,
            bank_id=self.agent_id,
            borrower_id=borrower_id,
            amount=requested_amount,
            description=f"loan {contract.loan_id}",
        )

        self.loans_issued_this_period = round_money(
            self.loans_issued_this_period + requested_amount
        )
        return contract

    def process_loan_payments(self, period: int) -> dict[str, float]:
        """Process all active loan payments for the period.

        Returns dict of borrower_id -> total_payment_made.
        """
        payments: dict[str, float] = {}

        for loan in self.loan_book.active_loans():
            payment_due = loan.periodic_payment
            borrower_bs = self.ledger.get_balance_sheet(loan.borrower_id)
            borrower_deposits = borrower_bs.get_account("deposits").balance

            # Pay what they can afford
            actual_payment = min(payment_due, max(0.0, borrower_deposits))

            if actual_payment > 0.01:
                interest_paid, principal_paid = loan.record_payment(actual_payment)

                # Interest payment: transfer from borrower to bank
                # transfer_deposits handles both deposit movement and equity
                # adjustments on both sides (borrower expense, bank income)
                if interest_paid > 0:
                    self.ledger.transfer_deposits(
                        period=period,
                        from_id=loan.borrower_id,
                        to_id=self.agent_id,
                        amount=interest_paid,
                        description=f"interest on {loan.loan_id}",
                    )
                    self.interest_income = round_money(self.interest_income + interest_paid)

                # Principal repayment: destroys money
                if principal_paid > 0:
                    self.ledger.repay_loan(
                        period=period,
                        bank_id=self.agent_id,
                        borrower_id=loan.borrower_id,
                        amount=principal_paid,
                        description=f"principal on {loan.loan_id}",
                    )

                payments[loan.borrower_id] = payments.get(loan.borrower_id, 0.0) + actual_payment
            else:
                loan.periods_delinquent += 1
                if loan.periods_delinquent >= max(1, self.default_threshold_periods // 2):
                    loan.mark_delinquent()

        return payments

    def process_defaults(self, period: int) -> list[str]:
        """Check for and process loan defaults. Returns list of defaulted loan IDs."""
        defaulted: list[str] = []

        for loan in self.loan_book.active_loans():
            if loan.periods_delinquent >= self.default_threshold_periods:
                loan.mark_default()
                loss = loan.remaining_balance

                self.ledger.write_off_loan(
                    period=period,
                    bank_id=self.agent_id,
                    borrower_id=loan.borrower_id,
                    amount=loss,
                    description=f"default {loan.loan_id}",
                )

                self.default_losses = round_money(self.default_losses + loss)
                defaulted.append(loan.loan_id)

        return defaulted

    def get_observation(self) -> dict[str, Any]:
        obs = super().get_observation()
        obs.update({
            "total_loans": self.total_loans,
            "total_deposits_liability": self.total_deposits_liability,
            "equity": self.equity_value,
            "capital_ratio": self.capital_ratio,
            "lending_rate": self.lending_rate,
            "interest_income": self.interest_income,
            "default_losses": self.default_losses,
            "active_loans_count": len(self.loan_book.active_loans()),
        })
        return obs
