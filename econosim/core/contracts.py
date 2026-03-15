"""
Loan and debt contracts with explicit state tracking.

Each LoanContract tracks principal, remaining balance, interest rate,
payment schedule, delinquency status, and default state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from econosim.core.accounting import round_money


class LoanStatus(Enum):
    ACTIVE = auto()
    DELINQUENT = auto()
    DEFAULTED = auto()
    PAID_OFF = auto()


@dataclass
class LoanContract:
    """A single loan between a bank and a borrower."""

    loan_id: str
    bank_id: str
    borrower_id: str
    principal: float
    interest_rate: float  # per-period rate (e.g. monthly)
    term_periods: int
    origination_period: int
    remaining_balance: float = 0.0
    periods_delinquent: int = 0
    status: LoanStatus = LoanStatus.ACTIVE
    total_interest_paid: float = 0.0
    total_principal_paid: float = 0.0

    def __post_init__(self) -> None:
        if self.remaining_balance == 0.0:
            self.remaining_balance = round_money(self.principal)

    @property
    def periodic_payment(self) -> float:
        """Fixed payment per period (amortizing loan)."""
        r = self.interest_rate
        n = self.term_periods
        if r == 0:
            return round_money(self.principal / n)
        payment = self.principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
        return round_money(payment)

    @property
    def interest_due(self) -> float:
        """Interest component of current period's payment."""
        return round_money(self.remaining_balance * self.interest_rate)

    @property
    def principal_due(self) -> float:
        """Principal component of current period's payment."""
        payment = self.periodic_payment
        interest = self.interest_due
        principal_part = min(payment - interest, self.remaining_balance)
        return round_money(max(0.0, principal_part))

    def record_payment(self, amount: float) -> tuple[float, float]:
        """Record a payment. Returns (interest_paid, principal_paid).

        Payment is applied to interest first, then principal.
        """
        interest = self.interest_due
        interest_paid = min(amount, interest)
        remainder = amount - interest_paid
        principal_paid = min(remainder, self.remaining_balance)

        self.remaining_balance = round_money(self.remaining_balance - principal_paid)
        self.total_interest_paid = round_money(self.total_interest_paid + interest_paid)
        self.total_principal_paid = round_money(self.total_principal_paid + principal_paid)

        if self.remaining_balance <= 0.01:
            self.remaining_balance = 0.0
            self.status = LoanStatus.PAID_OFF

        if amount >= self.periodic_payment - 0.01:
            self.periods_delinquent = 0
        else:
            self.periods_delinquent += 1

        return (interest_paid, principal_paid)

    def mark_delinquent(self) -> None:
        if self.status == LoanStatus.ACTIVE:
            self.status = LoanStatus.DELINQUENT

    def mark_default(self) -> None:
        self.status = LoanStatus.DEFAULTED

    @property
    def is_active(self) -> bool:
        return self.status in (LoanStatus.ACTIVE, LoanStatus.DELINQUENT)


class LoanBook:
    """Tracks all loans for the simulation. Owned by the bank."""

    def __init__(self) -> None:
        self._loans: dict[str, LoanContract] = {}
        self._next_id: int = 0

    def create_loan(
        self,
        bank_id: str,
        borrower_id: str,
        principal: float,
        interest_rate: float,
        term_periods: int,
        origination_period: int,
    ) -> LoanContract:
        loan_id = f"loan_{self._next_id:04d}"
        self._next_id += 1
        contract = LoanContract(
            loan_id=loan_id,
            bank_id=bank_id,
            borrower_id=borrower_id,
            principal=principal,
            interest_rate=interest_rate,
            term_periods=term_periods,
            origination_period=origination_period,
        )
        self._loans[loan_id] = contract
        return contract

    def get_loan(self, loan_id: str) -> LoanContract:
        return self._loans[loan_id]

    def active_loans(self) -> list[LoanContract]:
        return [l for l in self._loans.values() if l.is_active]

    def loans_for_borrower(self, borrower_id: str) -> list[LoanContract]:
        return [l for l in self._loans.values() if l.borrower_id == borrower_id]

    def active_loans_for_borrower(self, borrower_id: str) -> list[LoanContract]:
        return [l for l in self.loans_for_borrower(borrower_id) if l.is_active]

    def total_outstanding(self) -> float:
        return round_money(sum(l.remaining_balance for l in self.active_loans()))

    def total_outstanding_for_borrower(self, borrower_id: str) -> float:
        return round_money(
            sum(l.remaining_balance for l in self.active_loans_for_borrower(borrower_id))
        )

    def defaulted_loans(self) -> list[LoanContract]:
        return [l for l in self._loans.values() if l.status == LoanStatus.DEFAULTED]

    @property
    def all_loans(self) -> dict[str, LoanContract]:
        return dict(self._loans)
