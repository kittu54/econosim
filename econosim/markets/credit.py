"""
Credit market: firms (and optionally households) request loans from the bank.

MVP mechanism:
- Firms request loans based on cash flow needs
- Bank evaluates and approves/rejects via its rule-based logic
- Loan issuance creates money (endogenous money creation)
"""

from __future__ import annotations

from econosim.core.accounting import Ledger, round_money
from econosim.agents.firm import Firm
from econosim.agents.bank import Bank


class CreditMarket:
    """Processes credit applications each period."""

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        # Period stats
        self.applications: int = 0
        self.approvals: int = 0
        self.total_issued: float = 0.0
        self.total_rejected: float = 0.0

    def clear(
        self,
        firms: list[Firm],
        bank: Bank,
        period: int,
    ) -> None:
        """Process credit applications for one period.

        Firms apply for loans if they need cash for operations.
        Bank approves or rejects based on capital adequacy and creditworthiness.
        """
        self.applications = 0
        self.approvals = 0
        self.total_issued = 0.0
        self.total_rejected = 0.0

        for firm in firms:
            # Simple borrowing rule: firm borrows if deposits < wage bill needed
            expected_wage_bill = firm.posted_wage * max(firm.vacancies, 1)
            cash_shortfall = expected_wage_bill - firm.deposits

            if cash_shortfall <= 0:
                continue
            if not firm.can_borrow(cash_shortfall):
                continue

            self.applications += 1
            requested = round_money(cash_shortfall * 1.2)  # small buffer

            contract = bank.approve_loan(
                borrower_id=firm.agent_id,
                requested_amount=requested,
                borrower_deposits=firm.deposits,
                borrower_debt=firm.total_debt,
                period=period,
            )

            if contract is not None:
                self.approvals += 1
                self.total_issued = round_money(self.total_issued + contract.principal)
            else:
                self.total_rejected = round_money(self.total_rejected + requested)
