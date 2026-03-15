"""
Core accounting primitives for stock-flow-consistent economic simulation.

All monetary values are stored as floats rounded to 2 decimal places.
Every transfer is double-entry: one account debited, one credited.
No money is created or destroyed except through explicit mechanisms
(bank lending, government spending, default write-offs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

PRECISION = 2


def round_money(value: float) -> float:
    return round(value, PRECISION)


class AccountType(Enum):
    ASSET = auto()
    LIABILITY = auto()
    EQUITY = auto()
    REVENUE = auto()
    EXPENSE = auto()


@dataclass
class Account:
    """A single named account with a balance and type.

    Debit/credit semantics follow standard accounting:
    - ASSET / EXPENSE accounts increase on debit, decrease on credit.
    - LIABILITY / EQUITY / REVENUE accounts increase on credit, decrease on debit.
    """

    name: str
    account_type: AccountType
    balance: float = 0.0
    owner_id: str = ""

    def debit(self, amount: float) -> None:
        if amount < 0:
            raise ValueError(f"Debit amount must be non-negative, got {amount}")
        if self.account_type in (AccountType.ASSET, AccountType.EXPENSE):
            self.balance = round_money(self.balance + amount)
        else:
            self.balance = round_money(self.balance - amount)

    def credit(self, amount: float) -> None:
        if amount < 0:
            raise ValueError(f"Credit amount must be non-negative, got {amount}")
        if self.account_type in (AccountType.ASSET, AccountType.EXPENSE):
            self.balance = round_money(self.balance - amount)
        else:
            self.balance = round_money(self.balance + amount)

    def __repr__(self) -> str:
        return f"Account({self.name}, {self.account_type.name}, {self.balance:.2f})"


@dataclass(frozen=True)
class Transaction:
    """An immutable record of a double-entry transaction."""

    tx_id: int
    period: int
    debit_owner: str
    debit_account: str
    credit_owner: str
    credit_account: str
    amount: float
    description: str


class BalanceSheet:
    """Maintains a set of accounts for a single entity and enforces the
    accounting identity: Assets - Liabilities = Equity."""

    def __init__(self, owner_id: str) -> None:
        self.owner_id = owner_id
        self._accounts: dict[str, Account] = {}

    def add_account(
        self, name: str, account_type: AccountType, initial_balance: float = 0.0
    ) -> Account:
        if name in self._accounts:
            raise ValueError(f"Account '{name}' already exists for {self.owner_id}")
        acct = Account(
            name=name,
            account_type=account_type,
            balance=round_money(initial_balance),
            owner_id=self.owner_id,
        )
        self._accounts[name] = acct
        return acct

    def get_account(self, name: str) -> Account:
        if name not in self._accounts:
            raise KeyError(f"Account '{name}' not found for {self.owner_id}")
        return self._accounts[name]

    def has_account(self, name: str) -> bool:
        return name in self._accounts

    @property
    def total_assets(self) -> float:
        return round_money(
            sum(
                a.balance
                for a in self._accounts.values()
                if a.account_type == AccountType.ASSET
            )
        )

    @property
    def total_liabilities(self) -> float:
        return round_money(
            sum(
                a.balance
                for a in self._accounts.values()
                if a.account_type == AccountType.LIABILITY
            )
        )

    @property
    def total_equity(self) -> float:
        return round_money(
            sum(
                a.balance
                for a in self._accounts.values()
                if a.account_type == AccountType.EQUITY
            )
        )

    @property
    def net_worth(self) -> float:
        return round_money(self.total_assets - self.total_liabilities)

    def check_balanced(self, tolerance: float = 0.01) -> bool:
        return abs(self.net_worth - self.total_equity) <= tolerance

    def summary(self) -> dict:
        return {
            "owner": self.owner_id,
            "total_assets": self.total_assets,
            "total_liabilities": self.total_liabilities,
            "total_equity": self.total_equity,
            "net_worth": self.net_worth,
            "balanced": self.check_balanced(),
        }

    def accounts_by_type(self, account_type: AccountType) -> list[Account]:
        return [a for a in self._accounts.values() if a.account_type == account_type]

    def all_accounts(self) -> dict[str, Account]:
        return dict(self._accounts)

    def __repr__(self) -> str:
        return (
            f"BalanceSheet({self.owner_id}: "
            f"A={self.total_assets:.2f}, L={self.total_liabilities:.2f}, "
            f"E={self.total_equity:.2f}, balanced={self.check_balanced()})"
        )


class Ledger:
    """Central ledger that records all transactions and provides the only
    mechanism for moving money between accounts.

    All monetary flows MUST go through the ledger to maintain auditability
    and stock-flow consistency.
    """

    def __init__(self) -> None:
        self._transactions: list[Transaction] = []
        self._next_tx_id: int = 0
        self._balance_sheets: dict[str, BalanceSheet] = {}

    def register_balance_sheet(self, bs: BalanceSheet) -> None:
        self._balance_sheets[bs.owner_id] = bs

    def get_balance_sheet(self, owner_id: str) -> BalanceSheet:
        if owner_id not in self._balance_sheets:
            raise KeyError(f"No balance sheet registered for '{owner_id}'")
        return self._balance_sheets[owner_id]

    @property
    def transactions(self) -> list[Transaction]:
        return list(self._transactions)

    def post(
        self,
        period: int,
        debit_owner: str,
        debit_account: str,
        credit_owner: str,
        credit_account: str,
        amount: float,
        description: str,
    ) -> Optional[Transaction]:
        """Post a double-entry transaction.

        Debits the specified account on the debit_owner's balance sheet.
        Credits the specified account on the credit_owner's balance sheet.
        Returns the Transaction record, or None if amount is zero.
        """
        amount = round_money(amount)
        if amount == 0:
            return None
        if amount < 0:
            raise ValueError(f"Cannot post negative amount: {amount}")

        debit_bs = self.get_balance_sheet(debit_owner)
        credit_bs = self.get_balance_sheet(credit_owner)

        debit_acct = debit_bs.get_account(debit_account)
        credit_acct = credit_bs.get_account(credit_account)

        tx = Transaction(
            tx_id=self._next_tx_id,
            period=period,
            debit_owner=debit_owner,
            debit_account=debit_account,
            credit_owner=credit_owner,
            credit_account=credit_account,
            amount=amount,
            description=description,
        )

        debit_acct.debit(amount)
        credit_acct.credit(amount)

        self._transactions.append(tx)
        self._next_tx_id += 1
        return tx

    def issue_loan(
        self,
        period: int,
        bank_id: str,
        borrower_id: str,
        amount: float,
        description: str = "loan issuance",
    ) -> list[Transaction]:
        """Create money via bank lending (endogenous money creation).

        Double-entry on the bank side:
          - Bank's 'loans' account (ASSET) increases (debit)
          - Bank's 'deposits' account (LIABILITY) increases (credit)

        Double-entry on the borrower side:
          - Borrower's 'deposits' account (ASSET) increases (debit)
          - Borrower's 'loans_payable' account (LIABILITY) increases (credit)
        """
        txs: list[Transaction] = []

        # Bank side: loans asset up, deposit liability up
        tx1 = self.post(
            period=period,
            debit_owner=bank_id,
            debit_account="loans",
            credit_owner=bank_id,
            credit_account="deposits",
            amount=amount,
            description=f"{description} (bank side)",
        )
        if tx1:
            txs.append(tx1)

        # Borrower side: deposit asset up, loan liability up
        tx2 = self.post(
            period=period,
            debit_owner=borrower_id,
            debit_account="deposits",
            credit_owner=borrower_id,
            credit_account="loans_payable",
            amount=amount,
            description=f"{description} (borrower side)",
        )
        if tx2:
            txs.append(tx2)

        return txs

    def repay_loan(
        self,
        period: int,
        bank_id: str,
        borrower_id: str,
        amount: float,
        description: str = "loan repayment",
    ) -> list[Transaction]:
        """Destroy money via loan repayment (reverse of issuance).

        Bank side: deposits liability down, loans asset down.
        Borrower side: loans_payable liability down, deposits asset down.
        """
        txs: list[Transaction] = []

        # Bank side: deposit liability down (debit), loans asset down (credit)
        tx1 = self.post(
            period=period,
            debit_owner=bank_id,
            debit_account="deposits",
            credit_owner=bank_id,
            credit_account="loans",
            amount=amount,
            description=f"{description} (bank side)",
        )
        if tx1:
            txs.append(tx1)

        # Borrower side: loans_payable down (debit), deposits down (credit)
        tx2 = self.post(
            period=period,
            debit_owner=borrower_id,
            debit_account="loans_payable",
            credit_owner=borrower_id,
            credit_account="deposits",
            amount=amount,
            description=f"{description} (borrower side)",
        )
        if tx2:
            txs.append(tx2)

        return txs

    def transfer_deposits(
        self,
        period: int,
        from_id: str,
        to_id: str,
        amount: float,
        description: str,
    ) -> list[Transaction]:
        """Transfer deposits between two entities, preserving A - L = E.

        Implemented as two within-entity double-entry transactions:

        Sender (expense):
          Debit equity (equity decreases), Credit deposits (asset decreases)

        Receiver (income):
          Debit deposits (asset increases), Credit equity (equity increases)

        This ensures every entity's balance sheet remains balanced after
        any income/expense flow (wages, consumption, taxes, transfers, revenue).
        """
        txs: list[Transaction] = []
        amount = round_money(amount)
        if amount <= 0:
            return txs

        # Sender: equity down, deposits down
        tx1 = self.post(
            period=period,
            debit_owner=from_id,
            debit_account="equity",
            credit_owner=from_id,
            credit_account="deposits",
            amount=amount,
            description=f"{description} (sender)",
        )
        if tx1:
            txs.append(tx1)

        # Receiver: deposits up, equity up
        tx2 = self.post(
            period=period,
            debit_owner=to_id,
            debit_account="deposits",
            credit_owner=to_id,
            credit_account="equity",
            amount=amount,
            description=f"{description} (receiver)",
        )
        if tx2:
            txs.append(tx2)

        return txs

    def write_off_loan(
        self,
        period: int,
        bank_id: str,
        borrower_id: str,
        amount: float,
        description: str = "loan default write-off",
    ) -> list[Transaction]:
        """Write off a defaulted loan.

        Bank side: equity decreases (absorbs loss), loans asset decreases.
        Borrower side: loans_payable decreases, equity increases (debt forgiven).
        """
        txs: list[Transaction] = []

        # Bank: equity down (debit), loans down (credit)
        tx1 = self.post(
            period=period,
            debit_owner=bank_id,
            debit_account="equity",
            credit_owner=bank_id,
            credit_account="loans",
            amount=amount,
            description=f"{description} (bank loss)",
        )
        if tx1:
            txs.append(tx1)

        # Borrower: loans_payable down (debit), equity up (credit)
        tx2 = self.post(
            period=period,
            debit_owner=borrower_id,
            debit_account="loans_payable",
            credit_owner=borrower_id,
            credit_account="equity",
            amount=amount,
            description=f"{description} (borrower relief)",
        )
        if tx2:
            txs.append(tx2)

        return txs

    def get_transactions_for_period(self, period: int) -> list[Transaction]:
        return [t for t in self._transactions if t.period == period]

    def get_transactions_for_owner(self, owner_id: str) -> list[Transaction]:
        return [
            t
            for t in self._transactions
            if t.debit_owner == owner_id or t.credit_owner == owner_id
        ]

    def validate_all_balanced(self, tolerance: float = 0.01) -> dict[str, bool]:
        return {
            owner_id: bs.check_balanced(tolerance)
            for owner_id, bs in self._balance_sheets.items()
        }
