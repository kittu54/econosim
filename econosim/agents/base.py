"""
Base agent class providing balance-sheet integration and common interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from econosim.core.accounting import AccountType, BalanceSheet, Ledger


class BaseAgent(ABC):
    """Abstract base for all economic agents."""

    def __init__(self, agent_id: str, agent_type: str, ledger: Ledger) -> None:
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.ledger = ledger
        self.balance_sheet = BalanceSheet(owner_id=agent_id)
        ledger.register_balance_sheet(self.balance_sheet)
        self._setup_accounts()

    @abstractmethod
    def _setup_accounts(self) -> None:
        """Create the accounts this agent type needs on its balance sheet."""
        ...

    @property
    def deposits(self) -> float:
        return self.balance_sheet.get_account("deposits").balance

    def get_observation(self) -> dict[str, Any]:
        """Return observable state for metrics/RL. Override per agent type."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "deposits": self.deposits,
            "net_worth": self.balance_sheet.net_worth,
        }

    def __repr__(self) -> str:
        return f"{self.agent_type}({self.agent_id}, deposits={self.deposits:.2f})"
