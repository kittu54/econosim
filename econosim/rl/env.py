"""
RL environment wrapper for the economic simulation.

Provides a Gymnasium-compatible interface for training agents
on top of the rule-based simulation engine.

This is a scaffold for Phase 3. The MVP simulation runs entirely
with rule-based agents; this interface defines the contract for
future RL integration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class EconEnvInterface(ABC):
    """Abstract interface for RL environment wrapping the simulation.

    Follows Gymnasium's reset/step convention.
    Concrete implementations will be built in Phase 3.
    """

    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment. Returns (observation, info)."""
        ...

    @abstractmethod
    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Take one step. Returns (observation, reward, terminated, truncated, info)."""
        ...

    @abstractmethod
    def observation_space_spec(self) -> dict[str, Any]:
        """Return a description of the observation space."""
        ...

    @abstractmethod
    def action_space_spec(self) -> dict[str, Any]:
        """Return a description of the action space."""
        ...


# ── Role-specific observation/action definitions ─────────────────

FIRM_OBSERVATION_KEYS = [
    "deposits", "inventory", "price", "posted_wage", "workers_count",
    "revenue", "wage_bill", "units_sold", "total_debt",
    "avg_market_price", "unemployment_rate",
]

FIRM_ACTION_KEYS = [
    "price_adjustment",      # float: multiplicative factor on current price
    "wage_adjustment",       # float: multiplicative factor on current wage
    "vacancy_target",        # int: number of workers to hire
]

HOUSEHOLD_OBSERVATION_KEYS = [
    "deposits", "employed", "wage_income", "consumption_spending",
    "total_debt", "avg_price", "unemployment_rate",
]

HOUSEHOLD_ACTION_KEYS = [
    "consumption_fraction",  # float [0,1]: fraction of disposable income to consume
    "labor_participation",   # bool: whether to seek work
]

BANK_OBSERVATION_KEYS = [
    "total_loans", "total_deposits_liability", "equity", "capital_ratio",
    "interest_income", "default_losses", "active_loans_count",
    "avg_unemployment", "avg_gdp",
]

BANK_ACTION_KEYS = [
    "lending_rate_adjustment",     # float: adjustment to lending rate
    "capital_adequacy_target",     # float: target capital ratio
]

GOVERNMENT_OBSERVATION_KEYS = [
    "deposits", "tax_revenue", "transfers_paid", "goods_spending",
    "budget_balance", "unemployment_rate", "gdp", "avg_price",
]

GOVERNMENT_ACTION_KEYS = [
    "tax_rate_adjustment",         # float: adjustment to income tax rate
    "transfer_adjustment",         # float: adjustment to transfer amount
    "spending_adjustment",         # float: adjustment to government spending
]
