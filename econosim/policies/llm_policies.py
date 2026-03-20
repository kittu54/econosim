"""LLM-powered policy implementations for all agent types.

Each policy uses an LLM to make economic decisions based on the current
state, agent personality, and memory of past decisions/outcomes. These
implement the same policy interfaces as rule-based policies, so they
can be swapped in without changing the engine.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from econosim.llm.client import LLMClient
from econosim.llm.memory import AgentMemory
from econosim.llm.prompts import (
    EconomicPersonality,
    PromptTemplate,
    FIRM_PERSONALITIES,
    HOUSEHOLD_PERSONALITIES,
    BANK_PERSONALITIES,
    GOVERNMENT_PERSONALITIES,
)
from econosim.policies.interfaces import (
    FirmPolicy,
    FirmAction,
    FirmState,
    HouseholdPolicy,
    HouseholdAction,
    HouseholdState,
    BankPolicy,
    BankAction,
    BankState,
    GovernmentPolicy,
    GovernmentAction,
    GovernmentState,
    MacroState,
)

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float, low: float, high: float) -> float:
    """Safely extract a float from LLM output, clamping to bounds."""
    try:
        v = float(value)
        return max(low, min(high, v))
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int, low: int = 0) -> int:
    """Safely extract an int from LLM output."""
    try:
        v = int(float(value))
        return max(low, v)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool) -> bool:
    """Safely extract a bool from LLM output."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "yes", "1")
    return default


class LLMFirmPolicy(FirmPolicy):
    """LLM-powered firm decision-making policy."""

    def __init__(
        self,
        client: LLMClient,
        personality: EconomicPersonality | None = None,
        memory: AgentMemory | None = None,
    ) -> None:
        self.client = client
        self.personality = personality or FIRM_PERSONALITIES[1]  # Established Enterprise
        self.memory = memory or AgentMemory(max_entries=30, agent_id="firm")

    def act(self, firm_state: FirmState, macro_state: MacroState) -> FirmAction:
        system = PromptTemplate.get_system_prompt("firm", self.personality)
        prompt = PromptTemplate.get_decision_prompt(
            "firm",
            period=macro_state.period,
            inventory=firm_state.inventory,
            deposits=firm_state.deposits,
            price=firm_state.price,
            wage=firm_state.posted_wage,
            workers=firm_state.workers_count,
            revenue=firm_state.revenue,
            units_sold=firm_state.units_sold,
            avg_price=macro_state.avg_price,
            unemployment_rate=macro_state.unemployment_rate,
            gdp_growth=macro_state.gdp_growth,
            interest_rate=macro_state.lending_rate,
            memory=self.memory.to_prompt_text(max_entries=10),
        )

        try:
            response = self.client.complete(prompt, system=system, json_mode=True)
            data = response.as_json()
        except Exception as e:
            logger.warning(f"LLM firm policy failed: {e}; using defaults")
            data = {}

        action = FirmAction(
            vacancies=_safe_int(data.get("vacancies"), 1, 0),
            price_adjustment=_safe_float(data.get("price_adjustment"), 1.0, 0.8, 1.2),
            wage_adjustment=_safe_float(data.get("wage_adjustment"), 1.0, 0.9, 1.1),
            loan_request=_safe_float(data.get("loan_request"), 0.0, 0.0, 100000.0),
        )

        reasoning = data.get("reasoning", "no reasoning provided")
        self.memory.add(
            period=macro_state.period,
            category="decision",
            content=f"Set vacancies={action.vacancies}, price_adj={action.price_adjustment:.2f}, "
            f"wage_adj={action.wage_adjustment:.2f}, loan={action.loan_request:.0f}. "
            f"Reason: {reasoning}",
            importance=2.0,
        )

        return action


class LLMHouseholdPolicy(HouseholdPolicy):
    """LLM-powered household decision-making policy."""

    def __init__(
        self,
        client: LLMClient,
        personality: EconomicPersonality | None = None,
        memory: AgentMemory | None = None,
    ) -> None:
        self.client = client
        self.personality = personality or HOUSEHOLD_PERSONALITIES[1]  # Young Professional
        self.memory = memory or AgentMemory(max_entries=30, agent_id="household")

    def act(self, hh_state: HouseholdState, macro_state: MacroState) -> HouseholdAction:
        system = PromptTemplate.get_system_prompt("household", self.personality)
        prompt = PromptTemplate.get_decision_prompt(
            "household",
            period=macro_state.period,
            deposits=hh_state.deposits,
            employed="employed" if hh_state.employed else "unemployed",
            wage=hh_state.wage_income,
            reservation_wage=hh_state.reservation_wage,
            avg_price=macro_state.avg_price,
            unemployment_rate=macro_state.unemployment_rate,
            gdp_growth=macro_state.gdp_growth,
            inflation_rate=macro_state.inflation_rate,
            memory=self.memory.to_prompt_text(max_entries=10),
        )

        try:
            response = self.client.complete(prompt, system=system, json_mode=True)
            data = response.as_json()
        except Exception as e:
            logger.warning(f"LLM household policy failed: {e}; using defaults")
            data = {}

        action = HouseholdAction(
            consumption_fraction=_safe_float(data.get("consumption_fraction"), 0.8, 0.0, 1.0),
            labor_participation=_safe_bool(data.get("labor_participation"), True),
            reservation_wage_adjustment=_safe_float(
                data.get("reservation_wage_adjustment"), 1.0, 0.8, 1.2
            ),
        )

        reasoning = data.get("reasoning", "no reasoning provided")
        self.memory.add(
            period=macro_state.period,
            category="decision",
            content=f"Consume {action.consumption_fraction:.0%} of budget, "
            f"labor={'yes' if action.labor_participation else 'no'}, "
            f"wage_adj={action.reservation_wage_adjustment:.2f}. "
            f"Reason: {reasoning}",
            importance=2.0,
        )

        return action


class LLMBankPolicy(BankPolicy):
    """LLM-powered bank decision-making policy."""

    def __init__(
        self,
        client: LLMClient,
        personality: EconomicPersonality | None = None,
        memory: AgentMemory | None = None,
    ) -> None:
        self.client = client
        self.personality = personality or BANK_PERSONALITIES[0]  # Prudent Lender
        self.memory = memory or AgentMemory(max_entries=30, agent_id="bank")

    def act(self, bank_state: BankState, macro_state: MacroState) -> BankAction:
        system = PromptTemplate.get_system_prompt("bank", self.personality)

        default_rate = 0.0
        if bank_state.active_loans_count > 0:
            default_rate = bank_state.default_losses / max(bank_state.total_loans, 1.0)

        prompt = PromptTemplate.get_decision_prompt(
            "bank",
            period=macro_state.period,
            total_assets=bank_state.total_loans + bank_state.equity,
            total_deposits=bank_state.total_deposits,
            capital_ratio=bank_state.capital_ratio,
            base_rate=bank_state.base_interest_rate,
            risk_premium=bank_state.risk_premium,
            default_rate=default_rate,
            outstanding_loans=bank_state.total_loans,
            gdp_growth=macro_state.gdp_growth,
            inflation_rate=macro_state.inflation_rate,
            unemployment_rate=macro_state.unemployment_rate,
            memory=self.memory.to_prompt_text(max_entries=10),
        )

        try:
            response = self.client.complete(prompt, system=system, json_mode=True)
            data = response.as_json()
        except Exception as e:
            logger.warning(f"LLM bank policy failed: {e}; using defaults")
            data = {}

        action = BankAction(
            base_rate_adjustment=_safe_float(data.get("base_rate_adjustment"), 1.0, 0.8, 1.2),
            capital_target_adjustment=_safe_float(
                data.get("capital_target_adjustment"), 1.0, 0.9, 1.1
            ),
            risk_premium_adjustment=_safe_float(
                data.get("risk_premium_adjustment"), 1.0, 0.8, 1.2
            ),
        )

        reasoning = data.get("reasoning", "no reasoning provided")
        self.memory.add(
            period=macro_state.period,
            category="decision",
            content=f"Rate_adj={action.base_rate_adjustment:.2f}, "
            f"cap_adj={action.capital_target_adjustment:.2f}, "
            f"risk_adj={action.risk_premium_adjustment:.2f}. "
            f"Reason: {reasoning}",
            importance=2.0,
        )

        return action


class LLMGovernmentPolicy(GovernmentPolicy):
    """LLM-powered government fiscal policy."""

    def __init__(
        self,
        client: LLMClient,
        personality: EconomicPersonality | None = None,
        memory: AgentMemory | None = None,
    ) -> None:
        self.client = client
        self.personality = personality or GOVERNMENT_PERSONALITIES[0]  # Keynesian
        self.memory = memory or AgentMemory(max_entries=30, agent_id="government")

    def act(self, govt_state: GovernmentState, macro_state: MacroState) -> GovernmentAction:
        system = PromptTemplate.get_system_prompt("government", self.personality)
        prompt = PromptTemplate.get_decision_prompt(
            "government",
            period=macro_state.period,
            gdp=macro_state.gdp,
            gdp_growth=macro_state.gdp_growth,
            unemployment_rate=macro_state.unemployment_rate,
            inflation_rate=macro_state.inflation_rate,
            gini=0.3,  # default; could be passed via macro_state extension
            tax_revenue=govt_state.tax_revenue,
            spending=govt_state.goods_spending,
            budget_balance=govt_state.budget_balance,
            govt_deposits=govt_state.deposits,
            money_created=0.0,
            memory=self.memory.to_prompt_text(max_entries=10),
        )

        try:
            response = self.client.complete(prompt, system=system, json_mode=True)
            data = response.as_json()
        except Exception as e:
            logger.warning(f"LLM government policy failed: {e}; using defaults")
            data = {}

        action = GovernmentAction(
            tax_rate=_safe_float(data.get("tax_rate"), 0.2, 0.0, 0.5),
            transfer_per_unemployed=_safe_float(
                data.get("transfer_per_unemployed"), 50.0, 0.0, 500.0
            ),
            spending_per_period=_safe_float(
                data.get("spending_per_period"), 2000.0, 0.0, 10000.0
            ),
        )

        reasoning = data.get("reasoning", "no reasoning provided")
        self.memory.add(
            period=macro_state.period,
            category="decision",
            content=f"Tax={action.tax_rate:.1%}, transfers=${action.transfer_per_unemployed:.0f}, "
            f"spending=${action.spending_per_period:.0f}. "
            f"Reason: {reasoning}",
            importance=3.0,
        )

        return action
