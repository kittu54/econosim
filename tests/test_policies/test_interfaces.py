"""Tests for policy interfaces and rule-based policies."""

import pytest

from econosim.policies.interfaces import (
    FirmPolicy, FirmState, FirmAction,
    HouseholdPolicy, HouseholdState, HouseholdAction,
    BankPolicy, BankState, BankAction,
    GovernmentPolicy, GovernmentState, GovernmentAction,
    MacroState,
)
from econosim.policies.rule_based import (
    RuleBasedFirmPolicy,
    RuleBasedHouseholdPolicy,
    RuleBasedBankPolicy,
    RuleBasedGovernmentPolicy,
)


class TestFirmPolicy:
    def test_rule_based_produces_action(self):
        policy = RuleBasedFirmPolicy()
        state = FirmState(
            deposits=10000, inventory=50, price=10, posted_wage=100,
            workers_count=3, revenue=500, prev_revenue=500,
            units_sold=50, prev_units_sold=50, total_debt=0,
            equity=10000, labor_productivity=5, target_inventory_ratio=0.2,
        )
        macro = MacroState()
        action = policy.act(state, macro)

        assert isinstance(action, FirmAction)
        assert action.vacancies >= 0
        assert action.price_adjustment > 0

    def test_vacancy_decision_respects_affordability(self):
        policy = RuleBasedFirmPolicy()
        state = FirmState(
            deposits=50, posted_wage=100,  # can't afford even 1 worker
            prev_units_sold=10, prev_revenue=100,
            price=10, inventory=5, labor_productivity=5,
            target_inventory_ratio=0.2, equity=50,
        )
        action = policy.act(state, MacroState())
        assert action.vacancies == 0

    def test_price_drops_when_high_inventory(self):
        policy = RuleBasedFirmPolicy()
        state = FirmState(
            deposits=10000, inventory=100, price=10,
            prev_units_sold=10, target_inventory_ratio=0.2,
            posted_wage=100, labor_productivity=5, equity=10000,
        )
        action = policy.act(state, MacroState())
        assert action.price_adjustment < 1.0  # price should decrease

    def test_price_rises_when_low_inventory(self):
        policy = RuleBasedFirmPolicy()
        state = FirmState(
            deposits=10000, inventory=0.5, price=10,
            prev_units_sold=10, target_inventory_ratio=0.2,
            posted_wage=100, labor_productivity=5, equity=10000,
        )
        action = policy.act(state, MacroState())
        assert action.price_adjustment > 1.0  # price should increase


class TestHouseholdPolicy:
    def test_rule_based_produces_action(self):
        policy = RuleBasedHouseholdPolicy()
        state = HouseholdState(deposits=1000, wage_income=100)
        action = policy.act(state, MacroState())

        assert isinstance(action, HouseholdAction)
        assert 0.0 <= action.consumption_fraction <= 1.0
        assert action.labor_participation is True

    def test_consumption_fraction_bounded(self):
        policy = RuleBasedHouseholdPolicy()
        state = HouseholdState(deposits=0.01, wage_income=0)
        action = policy.act(state, MacroState())
        assert 0.0 <= action.consumption_fraction <= 1.0


class TestBankPolicy:
    def test_rule_based_produces_action(self):
        policy = RuleBasedBankPolicy()
        state = BankState()
        action = policy.act(state, MacroState())

        assert isinstance(action, BankAction)
        assert action.base_rate_adjustment == 0.0  # passive


class TestGovernmentPolicy:
    def test_rule_based_produces_action(self):
        policy = RuleBasedGovernmentPolicy()
        state = GovernmentState(income_tax_rate=0.2, spending_per_period=2000)
        action = policy.act(state, MacroState())

        assert isinstance(action, GovernmentAction)
        assert action.tax_rate == 0.2
        assert action.spending_per_period == 2000


class TestPolicyNames:
    def test_policy_names(self):
        assert RuleBasedFirmPolicy().name() == "RuleBasedFirmPolicy"
        assert RuleBasedBankPolicy().name() == "RuleBasedBankPolicy"
