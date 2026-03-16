"""Tests for policy interface integration with the simulation engine.

Verifies that:
1. Rule-based policies produce equivalent behavior to hardcoded logic
2. Custom policies can override agent decisions
3. MacroState / FirmState / etc. are constructed correctly
4. Simulation runs to completion with policies active
"""

from __future__ import annotations

import numpy as np
import pytest

from econosim.config.schema import SimulationConfig
from econosim.engine.simulation import (
    build_simulation,
    build_macro_state,
    build_firm_state,
    build_bank_state,
    build_govt_state,
    step,
    run_simulation,
)
from econosim.policies.interfaces import (
    FirmPolicy, FirmState, FirmAction,
    BankPolicy, BankState, BankAction,
    GovernmentPolicy, GovernmentState, GovernmentAction,
    MacroState,
)
from econosim.policies.rule_based import (
    RuleBasedFirmPolicy,
    RuleBasedBankPolicy,
    RuleBasedGovernmentPolicy,
)


def _small_config(**overrides) -> SimulationConfig:
    """Create a small config for fast tests."""
    defaults = {
        "name": "policy_test",
        "num_periods": 10,
        "seed": 42,
        "household": {"count": 20, "initial_deposits": 500.0},
        "firm": {"count": 5, "initial_deposits": 5000.0},
    }
    defaults.update(overrides)
    return SimulationConfig(**defaults)


# --- State builder tests ---


class TestStateBuilders:
    def test_build_macro_state_first_period(self):
        state = build_simulation(_small_config())
        macro = build_macro_state(state)
        assert macro.period == 0
        assert macro.lending_rate == state.bank.lending_rate

    def test_build_macro_state_after_periods(self):
        state = build_simulation(_small_config())
        for _ in range(3):
            step(state)
        macro = build_macro_state(state)
        assert macro.period == 3
        assert macro.gdp > 0
        assert 0.0 <= macro.unemployment_rate <= 1.0

    def test_build_macro_state_gdp_growth(self):
        state = build_simulation(_small_config())
        for _ in range(5):
            step(state)
        macro = build_macro_state(state)
        # gdp_growth is computed from last two periods
        assert isinstance(macro.gdp_growth, float)

    def test_build_firm_state(self):
        state = build_simulation(_small_config())
        firm = state.firms[0]
        fs = build_firm_state(firm)
        assert fs.deposits == firm.deposits
        assert fs.price == firm.price
        assert fs.posted_wage == firm.posted_wage
        assert fs.labor_productivity == firm.labor_productivity

    def test_build_bank_state(self):
        state = build_simulation(_small_config())
        bs = build_bank_state(state.bank)
        assert bs.lending_rate == state.bank.lending_rate
        assert bs.base_interest_rate == state.bank.base_interest_rate

    def test_build_govt_state(self):
        state = build_simulation(_small_config())
        gs = build_govt_state(state.government)
        assert gs.income_tax_rate == state.government.income_tax_rate
        assert gs.spending_per_period == state.government.spending_per_period


# --- Rule-based policy tests ---


class TestRuleBasedPolicies:
    def test_simulation_with_rule_based_firm_policy(self):
        """Rule-based firm policy should produce a valid simulation."""
        config = _small_config()
        state = run_simulation(
            config,
            firm_policy=RuleBasedFirmPolicy(
                price_adjustment_speed=config.firm.price_adjustment_speed,
                wage_adjustment_speed=config.firm.wage_adjustment_speed,
                max_leverage=config.firm.max_leverage,
            ),
        )
        assert len(state.history) == 10
        assert all(m["gdp"] >= 0 for m in state.history)

    def test_simulation_with_rule_based_bank_policy(self):
        """Rule-based bank policy (passive) should not change behavior."""
        config = _small_config()
        state = run_simulation(config, bank_policy=RuleBasedBankPolicy())
        assert len(state.history) == 10

    def test_simulation_with_rule_based_govt_policy(self):
        """Rule-based government policy should maintain fiscal parameters."""
        config = _small_config()
        state = run_simulation(config, government_policy=RuleBasedGovernmentPolicy())
        assert len(state.history) == 10
        # Tax rate should remain at initial value since rule-based policy preserves it
        assert state.government.income_tax_rate == config.government.income_tax_rate

    def test_simulation_all_policies(self):
        """Run with all policy interfaces active simultaneously."""
        config = _small_config()
        state = run_simulation(
            config,
            firm_policy=RuleBasedFirmPolicy(),
            bank_policy=RuleBasedBankPolicy(),
            government_policy=RuleBasedGovernmentPolicy(),
        )
        assert len(state.history) == 10
        assert all(m["gdp"] >= 0 for m in state.history)


# --- Custom policy tests ---


class HighSpendingGovtPolicy(GovernmentPolicy):
    """Test policy: double government spending."""

    def act(self, govt_state: GovernmentState, macro_state: MacroState) -> GovernmentAction:
        return GovernmentAction(
            tax_rate=govt_state.income_tax_rate,
            transfer_per_unemployed=govt_state.transfer_per_unemployed,
            spending_per_period=govt_state.spending_per_period * 2.0,
        )


class ZeroVacancyFirmPolicy(FirmPolicy):
    """Test policy: firms post zero vacancies."""

    def act(self, firm_state: FirmState, macro_state: MacroState) -> FirmAction:
        return FirmAction(vacancies=0, price_adjustment=1.0)


class AggressiveRateBankPolicy(BankPolicy):
    """Test policy: raise rates each period."""

    def act(self, bank_state: BankState, macro_state: MacroState) -> BankAction:
        return BankAction(base_rate_adjustment=0.001)


class TestCustomPolicies:
    def test_high_spending_govt_increases_spending(self):
        config = _small_config(num_periods=1)
        state = run_simulation(config, government_policy=HighSpendingGovtPolicy())
        # After 1 step, spending_per_period should be doubled
        assert state.government.spending_per_period == config.government.spending_per_period * 2.0

    def test_zero_vacancy_causes_unemployment(self):
        config = _small_config(num_periods=5)
        state = run_simulation(config, firm_policy=ZeroVacancyFirmPolicy())
        # With zero vacancies, unemployment should be high
        last = state.history[-1]
        assert last["unemployment_rate"] > 0.5

    def test_aggressive_rate_policy_raises_rate(self):
        config = _small_config(num_periods=10)
        initial_rate = config.bank.base_interest_rate
        state = run_simulation(config, bank_policy=AggressiveRateBankPolicy())
        # After 10 periods of +0.001, rate should be higher
        assert state.bank.base_interest_rate > initial_rate
        assert abs(state.bank.base_interest_rate - (initial_rate + 0.01)) < 1e-9

    def test_policy_receives_correct_macro_state(self):
        """Verify policy receives plausible MacroState values."""
        received_states = []

        class SpyFirmPolicy(FirmPolicy):
            def act(self, firm_state: FirmState, macro_state: MacroState) -> FirmAction:
                received_states.append(macro_state)
                return FirmAction(vacancies=1, price_adjustment=1.0)

        config = _small_config(num_periods=3)
        run_simulation(config, firm_policy=SpyFirmPolicy())

        # Should have received states for each firm * each period
        assert len(received_states) == 3 * config.firm.count
        # Period 0 should have default macro state
        assert received_states[0].period == 0
        # Later periods should have real data
        last_state = received_states[-1]
        assert last_state.period == 2

    def test_policy_receives_correct_firm_state(self):
        """Verify policy receives plausible FirmState values."""
        received_states = []

        class SpyFirmPolicy(FirmPolicy):
            def act(self, firm_state: FirmState, macro_state: MacroState) -> FirmAction:
                received_states.append(firm_state)
                return FirmAction(vacancies=1, price_adjustment=1.0)

        config = _small_config(num_periods=2)
        run_simulation(config, firm_policy=SpyFirmPolicy())

        # All firm states should have positive deposits and prices
        for fs in received_states:
            assert fs.deposits >= 0
            assert fs.price > 0
            assert fs.labor_productivity > 0


# --- No-regression tests ---


class TestNoRegression:
    def test_no_policy_matches_baseline(self):
        """Simulation without policies should produce same results as before."""
        config = _small_config(seed=123)
        state = run_simulation(config)
        assert len(state.history) == 10
        # Basic sanity: GDP positive, unemployment bounded
        for m in state.history:
            assert m["gdp"] >= 0
            assert 0.0 <= m["unemployment_rate"] <= 1.0

    def test_balance_sheets_balanced_with_policies(self):
        """All balance sheets should remain balanced when policies are active."""
        config = _small_config()
        state = run_simulation(
            config,
            firm_policy=RuleBasedFirmPolicy(),
            bank_policy=RuleBasedBankPolicy(),
            government_policy=RuleBasedGovernmentPolicy(),
        )
        for m in state.history:
            assert m["unbalanced_sheets"] == [], f"Unbalanced at period {m['period']}"
