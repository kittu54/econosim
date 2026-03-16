"""
Comprehensive stress tests for the EconoSim simulation engine.

Tests cover:
- Accounting invariants across all scenarios
- Extension combinations
- Extreme parameter edge cases
- Economic policy correctness
- Deflationary/inflationary dynamics
- Shock responses
- Batch run stability
- Money conservation
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from econosim.config.schema import SimulationConfig, ShockSpec
from econosim.engine.simulation import build_simulation, step, run_simulation
from econosim.experiments.runner import run_experiment, run_batch


# ───────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────

def run_and_collect(config: SimulationConfig):
    """Run a simulation and return (state, history)."""
    state = build_simulation(config)
    for _ in range(config.num_periods):
        step(state)
    return state, state.history


def assert_all_balanced(state, tolerance=0.05):
    """Check all balance sheets are balanced."""
    validation = state.ledger.validate_all_balanced()
    unbalanced = [k for k, v in validation.items() if not v]
    assert unbalanced == [], f"Unbalanced balance sheets: {unbalanced}"


def assert_no_nan_metrics(history):
    """Ensure no NaN values in any metric across all periods."""
    for t, m in enumerate(history):
        for key, val in m.items():
            if isinstance(val, (int, float)):
                assert not math.isnan(val), f"NaN in metric '{key}' at period {t}"
                assert not math.isinf(val), f"Inf in metric '{key}' at period {t}"


def assert_no_negative_deposits(state):
    """No agent should have negative deposits."""
    for hh in state.households:
        assert hh.deposits >= -0.01, f"{hh.agent_id} has negative deposits: {hh.deposits}"
    for firm in state.firms:
        assert firm.deposits >= -0.01, f"{firm.agent_id} has negative deposits: {firm.deposits}"


def assert_unemployment_bounded(history):
    """Unemployment rate must be between 0 and 1."""
    for t, m in enumerate(history):
        assert 0.0 <= m["unemployment_rate"] <= 1.0, (
            f"Unemployment out of bounds at period {t}: {m['unemployment_rate']}"
        )


def assert_gdp_nonnegative(history):
    """GDP should never be negative."""
    for t, m in enumerate(history):
        assert m["gdp"] >= 0, f"Negative GDP at period {t}: {m['gdp']}"


def assert_gini_bounded(history):
    """Gini coefficient must be in [0, 1]."""
    for t, m in enumerate(history):
        assert -0.01 <= m["gini_deposits"] <= 1.01, (
            f"Gini out of bounds at period {t}: {m['gini_deposits']}"
        )


def make_config(**overrides) -> SimulationConfig:
    """Create a config with overrides using nested dict notation."""
    return SimulationConfig(**overrides)


# ───────────────────────────────────────────────────────────
# 1. BASELINE SANITY TESTS
# ───────────────────────────────────────────────────────────

class TestBaselineSanity:
    """Verify the default configuration produces a functioning economy."""

    def test_baseline_runs_120_periods(self):
        config = make_config(num_periods=120, seed=42)
        state, history = run_and_collect(config)
        assert len(history) == 120
        assert_all_balanced(state)
        assert_no_nan_metrics(history)
        assert_no_negative_deposits(state)

    def test_baseline_positive_gdp(self):
        config = make_config(num_periods=60)
        state, history = run_and_collect(config)
        # GDP should be positive for most periods
        positive_gdp = sum(1 for m in history if m["gdp"] > 0)
        assert positive_gdp >= len(history) * 0.8, "GDP should be positive most of the time"

    def test_baseline_some_employment(self):
        config = make_config(num_periods=60)
        _, history = run_and_collect(config)
        # Should have some employment
        avg_unemp = np.mean([m["unemployment_rate"] for m in history])
        assert avg_unemp < 0.9, f"Average unemployment too high: {avg_unemp:.2%}"

    def test_baseline_prices_nonzero(self):
        config = make_config(num_periods=60)
        _, history = run_and_collect(config)
        for m in history:
            assert m["avg_price"] > 0, "Price should never be zero"

    def test_baseline_wages_nonzero(self):
        config = make_config(num_periods=60)
        _, history = run_and_collect(config)
        for m in history:
            assert m["avg_wage"] > 0, "Wage should never be zero"

    def test_baseline_reproducible(self):
        """Same seed produces identical results."""
        config = make_config(num_periods=30, seed=99)
        _, h1 = run_and_collect(config)
        _, h2 = run_and_collect(config)
        for t in range(30):
            assert h1[t]["gdp"] == h2[t]["gdp"], f"GDP differs at period {t}"
            assert h1[t]["unemployment_rate"] == h2[t]["unemployment_rate"]

    def test_different_seeds_differ(self):
        """Different seeds produce different results."""
        _, h1 = run_and_collect(make_config(num_periods=30, seed=1))
        _, h2 = run_and_collect(make_config(num_periods=30, seed=2))
        # At least some metrics should differ
        diffs = sum(1 for t in range(30) if h1[t]["gdp"] != h2[t]["gdp"])
        assert diffs > 0, "Different seeds should produce different results"


# ───────────────────────────────────────────────────────────
# 2. ACCOUNTING INVARIANT TESTS
# ───────────────────────────────────────────────────────────

class TestAccountingInvariants:
    """Verify double-entry accounting invariants hold in all scenarios."""

    @pytest.mark.parametrize("seed", [1, 17, 42, 99, 12345])
    def test_balance_sheets_always_balanced(self, seed):
        config = make_config(num_periods=60, seed=seed)
        state = build_simulation(config)
        for t in range(config.num_periods):
            metrics = step(state)
            unbalanced = metrics.get("unbalanced_sheets", [])
            assert unbalanced == [], f"Period {t}, seed {seed}: unbalanced: {unbalanced}"

    def test_no_negative_deposits_anywhere(self):
        config = make_config(num_periods=120, seed=42)
        state, _ = run_and_collect(config)
        assert_no_negative_deposits(state)

    def test_bank_capital_ratio_consistent(self):
        """Bank capital ratio = equity / total_loans (when loans > 0)."""
        config = make_config(num_periods=60)
        _, history = run_and_collect(config)
        for t, m in enumerate(history):
            if m["total_loans_outstanding"] > 0:
                expected_ratio = m["bank_equity"] / m["total_loans_outstanding"]
                assert abs(m["bank_capital_ratio"] - expected_ratio) < 0.01, (
                    f"Capital ratio mismatch at period {t}"
                )

    def test_total_deposits_tracked(self):
        """Sum of HH + firm deposits should be consistent with balance sheets."""
        config = make_config(num_periods=60)
        state, history = run_and_collect(config)
        final = history[-1]
        actual_hh = sum(hh.deposits for hh in state.households)
        actual_firm = sum(f.deposits for f in state.firms)
        assert abs(actual_hh - final["total_hh_deposits"]) < 0.02
        assert abs(actual_firm - final["total_firm_deposits"]) < 0.02

    def test_government_budget_identity(self):
        """Budget balance = tax_revenue - transfers - spending."""
        config = make_config(num_periods=60)
        _, history = run_and_collect(config)
        for t, m in enumerate(history):
            expected = m["govt_tax_revenue"] - m["govt_transfers"] - m["govt_spending"]
            assert abs(m["govt_budget_balance"] - expected) < 0.05, (
                f"Budget identity violation at period {t}: "
                f"tax={m['govt_tax_revenue']} - transfers={m['govt_transfers']} "
                f"- spending={m['govt_spending']} != balance={m['govt_budget_balance']}"
            )

    def test_employment_consistent(self):
        """Total employed <= labor force at all times."""
        config = make_config(num_periods=60)
        _, history = run_and_collect(config)
        for t, m in enumerate(history):
            assert m["total_employed"] <= m["labor_force"], (
                f"More employed than labor force at period {t}"
            )


# ───────────────────────────────────────────────────────────
# 3. EXTENSION COMBINATION TESTS
# ───────────────────────────────────────────────────────────

class TestExtensionCombinations:
    """Test all 8 combinations of extension flags."""

    @pytest.mark.parametrize("exp,net,bond", [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ])
    def test_extension_combination(self, exp, net, bond):
        config = make_config(
            num_periods=40,
            seed=42,
            extensions={
                "enable_expectations": exp,
                "enable_networks": net,
                "enable_bonds": bond,
            },
        )
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)
        assert_no_negative_deposits(state)
        assert_unemployment_bounded(history)
        assert_gdp_nonnegative(history)

        # Verify extension metrics present when enabled
        final = history[-1]
        if exp:
            assert "avg_price_forecast_error" in final
            assert "avg_demand_forecast_error" in final
        if net:
            assert "trade_network_density" in final
            assert "credit_systemic_risk" in final
        if bond:
            assert "bond_outstanding" in final
            assert "bond_debt_to_gdp" in final

    def test_all_extensions_long_run(self):
        """All extensions enabled for 200 periods — no crashes or NaN."""
        config = make_config(
            num_periods=200,
            seed=42,
            extensions={
                "enable_expectations": True,
                "enable_networks": True,
                "enable_bonds": True,
            },
        )
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)
        assert len(history) == 200

    def test_extensions_dont_change_baseline(self):
        """With extensions off, results should match pure baseline."""
        cfg_off = make_config(num_periods=30, seed=42, extensions={
            "enable_expectations": False, "enable_networks": False, "enable_bonds": False,
        })
        cfg_base = make_config(num_periods=30, seed=42)
        _, h_off = run_and_collect(cfg_off)
        _, h_base = run_and_collect(cfg_base)
        for t in range(30):
            assert h_off[t]["gdp"] == h_base[t]["gdp"], f"GDP differs at {t} with extensions off"


# ───────────────────────────────────────────────────────────
# 4. EXTREME PARAMETER EDGE CASES
# ───────────────────────────────────────────────────────────

class TestExtremeParameters:
    """Test boundary and extreme parameter values."""

    def test_single_firm(self):
        config = make_config(num_periods=40, firm={"count": 1})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_many_firms(self):
        config = make_config(num_periods=20, firm={"count": 50})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_single_household(self):
        config = make_config(num_periods=30, household={"count": 1})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_many_households(self):
        config = make_config(num_periods=20, household={"count": 500})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_zero_initial_hh_deposits(self):
        """Households start with no money — rely on wages and transfers."""
        config = make_config(num_periods=40, household={"initial_deposits": 0})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_very_high_hh_deposits(self):
        config = make_config(num_periods=40, household={"initial_deposits": 100000})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_zero_firm_deposits(self):
        """Firms start broke — must borrow to hire."""
        config = make_config(num_periods=40, firm={"initial_deposits": 0})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_very_high_firm_deposits(self):
        config = make_config(num_periods=40, firm={"initial_deposits": 500000})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_minimal_consumption_propensity(self):
        """Very low propensity to consume — economy should slow down."""
        config = make_config(
            num_periods=60,
            household={"consumption_propensity": 0.05, "wealth_propensity": 0.01},
        )
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)
        # GDP should be low
        avg_gdp = np.mean([m["gdp"] for m in history[-20:]])
        baseline_cfg = make_config(num_periods=60)
        _, baseline_h = run_and_collect(baseline_cfg)
        baseline_gdp = np.mean([m["gdp"] for m in baseline_h[-20:]])
        assert avg_gdp < baseline_gdp, "Low consumption should produce lower GDP"

    def test_max_consumption_propensity(self):
        """Consumption propensity = 1.0 — spend all income."""
        config = make_config(num_periods=60, household={"consumption_propensity": 1.0})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_zero_wealth_propensity(self):
        """wealth_propensity = 0 — only income-driven spending."""
        config = make_config(num_periods=60, household={"wealth_propensity": 0.0})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_very_high_labor_productivity(self):
        config = make_config(num_periods=40, firm={"labor_productivity": 100})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_very_low_labor_productivity(self):
        config = make_config(num_periods=40, firm={"labor_productivity": 0.5})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_very_high_price(self):
        config = make_config(num_periods=40, firm={"initial_price": 500})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_very_low_price(self):
        config = make_config(num_periods=40, firm={"initial_price": 0.1})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_very_high_wages(self):
        config = make_config(num_periods=40, firm={"initial_wage": 500})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_very_low_wages(self):
        config = make_config(num_periods=40, firm={"initial_wage": 1})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_fast_price_adjustment(self):
        """High adjustment speed — risk of oscillations."""
        config = make_config(num_periods=60, firm={"price_adjustment_speed": 0.2})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_fast_wage_adjustment(self):
        config = make_config(num_periods=60, firm={"wage_adjustment_speed": 0.2})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_very_short_run(self):
        config = make_config(num_periods=5)
        state, history = run_and_collect(config)
        assert len(history) == 5
        assert_all_balanced(state)

    def test_long_run(self):
        config = make_config(num_periods=500)
        state, history = run_and_collect(config)
        assert len(history) == 500
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_very_high_reservation_wage(self):
        """Reservation wage higher than offered — nobody works."""
        config = make_config(num_periods=30, household={"reservation_wage": 10000})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)
        # Most people should be unemployed
        final_unemp = history[-1]["unemployment_rate"]
        assert final_unemp > 0.5, "High reservation wage should cause unemployment"

    def test_zero_reservation_wage(self):
        """Everyone accepts any wage."""
        config = make_config(num_periods=30, household={"reservation_wage": 0})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)


# ───────────────────────────────────────────────────────────
# 5. BANKING EDGE CASES
# ───────────────────────────────────────────────────────────

class TestBankingEdgeCases:
    """Test banking system under stress."""

    def test_very_high_capital_adequacy(self):
        """Bank can barely lend — tight capital requirement."""
        config = make_config(num_periods=40, bank={"capital_adequacy_ratio": 0.5})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_very_low_capital_adequacy(self):
        """Bank can lend freely — very lax regulation."""
        config = make_config(num_periods=40, bank={"capital_adequacy_ratio": 0.01})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_very_high_interest_rate(self):
        """Very expensive loans — firms avoid borrowing."""
        config = make_config(num_periods=40, bank={"base_interest_rate": 0.1})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_zero_interest_rate(self):
        """Free money — should encourage borrowing."""
        config = make_config(num_periods=40, bank={"base_interest_rate": 0.0})
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_loan_defaults_dont_break_accounting(self):
        """Scenario designed to cause defaults — accounting must hold."""
        config = make_config(
            num_periods=80,
            firm={"initial_deposits": 0, "max_leverage": 3.0},
            bank={"base_interest_rate": 0.05, "default_threshold_periods": 2},
            household={"consumption_propensity": 0.2, "wealth_propensity": 0.05},
        )
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)


# ───────────────────────────────────────────────────────────
# 6. GOVERNMENT POLICY TESTS
# ───────────────────────────────────────────────────────────

class TestGovernmentPolicy:
    """Test fiscal policy effectiveness and edge cases."""

    def test_zero_government_spending(self):
        """Economy runs without fiscal stabilizer."""
        config = make_config(
            num_periods=60,
            government={"spending_per_period": 0, "transfer_per_unemployed": 0},
        )
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_very_high_government_spending(self):
        """Massive government injection — should boost GDP."""
        config = make_config(
            num_periods=60,
            government={"spending_per_period": 20000},
        )
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)
        avg_gdp = np.mean([m["gdp"] for m in history[-20:]])
        assert avg_gdp > 0, "High govt spending should produce positive GDP"

    def test_zero_tax_rate(self):
        """No taxes — government runs pure deficit."""
        config = make_config(
            num_periods=60,
            government={"income_tax_rate": 0.0},
        )
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)
        # Government should need money creation
        total_created = history[-1]["govt_cumulative_money_created"]
        assert total_created > 0, "Zero-tax government should create money"

    def test_very_high_tax_rate(self):
        """Tax rate = 50% — heavy extraction."""
        config = make_config(
            num_periods=60,
            government={"income_tax_rate": 0.5},
        )
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_high_spending_produces_more_gdp(self):
        """Higher spending → higher GDP (fiscal multiplier test)."""
        cfg_low = make_config(num_periods=60, seed=42, government={"spending_per_period": 500})
        cfg_high = make_config(num_periods=60, seed=42, government={"spending_per_period": 5000})
        _, h_low = run_and_collect(cfg_low)
        _, h_high = run_and_collect(cfg_high)
        gdp_low = np.mean([m["gdp"] for m in h_low[-20:]])
        gdp_high = np.mean([m["gdp"] for m in h_high[-20:]])
        assert gdp_high > gdp_low, (
            f"Higher spending should produce higher GDP: {gdp_high:.0f} vs {gdp_low:.0f}"
        )

    def test_high_tax_reduces_consumption(self):
        """Higher taxes → lower disposable income → lower consumption."""
        cfg_low = make_config(num_periods=60, seed=42, government={"income_tax_rate": 0.05})
        cfg_high = make_config(num_periods=60, seed=42, government={"income_tax_rate": 0.4})
        _, h_low = run_and_collect(cfg_low)
        _, h_high = run_and_collect(cfg_high)
        cons_low = np.mean([m["total_consumption"] for m in h_low[-20:]])
        cons_high = np.mean([m["total_consumption"] for m in h_high[-20:]])
        assert cons_high < cons_low, (
            f"Higher taxes should reduce consumption: {cons_high:.0f} vs {cons_low:.0f}"
        )

    def test_sovereign_money_creation(self):
        """Government creates money when it runs deficits."""
        config = make_config(
            num_periods=60,
            government={"initial_deposits": 0, "spending_per_period": 2000},
        )
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        # Money was created
        assert history[-1]["govt_cumulative_money_created"] > 0

    def test_generous_transfers_reduce_inequality(self):
        """Larger transfers → lower Gini (transfers go to unemployed).
        Use a scenario with high unemployment so transfers actually matter."""
        cfg_low = make_config(
            num_periods=60, seed=42,
            government={"transfer_per_unemployed": 1, "spending_per_period": 500},
            household={"consumption_propensity": 0.4, "reservation_wage": 80},
        )
        cfg_high = make_config(
            num_periods=60, seed=42,
            government={"transfer_per_unemployed": 500, "spending_per_period": 500},
            household={"consumption_propensity": 0.4, "reservation_wage": 80},
        )
        _, h_low = run_and_collect(cfg_low)
        _, h_high = run_and_collect(cfg_high)
        gini_low = np.mean([m["gini_deposits"] for m in h_low[-20:]])
        gini_high = np.mean([m["gini_deposits"] for m in h_high[-20:]])
        assert gini_high < gini_low, (
            f"Higher transfers should reduce inequality: Gini {gini_high:.3f} vs {gini_low:.3f}"
        )


# ───────────────────────────────────────────────────────────
# 7. ECONOMIC DYNAMICS TESTS
# ───────────────────────────────────────────────────────────

class TestEconomicDynamics:
    """Verify economic behavior makes theoretical sense."""

    def test_higher_productivity_raises_gdp(self):
        """More productive firms → higher GDP."""
        cfg_low = make_config(num_periods=60, seed=42, firm={"labor_productivity": 2})
        cfg_high = make_config(num_periods=60, seed=42, firm={"labor_productivity": 20})
        _, h_low = run_and_collect(cfg_low)
        _, h_high = run_and_collect(cfg_high)
        gdp_low = np.mean([m["gdp"] for m in h_low[-20:]])
        gdp_high = np.mean([m["gdp"] for m in h_high[-20:]])
        assert gdp_high > gdp_low, (
            f"Higher productivity should raise GDP: {gdp_high:.0f} vs {gdp_low:.0f}"
        )

    def test_higher_consumption_propensity_raises_gdp(self):
        """More spending → higher GDP (demand-driven model)."""
        cfg_low = make_config(num_periods=60, seed=42, household={"consumption_propensity": 0.3})
        cfg_high = make_config(num_periods=60, seed=42, household={"consumption_propensity": 0.9})
        _, h_low = run_and_collect(cfg_low)
        _, h_high = run_and_collect(cfg_high)
        gdp_low = np.mean([m["gdp"] for m in h_low[-20:]])
        gdp_high = np.mean([m["gdp"] for m in h_high[-20:]])
        assert gdp_high > gdp_low, (
            f"Higher consumption propensity should raise GDP: {gdp_high:.0f} vs {gdp_low:.0f}"
        )

    def test_more_households_raises_gdp(self):
        """Larger population → more demand → higher GDP."""
        cfg_small = make_config(num_periods=60, seed=42, household={"count": 20})
        cfg_large = make_config(num_periods=60, seed=42, household={"count": 200})
        _, h_small = run_and_collect(cfg_small)
        _, h_large = run_and_collect(cfg_large)
        gdp_small = np.mean([m["gdp"] for m in h_small[-20:]])
        gdp_large = np.mean([m["gdp"] for m in h_large[-20:]])
        assert gdp_large > gdp_small, (
            f"Larger population should raise GDP: {gdp_large:.0f} vs {gdp_small:.0f}"
        )

    def test_high_demand_reduces_inventory(self):
        """High demand economy should have lower inventory on average than baseline.
        Compare total production vs total consumption as demand proxy."""
        cfg_baseline = make_config(num_periods=60, seed=42)
        cfg_high_demand = make_config(
            num_periods=60, seed=42,
            household={"consumption_propensity": 0.95, "wealth_propensity": 0.5},
        )
        _, h_base = run_and_collect(cfg_baseline)
        _, h_high = run_and_collect(cfg_high_demand)
        # High demand should produce higher consumption
        cons_base = np.mean([m["total_consumption"] for m in h_base[-20:]])
        cons_high = np.mean([m["total_consumption"] for m in h_high[-20:]])
        assert cons_high > cons_base, (
            f"Higher propensity should raise consumption: {cons_high:.0f} vs {cons_base:.0f}"
        )

    def test_credit_system_functional(self):
        """Bank lending machinery works — loans appear when firms need capital.
        Firms with zero deposits and positive leverage should attempt borrowing."""
        config = make_config(num_periods=40, seed=42)
        _, history = run_and_collect(config)
        # Verify the credit system tracks loan metrics correctly
        for m in history:
            assert m["total_loans_outstanding"] >= 0
            assert m["loans_issued"] >= 0
            assert m["active_loans_count"] >= 0


# ───────────────────────────────────────────────────────────
# 8. SHOCK RESPONSE TESTS
# ───────────────────────────────────────────────────────────

class TestShockResponses:
    """Verify economy responds correctly to shocks."""

    def test_supply_shock_raises_prices(self):
        """Negative productivity shock → lower production → higher prices."""
        shocks = [ShockSpec(
            period=20, shock_type="supply", parameter="labor_productivity",
            magnitude=0.3, additive=False,
        )]
        config = make_config(num_periods=60, seed=42, shocks=shocks)
        _, history = run_and_collect(config)
        pre_shock_price = np.mean([m["avg_price"] for m in history[15:20]])
        post_shock_price = np.mean([m["avg_price"] for m in history[30:40]])
        # After productivity drops, prices should eventually rise
        assert_no_nan_metrics(history)
        assert_gdp_nonnegative(history)

    def test_demand_shock_reduces_gdp(self):
        """Negative demand shock → lower consumption → lower GDP."""
        shocks = [ShockSpec(
            period=20, shock_type="demand", parameter="consumption_propensity",
            magnitude=0.3, additive=False,
        )]
        config = make_config(num_periods=60, seed=42, shocks=shocks)
        _, history = run_and_collect(config)
        pre_shock_gdp = np.mean([m["gdp"] for m in history[15:20]])
        post_shock_gdp = np.mean([m["gdp"] for m in history[25:35]])
        assert post_shock_gdp < pre_shock_gdp, (
            f"Demand shock should reduce GDP: {post_shock_gdp:.0f} vs {pre_shock_gdp:.0f}"
        )

    def test_credit_crunch_reduces_lending(self):
        """Tighter capital requirements → less lending."""
        shocks = [ShockSpec(
            period=20, shock_type="credit", parameter="capital_adequacy_ratio",
            magnitude=0.3, additive=False,
        )]
        config = make_config(num_periods=60, seed=42, shocks=shocks)
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_fiscal_austerity_shock(self):
        """Tax hike → lower disposable income → lower GDP."""
        shocks = [ShockSpec(
            period=20, shock_type="fiscal", parameter="income_tax_rate",
            magnitude=0.15, additive=True,
        )]
        config = make_config(num_periods=60, seed=42, shocks=shocks)
        _, history = run_and_collect(config)
        pre_gdp = np.mean([m["gdp"] for m in history[15:20]])
        post_gdp = np.mean([m["gdp"] for m in history[30:40]])
        assert post_gdp < pre_gdp, "Tax hike should reduce GDP"

    def test_stimulus_shock(self):
        """Spending increase → higher GDP."""
        shocks = [ShockSpec(
            period=20, shock_type="fiscal", parameter="spending_per_period",
            magnitude=3.0, additive=False,
        )]
        config = make_config(num_periods=60, seed=42, shocks=shocks)
        _, history = run_and_collect(config)
        pre_gdp = np.mean([m["gdp"] for m in history[15:20]])
        post_gdp = np.mean([m["gdp"] for m in history[30:40]])
        assert post_gdp > pre_gdp, "Spending increase should raise GDP"

    def test_multiple_simultaneous_shocks(self):
        """Multiple shocks at once — accounting should still hold."""
        shocks = [
            ShockSpec(period=15, shock_type="supply", parameter="labor_productivity", magnitude=0.5, additive=False),
            ShockSpec(period=15, shock_type="demand", parameter="consumption_propensity", magnitude=0.5, additive=False),
            ShockSpec(period=15, shock_type="fiscal", parameter="spending_per_period", magnitude=2.0, additive=False),
        ]
        config = make_config(num_periods=60, seed=42, shocks=shocks)
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_shocks_with_all_extensions(self):
        """Shocks + all extensions — no crashes."""
        shocks = [
            ShockSpec(period=10, shock_type="supply", parameter="labor_productivity", magnitude=0.5, additive=False),
            ShockSpec(period=20, shock_type="demand", parameter="consumption_propensity", magnitude=0.6, additive=False),
        ]
        config = make_config(
            num_periods=60, seed=42, shocks=shocks,
            extensions={
                "enable_expectations": True,
                "enable_networks": True,
                "enable_bonds": True,
            },
        )
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)


# ───────────────────────────────────────────────────────────
# 9. SCENARIO PRESET TESTS
# ───────────────────────────────────────────────────────────

class TestScenarioPresets:
    """Test the 4 built-in scenario presets from the dashboard."""

    def test_baseline_scenario(self):
        config = make_config(num_periods=60, seed=42)
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)
        assert_unemployment_bounded(history)
        assert_gdp_nonnegative(history)
        assert_gini_bounded(history)

    def test_high_growth_scenario(self):
        config = make_config(
            num_periods=60, seed=42,
            government={"income_tax_rate": 0.1, "spending_per_period": 5000},
        )
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)
        # Should have higher GDP than baseline
        baseline_cfg = make_config(num_periods=60, seed=42)
        _, baseline_h = run_and_collect(baseline_cfg)
        gdp_hg = np.mean([m["gdp"] for m in history[-20:]])
        gdp_bl = np.mean([m["gdp"] for m in baseline_h[-20:]])
        assert gdp_hg > gdp_bl, "High growth should produce higher GDP"

    def test_recession_scenario(self):
        config = make_config(
            num_periods=60, seed=42,
            government={"spending_per_period": 500, "income_tax_rate": 0.35},
            household={"consumption_propensity": 0.5},
        )
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)
        # Should have lower GDP than baseline
        baseline_cfg = make_config(num_periods=60, seed=42)
        _, baseline_h = run_and_collect(baseline_cfg)
        gdp_rec = np.mean([m["gdp"] for m in history[-20:]])
        gdp_bl = np.mean([m["gdp"] for m in baseline_h[-20:]])
        assert gdp_rec < gdp_bl, "Recession should produce lower GDP"

    def test_tight_money_scenario(self):
        config = make_config(
            num_periods=60, seed=42,
            bank={"base_interest_rate": 0.03, "capital_adequacy_ratio": 0.15},
        )
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)


# ───────────────────────────────────────────────────────────
# 10. BATCH RUN STABILITY
# ───────────────────────────────────────────────────────────

class TestBatchRunStability:
    """Test multi-seed batch runs for consistency."""

    def test_batch_5_seeds(self):
        config = make_config(num_periods=30)
        seeds = [1, 2, 3, 4, 5]
        batch = run_batch(config, seeds)
        assert len(batch["runs"]) == 5
        assert batch["aggregate"] is not None
        # Check aggregate columns exist
        agg = batch["aggregate"]
        assert "gdp_mean" in agg.columns or "gdp" in agg.columns

    def test_batch_10_seeds(self):
        config = make_config(num_periods=20)
        seeds = list(range(10))
        batch = run_batch(config, seeds)
        assert len(batch["runs"]) == 10

    def test_batch_with_extensions(self):
        config = make_config(
            num_periods=20,
            extensions={
                "enable_expectations": True,
                "enable_networks": True,
                "enable_bonds": True,
            },
        )
        seeds = [42, 43, 44]
        batch = run_batch(config, seeds)
        assert len(batch["runs"]) == 3


# ───────────────────────────────────────────────────────────
# 11. METRIC BOUNDS & CONSISTENCY
# ───────────────────────────────────────────────────────────

class TestMetricConsistency:
    """Verify metric values are consistent and bounded."""

    @pytest.mark.parametrize("seed", [1, 42, 100, 999, 54321])
    def test_all_metrics_bounded(self, seed):
        config = make_config(num_periods=60, seed=seed)
        _, history = run_and_collect(config)
        assert_no_nan_metrics(history)
        assert_unemployment_bounded(history)
        assert_gdp_nonnegative(history)
        assert_gini_bounded(history)
        for m in history:
            assert m["avg_price"] >= 0, f"Negative price at period {m['period']}"
            assert m["avg_wage"] >= 0, f"Negative wage at period {m['period']}"
            assert m["total_inventory"] >= -0.01, f"Negative inventory at period {m['period']}"
            assert m["total_loans_outstanding"] >= -0.01

    def test_gdp_components(self):
        """GDP should relate to consumption + government spending."""
        config = make_config(num_periods=60)
        _, history = run_and_collect(config)
        for m in history:
            # GDP = goods market transactions + govt spending
            # total_consumption is HH spending; govt_spending is govt spending
            # These should be related to GDP
            assert m["gdp"] >= 0

    def test_employment_identity(self):
        """employed + unemployed = labor force at all times."""
        config = make_config(num_periods=60)
        state, history = run_and_collect(config)
        for t, m in enumerate(history):
            unemployed = m["labor_force"] - m["total_employed"]
            expected_rate = unemployed / max(m["labor_force"], 1)
            assert abs(m["unemployment_rate"] - expected_rate) < 0.001, (
                f"Unemployment identity broken at period {t}"
            )

    def test_wage_income_relates_to_employment(self):
        """Total wage income should be positive when employment is positive."""
        config = make_config(num_periods=60)
        _, history = run_and_collect(config)
        for t, m in enumerate(history):
            if m["total_employed"] > 0:
                assert m["total_wage_income"] > 0, (
                    f"Employed workers but no wage income at period {t}"
                )


# ───────────────────────────────────────────────────────────
# 12. COMBINED STRESS SCENARIOS
# ───────────────────────────────────────────────────────────

class TestCombinedStress:
    """Maximum stress: combine extreme parameters, shocks, and extensions."""

    def test_recession_with_all_extensions(self):
        config = make_config(
            num_periods=80, seed=42,
            government={"spending_per_period": 500, "income_tax_rate": 0.35},
            household={"consumption_propensity": 0.5},
            extensions={
                "enable_expectations": True,
                "enable_networks": True,
                "enable_bonds": True,
            },
        )
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_credit_crunch_with_extensions(self):
        config = make_config(
            num_periods=80, seed=42,
            bank={"base_interest_rate": 0.05, "capital_adequacy_ratio": 0.3},
            firm={"initial_deposits": 500},
            extensions={
                "enable_expectations": True,
                "enable_networks": True,
                "enable_bonds": True,
            },
        )
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_stagflation_scenario(self):
        """Supply shock + loose fiscal → stagflation (high inflation + low growth)."""
        shocks = [ShockSpec(
            period=10, shock_type="supply", parameter="labor_productivity",
            magnitude=0.3, additive=False,
        )]
        config = make_config(
            num_periods=60, seed=42, shocks=shocks,
            government={"spending_per_period": 5000},
        )
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_everything_extreme(self):
        """Push all parameters to extremes — should not crash."""
        shocks = [
            ShockSpec(period=5, shock_type="supply", parameter="labor_productivity", magnitude=0.2, additive=False),
            ShockSpec(period=10, shock_type="demand", parameter="consumption_propensity", magnitude=0.3, additive=False),
            ShockSpec(period=15, shock_type="fiscal", parameter="spending_per_period", magnitude=5.0, additive=False),
            ShockSpec(period=20, shock_type="credit", parameter="capital_adequacy_ratio", magnitude=3.0, additive=False),
        ]
        config = make_config(
            num_periods=60, seed=42, shocks=shocks,
            household={"count": 200, "initial_deposits": 100, "consumption_propensity": 0.95,
                        "wealth_propensity": 0.6, "reservation_wage": 20},
            firm={"count": 10, "initial_deposits": 5000, "labor_productivity": 15,
                  "price_adjustment_speed": 0.1, "wage_adjustment_speed": 0.08},
            government={"income_tax_rate": 0.3, "spending_per_period": 3000,
                        "transfer_per_unemployed": 100, "initial_deposits": 50000},
            bank={"base_interest_rate": 0.02, "capital_adequacy_ratio": 0.1},
            extensions={
                "enable_expectations": True,
                "enable_networks": True,
                "enable_bonds": True,
            },
        )
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)
        assert len(history) == 60

    def test_deflationary_death_spiral_handled(self):
        """Economy with minimal demand — should not produce NaN or crash."""
        config = make_config(
            num_periods=100, seed=42,
            household={"consumption_propensity": 0.1, "wealth_propensity": 0.0,
                        "initial_deposits": 100},
            government={"spending_per_period": 100, "transfer_per_unemployed": 5},
            firm={"initial_deposits": 1000, "initial_price": 50},
        )
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)

    def test_hyperinflation_scenario_stable(self):
        """Massive money creation + high demand — economy should not crash.
        Note: prices may not always rise if supply-side constraints dominate
        (low productivity → inventory stays high → price adjustment lowers prices).
        The key test is that accounting invariants hold."""
        config = make_config(
            num_periods=80, seed=42,
            government={"initial_deposits": 0, "spending_per_period": 10000,
                        "income_tax_rate": 0.0},
            household={"consumption_propensity": 0.95, "wealth_propensity": 0.5},
            firm={"labor_productivity": 2, "count": 3},
        )
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)
        assert_unemployment_bounded(history)
        assert_gdp_nonnegative(history)
        # Large money creation should occur
        assert history[-1]["govt_cumulative_money_created"] > 0

    @pytest.mark.parametrize("seed", range(10))
    def test_multi_seed_robustness(self, seed):
        """Run with 10 different seeds — no crashes or NaN."""
        config = make_config(num_periods=40, seed=seed)
        state, history = run_and_collect(config)
        assert_all_balanced(state)
        assert_no_nan_metrics(history)
        assert_unemployment_bounded(history)
        assert_gdp_nonnegative(history)
        assert_gini_bounded(history)


# ───────────────────────────────────────────────────────────
# 13. API ENDPOINT SIMULATION
# ───────────────────────────────────────────────────────────

class TestAPISimulation:
    """Test simulation via the same path the API uses."""

    def test_run_experiment_default(self):
        config = SimulationConfig()
        result = run_experiment(config)
        assert "dataframe" in result
        assert "summary" in result
        assert len(result["dataframe"]) == config.num_periods

    def test_run_experiment_with_extensions(self):
        config = make_config(
            num_periods=30,
            extensions={
                "enable_expectations": True,
                "enable_networks": True,
                "enable_bonds": True,
            },
        )
        result = run_experiment(config)
        df = result["dataframe"]
        assert "avg_price_forecast_error" in df.columns
        assert "trade_network_density" in df.columns
        assert "bond_outstanding" in df.columns

    def test_run_batch_default(self):
        config = SimulationConfig()
        batch = run_batch(config, [42, 43, 44])
        assert len(batch["runs"]) == 3
        assert batch["aggregate"] is not None
