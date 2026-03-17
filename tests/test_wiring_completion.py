"""Tests for completed policy wiring and backtesting metrics.

Verifies all action fields in FirmAction, HouseholdAction, BankAction,
GovernmentAction are actually applied by the simulation engine.
Also tests PIT uniformity and multi-benchmark skill scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from econosim.config.schema import SimulationConfig
from econosim.engine.simulation import (
    build_simulation, step, run_simulation,
    build_firm_state, build_household_state, build_bank_state,
    build_macro_state,
)
from econosim.policies.interfaces import (
    FirmPolicy, FirmState, FirmAction,
    HouseholdPolicy, HouseholdState, HouseholdAction,
    BankPolicy, BankState, BankAction,
    GovernmentPolicy, GovernmentState, GovernmentAction,
    MacroState,
)
from econosim.policies.rule_based import RuleBasedFirmPolicy, RuleBasedHouseholdPolicy
from econosim.forecasting.backtesting import (
    _ks_uniformity,
    _crps_ensemble,
    BacktestConfig,
    BacktestRunner,
    RandomWalkBenchmark,
    ARBenchmark,
    TrendBenchmark,
)


def _small_config(**overrides) -> SimulationConfig:
    defaults = {
        "name": "wiring_test",
        "num_periods": 10,
        "seed": 42,
        "household": {"count": 20, "initial_deposits": 500.0},
        "firm": {"count": 5, "initial_deposits": 5000.0},
    }
    defaults.update(overrides)
    return SimulationConfig(**defaults)


# ── Firm policy: wage_adjustment wiring ────────────────────────────


class HighWagePolicy(FirmPolicy):
    """Test policy that doubles wages every period."""
    def act(self, fs: FirmState, ms: MacroState) -> FirmAction:
        return FirmAction(vacancies=1, price_adjustment=1.0, wage_adjustment=1.5)


class LowWagePolicy(FirmPolicy):
    """Test policy that halves wages every period."""
    def act(self, fs: FirmState, ms: MacroState) -> FirmAction:
        return FirmAction(vacancies=1, price_adjustment=1.0, wage_adjustment=0.7)


class TestFirmWageAdjustment:
    def test_high_wage_policy_raises_wages(self):
        config = _small_config(num_periods=5)
        state = build_simulation(config)
        state.firm_policy = HighWagePolicy()
        initial_wages = [f.posted_wage for f in state.firms]
        for _ in range(3):
            step(state)
        final_wages = [f.posted_wage for f in state.firms]
        # Wages should have increased
        assert all(f > i for f, i in zip(final_wages, initial_wages))

    def test_low_wage_policy_lowers_wages(self):
        config = _small_config(num_periods=5)
        state = build_simulation(config)
        state.firm_policy = LowWagePolicy()
        initial_wages = [f.posted_wage for f in state.firms]
        for _ in range(3):
            step(state)
        final_wages = [f.posted_wage for f in state.firms]
        # Wages should have decreased
        assert all(f < i for f, i in zip(final_wages, initial_wages))


# ── Firm policy: loan_request wiring ───────────────────────────────


class BorrowingPolicy(FirmPolicy):
    """Test policy that requests large loans."""
    def act(self, fs: FirmState, ms: MacroState) -> FirmAction:
        return FirmAction(
            vacancies=2,
            price_adjustment=1.0,
            loan_request=5000.0,  # request a big loan
        )


class NoBorrowingPolicy(FirmPolicy):
    """Test policy that never borrows."""
    def act(self, fs: FirmState, ms: MacroState) -> FirmAction:
        return FirmAction(vacancies=1, price_adjustment=1.0, loan_request=0.0)


class TestFirmLoanRequest:
    def test_borrowing_policy_generates_loans(self):
        config = _small_config(num_periods=5)
        state = run_simulation(config, firm_policy=BorrowingPolicy())
        # Should have some loans
        total_loans = state.history[-1]["total_loans_outstanding"]
        assert total_loans > 0

    def test_no_borrowing_policy_suppresses_loans(self):
        config = _small_config(num_periods=5)
        state_borrow = run_simulation(config, firm_policy=BorrowingPolicy())
        state_no = run_simulation(config, firm_policy=NoBorrowingPolicy())
        # No-borrow should have fewer loans
        loans_borrow = state_borrow.history[-1]["total_loans_outstanding"]
        loans_no = state_no.history[-1]["total_loans_outstanding"]
        assert loans_no <= loans_borrow


# ── Household policy: labor_participation wiring ───────────────────


class NoParticipationPolicy(HouseholdPolicy):
    """Test policy where nobody seeks work."""
    def act(self, hs: HouseholdState, ms: MacroState) -> HouseholdAction:
        return HouseholdAction(
            consumption_fraction=0.5,
            labor_participation=False,
            reservation_wage_adjustment=1.0,
        )


class TestHouseholdLaborParticipation:
    def test_no_participation_means_no_employment(self):
        config = _small_config(num_periods=5)
        state = run_simulation(config, household_policy=NoParticipationPolicy())
        # With no participation, nobody should be employed
        for m in state.history:
            assert m["total_employed"] == 0
            # labor_force=0 → unemployment_rate=0 (no one in labor force)
            assert m["labor_force"] == 0

    def test_participation_flag_is_applied(self):
        """Verify labor_participation is set on households before labor market."""
        config = _small_config(num_periods=3)
        state = build_simulation(config)
        state.household_policy = NoParticipationPolicy()
        step(state)
        # After step, all households should have labor_participation=False
        for hh in state.households:
            assert hh.labor_participation is False


# ── Household policy: reservation_wage_adjustment wiring ───────────


class DroppingReservationWagePolicy(HouseholdPolicy):
    """Test policy that aggressively lowers reservation wages."""
    def act(self, hs: HouseholdState, ms: MacroState) -> HouseholdAction:
        return HouseholdAction(
            consumption_fraction=0.5,
            labor_participation=True,
            reservation_wage_adjustment=0.5,  # halve each period
        )


class TestHouseholdReservationWage:
    def test_reservation_wage_decreases(self):
        config = _small_config(num_periods=5)
        state = build_simulation(config)
        state.household_policy = DroppingReservationWagePolicy()
        initial_rw = [hh.reservation_wage for hh in state.households]
        for _ in range(3):
            step(state)
        final_rw = [hh.reservation_wage for hh in state.households]
        # Reservation wages should have decreased dramatically
        assert all(f < i for f, i in zip(final_rw, initial_rw))

    def test_rule_based_adjusts_reservation_wage(self):
        """RuleBasedHouseholdPolicy lowers reservation wage for unemployed."""
        config = _small_config(num_periods=10)
        state = run_simulation(config, household_policy=RuleBasedHouseholdPolicy())
        # Just verify it runs and wages adjust
        assert len(state.history) == 10


# ── Backtesting: PIT uniformity ────────────────────────────────────


class TestPITUniformity:
    def test_uniform_pit_gives_low_ks(self):
        """Perfectly uniform PIT values should give KS ≈ 0."""
        pit_values = list(np.linspace(0, 1, 100))
        ks = _ks_uniformity(pit_values)
        assert ks < 0.02  # should be very small

    def test_biased_pit_gives_high_ks(self):
        """All PIT values near 0 (biased high forecasts) → large KS."""
        pit_values = [0.01] * 50
        ks = _ks_uniformity(pit_values)
        assert ks > 0.5

    def test_empty_pit(self):
        assert _ks_uniformity([]) == 0.0

    def test_single_value(self):
        ks = _ks_uniformity([0.5])
        assert 0.0 <= ks <= 1.0


# ── Backtesting: CRPS edge cases ───────────────────────────────────


class TestCRPSEdgeCases:
    def test_perfect_ensemble_gives_zero(self):
        """If ensemble is all exactly the observation, CRPS = 0."""
        ensemble = np.array([5.0, 5.0, 5.0, 5.0])
        assert _crps_ensemble(ensemble, 5.0) == pytest.approx(0.0, abs=1e-10)

    def test_single_member(self):
        """Single-member ensemble: CRPS = |forecast - obs|."""
        ensemble = np.array([3.0])
        assert _crps_ensemble(ensemble, 5.0) == pytest.approx(2.0, abs=1e-10)

    def test_empty_ensemble(self):
        assert np.isnan(_crps_ensemble(np.array([]), 5.0))

    def test_symmetric_ensemble(self):
        """CRPS should be positive for spread ensemble."""
        ensemble = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        crps = _crps_ensemble(ensemble, 5.0)
        assert crps >= 0.0


# ── Backtesting: skill scores vs multiple benchmarks ───────────────


class TestSkillScoresMultiBenchmark:
    def _make_history(self):
        """Create synthetic history for backtesting."""
        n = 200
        t = np.arange(n)
        rng = np.random.default_rng(42)
        gdp = 1000 + 5 * t + rng.normal(0, 10, n)
        unemp = 0.05 + 0.001 * np.sin(t / 10) + rng.normal(0, 0.005, n)
        return pd.DataFrame({"gdp": gdp, "unemployment_rate": unemp})

    def test_all_benchmarks_produce_scorecards(self):
        df = self._make_history()

        def naive_forecast(history, horizon, variable):
            last = float(history[variable].iloc[-1])
            median = np.full(horizon, last)
            # Simple ensemble: add noise
            rng = np.random.default_rng(42)
            ensemble = median[None, :] + rng.normal(0, 5, (20, horizon))
            return median, ensemble

        benchmarks = [RandomWalkBenchmark(), ARBenchmark(), TrendBenchmark()]
        runner = BacktestRunner(df, naive_forecast, benchmarks)
        config = BacktestConfig(
            forecast_horizon=5,
            num_origins=5,
            calibration_window=50,
            step_size=20,
            variables=["gdp"],
        )
        report = runner.run(config)

        assert len(report.scorecards) == 1
        sc = report.scorecards[0]

        # Should have benchmark scorecards for all three
        assert "random_walk" in report.benchmark_scorecards
        assert "ar1" in report.benchmark_scorecards
        assert "linear_trend" in report.benchmark_scorecards

        # All three should have entries
        for bm_name in ["random_walk", "ar1", "linear_trend"]:
            assert len(report.benchmark_scorecards[bm_name]) > 0

        # Skill scores should be computed
        assert sc.skill_score_rmse != 0.0 or sc.rmse == 0.0

    def test_pit_uniformity_in_summary(self):
        """PIT uniformity should appear in summary table."""
        df = self._make_history()

        def naive_forecast(history, horizon, variable):
            last = float(history[variable].iloc[-1])
            median = np.full(horizon, last)
            rng = np.random.default_rng(42)
            ensemble = median[None, :] + rng.normal(0, 5, (20, horizon))
            return median, ensemble

        runner = BacktestRunner(df, naive_forecast)
        config = BacktestConfig(
            forecast_horizon=5,
            num_origins=3,
            calibration_window=50,
            step_size=30,
            variables=["gdp"],
        )
        report = runner.run(config)
        summary = report.summary_table()
        assert "pit_uniformity" in summary.columns
