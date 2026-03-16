"""Tests for the full policy pipeline: household policy, calibration, forecasting, RL env.

Verifies:
1. Household policy overrides consumption in goods market
2. Calibration engine accepts and uses policies
3. Forecasting engine accepts and uses policies
4. RL environment runs with all three roles
5. End-to-end: calibrate → forecast with policies
"""

from __future__ import annotations

import numpy as np
import pytest

from econosim.config.schema import SimulationConfig
from econosim.engine.simulation import (
    build_simulation,
    build_household_state,
    step,
    run_simulation,
)
from econosim.policies.interfaces import (
    HouseholdPolicy, HouseholdState, HouseholdAction,
    FirmPolicy, FirmState, FirmAction,
    GovernmentPolicy, GovernmentState, GovernmentAction,
    MacroState,
)
from econosim.policies.rule_based import (
    RuleBasedFirmPolicy,
    RuleBasedHouseholdPolicy,
    RuleBasedBankPolicy,
    RuleBasedGovernmentPolicy,
)
from econosim.calibration.engine import (
    SimulationObjective,
    SmmCalibrator,
    CalibrationProfile,
)
from econosim.calibration.parameters import default_macro_registry
from econosim.calibration.moments import default_us_moments
from econosim.forecasting.engine import (
    ForecastConfig,
    ForecastEnsembleRunner,
    ScenarioSpec,
)
from econosim.rl.macro_env import MacroEnv


def _small_config(**overrides) -> SimulationConfig:
    defaults = {
        "name": "pipeline_test",
        "num_periods": 10,
        "seed": 42,
        "household": {"count": 20, "initial_deposits": 500.0},
        "firm": {"count": 5, "initial_deposits": 5000.0},
    }
    defaults.update(overrides)
    return SimulationConfig(**defaults)


# --- Household policy tests ---


class ZeroConsumptionPolicy(HouseholdPolicy):
    """Test policy: households don't consume."""
    def act(self, hh_state: HouseholdState, macro_state: MacroState) -> HouseholdAction:
        return HouseholdAction(consumption_fraction=0.0)


class FullConsumptionPolicy(HouseholdPolicy):
    """Test policy: households spend everything."""
    def act(self, hh_state: HouseholdState, macro_state: MacroState) -> HouseholdAction:
        return HouseholdAction(consumption_fraction=1.0)


class TestHouseholdPolicy:
    def test_zero_consumption_suppresses_spending(self):
        config = _small_config(num_periods=5)
        state = run_simulation(config, household_policy=ZeroConsumptionPolicy())
        # With zero consumption, total_consumption should be 0
        for m in state.history:
            assert m["total_consumption"] == 0.0

    def test_full_consumption_maximizes_spending(self):
        config = _small_config(num_periods=5)
        state_full = run_simulation(config, household_policy=FullConsumptionPolicy())
        state_zero = run_simulation(config, household_policy=ZeroConsumptionPolicy())
        # Full should have more consumption than zero
        total_full = sum(m["total_consumption"] for m in state_full.history)
        total_zero = sum(m["total_consumption"] for m in state_zero.history)
        assert total_full > total_zero

    def test_rule_based_household_policy(self):
        config = _small_config()
        state = run_simulation(config, household_policy=RuleBasedHouseholdPolicy())
        assert len(state.history) == 10
        assert all(m["gdp"] >= 0 for m in state.history)

    def test_build_household_state(self):
        state = build_simulation(_small_config())
        hh = state.households[0]
        hs = build_household_state(hh)
        assert hs.deposits == hh.deposits
        assert hs.consumption_propensity == hh.consumption_propensity

    def test_all_four_policies_together(self):
        config = _small_config()
        state = run_simulation(
            config,
            firm_policy=RuleBasedFirmPolicy(),
            household_policy=RuleBasedHouseholdPolicy(),
            bank_policy=RuleBasedBankPolicy(),
            government_policy=RuleBasedGovernmentPolicy(),
        )
        assert len(state.history) == 10
        for m in state.history:
            assert m["unbalanced_sheets"] == []


# --- Calibration with policies tests ---


class TestCalibrationWithPolicies:
    def test_objective_accepts_policies(self):
        config = _small_config(num_periods=30)
        registry = default_macro_registry()
        moments = default_us_moments()
        profile = CalibrationProfile(
            num_simulations=1,
            num_periods=30,
            burn_in=5,
            max_iterations=1,
            verbose=False,
        )
        policies = {"firm_policy": RuleBasedFirmPolicy()}

        obj = SimulationObjective(
            base_config=config,
            registry=registry,
            moments=moments,
            profile=profile,
            policies=policies,
        )

        # Should evaluate without error
        defaults = registry.from_vector(registry.calibrated_defaults())
        val, sim_moments = obj.evaluate(defaults)
        assert np.isfinite(val)
        assert len(sim_moments) == len(moments)

    def test_calibration_runs_with_policies(self):
        config = _small_config(num_periods=30)
        registry = default_macro_registry()
        moments = default_us_moments()
        profile = CalibrationProfile(
            num_simulations=1,
            num_periods=30,
            burn_in=5,
            max_iterations=2,
            verbose=False,
        )

        obj = SimulationObjective(
            base_config=config,
            registry=registry,
            moments=moments,
            profile=profile,
            policies={"government_policy": RuleBasedGovernmentPolicy()},
        )

        calibrator = SmmCalibrator(obj, registry, profile)
        result = calibrator.calibrate()
        assert result.num_evaluations > 0
        assert np.isfinite(result.weighted_objective)


# --- Forecasting with policies tests ---


class TestForecastingWithPolicies:
    def test_forecast_runner_accepts_policies(self):
        config = _small_config()
        runner = ForecastEnsembleRunner(
            base_config=config,
            policies={"firm_policy": RuleBasedFirmPolicy()},
        )
        fc = ForecastConfig(
            horizon=5,
            num_parameter_draws=2,
            num_shock_draws=1,
            burn_in=5,
            seed=42,
        )
        forecast = runner.forecast(fc)
        assert forecast.num_paths > 0
        assert "gdp" in forecast.paths

    def test_forecast_with_all_policies(self):
        config = _small_config()
        policies = {
            "firm_policy": RuleBasedFirmPolicy(),
            "household_policy": RuleBasedHouseholdPolicy(),
            "bank_policy": RuleBasedBankPolicy(),
            "government_policy": RuleBasedGovernmentPolicy(),
        }
        runner = ForecastEnsembleRunner(base_config=config, policies=policies)
        fc = ForecastConfig(
            horizon=5,
            num_parameter_draws=2,
            num_shock_draws=1,
            burn_in=5,
            seed=42,
        )
        forecast = runner.forecast(fc)
        assert forecast.num_paths > 0


# --- RL Environment tests ---


class TestMacroEnv:
    def test_govt_env_reset_and_step(self):
        config = _small_config()
        env = MacroEnv(config=config, role="government", max_steps=10)
        obs, info = env.reset(seed=42)

        assert "macro" in obs
        assert "role" in obs
        assert "deposits" in obs["role"]

        # Take a step with default action
        action = {"tax_rate": 0.2, "spending_per_period": 2000.0}
        obs2, reward, terminated, truncated, info2 = env.step(action)
        assert isinstance(reward, float)
        assert not terminated

    def test_bank_env_reset_and_step(self):
        config = _small_config()
        env = MacroEnv(config=config, role="bank", max_steps=5)
        obs, _ = env.reset(seed=42)
        assert "lending_rate" in obs["role"]

        action = {"base_rate_adjustment": 0.001}
        obs2, reward, _, _, _ = env.step(action)
        assert isinstance(reward, float)

    def test_firm_env_reset_and_step(self):
        config = _small_config()
        env = MacroEnv(config=config, role="firm", max_steps=5)
        obs, _ = env.reset(seed=42)
        assert "avg_price" in obs["role"]

        action = {"vacancies": 2, "price_adjustment": 1.01}
        obs2, reward, _, _, _ = env.step(action)
        assert isinstance(reward, float)

    def test_env_runs_to_truncation(self):
        config = _small_config()
        env = MacroEnv(config=config, role="government", max_steps=5)
        obs, _ = env.reset(seed=42)

        for _ in range(5):
            action = {"tax_rate": 0.2}
            obs, reward, terminated, truncated, info = env.step(action)

        assert truncated

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="role must be"):
            MacroEnv(role="central_bank")

    def test_to_flat_obs(self):
        config = _small_config()
        env = MacroEnv(config=config, role="government", max_steps=5)
        obs, _ = env.reset(seed=42)
        flat = env.to_flat_obs(obs)
        assert isinstance(flat, np.ndarray)
        assert flat.dtype == np.float32
        assert len(flat) > 0

    def test_observation_space_spec(self):
        env = MacroEnv(role="government")
        spec = env.observation_space_spec()
        assert "macro" in spec

    def test_action_space_spec(self):
        env = MacroEnv(role="government")
        spec = env.action_space_spec()
        assert "tax_rate" in spec

        env2 = MacroEnv(role="bank")
        spec2 = env2.action_space_spec()
        assert "base_rate_adjustment" in spec2


# --- End-to-end pipeline test ---


class TestEndToEndPipeline:
    def test_calibrate_then_forecast_with_policies(self):
        """Full pipeline: calibrate with policies, then forecast with policies."""
        config = _small_config(num_periods=30)
        policies = {"firm_policy": RuleBasedFirmPolicy()}

        # Calibrate
        registry = default_macro_registry()
        moments = default_us_moments()
        profile = CalibrationProfile(
            num_simulations=1,
            num_periods=30,
            burn_in=5,
            max_iterations=2,
            verbose=False,
        )
        obj = SimulationObjective(
            base_config=config,
            registry=registry,
            moments=moments,
            profile=profile,
            policies=policies,
        )
        calibrator = SmmCalibrator(obj, registry, profile)
        cal_result = calibrator.calibrate()

        # Forecast using calibration result
        runner = ForecastEnsembleRunner(
            base_config=config,
            calibration_result=cal_result,
            policies=policies,
        )
        fc = ForecastConfig(
            horizon=5,
            num_parameter_draws=2,
            num_shock_draws=1,
            burn_in=5,
            seed=42,
        )
        forecast = runner.forecast(fc)

        assert forecast.num_paths > 0
        assert "gdp" in forecast.paths
        assert forecast.elapsed_seconds > 0
