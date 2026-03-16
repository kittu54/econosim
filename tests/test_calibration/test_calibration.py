"""Tests for calibration engine, parameters, and moments."""

import numpy as np
import pandas as pd
import pytest

from econosim.calibration.parameters import (
    ParameterSpec,
    ParameterStatus,
    ParameterRegistry,
    Prior,
    PriorType,
    default_macro_registry,
)
from econosim.calibration.moments import (
    MomentDefinition,
    MomentSet,
    default_us_moments,
)
from econosim.calibration.engine import (
    CalibrationProfile,
    SimulationObjective,
    SmmCalibrator,
    CalibrationResult,
)
from econosim.config.schema import SimulationConfig


# --- Parameter tests ---


class TestPrior:
    def test_uniform_log_pdf(self):
        p = Prior.uniform(0, 1)
        assert p.log_pdf(0.5) == pytest.approx(0.0)  # log(1) = 0
        assert p.log_pdf(1.5) == -np.inf

    def test_uniform_sample(self):
        p = Prior.uniform(0, 1)
        rng = np.random.default_rng(42)
        samples = p.sample(rng, 100)
        assert len(samples) == 100
        assert all(0 <= s <= 1 for s in samples)

    def test_normal_log_pdf(self):
        p = Prior.normal(0, 1)
        # Log PDF at mean should be highest
        assert p.log_pdf(0) > p.log_pdf(3)

    def test_beta_sample(self):
        p = Prior.beta(2, 5)
        rng = np.random.default_rng(42)
        samples = p.sample(rng, 100)
        assert all(0 <= s <= 1 for s in samples)


class TestParameterSpec:
    def test_identity_transform(self):
        p = ParameterSpec(name="x", config_path="a.b", transform="identity")
        assert p.to_unconstrained(5.0) == 5.0
        assert p.from_unconstrained(5.0) == 5.0

    def test_log_transform(self):
        p = ParameterSpec(
            name="x", config_path="a.b", transform="log",
            lower_bound=0.01, upper_bound=100,
        )
        x = 5.0
        y = p.to_unconstrained(x)
        x_back = p.from_unconstrained(y)
        assert abs(x - x_back) < 1e-6

    def test_logit_transform(self):
        p = ParameterSpec(
            name="x", config_path="a.b", transform="logit",
            lower_bound=0, upper_bound=1,
        )
        x = 0.7
        y = p.to_unconstrained(x)
        x_back = p.from_unconstrained(y)
        assert abs(x - x_back) < 1e-6

    def test_bounds_enforced(self):
        p = ParameterSpec(
            name="x", config_path="a.b", lower_bound=0, upper_bound=10,
        )
        assert p.from_unconstrained(15) == 10
        assert p.from_unconstrained(-5) == 0


class TestParameterRegistry:
    def test_register_and_get(self):
        reg = ParameterRegistry()
        spec = ParameterSpec(
            name="test_param",
            config_path="firm.labor_productivity",
            status=ParameterStatus.CALIBRATED,
            default_value=5.0,
        )
        reg.register(spec)
        assert reg.get("test_param") == spec

    def test_calibrated_filter(self):
        reg = ParameterRegistry()
        reg.register(ParameterSpec(name="a", config_path="x.a", status=ParameterStatus.CALIBRATED))
        reg.register(ParameterSpec(name="b", config_path="x.b", status=ParameterStatus.FIXED))
        reg.register(ParameterSpec(name="c", config_path="x.c", status=ParameterStatus.CALIBRATED))

        assert len(reg.calibrated) == 2
        assert len(reg.fixed) == 1
        assert "a" in reg.calibrated_names()
        assert "c" in reg.calibrated_names()

    def test_vector_roundtrip(self):
        reg = ParameterRegistry()
        reg.register(ParameterSpec(name="a", config_path="x.a", status=ParameterStatus.CALIBRATED, default_value=1.0))
        reg.register(ParameterSpec(name="b", config_path="x.b", status=ParameterStatus.CALIBRATED, default_value=2.0))

        vec = reg.calibrated_defaults()
        d = reg.from_vector(vec)
        vec2 = reg.to_vector(d)
        np.testing.assert_array_almost_equal(vec, vec2)

    def test_default_registry(self):
        reg = default_macro_registry()
        assert len(reg.calibrated) >= 5
        assert "consumption_propensity" in reg.calibrated_names()
        assert "labor_productivity" in reg.calibrated_names()


# --- Moment tests ---


class TestMomentDefinition:
    def test_compute_mean(self):
        m = MomentDefinition(
            name="test", description="test",
            empirical_value=0.05,
            series_name="urate",
            statistic="mean",
            burn_in=0,
        )
        df = pd.DataFrame({"urate": [0.04, 0.05, 0.06, 0.05, 0.04]})
        assert abs(m.compute(df) - 0.048) < 0.001

    def test_compute_std(self):
        m = MomentDefinition(
            name="test", description="test",
            empirical_value=0.01,
            series_name="x",
            statistic="std",
            burn_in=0,
        )
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        result = m.compute(df)
        assert result > 0

    def test_compute_custom_fn(self):
        m = MomentDefinition(
            name="test", description="test",
            empirical_value=0.5,
            compute_fn=lambda df: float(df["a"].mean() / df["b"].mean()),
        )
        df = pd.DataFrame({"a": [10, 20], "b": [20, 40]})
        assert abs(m.compute(df) - 0.5) < 0.001

    def test_burn_in(self):
        m = MomentDefinition(
            name="test", description="test",
            empirical_value=5.0,
            series_name="x",
            statistic="mean",
            burn_in=5,
        )
        df = pd.DataFrame({"x": list(range(10))})
        # After burn_in=5, values are [5,6,7,8,9], mean = 7
        assert abs(m.compute(df) - 7.0) < 0.001


class TestMomentSet:
    def test_compute_all(self):
        ms = MomentSet("test")
        ms.add(MomentDefinition("m1", "test", 1.0, series_name="x", statistic="mean", burn_in=0))
        ms.add(MomentDefinition("m2", "test", 2.0, series_name="x", statistic="std", burn_in=0))

        df = pd.DataFrame({"x": np.random.normal(1.0, 2.0, 100)})
        values = ms.compute_all(df)
        assert len(values) == 2

    def test_weighting_matrix(self):
        ms = MomentSet("test")
        ms.add(MomentDefinition("m1", "test", 1.0, empirical_std=0.1))
        ms.add(MomentDefinition("m2", "test", 2.0, empirical_std=0.2))

        W = ms.weighting_matrix("inverse_variance")
        assert W.shape == (2, 2)
        assert W[0, 0] > W[1, 1]  # smaller std → higher weight

    def test_default_us_moments(self):
        ms = default_us_moments()
        assert len(ms) >= 8
        assert "gdp_growth_mean" in ms.names
        assert "unemployment_mean" in ms.names


# --- Calibration engine tests ---


class TestCalibrationProfile:
    def test_default_profile(self):
        p = CalibrationProfile()
        assert p.num_simulations >= 1
        assert p.max_iterations > 0


class TestSimulationObjective:
    def test_evaluate(self):
        config = SimulationConfig(num_periods=40, seed=42)
        registry = default_macro_registry()
        moments = default_us_moments()
        profile = CalibrationProfile(
            num_simulations=1,
            num_periods=40,
            max_iterations=5,
            verbose=False,
        )

        objective = SimulationObjective(config, registry, moments, profile)
        params = {p.name: p.default_value for p in registry.calibrated}
        obj_val, sim_moments = objective.evaluate(params)

        assert np.isfinite(obj_val)
        assert len(sim_moments) == len(moments)

    def test_callable_interface(self):
        config = SimulationConfig(num_periods=30, seed=42)
        registry = default_macro_registry()
        moments = default_us_moments()
        profile = CalibrationProfile(
            num_simulations=1, num_periods=30,
            max_iterations=2, verbose=False,
        )

        objective = SimulationObjective(config, registry, moments, profile)
        x0 = registry.calibrated_defaults()
        val = objective(x0)
        assert np.isfinite(val)


class TestSmmCalibrator:
    def test_calibrate_runs(self):
        """Smoke test: calibration runs and produces a result."""
        config = SimulationConfig(num_periods=30, seed=42)
        registry = default_macro_registry()
        moments = default_us_moments()
        profile = CalibrationProfile(
            num_simulations=1,
            num_periods=30,
            max_iterations=3,  # minimal for speed
            verbose=False,
            optimizer="nelder-mead",
        )

        objective = SimulationObjective(config, registry, moments, profile)
        calibrator = SmmCalibrator(objective, registry, profile)
        result = calibrator.calibrate()

        assert isinstance(result, CalibrationResult)
        assert result.method == "SMM"
        assert len(result.estimated_params) > 0
        assert len(result.moment_fit) > 0
        assert result.elapsed_seconds > 0

    def test_result_summary(self):
        result = CalibrationResult(
            profile_name="test",
            method="SMM",
            estimated_params={"a": 1.0, "b": 2.0},
            moment_fit={"m1": {"empirical": 1.0, "simulated": 1.1, "error": 0.1}},
            weighted_objective=0.5,
            num_evaluations=10,
            elapsed_seconds=5.0,
            converged=True,
            seed=42,
        )
        summary = result.summary()
        assert "SMM" in summary
        assert "Converged: True" in summary
