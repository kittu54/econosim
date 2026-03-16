"""Tests for the forecasting engine and backtesting framework."""

import numpy as np
import pandas as pd
import pytest

from econosim.config.schema import SimulationConfig
from econosim.forecasting.engine import (
    ForecastConfig,
    ForecastEnsembleRunner,
    DensityForecast,
    ScenarioSpec,
    ShockProcess,
)
from econosim.forecasting.backtesting import (
    BacktestConfig,
    BacktestRunner,
    ForecastScorecard,
    EvaluationReport,
    RandomWalkBenchmark,
    ARBenchmark,
    TrendBenchmark,
    _crps_ensemble,
)


class TestShockProcess:
    def test_constant_shock(self):
        proc = ShockProcess(shock_type="supply", parameter="labor_productivity", mean=1.0, std=0.0)
        path = proc.generate_path(10, np.random.default_rng(42))
        assert all(x == 1.0 for x in path)

    def test_stochastic_shock(self):
        proc = ShockProcess(
            shock_type="demand", parameter="consumption_propensity",
            mean=1.0, std=0.05, persistence=0.5,
        )
        path = proc.generate_path(20, np.random.default_rng(42))
        assert len(path) == 20
        assert not all(x == 1.0 for x in path)  # should have variation

    def test_length(self):
        proc = ShockProcess(shock_type="supply", parameter="x", std=0.1)
        assert len(proc.generate_path(5, np.random.default_rng(42))) == 5
        assert len(proc.generate_path(50, np.random.default_rng(42))) == 50


class TestDensityForecast:
    def _make_forecast(self, n_paths=50, horizon=12) -> DensityForecast:
        rng = np.random.default_rng(42)
        paths = {
            "gdp": rng.normal(1000, 50, (n_paths, horizon)),
            "unemployment_rate": rng.beta(2, 20, (n_paths, horizon)),
            "gdp_growth": rng.normal(0.01, 0.02, (n_paths, horizon)),
            "inflation_rate": rng.normal(0.02, 0.01, (n_paths, horizon)),
        }
        return DensityForecast(
            config={},
            scenario_name="test",
            variables=list(paths.keys()),
            horizon=horizon,
            num_paths=n_paths,
            paths=paths,
        )

    def test_compute_quantiles(self):
        fc = self._make_forecast()
        fc.compute_quantiles([0.1, 0.5, 0.9])

        assert 0.1 in fc.quantiles["gdp"]
        assert 0.5 in fc.quantiles["gdp"]
        assert 0.9 in fc.quantiles["gdp"]

        # q10 < q50 < q90
        assert fc.quantiles["gdp"][0.1][0] < fc.quantiles["gdp"][0.9][0]

    def test_median_path(self):
        fc = self._make_forecast()
        fc.compute_quantiles()
        median = fc.median_path("gdp")
        assert len(median) == 12
        assert all(np.isfinite(median))

    def test_event_probabilities(self):
        fc = self._make_forecast()
        fc.compute_event_probabilities()
        assert "high_inflation_probability" in fc.event_probs
        assert 0.0 <= fc.event_probs["high_inflation_probability"] <= 1.0

    def test_to_dataframe(self):
        fc = self._make_forecast()
        fc.compute_quantiles([0.1, 0.5, 0.9])
        df = fc.to_dataframe()
        assert len(df) > 0
        assert "variable" in df.columns
        assert "horizon" in df.columns


class TestForecastEnsembleRunner:
    def test_forecast_runs(self):
        """Smoke test: ensemble forecast completes."""
        config = SimulationConfig(num_periods=20, seed=42)
        runner = ForecastEnsembleRunner(config)
        fc_config = ForecastConfig(
            horizon=5,
            num_parameter_draws=2,
            num_shock_draws=2,
            burn_in=10,
            variables=["gdp", "unemployment_rate"],
        )
        result = runner.forecast(fc_config)

        assert isinstance(result, DensityForecast)
        assert result.num_paths > 0
        assert result.horizon == 5
        assert "gdp" in result.paths

    def test_scenario_forecast(self):
        config = SimulationConfig(num_periods=20, seed=42)
        runner = ForecastEnsembleRunner(config)

        scenario = ScenarioSpec(
            name="recession",
            shock_processes=[
                ShockProcess(
                    shock_type="demand",
                    parameter="consumption_propensity",
                    mean=0.9, std=0.05,
                ),
            ],
        )

        fc_config = ForecastConfig(
            horizon=5, num_parameter_draws=2, num_shock_draws=2,
            burn_in=10, variables=["gdp"],
        )

        result = runner.forecast(fc_config, scenario)
        assert result.scenario_name == "recession"


# --- Backtesting tests ---


class TestBenchmarkModels:
    def _make_history(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "gdp": np.cumsum(np.random.normal(10, 2, n)) + 1000,
            "unemployment_rate": 0.05 + 0.01 * np.sin(np.arange(n) / 5),
        })

    def test_random_walk(self):
        bm = RandomWalkBenchmark()
        history = self._make_history()
        fc = bm.forecast(history, 10, "gdp")
        assert len(fc) == 10
        assert all(fc == history["gdp"].iloc[-1])

    def test_ar_forecast(self):
        bm = ARBenchmark()
        history = self._make_history()
        fc = bm.forecast(history, 10, "gdp")
        assert len(fc) == 10
        assert all(np.isfinite(fc))

    def test_trend_forecast(self):
        bm = TrendBenchmark()
        history = self._make_history()
        fc = bm.forecast(history, 10, "gdp")
        assert len(fc) == 10
        assert all(np.isfinite(fc))

    def test_benchmark_empty_history(self):
        bm = RandomWalkBenchmark()
        history = pd.DataFrame(columns=["gdp"])
        fc = bm.forecast(history, 5, "gdp")
        assert all(np.isnan(fc))


class TestCRPS:
    def test_perfect_forecast(self):
        # Ensemble concentrated at observation
        ensemble = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        crps = _crps_ensemble(ensemble, 5.0)
        assert abs(crps) < 0.01

    def test_biased_forecast(self):
        ensemble = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        crps = _crps_ensemble(ensemble, 5.0)
        assert crps > 0  # worse than perfect


class TestBacktestRunner:
    def test_backtest_with_naive_forecast(self):
        """Smoke test: backtest runs with a simple forecast function."""
        np.random.seed(42)
        n = 120
        history = pd.DataFrame({
            "gdp": np.cumsum(np.random.normal(10, 2, n)) + 1000,
            "unemployment_rate": 0.05 + 0.01 * np.sin(np.arange(n) / 5),
        })

        def naive_forecast(hist, horizon, variable):
            last = float(hist[variable].iloc[-1])
            median = np.full(horizon, last)
            rng = np.random.default_rng(42)
            ensemble = rng.normal(last, abs(last) * 0.05, (20, horizon))
            return median, ensemble

        runner = BacktestRunner(history, naive_forecast)
        config = BacktestConfig(
            forecast_horizon=10,
            num_origins=5,
            calibration_window=40,
            step_size=10,
            variables=["gdp"],
        )
        report = runner.run(config)

        assert isinstance(report, EvaluationReport)
        assert len(report.scorecards) > 0
        assert report.scorecards[0].rmse > 0
        assert report.elapsed_seconds > 0

    def test_evaluation_report_summary(self):
        sc = ForecastScorecard(
            variable="gdp", horizon=10,
            rmse=5.0, mae=4.0, crps=3.0,
            coverage_90=0.85, coverage_50=0.45,
            num_origins=10,
        )
        report = EvaluationReport(name="test", scorecards=[sc])
        table = report.summary_table()
        assert len(table) == 1
        assert table.iloc[0]["rmse"] == 5.0
