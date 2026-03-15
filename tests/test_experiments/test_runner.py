"""Tests for the experiment runner, batch execution, and parameter sweeps."""

from __future__ import annotations

import pytest
import pandas as pd

from econosim.config.schema import SimulationConfig
from econosim.experiments.runner import run_experiment, run_batch, run_parameter_sweep
from econosim.metrics.collector import (
    enrich_dataframe,
    aggregate_runs,
    compare_scenarios,
    summary_statistics,
)


def _small_config(**overrides) -> SimulationConfig:
    defaults = dict(
        num_periods=5,
        seed=42,
        household={"count": 10},
        firm={"count": 2},
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


class TestRunExperiment:
    def test_returns_expected_keys(self):
        result = run_experiment(_small_config())
        assert "name" in result
        assert "seed" in result
        assert "summary" in result
        assert "final_metrics" in result
        assert "dataframe" in result

    def test_dataframe_has_periods(self):
        result = run_experiment(_small_config(num_periods=5))
        df = result["dataframe"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_enriched_columns(self):
        result = run_experiment(_small_config(num_periods=5))
        df = result["dataframe"]
        assert "inflation_rate" in df.columns
        assert "gdp_growth" in df.columns

    def test_summary_has_gdp(self):
        result = run_experiment(_small_config(num_periods=5))
        assert "gdp" in result["summary"]
        assert "mean" in result["summary"]["gdp"]


class TestRunBatch:
    def test_batch_returns_aggregate(self):
        batch = run_batch(_small_config(), seeds=[42, 43])
        assert "aggregate" in batch
        assert "runs" in batch
        assert len(batch["runs"]) == 2

    def test_aggregate_has_ci_bands(self):
        batch = run_batch(_small_config(), seeds=[42, 43, 44])
        agg = batch["aggregate"]
        assert "gdp_mean" in agg.columns
        assert "gdp_std" in agg.columns
        assert "gdp_lo" in agg.columns
        assert "gdp_hi" in agg.columns

    def test_aggregate_indexed_by_period(self):
        batch = run_batch(_small_config(num_periods=5), seeds=[42, 43])
        agg = batch["aggregate"]
        assert len(agg) == 5

    def test_different_seeds_produce_different_runs(self):
        batch = run_batch(_small_config(num_periods=5), seeds=[1, 2])
        df1 = batch["runs"][0]["dataframe"]
        df2 = batch["runs"][1]["dataframe"]
        assert not df1["gdp"].equals(df2["gdp"])


class TestParameterSweep:
    def test_sweep_single_param(self):
        result = run_parameter_sweep(
            _small_config(num_periods=3),
            sweep_params={"household.wealth_propensity": [0.2, 0.4]},
            seeds=[42],
        )
        assert len(result["combinations"]) == 2
        assert len(result["results"]) == 2
        assert isinstance(result["comparison"], pd.DataFrame)

    def test_sweep_two_params(self):
        result = run_parameter_sweep(
            _small_config(num_periods=3),
            sweep_params={
                "household.wealth_propensity": [0.2, 0.4],
                "government.spending_per_period": [1000, 2000],
            },
            seeds=[42],
        )
        assert len(result["combinations"]) == 4  # 2x2

    def test_comparison_has_required_columns(self):
        result = run_parameter_sweep(
            _small_config(num_periods=3),
            sweep_params={"household.wealth_propensity": [0.2, 0.4]},
            seeds=[42],
        )
        comp = result["comparison"]
        assert "scenario" in comp.columns
        assert "metric" in comp.columns
        assert "mean" in comp.columns


class TestMetricsCollector:
    def test_enrich_adds_columns(self):
        data = {"period": [0, 1, 2], "gdp": [100, 110, 105], "avg_price": [10.0, 10.5, 10.2]}
        df = pd.DataFrame(data).set_index("period")
        enriched = enrich_dataframe(df)
        assert "inflation_rate" in enriched.columns
        assert "gdp_growth" in enriched.columns

    def test_aggregate_runs_basic(self):
        df1 = pd.DataFrame({"period": [0, 1], "gdp": [100, 110]}).set_index("period")
        df2 = pd.DataFrame({"period": [0, 1], "gdp": [90, 120]}).set_index("period")
        agg = aggregate_runs([df1, df2])
        assert "gdp_mean" in agg.columns
        assert abs(agg.loc[0, "gdp_mean"] - 95.0) < 0.01  # (100+90)/2

    def test_compare_scenarios(self):
        df1 = pd.DataFrame({"period": [0, 1], "gdp": [100, 110]}).set_index("period")
        df2 = pd.DataFrame({"period": [0, 1], "gdp": [80, 90]}).set_index("period")
        agg1 = aggregate_runs([df1])
        agg2 = aggregate_runs([df2])
        comp = compare_scenarios({"baseline": agg1, "shock": agg2}, metrics=["gdp"])
        assert len(comp) == 4  # 2 scenarios x 2 periods
        assert set(comp["scenario"].unique()) == {"baseline", "shock"}

    def test_summary_statistics_keys(self):
        data = {"period": [0, 1, 2], "gdp": [100, 110, 105]}
        df = pd.DataFrame(data).set_index("period")
        stats = summary_statistics(df)
        assert "gdp" in stats
        for key in ("mean", "std", "min", "max", "final"):
            assert key in stats["gdp"]
