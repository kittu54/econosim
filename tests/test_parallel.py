"""Tests for parallel execution utilities."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from econosim.config.schema import SimulationConfig
from econosim.parallel import run_simulations_parallel, parallel_moment_evaluation


class TestParallelSimulations:
    def _make_config(self, seed: int = 42, periods: int = 20) -> SimulationConfig:
        return SimulationConfig(num_periods=periods, seed=seed)

    def test_single_simulation(self):
        configs = [self._make_config()]
        results = run_simulations_parallel(configs, max_workers=1)
        assert len(results) == 1
        assert isinstance(results[0], pd.DataFrame)
        assert len(results[0]) == 20

    def test_multiple_simulations_sequential(self):
        configs = [self._make_config(seed=i) for i in range(3)]
        results = run_simulations_parallel(configs, max_workers=1)
        assert len(results) == 3
        assert all(r is not None for r in results)

    def test_multiple_simulations_parallel(self):
        configs = [self._make_config(seed=i) for i in range(4)]
        results = run_simulations_parallel(configs, max_workers=2)
        assert len(results) == 4
        assert all(r is not None for r in results)
        # Different seeds should produce different results
        gdp_finals = [r["gdp"].iloc[-1] for r in results]
        assert len(set(round(g, 2) for g in gdp_finals)) > 1

    def test_empty_configs(self):
        results = run_simulations_parallel([])
        assert results == []

    def test_result_order_preserved(self):
        configs = [self._make_config(seed=i, periods=20) for i in range(5)]
        results = run_simulations_parallel(configs, max_workers=2)
        # Verify order matches by checking seeds produce consistent GDP paths
        for i, (cfg, df) in enumerate(zip(configs, results)):
            assert df is not None
            # Re-run sequentially to verify match
            single = run_simulations_parallel([cfg], max_workers=1)
            np.testing.assert_array_almost_equal(
                df["gdp"].values, single[0]["gdp"].values
            )

    def test_parallel_moment_evaluation(self):
        config = self._make_config()
        seeds = [100, 200, 300]
        dfs = parallel_moment_evaluation(config, seeds, num_periods=20, max_workers=2)
        assert len(dfs) == 3
        assert all(isinstance(df, pd.DataFrame) for df in dfs)


class TestParallelBatchRunner:
    def test_batch_parallel(self):
        from econosim.experiments.runner import run_batch
        config = SimulationConfig(num_periods=20, seed=42)
        result = run_batch(config, seeds=[1, 2, 3], parallel=True, max_workers=2)
        assert len(result["runs"]) == 3
        assert result["aggregate"] is not None

    def test_batch_sequential_matches_parallel(self):
        from econosim.experiments.runner import run_batch
        config = SimulationConfig(num_periods=20, seed=42)
        seq = run_batch(config, seeds=[10, 20], parallel=False)
        par = run_batch(config, seeds=[10, 20], parallel=True, max_workers=2)
        # Same number of runs
        assert len(seq["runs"]) == len(par["runs"])
        # Same GDP values for same seeds
        for s, p in zip(seq["runs"], par["runs"]):
            np.testing.assert_array_almost_equal(
                s["dataframe"]["gdp"].values,
                p["dataframe"]["gdp"].values,
            )
