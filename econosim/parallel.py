"""Parallel execution utilities for EconoSim.

Provides a unified parallel runner that distributes independent simulation
runs across multiple processes. Used by calibration, forecasting, and batch
experiment runners.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np
import pandas as pd

from econosim.config.schema import SimulationConfig

logger = logging.getLogger(__name__)

# Default to leaving one core free, minimum 1 worker
DEFAULT_MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)


def _run_single_sim(config_dict: dict[str, Any]) -> pd.DataFrame:
    """Worker function that runs a single simulation from a config dict.

    Must be a top-level function for pickling by multiprocessing.
    """
    from econosim.engine.simulation import build_simulation, step
    from econosim.metrics.collector import history_to_dataframe, enrich_dataframe

    config = SimulationConfig(**config_dict)
    state = build_simulation(config)
    for _ in range(config.num_periods):
        step(state)
    return enrich_dataframe(history_to_dataframe(state.history))


def run_simulations_parallel(
    configs: list[SimulationConfig],
    max_workers: int | None = None,
) -> list[pd.DataFrame | None]:
    """Run multiple independent simulations in parallel.

    Args:
        configs: List of simulation configurations.
        max_workers: Max parallel processes. Defaults to cpu_count - 1.

    Returns:
        List of DataFrames (or None for failed runs), in same order as configs.
    """
    if not configs:
        return []

    max_workers = max_workers or DEFAULT_MAX_WORKERS

    # For single config or single worker, just run sequentially
    if len(configs) == 1 or max_workers <= 1:
        results: list[pd.DataFrame | None] = []
        for cfg in configs:
            try:
                results.append(_run_single_sim(cfg.model_dump()))
            except Exception as e:
                logger.warning(f"Simulation failed: {e}")
                results.append(None)
        return results

    # Parallel execution
    config_dicts = [cfg.model_dump() for cfg in configs]
    results_ordered: list[pd.DataFrame | None] = [None] * len(configs)

    workers = min(max_workers, len(configs))
    logger.info(f"Running {len(configs)} simulations across {workers} workers")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_run_single_sim, cd): i
            for i, cd in enumerate(config_dicts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results_ordered[idx] = future.result()
            except Exception as e:
                logger.warning(f"Simulation {idx} failed: {e}")
                results_ordered[idx] = None

    succeeded = sum(1 for r in results_ordered if r is not None)
    logger.info(f"Parallel run complete: {succeeded}/{len(configs)} succeeded")
    return results_ordered


def parallel_moment_evaluation(
    base_config: SimulationConfig,
    seeds: list[int],
    num_periods: int,
    max_workers: int | None = None,
) -> list[pd.DataFrame]:
    """Run simulations for moment evaluation (used by calibration).

    Args:
        base_config: Base config with calibrated parameters applied.
        seeds: List of seeds to run.
        num_periods: Number of periods per simulation.
        max_workers: Max parallel processes.

    Returns:
        List of DataFrames from successful runs.
    """
    configs = [
        base_config.model_copy(update={"seed": seed, "num_periods": num_periods})
        for seed in seeds
    ]
    results = run_simulations_parallel(configs, max_workers=max_workers)
    return [r for r in results if r is not None]
