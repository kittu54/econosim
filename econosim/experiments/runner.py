"""
Experiment runner: loads configs, executes simulation runs,
persists results, and supports batch execution with different seeds.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from econosim.config.schema import SimulationConfig
from econosim.engine.simulation import build_simulation, step
from econosim.metrics.collector import (
    aggregate_runs,
    compare_scenarios,
    enrich_dataframe,
    export_results,
    history_to_dataframe,
    summary_statistics,
)

logger = logging.getLogger(__name__)


def load_config_from_yaml(path: str | Path) -> SimulationConfig:
    """Load a SimulationConfig from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return SimulationConfig(**data)


def run_experiment(
    config: SimulationConfig,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run a single simulation experiment and return results summary.

    Returns dict with keys: name, seed, num_periods, summary, final_metrics, dataframe.
    """
    state = build_simulation(config)

    for t in range(config.num_periods):
        step(state)

    df = enrich_dataframe(history_to_dataframe(state.history))
    stats = summary_statistics(df)

    result: dict[str, Any] = {
        "name": config.name,
        "seed": config.seed,
        "num_periods": config.num_periods,
        "summary": stats,
        "final_metrics": state.history[-1] if state.history else {},
        "dataframe": df,
    }

    if output_dir:
        out = Path(output_dir) / config.name
        out.mkdir(parents=True, exist_ok=True)
        export_results(df, out, config.name)
        serialisable = {k: v for k, v in result.items() if k != "dataframe"}
        with open(out / "summary.json", "w") as f:
            json.dump(serialisable, f, indent=2, default=str)
        logger.info(f"Results saved to {out}")

    return result


def run_batch(
    base_config: SimulationConfig,
    seeds: list[int],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the same config with multiple seeds for statistical analysis.

    Returns dict with keys: name, seeds, runs (list of results),
    aggregate (aggregated DataFrame with CI bands), summary.
    """
    runs: list[dict[str, Any]] = []
    dataframes: list[pd.DataFrame] = []

    for seed in seeds:
        config = base_config.model_copy(update={"seed": seed, "name": f"{base_config.name}_s{seed}"})
        result = run_experiment(config, output_dir)
        runs.append(result)
        dataframes.append(result["dataframe"])
        logger.info(f"Completed run with seed={seed}")

    agg = aggregate_runs(dataframes)

    if output_dir:
        out = Path(output_dir) / base_config.name
        out.mkdir(parents=True, exist_ok=True)
        agg.to_csv(out / f"{base_config.name}_aggregate.csv")

    return {
        "name": base_config.name,
        "seeds": seeds,
        "runs": runs,
        "aggregate": agg,
    }


def run_parameter_sweep(
    base_config: SimulationConfig,
    sweep_params: dict[str, list[Any]],
    seeds: list[int] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Sweep over parameter combinations and run batch experiments.

    Args:
        base_config: base configuration to modify
        sweep_params: dict of dotted parameter paths to lists of values, e.g.
            {'household.wealth_propensity': [0.2, 0.3, 0.4],
             'government.spending_per_period': [1000, 2000, 3000]}
        seeds: list of seeds per combination (default [42])
        output_dir: optional output directory

    Returns dict with keys: sweep_params, combinations (list of param dicts),
    results (list of batch results), comparison (DataFrame).
    """
    if seeds is None:
        seeds = [42]

    # Generate all combinations
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    combinations = list(itertools.product(*param_values))

    logger.info(f"Parameter sweep: {len(combinations)} combinations x {len(seeds)} seeds")

    batch_results: list[dict[str, Any]] = []
    scenario_aggs: dict[str, pd.DataFrame] = {}

    for combo in combinations:
        # Build config by deep-copying and setting individual fields
        label_parts = []
        config = base_config.model_copy(deep=True)
        for pname, pval in zip(param_names, combo):
            parts = pname.split(".")
            if len(parts) == 2:
                section, key = parts
                sub_model = getattr(config, section)
                setattr(sub_model, key, pval)
            else:
                setattr(config, pname, pval)
            label_parts.append(f"{parts[-1]}={pval}")

        label = "_".join(label_parts)
        config.name = label

        batch = run_batch(config, seeds, output_dir)
        batch["params"] = dict(zip(param_names, combo))
        batch_results.append(batch)
        scenario_aggs[label] = batch["aggregate"]

        logger.info(f"Sweep: completed {label}")

    comparison = compare_scenarios(scenario_aggs)

    if output_dir:
        out = Path(output_dir) / "sweep"
        out.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(out / "comparison.csv", index=False)

    return {
        "sweep_params": sweep_params,
        "combinations": [dict(zip(param_names, c)) for c in combinations],
        "results": batch_results,
        "comparison": comparison,
    }
