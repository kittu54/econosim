"""
Metrics collection and analysis utilities.

Provides tools to convert simulation history into DataFrames,
compute derived indicators, and export results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from econosim.core.accounting import round_money


def history_to_dataframe(history: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert simulation history (list of period metric dicts) to a DataFrame."""
    if not history:
        return pd.DataFrame()
    df = pd.DataFrame(history)
    df = df.set_index("period")
    return df


def compute_inflation(df: pd.DataFrame, price_col: str = "avg_price") -> pd.Series:
    """Compute period-over-period inflation rate from price series."""
    if price_col not in df.columns or len(df) < 2:
        return pd.Series(dtype=float)
    return df[price_col].pct_change().rename("inflation_rate")


def compute_gdp_growth(df: pd.DataFrame, gdp_col: str = "gdp") -> pd.Series:
    """Compute period-over-period GDP growth rate."""
    if gdp_col not in df.columns or len(df) < 2:
        return pd.Series(dtype=float)
    return df[gdp_col].pct_change().rename("gdp_growth")


def compute_velocity(df: pd.DataFrame) -> pd.Series:
    """Compute velocity of money = GDP / total deposits."""
    total_deposits = df.get("total_hh_deposits", 0) + df.get("total_firm_deposits", 0)
    total_deposits = total_deposits.replace(0, np.nan)
    return (df["gdp"] / total_deposits).rename("velocity")


def summary_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics over the full simulation run."""
    if df.empty:
        return {}

    stats: dict[str, Any] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats[col] = {
            "mean": round(float(df[col].mean()), 4),
            "std": round(float(df[col].std()), 4),
            "min": round(float(df[col].min()), 4),
            "max": round(float(df[col].max()), 4),
            "final": round(float(df[col].iloc[-1]), 4),
        }
    return stats


def export_results(
    df: pd.DataFrame,
    output_dir: str | Path,
    name: str = "simulation",
) -> Path:
    """Export simulation results to CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f"{name}_metrics.csv"
    df.to_csv(csv_path)
    return csv_path


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns (inflation, GDP growth, velocity) to a metrics DataFrame."""
    if df.empty:
        return df
    df = df.copy()
    if "avg_price" in df.columns:
        df["inflation_rate"] = df["avg_price"].pct_change()
    if "gdp" in df.columns:
        df["gdp_growth"] = df["gdp"].pct_change()
    total_dep = df.get("total_hh_deposits", 0) + df.get("total_firm_deposits", 0)
    if isinstance(total_dep, pd.Series) and (total_dep > 0).any():
        df["velocity"] = df["gdp"] / total_dep.replace(0, np.nan)
    return df


def aggregate_runs(
    run_dataframes: list[pd.DataFrame],
    ci: float = 0.95,
) -> pd.DataFrame:
    """Aggregate multiple run DataFrames into mean/std/CI bands per period.

    Returns a DataFrame indexed by period with columns like
    'gdp_mean', 'gdp_std', 'gdp_lo', 'gdp_hi' for each numeric column.
    """
    if not run_dataframes:
        return pd.DataFrame()

    from scipy import stats as sp_stats

    # Stack all runs into a single frame with a 'run' level
    stacked = pd.concat(
        {i: df for i, df in enumerate(run_dataframes)},
        names=["run", "period"],
    )
    numeric_cols = stacked.select_dtypes(include=[np.number]).columns.tolist()
    grouped = stacked[numeric_cols].groupby(level="period")

    n_runs = len(run_dataframes)
    z = sp_stats.norm.ppf(0.5 + ci / 2)

    parts: dict[str, pd.Series] = {}
    for col in numeric_cols:
        means = grouped[col].mean()
        stds = grouped[col].std(ddof=1).fillna(0)
        se = stds / np.sqrt(n_runs)
        parts[f"{col}_mean"] = means
        parts[f"{col}_std"] = stds
        parts[f"{col}_lo"] = means - z * se
        parts[f"{col}_hi"] = means + z * se

    return pd.DataFrame(parts)


def compare_scenarios(
    scenario_aggs: dict[str, pd.DataFrame],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Compare aggregated results across scenarios.

    Args:
        scenario_aggs: {scenario_name: aggregated_df from aggregate_runs}
        metrics: which metrics to compare (defaults to common macro indicators)

    Returns a long-form DataFrame with columns: scenario, period, metric, mean, std, lo, hi
    """
    if metrics is None:
        metrics = ["gdp", "unemployment_rate", "avg_price", "gini_deposits", "total_loans_outstanding"]

    rows = []
    for name, agg_df in scenario_aggs.items():
        for metric in metrics:
            mean_col = f"{metric}_mean"
            if mean_col not in agg_df.columns:
                continue
            for period in agg_df.index:
                rows.append({
                    "scenario": name,
                    "period": period,
                    "metric": metric,
                    "mean": agg_df.loc[period, mean_col],
                    "std": agg_df.loc[period, f"{metric}_std"],
                    "lo": agg_df.loc[period, f"{metric}_lo"],
                    "hi": agg_df.loc[period, f"{metric}_hi"],
                })
    return pd.DataFrame(rows)
