"""Backtesting and forecast evaluation framework.

Implements rolling-origin evaluation with benchmark comparisons.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BacktestConfig(BaseModel):
    """Configuration for a backtesting run."""

    name: str = "default_backtest"
    forecast_horizon: int = 12
    num_origins: int = 10  # number of rolling forecast origins
    calibration_window: int = 60  # periods used for calibration at each origin
    step_size: int = 4  # periods between origins
    variables: list[str] = Field(default_factory=lambda: [
        "gdp", "unemployment_rate", "avg_price", "inflation_rate",
    ])
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    num_ensemble_paths: int = 50


@dataclass
class ForecastScorecard:
    """Forecast evaluation metrics for a single variable."""

    variable: str
    horizon: int

    # Point forecast metrics (using median)
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    bias: float = 0.0

    # Distributional metrics
    crps: float = 0.0  # Continuous Ranked Probability Score
    coverage_90: float = 0.0  # fraction of actuals in 90% interval
    coverage_50: float = 0.0  # fraction of actuals in 50% interval
    pit_uniformity: float = 0.0  # Probability Integral Transform calibration

    # Relative metrics (vs benchmark)
    skill_score_rmse: float = 0.0  # 1 - RMSE/RMSE_benchmark
    skill_score_crps: float = 0.0

    num_origins: int = 0


@dataclass
class EvaluationReport:
    """Complete backtest evaluation report."""

    name: str
    scorecards: list[ForecastScorecard] = field(default_factory=list)
    benchmark_scorecards: dict[str, list[ForecastScorecard]] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary_table(self) -> pd.DataFrame:
        """Create summary table of forecast performance."""
        rows = []
        for sc in self.scorecards:
            row = {
                "variable": sc.variable,
                "horizon": sc.horizon,
                "rmse": sc.rmse,
                "mae": sc.mae,
                "crps": sc.crps,
                "coverage_90": sc.coverage_90,
                "coverage_50": sc.coverage_50,
                "skill_rmse": sc.skill_score_rmse,
                "skill_crps": sc.skill_score_crps,
                "n_origins": sc.num_origins,
            }
            rows.append(row)
        return pd.DataFrame(rows)


class BenchmarkModel:
    """Interface for benchmark forecast models."""

    def __init__(self, name: str = "benchmark") -> None:
        self.name = name

    def forecast(
        self,
        history: pd.DataFrame,
        horizon: int,
        variable: str,
    ) -> np.ndarray:
        """Produce point forecast from historical data.

        Args:
            history: observed data up to forecast origin
            horizon: number of periods ahead
            variable: column to forecast

        Returns:
            np.ndarray of shape (horizon,) with point forecasts
        """
        raise NotImplementedError


class RandomWalkBenchmark(BenchmarkModel):
    """Random walk (no-change) forecast: f(t+h) = y(t)."""

    def __init__(self) -> None:
        super().__init__("random_walk")

    def forecast(self, history: pd.DataFrame, horizon: int, variable: str) -> np.ndarray:
        if variable not in history.columns or len(history) == 0:
            return np.full(horizon, np.nan)
        last_value = float(history[variable].iloc[-1])
        return np.full(horizon, last_value)


class ARBenchmark(BenchmarkModel):
    """Simple AR(1) forecast."""

    def __init__(self) -> None:
        super().__init__("ar1")

    def forecast(self, history: pd.DataFrame, horizon: int, variable: str) -> np.ndarray:
        if variable not in history.columns or len(history) < 10:
            return np.full(horizon, np.nan)

        y = history[variable].dropna().values
        if len(y) < 10:
            return np.full(horizon, np.nan)

        # Simple AR(1): y_t = c + phi * y_{t-1}
        y_lag = y[:-1]
        y_cur = y[1:]
        if np.std(y_lag) < 1e-10:
            return np.full(horizon, float(np.mean(y)))

        phi = np.corrcoef(y_cur, y_lag)[0, 1]
        c = np.mean(y_cur) - phi * np.mean(y_lag)

        forecasts = np.zeros(horizon)
        forecasts[0] = c + phi * y[-1]
        for t in range(1, horizon):
            forecasts[t] = c + phi * forecasts[t - 1]
        return forecasts


class TrendBenchmark(BenchmarkModel):
    """Simple linear trend forecast."""

    def __init__(self) -> None:
        super().__init__("linear_trend")

    def forecast(self, history: pd.DataFrame, horizon: int, variable: str) -> np.ndarray:
        if variable not in history.columns or len(history) < 5:
            return np.full(horizon, np.nan)

        y = history[variable].dropna().values
        if len(y) < 5:
            return np.full(horizon, np.nan)

        t = np.arange(len(y))
        coeffs = np.polyfit(t, y, 1)
        future_t = np.arange(len(y), len(y) + horizon)
        return np.polyval(coeffs, future_t)


class BacktestRunner:
    """Runs rolling-origin forecast evaluation."""

    def __init__(
        self,
        full_history: pd.DataFrame,
        forecast_fn: Callable[[pd.DataFrame, int, str], tuple[np.ndarray, np.ndarray | None]],
        benchmarks: list[BenchmarkModel] | None = None,
    ) -> None:
        """
        Args:
            full_history: Complete observed/simulated data
            forecast_fn: Callable(history_df, horizon, variable) ->
                         (median_forecast, ensemble_paths or None)
                         ensemble_paths shape: (n_paths, horizon)
            benchmarks: List of benchmark models to compare against
        """
        self.full_history = full_history
        self.forecast_fn = forecast_fn
        self.benchmarks = benchmarks or [RandomWalkBenchmark(), ARBenchmark(), TrendBenchmark()]

    def run(self, config: BacktestConfig) -> EvaluationReport:
        """Run full backtesting evaluation."""
        start_time = time.monotonic()

        # Determine forecast origins
        max_origin = len(self.full_history) - config.forecast_horizon
        origins = list(range(
            config.calibration_window,
            max_origin,
            config.step_size,
        ))[:config.num_origins]

        if not origins:
            logger.warning("No valid forecast origins found")
            return EvaluationReport(name=config.name)

        # Collect forecasts and actuals
        scorecards = []
        benchmark_scorecards: dict[str, list[ForecastScorecard]] = {
            bm.name: [] for bm in self.benchmarks
        }

        for variable in config.variables:
            if variable not in self.full_history.columns:
                continue

            forecast_errors = []
            actual_values = []
            forecast_values = []
            ensemble_collections = []

            # Benchmark forecasts
            bm_errors: dict[str, list[float]] = {bm.name: [] for bm in self.benchmarks}

            for origin in origins:
                history = self.full_history.iloc[:origin]
                actual = self.full_history[variable].iloc[
                    origin:origin + config.forecast_horizon
                ].values

                if len(actual) < config.forecast_horizon:
                    continue

                # Model forecast
                try:
                    median_fc, ensemble = self.forecast_fn(
                        history, config.forecast_horizon, variable
                    )
                    forecast_values.append(median_fc)
                    actual_values.append(actual)
                    forecast_errors.append(median_fc - actual)
                    if ensemble is not None:
                        ensemble_collections.append(ensemble)
                except Exception as e:
                    logger.warning(f"Forecast failed at origin {origin}: {e}")
                    continue

                # Benchmark forecasts
                for bm in self.benchmarks:
                    try:
                        bm_fc = bm.forecast(history, config.forecast_horizon, variable)
                        bm_errors[bm.name].append(bm_fc - actual)
                    except Exception:
                        pass

            if not forecast_errors:
                continue

            # Compute scorecard
            errors_array = np.array(forecast_errors)
            actuals_array = np.array(actual_values)
            forecasts_array = np.array(forecast_values)

            sc = ForecastScorecard(
                variable=variable,
                horizon=config.forecast_horizon,
                num_origins=len(forecast_errors),
            )

            # Point forecast metrics (averaged across origins and horizons)
            sc.rmse = float(np.sqrt(np.mean(errors_array ** 2)))
            sc.mae = float(np.mean(np.abs(errors_array)))
            denom = np.abs(actuals_array)
            denom[denom < 1e-10] = np.nan
            sc.mape = float(np.nanmean(np.abs(errors_array) / denom))
            sc.bias = float(np.mean(errors_array))

            # Coverage and CRPS (if ensemble available)
            if ensemble_collections:
                ensembles = np.array(ensemble_collections)  # (n_origins, n_paths, horizon)
                coverages_90 = []
                coverages_50 = []
                crps_values = []

                for oi in range(len(ensemble_collections)):
                    ens = ensemble_collections[oi]  # (n_paths, horizon)
                    act = actuals_array[oi]
                    for t in range(min(len(act), ens.shape[1])):
                        samples = ens[:, t]
                        q05, q25, q75, q95 = np.quantile(samples, [0.05, 0.25, 0.75, 0.95])
                        coverages_90.append(q05 <= act[t] <= q95)
                        coverages_50.append(q25 <= act[t] <= q75)
                        crps_values.append(_crps_ensemble(samples, act[t]))

                sc.coverage_90 = float(np.mean(coverages_90))
                sc.coverage_50 = float(np.mean(coverages_50))
                sc.crps = float(np.mean(crps_values))

            # Skill scores vs random walk
            for bm in self.benchmarks:
                if bm.name == "random_walk" and bm_errors[bm.name]:
                    bm_rmse = float(np.sqrt(np.mean(np.array(bm_errors[bm.name]) ** 2)))
                    sc.skill_score_rmse = 1.0 - sc.rmse / max(bm_rmse, 1e-10)
                    break

            scorecards.append(sc)

            # Benchmark scorecards
            for bm in self.benchmarks:
                if bm_errors[bm.name]:
                    bm_err = np.array(bm_errors[bm.name])
                    bm_sc = ForecastScorecard(
                        variable=variable,
                        horizon=config.forecast_horizon,
                        num_origins=len(bm_errors[bm.name]),
                        rmse=float(np.sqrt(np.mean(bm_err ** 2))),
                        mae=float(np.mean(np.abs(bm_err))),
                        bias=float(np.mean(bm_err)),
                    )
                    benchmark_scorecards[bm.name].append(bm_sc)

        elapsed = time.monotonic() - start_time

        return EvaluationReport(
            name=config.name,
            scorecards=scorecards,
            benchmark_scorecards=benchmark_scorecards,
            elapsed_seconds=elapsed,
            metadata={
                "num_origins": len(origins),
                "origins": origins,
            },
        )


def _crps_ensemble(ensemble: np.ndarray, observation: float) -> float:
    """Compute CRPS (Continuous Ranked Probability Score) for an ensemble forecast.

    CRPS = E|X - y| - 0.5 * E|X - X'|
    where X, X' are independent draws from the forecast distribution.
    """
    n = len(ensemble)
    if n == 0:
        return np.nan

    abs_diff = np.mean(np.abs(ensemble - observation))
    # Efficient computation of E|X - X'|
    sorted_ens = np.sort(ensemble)
    spread = 0.0
    for i in range(n):
        spread += (2 * i - n + 1) * sorted_ens[i]
    spread = spread / (n * (n - 1)) if n > 1 else 0.0

    return abs_diff - abs(spread)
