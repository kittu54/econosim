"""Probabilistic forecasting engine.

Produces distributional forecasts by:
1. Sampling calibrated parameter draws (from posterior or bootstrap)
2. Sampling stochastic shock paths
3. Running simulation ensembles forward
4. Aggregating into predictive distributions

Key outputs: median paths, quantile bands, event probabilities.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from econosim.config.schema import SimulationConfig, ShockSpec
from econosim.calibration.engine import CalibrationResult

logger = logging.getLogger(__name__)


class ShockProcess(BaseModel):
    """Specification for stochastic shock generation during forecasts."""

    shock_type: str  # "supply", "demand", "credit", "fiscal"
    parameter: str
    mean: float = 1.0  # mean of multiplicative shock
    std: float = 0.0   # std of multiplicative shock
    persistence: float = 0.0  # AR(1) coefficient
    additive: bool = False

    def generate_path(
        self,
        horizon: int,
        rng: np.random.Generator,
    ) -> list[float]:
        """Generate a shock path over the forecast horizon."""
        if self.std == 0:
            return [self.mean] * horizon

        path = []
        prev = 0.0
        for t in range(horizon):
            innovation = rng.normal(0, self.std)
            shock = self.persistence * prev + innovation
            prev = shock
            path.append(self.mean + shock)
        return path


class ScenarioSpec(BaseModel):
    """Specification for a forecast scenario."""

    name: str = "baseline"
    description: str = ""
    shocks: list[ShockSpec] = Field(default_factory=list)
    shock_processes: list[ShockProcess] = Field(default_factory=list)
    parameter_overrides: dict[str, float] = Field(default_factory=dict)


class ForecastConfig(BaseModel):
    """Configuration for a forecast run."""

    horizon: int = 24  # forecast periods ahead
    num_parameter_draws: int = 50  # from calibration posterior
    num_shock_draws: int = 10  # per parameter draw
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    variables: list[str] = Field(default_factory=lambda: [
        "gdp", "unemployment_rate", "avg_price", "total_loans_outstanding",
        "inflation_rate", "gdp_growth", "gini_deposits",
    ])
    seed: int = 42
    burn_in: int = 20  # warm-up periods before forecast starts
    scenarios: list[ScenarioSpec] = Field(default_factory=lambda: [ScenarioSpec()])


@dataclass
class DensityForecast:
    """Output of a probabilistic forecast."""

    config: dict[str, Any]
    scenario_name: str
    variables: list[str]
    horizon: int
    num_paths: int

    # paths[variable] = np.ndarray of shape (num_paths, horizon)
    paths: dict[str, np.ndarray] = field(default_factory=dict)

    # Computed summary statistics
    quantile_levels: list[float] = field(default_factory=list)
    # quantiles[variable] = dict of {level: np.ndarray(horizon)}
    quantiles: dict[str, dict[float, np.ndarray]] = field(default_factory=dict)

    # Event probabilities (e.g., P(recession), P(inflation > 4%))
    event_probs: dict[str, float] = field(default_factory=dict)

    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_quantiles(self, levels: list[float] | None = None) -> None:
        """Compute quantile paths from raw simulation paths."""
        if levels is None:
            levels = self.quantile_levels or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.quantile_levels = levels

        for var, path_arr in self.paths.items():
            self.quantiles[var] = {}
            for q in levels:
                self.quantiles[var][q] = np.nanquantile(path_arr, q, axis=0)

    def compute_event_probabilities(self) -> None:
        """Compute probabilities of key economic events."""
        # Recession: any period with negative GDP growth
        if "gdp_growth" in self.paths:
            paths = self.paths["gdp_growth"]
            # Recession = 2 consecutive periods of negative growth
            recession_count = 0
            for i in range(paths.shape[0]):
                path = paths[i]
                for t in range(1, len(path)):
                    if path[t] < 0 and path[t - 1] < 0:
                        recession_count += 1
                        break
            self.event_probs["recession_probability"] = recession_count / max(paths.shape[0], 1)

        # High inflation: inflation > 5% annualized at any point
        if "inflation_rate" in self.paths:
            paths = self.paths["inflation_rate"]
            high_inflation = np.any(paths > 0.05, axis=1).mean()
            self.event_probs["high_inflation_probability"] = float(high_inflation)

        # Banking stress: bank capital ratio < 5% at any point
        if "bank_capital_ratio" in self.paths:
            paths = self.paths["bank_capital_ratio"]
            bank_stress = np.any(paths < 0.05, axis=1).mean()
            self.event_probs["banking_stress_probability"] = float(bank_stress)

        # High unemployment: unemployment > 10%
        if "unemployment_rate" in self.paths:
            paths = self.paths["unemployment_rate"]
            high_unemp = np.any(paths > 0.10, axis=1).mean()
            self.event_probs["high_unemployment_probability"] = float(high_unemp)

    def median_path(self, variable: str) -> np.ndarray:
        if variable in self.quantiles and 0.5 in self.quantiles[variable]:
            return self.quantiles[variable][0.5]
        if variable in self.paths:
            return np.nanmedian(self.paths[variable], axis=0)
        return np.array([])

    def to_dataframe(self) -> pd.DataFrame:
        """Convert quantile forecasts to a DataFrame."""
        rows = []
        for var in self.variables:
            if var not in self.quantiles:
                continue
            for t in range(self.horizon):
                row = {"variable": var, "horizon": t + 1}
                for q, vals in self.quantiles[var].items():
                    row[f"q{int(q*100):02d}"] = float(vals[t])
                rows.append(row)
        return pd.DataFrame(rows)


class ForecastEnsembleRunner:
    """Runs forecast ensembles from calibrated parameters."""

    def __init__(
        self,
        base_config: SimulationConfig,
        calibration_result: CalibrationResult | None = None,
        sim_runner: Callable[[SimulationConfig], pd.DataFrame] | None = None,
        policies: dict[str, Any] | None = None,
    ) -> None:
        self.base_config = base_config
        self.calibration = calibration_result
        self.policies = policies or {}
        self._sim_runner = sim_runner or self._make_policy_runner()

        # Override if user provided explicit runner
        if sim_runner is not None:
            self._sim_runner = sim_runner

    def _make_policy_runner(self) -> Callable[[SimulationConfig], pd.DataFrame]:
        """Create a sim runner that passes policies to the simulation."""
        policies = self.policies

        def runner(config: SimulationConfig) -> pd.DataFrame:
            from econosim.engine.simulation import build_simulation, step
            from econosim.metrics.collector import history_to_dataframe, enrich_dataframe

            state = build_simulation(config)
            state.firm_policy = policies.get("firm_policy")
            state.household_policy = policies.get("household_policy")
            state.bank_policy = policies.get("bank_policy")
            state.government_policy = policies.get("government_policy")
            for _ in range(config.num_periods):
                step(state)
            return enrich_dataframe(history_to_dataframe(state.history))

        return runner

    def forecast(
        self,
        config: ForecastConfig,
        scenario: ScenarioSpec | None = None,
    ) -> DensityForecast:
        """Run a probabilistic forecast ensemble.

        Steps:
        1. Generate parameter draws (from posterior or point estimate)
        2. Generate shock paths
        3. Run simulation for each (param, shock) combination
        4. Collect forecast paths
        5. Compute quantiles and event probabilities
        """
        start_time = time.monotonic()
        scenario = scenario or ScenarioSpec()
        rng = np.random.default_rng(config.seed)

        # Generate parameter draws
        if (
            self.calibration is not None
            and self.calibration.posterior_samples is not None
        ):
            burn = self.calibration.metadata.get("burn_in", 250)
            posterior = self.calibration.posterior_samples[burn:]
            indices = rng.choice(len(posterior), size=config.num_parameter_draws, replace=True)
            param_draws = posterior[indices]
        else:
            # Use point estimate (or defaults) with small perturbations
            if self.calibration is not None:
                base_params = np.array(list(self.calibration.estimated_params.values()))
            else:
                base_params = np.zeros(0)

            if len(base_params) > 0:
                param_draws = np.tile(base_params, (config.num_parameter_draws, 1))
                # Add small noise for uncertainty
                param_draws += rng.normal(0, 0.01 * np.abs(base_params) + 1e-6, param_draws.shape)
            else:
                param_draws = [None] * config.num_parameter_draws

        # Run ensemble
        all_paths: dict[str, list[np.ndarray]] = {v: [] for v in config.variables}
        total_runs = config.num_parameter_draws * config.num_shock_draws

        for p_idx in range(config.num_parameter_draws):
            for s_idx in range(config.num_shock_draws):
                run_seed = int(rng.integers(0, 1_000_000))
                sim_config = self._build_run_config(
                    config, scenario, param_draws, p_idx, run_seed
                )

                # Add stochastic shocks from shock processes
                if scenario.shock_processes:
                    self._add_shock_path(sim_config, scenario, config, rng)

                try:
                    df = self._sim_runner(sim_config)
                    # Extract forecast horizon from end of simulation
                    forecast_start = max(0, len(df) - config.horizon)
                    for var in config.variables:
                        if var in df.columns:
                            path = df[var].iloc[forecast_start:].values
                            if len(path) == config.horizon:
                                all_paths[var].append(path)
                            else:
                                # Pad if needed
                                padded = np.full(config.horizon, np.nan)
                                padded[:len(path)] = path
                                all_paths[var].append(padded)
                except Exception as e:
                    logger.warning(f"Forecast run failed: {e}")
                    continue

        # Build DensityForecast
        forecast = DensityForecast(
            config=config.model_dump(),
            scenario_name=scenario.name,
            variables=config.variables,
            horizon=config.horizon,
            num_paths=len(all_paths.get(config.variables[0], [])),
            quantile_levels=config.quantiles,
        )

        for var in config.variables:
            if all_paths[var]:
                forecast.paths[var] = np.array(all_paths[var])
            else:
                forecast.paths[var] = np.full((1, config.horizon), np.nan)

        forecast.compute_quantiles(config.quantiles)
        forecast.compute_event_probabilities()
        forecast.elapsed_seconds = time.monotonic() - start_time

        logger.info(
            f"Forecast complete: {forecast.num_paths} paths, "
            f"{forecast.elapsed_seconds:.1f}s, "
            f"scenario={scenario.name}"
        )

        return forecast

    def _build_run_config(
        self,
        forecast_config: ForecastConfig,
        scenario: ScenarioSpec,
        param_draws: Any,
        p_idx: int,
        seed: int,
    ) -> SimulationConfig:
        """Build a simulation config for one forecast run."""
        config = self.base_config.model_copy(deep=True)
        config.seed = seed
        config.num_periods = forecast_config.burn_in + forecast_config.horizon

        # Apply calibrated parameters
        if self.calibration is not None and param_draws is not None:
            if hasattr(param_draws, '__getitem__') and param_draws[p_idx] is not None:
                param_names = list(self.calibration.estimated_params.keys())
                if len(param_names) > 0:
                    draw = param_draws[p_idx]
                    for i, name in enumerate(param_names):
                        if i < len(draw):
                            self._set_config_param(config, name, float(draw[i]))

        # Apply scenario parameter overrides
        for param_name, value in scenario.parameter_overrides.items():
            self._set_config_param(config, param_name, value)

        # Apply deterministic scenario shocks
        config.shocks = list(config.shocks) + list(scenario.shocks)

        return config

    def _add_shock_path(
        self,
        config: SimulationConfig,
        scenario: ScenarioSpec,
        forecast_config: ForecastConfig,
        rng: np.random.Generator,
    ) -> None:
        """Add stochastic shock path to config."""
        for proc in scenario.shock_processes:
            path = proc.generate_path(forecast_config.horizon, rng)
            for t, mag in enumerate(path):
                if abs(mag - proc.mean) > 1e-8:  # only add non-trivial shocks
                    config.shocks.append(ShockSpec(
                        period=forecast_config.burn_in + t,
                        shock_type=proc.shock_type,
                        parameter=proc.parameter,
                        magnitude=mag,
                        additive=proc.additive,
                    ))

    @staticmethod
    def _set_config_param(config: SimulationConfig, name: str, value: float) -> None:
        """Set a parameter on the config by dotted path or name."""
        # Try dotted path first
        parts = name.split(".")
        if len(parts) == 2:
            section, key = parts
            if hasattr(config, section):
                sub = getattr(config, section)
                if hasattr(sub, key):
                    setattr(sub, key, value)
                    return
        # Try direct attribute
        if hasattr(config, name):
            setattr(config, name, value)


def _default_forecast_runner(config: SimulationConfig) -> pd.DataFrame:
    """Default simulation runner for forecasting."""
    from econosim.engine.simulation import build_simulation, step
    from econosim.metrics.collector import history_to_dataframe, enrich_dataframe

    state = build_simulation(config)
    for _ in range(config.num_periods):
        step(state)
    return enrich_dataframe(history_to_dataframe(state.history))
