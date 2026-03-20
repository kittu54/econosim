"""
EconoSim FastAPI backend — serves simulation data, calibration,
forecasting, and data management to the Next.js frontend.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure econosim package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from econosim.config.schema import SimulationConfig
from econosim.experiments.runner import run_experiment, run_batch
from econosim.metrics.collector import enrich_dataframe

app = FastAPI(title="EconoSim API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────


class HouseholdParams(BaseModel):
    count: int = 100
    initial_deposits: float = 1000.0
    consumption_propensity: float = 0.8
    wealth_propensity: float = 0.4
    reservation_wage: float = 50.0


class FirmParams(BaseModel):
    count: int = 5
    initial_deposits: float = 15000.0
    initial_price: float = 10.0
    initial_wage: float = 60.0
    labor_productivity: float = 8.0
    price_adjustment_speed: float = 0.03
    wage_adjustment_speed: float = 0.02


class GovernmentParams(BaseModel):
    income_tax_rate: float = 0.2
    transfer_per_unemployed: float = 50.0
    spending_per_period: float = 2000.0
    initial_deposits: float = 100000.0


class BankParams(BaseModel):
    base_interest_rate: float = 0.005
    capital_adequacy_ratio: float = 0.08


class ExtensionParams(BaseModel):
    enable_expectations: bool = False
    enable_networks: bool = False
    enable_bonds: bool = False


class SimulationRequest(BaseModel):
    num_periods: int = Field(60, ge=5, le=500)
    seed: int = Field(42, ge=0, le=99999)
    n_seeds: int = Field(1, ge=1, le=20)
    household: HouseholdParams = Field(default_factory=HouseholdParams)
    firm: FirmParams = Field(default_factory=FirmParams)
    government: GovernmentParams = Field(default_factory=GovernmentParams)
    bank: BankParams = Field(default_factory=BankParams)
    extensions: ExtensionParams = Field(default_factory=ExtensionParams)


class SimulationResponse(BaseModel):
    periods: list[dict[str, Any]]
    summary: dict[str, Any]
    config: dict[str, Any]
    has_ci: bool = False
    aggregate: list[dict[str, Any]] | None = None


# ── Endpoints ─────────────────────────────────────────────────────


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/defaults")
def defaults():
    """Return default simulation configuration."""
    config = SimulationConfig()
    return config.model_dump()


@app.post("/api/simulate", response_model=SimulationResponse)
def simulate(req: SimulationRequest):
    """Run a simulation with the given parameters."""
    try:
        config = SimulationConfig(
            num_periods=req.num_periods,
            seed=req.seed,
            household=req.household.model_dump(),
            firm=req.firm.model_dump(),
            government=req.government.model_dump(),
            bank=req.bank.model_dump(),
            extensions=req.extensions.model_dump(),
        )

        if req.n_seeds == 1:
            result = run_experiment(config)
            df = result["dataframe"]
            periods = df.reset_index().to_dict(orient="records")
            return SimulationResponse(
                periods=periods,
                summary=result["summary"],
                config=config.model_dump(),
                has_ci=False,
            )
        else:
            seeds_list = list(range(req.seed, req.seed + req.n_seeds))
            batch = run_batch(config, seeds_list)
            df = batch["runs"][0]["dataframe"]
            agg_df = batch["aggregate"]
            periods = df.reset_index().to_dict(orient="records")
            aggregate = agg_df.reset_index().to_dict(orient="records")
            return SimulationResponse(
                periods=periods,
                summary=batch["runs"][0]["summary"],
                config=config.model_dump(),
                has_ci=True,
                aggregate=aggregate,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Calibration endpoints ─────────────────────────────────────────


class CalibrationRequest(BaseModel):
    num_simulations: int = Field(5, ge=1, le=50)
    num_periods: int = Field(120, ge=20, le=500)
    method: str = Field("smm", pattern="^(smm|bayesian)$")
    max_iterations: int = Field(100, ge=10, le=1000)
    seed: int = Field(42, ge=0)
    household: HouseholdParams = Field(default_factory=HouseholdParams)
    firm: FirmParams = Field(default_factory=FirmParams)
    government: GovernmentParams = Field(default_factory=GovernmentParams)
    bank: BankParams = Field(default_factory=BankParams)


@app.post("/api/calibrate")
def calibrate(req: CalibrationRequest):
    """Run calibration to estimate structural parameters."""
    try:
        from econosim.calibration.parameters import default_macro_registry
        from econosim.calibration.moments import default_us_moments
        from econosim.calibration.engine import (
            CalibrationProfile,
            SimulationObjective,
            SmmCalibrator,
            BayesianCalibrator,
        )

        config = SimulationConfig(
            num_periods=req.num_periods,
            seed=req.seed,
            household=req.household.model_dump(),
            firm=req.firm.model_dump(),
            government=req.government.model_dump(),
            bank=req.bank.model_dump(),
        )

        registry = default_macro_registry()
        moments = default_us_moments()
        profile = CalibrationProfile(
            num_simulations=req.num_simulations,
            num_periods=req.num_periods,
            max_iterations=req.max_iterations,
            seed_base=req.seed,
        )

        objective = SimulationObjective(config, registry, moments, profile)

        if req.method == "bayesian":
            calibrator = BayesianCalibrator(
                objective, registry, profile, num_samples=req.max_iterations
            )
        else:
            calibrator = SmmCalibrator(objective, registry, profile)

        result = calibrator.calibrate()

        response = {
            "method": result.method,
            "converged": result.converged,
            "objective": result.weighted_objective,
            "estimated_params": result.estimated_params,
            "moment_fit": result.moment_fit,
            "num_evaluations": result.num_evaluations,
            "elapsed_seconds": result.elapsed_seconds,
        }

        # Add convergence warning if not converged
        if not result.converged:
            response["warning"] = (
                f"Calibration did not converge after {result.num_evaluations} evaluations. "
                f"Final objective: {result.weighted_objective:.6f}. "
                "Consider increasing max_iterations or adjusting parameter bounds."
            )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Forecast endpoints ────────────────────────────────────────────


VALID_SCENARIOS = {"baseline", "recession", "high_growth", "tight_money"}

_SCENARIO_OVERRIDES: dict[str, dict[str, float]] = {
    "baseline": {},
    "recession": {
        "household.consumption_propensity": 0.5,
        "firm.labor_productivity": 5.0,
    },
    "high_growth": {
        "household.consumption_propensity": 0.95,
        "firm.labor_productivity": 12.0,
    },
    "tight_money": {
        "bank.base_interest_rate": 0.03,
    },
}


class ForecastRequest(BaseModel):
    horizon: int = Field(24, ge=1, le=120)
    num_parameter_draws: int = Field(20, ge=1, le=200)
    num_shock_draws: int = Field(5, ge=1, le=50)
    seed: int = Field(42, ge=0)
    scenario_name: str = "baseline"
    household: HouseholdParams = Field(default_factory=HouseholdParams)
    firm: FirmParams = Field(default_factory=FirmParams)
    government: GovernmentParams = Field(default_factory=GovernmentParams)
    bank: BankParams = Field(default_factory=BankParams)


@app.post("/api/forecast")
def forecast(req: ForecastRequest):
    """Run probabilistic forecast ensemble."""
    # Validate scenario name
    if req.scenario_name not in VALID_SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario '{req.scenario_name}'. "
            f"Valid scenarios: {sorted(VALID_SCENARIOS)}",
        )

    try:
        from econosim.forecasting.engine import (
            ForecastConfig,
            ForecastEnsembleRunner,
            ScenarioSpec,
        )

        config = SimulationConfig(
            seed=req.seed,
            household=req.household.model_dump(),
            firm=req.firm.model_dump(),
            government=req.government.model_dump(),
            bank=req.bank.model_dump(),
        )

        forecast_config = ForecastConfig(
            horizon=req.horizon,
            num_parameter_draws=req.num_parameter_draws,
            num_shock_draws=req.num_shock_draws,
            seed=req.seed,
        )

        scenario = ScenarioSpec(
            name=req.scenario_name,
            parameter_overrides=_SCENARIO_OVERRIDES.get(req.scenario_name, {}),
        )

        runner = ForecastEnsembleRunner(config)
        result = runner.forecast(forecast_config, scenario)

        return {
            "scenario": result.scenario_name,
            "horizon": result.horizon,
            "num_paths": result.num_paths,
            "elapsed_seconds": result.elapsed_seconds,
            "event_probabilities": result.event_probs,
            "quantiles": result.to_dataframe().to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Backtest endpoints ────────────────────────────────────────────


class BacktestRequest(BaseModel):
    forecast_horizon: int = Field(12, ge=1, le=60)
    num_origins: int = Field(10, ge=1, le=50)
    calibration_window: int = Field(60, ge=10, le=300)
    step_size: int = Field(4, ge=1, le=50)
    variables: list[str] = Field(
        default_factory=lambda: ["gdp", "unemployment_rate", "avg_price"]
    )
    num_periods: int = Field(200, ge=50, le=500)
    seed: int = Field(42, ge=0)
    household: HouseholdParams = Field(default_factory=HouseholdParams)
    firm: FirmParams = Field(default_factory=FirmParams)
    government: GovernmentParams = Field(default_factory=GovernmentParams)
    bank: BankParams = Field(default_factory=BankParams)


@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    """Run rolling-origin backtesting evaluation."""
    try:
        import numpy as np

        from econosim.forecasting.backtesting import (
            BacktestConfig,
            BacktestRunner,
            RandomWalkBenchmark,
            ARBenchmark,
            TrendBenchmark,
        )

        config = SimulationConfig(
            num_periods=req.num_periods,
            seed=req.seed,
            household=req.household.model_dump(),
            firm=req.firm.model_dump(),
            government=req.government.model_dump(),
            bank=req.bank.model_dump(),
        )

        # Generate simulation history for backtesting
        result = run_experiment(config)
        df = result["dataframe"]

        # Build forecast function using simulation-based naive forecast
        def sim_forecast(
            history: "pd.DataFrame", horizon: int, variable: str
        ) -> tuple["np.ndarray", "np.ndarray | None"]:
            import pandas as pd

            last = float(history[variable].iloc[-1])
            median = np.full(horizon, last)
            rng = np.random.default_rng(req.seed)
            ensemble = median[None, :] + rng.normal(0, abs(last) * 0.05 + 1.0, (20, horizon))
            return median, ensemble

        benchmarks = [RandomWalkBenchmark(), ARBenchmark(), TrendBenchmark()]
        runner = BacktestRunner(df, sim_forecast, benchmarks)

        bt_config = BacktestConfig(
            forecast_horizon=req.forecast_horizon,
            num_origins=req.num_origins,
            calibration_window=req.calibration_window,
            step_size=req.step_size,
            variables=req.variables,
        )

        report = runner.run(bt_config)
        summary = report.summary_table()

        return {
            "name": report.name,
            "elapsed_seconds": report.elapsed_seconds,
            "scorecards": [
                {
                    "variable": sc.variable,
                    "horizon": sc.horizon,
                    "rmse": sc.rmse,
                    "mae": sc.mae,
                    "mape": sc.mape,
                    "bias": sc.bias,
                    "crps": sc.crps,
                    "coverage_90": sc.coverage_90,
                    "coverage_50": sc.coverage_50,
                    "pit_uniformity": sc.pit_uniformity,
                    "skill_score_rmse": sc.skill_score_rmse,
                    "skill_score_crps": sc.skill_score_crps,
                    "num_origins": sc.num_origins,
                }
                for sc in report.scorecards
            ],
            "benchmark_scorecards": {
                bm_name: [
                    {
                        "variable": sc.variable,
                        "horizon": sc.horizon,
                        "rmse": sc.rmse,
                        "mae": sc.mae,
                        "bias": sc.bias,
                    }
                    for sc in scs
                ]
                for bm_name, scs in report.benchmark_scorecards.items()
            },
            "summary": summary.to_dict(orient="records") if not summary.empty else [],
            "metadata": report.metadata,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Data endpoints ────────────────────────────────────────────────


class DataPullRequest(BaseModel):
    series: dict[str, str] = Field(
        default_factory=lambda: {
            "gdp_real": "GDPC1",
            "unemployment": "UNRATE",
            "cpi": "CPIAUCSL",
        },
        description="Dict of {column_name: FRED_series_id}",
    )
    start_date: str = Field("2000-01-01", pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str | None = None
    frequency: str = Field("q", pattern="^(d|w|bw|m|q|sa|a)$")
    compute_moments: bool = False


@app.post("/api/data/pull")
def pull_data(req: DataPullRequest):
    """Pull FRED data series into an aligned DataFrame.

    Requires FRED_API_KEY environment variable to be set.
    Returns empty data with a warning if the API key is missing.
    """
    try:
        from econosim.data.pipelines import pull_series, compute_calibration_moments

        df = pull_series(
            series=req.series,
            start_date=req.start_date,
            end_date=req.end_date,
            frequency=req.frequency,
        )

        response: dict[str, Any] = {
            "num_series": len(df.columns),
            "num_observations": len(df),
            "columns": list(df.columns),
            "data": df.reset_index().to_dict(orient="records") if not df.empty else [],
        }

        if req.compute_moments and not df.empty:
            moments = compute_calibration_moments(df)
            response["moments"] = moments

        if df.empty:
            response["warning"] = (
                "No data returned. Ensure FRED_API_KEY environment variable is set."
            )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/series")
def list_data_series():
    """List available macro data series definitions."""
    from econosim.data.sources.fred import FRED_MACRO_SERIES
    return {"fred_series": FRED_MACRO_SERIES}


@app.get("/api/models")
def list_models():
    """List available model components."""
    return {
        "agents": ["household", "firm", "bank", "government"],
        "markets": ["labor", "goods", "credit"],
        "policies": ["rule_based", "rl", "transformer"],
        "calibration_methods": ["smm", "bayesian"],
        "forecast_methods": ["ensemble", "residual_transformer"],
        "benchmarks": ["random_walk", "ar1", "linear_trend"],
    }


@app.get("/api/measurement/series")
def list_measurement_series():
    """List measurement model series definitions."""
    from econosim.measurement.national_accounts import MEASUREMENT_SERIES
    return {
        name: {
            "description": s.description,
            "units": s.units,
            "source_variables": s.source_variables,
        }
        for name, s in MEASUREMENT_SERIES.items()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
