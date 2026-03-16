"""
EconoSim FastAPI backend — serves simulation data to the Next.js frontend.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException
from contextlib import asynccontextmanager

from api.database import init_db, get_db, SimulationRun

# Ensure econosim package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from econosim.config.schema import SimulationConfig
from econosim.experiments.runner import run_experiment, run_batch
from econosim.metrics.collector import enrich_dataframe

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database
    init_db()
    yield

app = FastAPI(title="EconoSim API", version="0.1.0", lifespan=lifespan)

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


# ── Database Models ───────────────────────────────────────────────


class SavedRunRequest(BaseModel):
    name: str
    config: dict[str, Any]
    summary: dict[str, Any]
    periods: list[dict[str, Any]]
    aggregate: list[dict[str, Any]] | None = None


class SavedRunResponse(BaseModel):
    id: str
    name: str
    created_at: str
    config: dict[str, Any]
    summary: dict[str, Any]
    # We omit periods in the base response to save bandwidth


class SavedRunDetailResponse(SavedRunResponse):
    periods: list[dict[str, Any]]
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


@app.post("/api/runs", response_model=SavedRunResponse)
def save_run(run_data: SavedRunRequest, db: Session = Depends(get_db)):
    db_run = SimulationRun(
        name=run_data.name,
        config=run_data.config,
        summary=run_data.summary,
        periods=run_data.periods,
        aggregate=run_data.aggregate
    )
    db.add(db_run)
    db.commit()
    db.refresh(db_run)
    
    return SavedRunResponse(
        id=db_run.id,
        name=db_run.name,
        created_at=db_run.created_at.isoformat(),
        config=db_run.config,
        summary=db_run.summary,
    )


@app.get("/api/runs", response_model=list[SavedRunResponse])
def list_runs(db: Session = Depends(get_db)):
    runs = db.query(SimulationRun).order_by(SimulationRun.created_at.desc()).all()
    return [
        SavedRunResponse(
            id=run.id,
            name=run.name,
            created_at=run.created_at.isoformat(),
            config=run.config,
            summary=run.summary,
        ) for run in runs
    ]


@app.get("/api/runs/{run_id}", response_model=SavedRunDetailResponse)
def get_run(run_id: str, db: Session = Depends(get_db)):
    run = db.query(SimulationRun).filter(SimulationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
        
    return SavedRunDetailResponse(
        id=run.id,
        name=run.name,
        created_at=run.created_at.isoformat(),
        config=run.config,
        summary=run.summary,
        periods=run.periods,
        aggregate=run.aggregate
    )


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str, db: Session = Depends(get_db)):
    run = db.query(SimulationRun).filter(SimulationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
        
    db.delete(run)
    db.commit()
    return {"status": "success"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
