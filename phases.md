# Macro World Model — Development Phases

> Roadmap for transforming EconoSim into a forecasting-grade macroeconomic world model.

---

## Phase M0 — Context Maintenance & Architecture Blueprint
**Status**: complete
**Dependencies**: None
**Success criteria**: phases.md and project_log.md exist, target architecture documented

---

## Phase M1 — Data Ingestion Layer
**Status**: complete
**Dependencies**: M0
**Scope**:
- FRED client with vintage/real-time support
- BEA client with NIPA table discovery
- IMF SDMX client with country panel support
- DataRegistry, DatasetVersion, DataStore (Parquet + metadata)
- Canonical internal schema, frequency alignment, missing value policy
**Success criteria**: Can pull GDP, CPI, unemployment, credit data from all 3 sources; data cached as Parquet with provenance metadata

---

## Phase M2 — Measurement Model / National Accounts
**Status**: complete
**Dependencies**: M1
**Scope**:
- NationalAccountsMapper: simulation state → observed macro series
- LaborMarketMetrics, FinancialSystemMetrics, PriceIndex/Deflator
- GDP decomposition (C + I + G), inflation, unemployment, credit growth
- Measurement config mapping observed series IDs to model outputs
**Success criteria**: Simulator produces GDP, inflation, unemployment, credit growth series comparable to FRED data

---

## Phase M3 — Economic Core Refactor + Policy Interfaces
**Status**: complete
**Dependencies**: M0
**Scope**:
- Formal policy interfaces (HouseholdPolicy, FirmPolicy, BankPolicy, GovernmentPolicy)
- Clean state-transition simulation loop with explicit stages
- Rule-based baseline policies extracted from current agent logic
- Policy swap without engine changes
- **Wired into engine**: Policies now drive agent decisions in the simulation loop via `_apply_firm_policy`, `_apply_bank_policy`, `_apply_govt_policy`
- State builders: `build_macro_state`, `build_firm_state`, `build_bank_state`, `build_govt_state`
- `run_simulation()` accepts optional policy arguments
**Success criteria**: Current behavior preserved, policies swappable, all existing tests pass

---

## Phase M4 — Calibration / Estimation Engine
**Status**: complete
**Dependencies**: M2, M3
**Scope**:
- Parameter registry with priors, bounds, calibrated-vs-fixed flags
- Moment definitions (GDP volatility, inflation persistence, unemployment mean, etc.)
- Simulated Method of Moments (SMM) calibrator
- Bayesian synthetic likelihood calibrator
- Common random numbers for variance reduction
- Diagnostics, convergence tools, posterior artifacts
**Success criteria**: Can estimate 5+ parameters to match US macro moments; parameter uncertainty quantified

---

## Phase M5 — Forecasting Engine
**Status**: complete
**Dependencies**: M4
**Scope**:
- ForecastConfig, ShockProcess, ScenarioSpec
- Posterior parameter sampling + stochastic shock path sampling
- ForecastEnsembleRunner producing DensityForecast
- Quantiles, fan charts, event probabilities (recession, high inflation, bank stress)
- Forecast artifact persistence
**Success criteria**: Produces GDP/inflation/unemployment fan charts with credible intervals

---

## Phase M6 — Backtesting / Evaluation
**Status**: complete
**Dependencies**: M5
**Scope**:
- Rolling-origin evaluation with expanding/sliding window calibration
- Point and distributional forecast metrics (RMSE, CRPS, PIT calibration)
- Benchmark models (random walk, AR, VAR, naive)
- Vintage-aware evaluation
- ForecastScorecard, EvaluationReport
**Success criteria**: Backtested over 10+ rolling origins; scores compared to benchmarks

---

## Phase M7 — Transformer Integration
**Status**: complete (prototype)
**Dependencies**: M5, M6
**Scope**:
- Macro forecasting head (residual correction over simulator)
- Agent policy transformer (optional learned policies)
- Simulation emulator/surrogate for fast calibration
- Training pipelines, evaluation, checkpoint management
**Success criteria**: Transformer residual model improves forecast CRPS vs simulator-only
**Note**: NumPy prototype complete; PyTorch production version pending

---

## Phase M8 — Scalability / Performance
**Status**: not_started
**Dependencies**: M4
**Scope**:
- Profile hotspots, vectorize hot loops
- Parallel Monte Carlo for calibration/forecasting
- Columnar storage for large batches
- Path from 100 to 100k agents
**Success criteria**: 1000-run ensemble completes in <5 min for 100-agent economy

---

## Phase M9 — API / Persistence / Dashboard Upgrade
**Status**: in_progress
**Dependencies**: M5, M6
**Scope**:
- Extended FastAPI endpoints for data pull, calibrate, forecast, backtest
- Run/dataset/model registries
- Experiment metadata, reproducibility hashes
- Dashboard: observed data browser, calibration diagnostics, forecast fan charts
**Success criteria**: End-to-end workflow accessible via API; dashboard shows forecast fan charts
**Note**: API endpoints complete; dashboard and persistence pending

---

## Current Status Summary
- **Phases M0-M7**: Complete (core implementation)
- **Phase M3**: All four policy interfaces (firm, household, bank, govt) wired into engine, calibration, forecasting, and RL
- **Phase M8 (performance)**: Not started
- **Phase M9 (platform)**: In progress (API done, dashboard pending)
- **Tests**: 544 passing
- **New modules**: data, measurement, policies, calibration, forecasting, learning, rl/macro_env

## Next Priority Actions
1. Run actual FRED data pulls and calibrate to US macro moments
2. Profile and parallelize calibration/forecasting runs
3. Build PyTorch transformer training for production
4. Add forecast fan charts to dashboard
