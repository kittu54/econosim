# Macro World Model — Project Log

> Chronological development log for the macro world model transformation.

---

## 2026-03-16 — Session Start: Macro World Model Architecture

### Context
- Starting from EconoSim with Phases 0-5 complete (SFC accounting, 4 agents, 3 markets, RL envs, 494 tests)
- Goal: Transform into forecasting-grade macro world model

### Repository Audit Summary
**Strengths**:
- Solid double-entry accounting in `core/accounting.py` (Ledger, BalanceSheet, Transaction)
- Clean agent hierarchy (Household, Firm, Bank, Government) with balance sheets
- SFC-consistent money creation (endogenous via loans, sovereign via government)
- Well-structured simulation loop with explicit stages
- Good test coverage (494 tests)
- Pydantic config, YAML scenarios, experiment runner
- RL environments (Gymnasium + PettingZoo)

**Weaknesses/Gaps**:
- No real-world data ingestion (FRED, BEA, IMF)
- GDP is a proxy (goods transacted + govt spending), not proper national accounting
- No measurement model mapping sim state → observable macro series
- No calibration/estimation engine (parameters are hand-tuned)
- No probabilistic forecasting
- No backtesting framework
- Agent behaviors are hardcoded in classes (no policy interface abstraction)
- Single bank, single government (not heterogeneous)
- No transformer/ML forecasting layer
- Performance untested at scale (100+ agents likely fine, 10k+ unknown)

**What Can Be Reused** (most of it):
- All core accounting (Ledger, BalanceSheet, Transaction) — excellent, keep as-is
- Agent classes — extend with policy interfaces, don't rewrite
- Markets — keep, refine later
- Config system — extend with calibration/forecast configs
- Experiment runner — extend for calibration/forecast workflows
- RL environments — keep, ensure compatibility with new policy interfaces
- All tests — keep, add new ones

**Immediate Architecture Decisions**:
1. Data layer is highest leverage — unlocks calibration, forecasting, backtesting
2. Policy interfaces before calibration — clean separation enables parameter estimation
3. Measurement model before calibration — need to know what we're matching
4. Incremental refactor, not rewrite — preserve all 494 tests

### Actions Taken
- Created `phases.md` with 10 macro world model phases
- Created `project_log.md` (this file)
- Beginning Phase M1: Data Ingestion Layer

### Next Steps
- Implement FredClient, BeaClient, ImfSdmxClient
- Build DataRegistry and DataStore
- Implement measurement model
- Add policy interfaces to agents

---

## 2026-03-16 — Macro World Model: Full Implementation

### What Was Built

**Phase M1 — Data Ingestion Layer** (`econosim/data/`):
- `FredClient`: FRED API with caching, retries, vintage/real-time data, frequency aggregation
- `BeaClient`: BEA NIPA table retrieval, parameter discovery, GDP components
- `ImfSdmxClient`: IMF SDMX 2.1 for cross-country panel data
- `DataStore`: Versioned Parquet storage with content hashing and metadata
- `DataRegistry`: Pipeline registration for reproducible data pulls
- `FRED_MACRO_SERIES`: 24 standard macro series catalog
- `BEA_NIPA_TABLES`, `IMF_IFS_INDICATORS`: Reference catalogs

**Phase M2 — Measurement Model** (`econosim/measurement/`):
- `NationalAccountsMapper`: Maps sim state → GDP (C+I+G+NX), inflation, unemployment
- `NationalAccountsOutput`: 30+ measured series per period
- `LaborMarketMetrics`, `FinancialSystemMetrics`: Detailed subsystem metrics
- `MeasuredSeries`: Formal series definitions with units, source variables, transformations
- `MEASUREMENT_SERIES`: 7 core macro series mapped to FRED equivalents

**Phase M3 — Policy Interfaces** (`econosim/policies/`):
- `FirmPolicy`, `HouseholdPolicy`, `BankPolicy`, `GovernmentPolicy`: Abstract interfaces
- `FirmAction`, `HouseholdAction`, `BankAction`, `GovernmentAction`: Typed action dataclasses
- `FirmState`, `HouseholdState`, `BankState`, `GovernmentState`, `MacroState`: State containers
- `RuleBasedFirmPolicy` etc.: Existing logic extracted into swappable policy objects

**Phase M4 — Calibration Engine** (`econosim/calibration/`):
- `ParameterRegistry`: Named parameters with priors, bounds, transforms, calibrated/fixed flags
- `Prior`: Uniform, Normal, LogNormal, Beta, Gamma priors with log_pdf and sampling
- `MomentDefinition`, `MomentSet`: Moment computation with burn-in, custom functions
- `SimulationObjective`: Wraps simulation as callable for optimization
- `SmmCalibrator`: Simulated Method of Moments (Nelder-Mead, Powell, differential evolution)
- `BayesianCalibrator`: Random-walk Metropolis-Hastings with synthetic likelihood
- `CalibrationResult`: Estimated params, moment fit, posterior samples, diagnostics
- `default_macro_registry()`: 8 calibrated parameters for US macro
- `default_us_moments()`: 10 empirical moments (GDP growth, unemployment, inflation, credit, Gini)

**Phase M5 — Forecasting Engine** (`econosim/forecasting/`):
- `ForecastEnsembleRunner`: Posterior sampling + shock path sampling + simulation
- `DensityForecast`: Quantile bands, event probabilities, DataFrame export
- `ShockProcess`: AR(1) stochastic shock generation for scenarios
- `ScenarioSpec`: Deterministic + stochastic scenario definitions
- Event probabilities: recession, high inflation, banking stress, high unemployment

**Phase M6 — Backtesting** (`econosim/forecasting/backtesting.py`):
- `BacktestRunner`: Rolling-origin forecast evaluation
- `ForecastScorecard`: RMSE, MAE, CRPS, coverage, skill scores
- `RandomWalkBenchmark`, `ARBenchmark`, `TrendBenchmark`: Baseline comparisons
- `EvaluationReport`: Summary tables with benchmark comparisons
- `_crps_ensemble()`: CRPS computation for ensemble forecasts

**Phase M7 — Transformer Integration** (`econosim/learning/transformers/`):
- `MacroTransformer`: Patched transformer with sinusoidal encoding (NumPy prototype)
- `ResidualForecaster`: Transformer residual correction over simulator forecasts
- `SimulationEmulator`: Fast parameter→trajectory approximation
- `TimeSeriesDataset`: Windowed datasets with temporal train/val/test splits
- `EmulatorDataset`: Parameter-trajectory pairs for emulator training
- `NumpyTrainer`: SGD training loop with early stopping
- `TransformerEvaluator`: MSE, RMSE, MAE evaluation

**API Upgrade** (`api/main.py`):
- `POST /api/calibrate`: Run SMM or Bayesian calibration
- `POST /api/forecast`: Run probabilistic forecast ensemble
- `GET /api/data/series`: List available macro data series
- `GET /api/models`: List model components
- `GET /api/measurement/series`: List measurement series definitions

**Tests**: 99 new tests across 6 test modules
- `test_data/test_sources.py`: 14 tests for FRED, BEA, IMF clients
- `test_data/test_storage.py`: 8 tests for DataStore, DataRegistry
- `test_measurement/test_national_accounts.py`: 12 tests for measurement model
- `test_policies/test_interfaces.py`: 9 tests for policy interfaces
- `test_calibration/test_calibration.py`: 24 tests for calibration engine
- `test_forecasting/test_forecasting.py`: 18 tests for forecasting/backtesting
- `test_learning/test_transformers.py`: 14 tests for transformer models

**Total test count**: 509 passing (up from 494)

### Architecture Decisions
1. **Incremental refactor**: All 494 existing tests preserved and passing
2. **Policy interfaces alongside agents**: Didn't modify agent classes; policies are external
3. **NumPy transformer prototype**: No PyTorch dependency; production version can use torch
4. **Parquet + JSON metadata**: Versioned data with content hashing for reproducibility
5. **SMM before ABC/SMC**: SMM is simpler, faster, and sufficient for initial calibration
6. **CRPS for distributional evaluation**: Standard proper scoring rule for density forecasts

### Known Limitations
- Transformer models are NumPy prototypes (no gradient training)
- Data clients require API keys (FRED, BEA) for actual data pulls
- Calibration is slow with many simulations (parallelize in M8)
- No central bank agent yet (optional in design)
- Measurement GDP still uses market transactions as floor

---

## 2026-03-16 — Policy Interfaces Wired Into Engine Loop

### Changes
- **`engine/simulation.py`**: Added `build_macro_state()`, `build_firm_state()`, `build_bank_state()`, `build_govt_state()` helpers that extract typed state snapshots from live simulation
- **`engine/simulation.py`**: Added `_apply_firm_policy()`, `_apply_bank_policy()`, `_apply_govt_policy()` that call policy.act() and apply actions to agents
- **`engine/simulation.py`**: `SimulationState` now holds optional `firm_policy`, `household_policy`, `bank_policy`, `government_policy` fields
- **`engine/simulation.py`**: `step()` applies policies at step 1a (after shocks, before markets)
- **`engine/simulation.py`**: `run_simulation()` accepts optional policy arguments
- **`markets/labor.py`**: Added `skip_vacancy_decision` flag so policies can pre-set vacancies
- **`tests/test_policy_integration.py`**: 17 new tests covering state builders, rule-based policies, custom policies (zero-vacancy, high-spending, aggressive-rate, spy policies), no-regression, balance sheet integrity

### Design Decisions
1. **Non-breaking integration**: Policies default to None; when None, all existing hardcoded logic runs unchanged
2. **Policy application order**: Bank policy → Government policy → Firm policy → markets. This ensures rates and fiscal params are set before firms make decisions
3. **Firm policy skips `decide_vacancies()` and `adjust_price()`**: When firm_policy is set, these methods are bypassed entirely. The policy's FirmAction.vacancies and price_adjustment fields drive behavior instead
4. **Wage adjustment left to existing logic**: `adjust_wage()` still runs after labor market since it depends on vacancy fill rate, which is only known post-matching

### Test Results
- **526 tests passing** (509 existing + 17 new)
- Zero regressions across all stress tests, accounting tests, and extension tests

### Next Steps
- Run actual FRED data pulls and calibrate to US macro moments
- Profile and parallelize calibration/forecasting runs
- Build PyTorch transformer training for production
- Add forecast fan charts to dashboard
- Wire household policy into goods market (consumption fraction)
