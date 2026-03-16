# EconoSim Development Phases

> Roadmap and status for the EconoSim multi-agent economic simulation platform.

---

## Phase 0 — Core Infrastructure ✅

**Status**: Complete

**What was built**:
- Project structure and packaging (`pyproject.toml`, editable install)
- Core accounting layer: `Account`, `AccountType`, `BalanceSheet`, `Ledger`, `Transaction`
- Loan contracts: `LoanContract`, `LoanBook`, `LoanStatus`
- Goods/inventory: `Inventory` with weighted-average costing
- All 4 agent models: `Household`, `Firm`, `Bank`, `Government`
- All 3 market modules: `LaborMarket`, `GoodsMarket`, `CreditMarket`
- Simulation engine: `build_simulation()`, `step()`, `run_simulation()`
- Shock system: supply, demand, credit, fiscal shocks via config
- Metrics engine: GDP, unemployment, inflation, Gini, credit metrics
- Config system: Pydantic schemas, YAML scenario files
- Experiment runner: single runs, batch runs with multiple seeds
- RL interface scaffold: `EconEnvInterface` ABC, observation/action specs

**Key outcomes**:
- 55 tests passing
- Stock-flow consistent double-entry accounting
- Endogenous money creation via loans
- Reproducible seeded runs
- All accounting invariants verified

---

## Phase 1 — Dynamics Stabilization ✅

**Status**: Complete

**What was fixed**:
- **Government spending as fiscal stabilizer**: Injects money even with zero inventory
- **Sovereign money creation**: `Government.ensure_solvency()` creates deposits as needed (MMT/SFC)
- **Price adjustment**: Only raises prices when `prev_units_sold > 0.1` (prevents infinite price increases in dead markets)
- **Revenue-based hiring**: `demand_estimate = max(prev_units_sold, prev_revenue/price)`
- **Minimum hiring floor**: Always try to hire 1 worker if affordable
- **Initial demand estimate**: Set `units_sold` and `revenue` so `reset_period_state()` propagates correctly
- **Buffer-stock consumption**: `C = α₁ × income + α₂ × wealth` (SFC standard)

**Key outcomes**:
- Economy no longer collapses to GDP=0
- Full employment sustained in early periods
- Mild deflation identified as calibration issue (acceptable for MVP)
- CLI entry point (`python -m econosim`)

---

## Phase 2 — Experiments & Visualization ✅

**Status**: Complete

**What was built**:
- **Enhanced batch runner**: Cross-run aggregation with 95% CI bands
- **Parameter sweep tooling**: `run_parameter_sweep()` with dotted param paths, cartesian product
- **Enriched metrics**: inflation rate, GDP growth, velocity of money
- **Scenario comparison**: `compare_scenarios()` long-form DataFrame
- **Streamlit dashboard**: Interactive config, Plotly charts, CI bands
- **Test suite**: Expanded to 70 tests passing

**Key outcomes**:
- Statistical analysis across multiple seeded runs
- Systematic parameter exploration
- Interactive visualization dashboard
- All tests passing (70)

---

## Phase 3 — RL Integration & UI Overhaul ✅

**Status**: Complete

**What was built**:
- **FirmEnv**: Gymnasium-compatible environment (`econosim/rl/firm_env.py`)
  - 14-dim continuous observation (firm state + macro indicators)
  - 3-dim continuous action (price ×0.8-1.2, wage ×0.9-1.1, vacancy fraction 0-1)
  - Monkey-patches firm decision methods per step
  - 3 reward functions: `profit`, `gdp`, `balanced`
- **Gymnasium registration**: `EconoSim-Firm-v0`
- **SB3 training script**: `scripts/train_firm_rl.py` (PPO, eval callback, baseline comparison)
- **Dashboard redesign**:
  - Dark sidebar with gradient background
  - Styled KPI cards with period-over-period deltas
  - 5 tabs: Macro, Labor & Production, **Government**, Money & Credit, Data
  - Government fiscal KPIs and sovereign money creation charts
  - Stacked area chart for deposit distribution
  - Column selector + formatted data tables
- **Government metrics**: Added tax revenue, transfers, spending, money creation to simulation output

**Key outcomes**:
- RL agent can control firm pricing, wages, and hiring
- Modern, polished UI with comprehensive macro/fiscal views
- Test suite expanded to 86 tests passing
- Ready for RL training runs

---

## Phase 3b — Multi-Agent RL Environments ✅

**Status**: Complete

**What was built**:
- **HouseholdEnv**: 12-dim obs, 2-dim action (consumption fraction, reservation wage mult)
  - Reward modes: `utility` (log consumption + savings), `consumption`, `balanced`
- **GovernmentEnv**: 12-dim obs, 3-dim action (tax rate, transfer mult, spending mult)
  - Reward modes: `welfare` (GDP - unemployment - inequality), `gdp`, `employment`, `balanced`
- **BankEnv**: 12-dim obs, 2-dim action (base interest rate, capital adequacy ratio)
  - Reward modes: `profit` (interest - defaults), `stability`, `growth`
- **EconoSimMultiAgentEnv**: PettingZoo parallel env with all 4 agents acting simultaneously
  - Each agent has its own observation/action space
  - Simultaneous actions applied before each simulation step
- All envs registered with Gymnasium (`EconoSim-{Firm,Household,Government,Bank}-v0`)

**Key outcomes**:
- Full set of single-agent RL environments for every agent type
- Multi-agent parallel environment for coordinated training
- Test suite expanded to 130 tests passing
- Ready for single-agent and multi-agent training

---

## Phase 3c — RL Training & Evaluation ✅

**Status**: Complete

**What was built**:
- **Unified training script** (`scripts/train_agent.py`): Train any of the 4 agent types
  - Supports PPO and A2C algorithms
  - 3 hyperparameter presets: default, conservative, aggressive
  - Optional observation & reward normalization via VecNormalize
  - Automatic baseline comparison with neutral actions
  - TensorBoard logging, model saving, results JSON export
- **Multi-agent training** (`scripts/train_multiagent.py`): Independent learners approach
  - `SingleAgentWrapper`: Wraps one agent's view of the PettingZoo multi-agent env for SB3
  - Sequential mode: Train agents one-at-a-time, using previously trained agents as fixed policies
  - Simultaneous mode: Round-robin training with periodic policy updates
  - Per-agent model saving and evaluation
- **Policy comparison** (`scripts/compare_policies.py`): RL vs rule-based across scenarios
  - 6 pre-defined scenarios: baseline, supply shock, demand shock, credit crunch, tax hike, stimulus
  - Multi-seed evaluation with statistical aggregation
  - Per-timestep trajectory tracking for all key metrics
  - Formatted comparison tables with win/loss indicators
  - Supports `--all-agents` mode for batch comparison
- **Observation normalization** (`econosim/rl/wrappers.py`):
  - `NormalizeObservation`: Running mean/std normalization with clipping
  - `NormalizeReward`: Discounted return normalization
  - `ScaleReward`: Fixed scaling factor
  - `ClipAction`: Safety clipping wrapper
  - `RecordEpisodeMetrics`: Per-episode macro metrics recording
  - All wrappers composable (stackable)
- **Hyperparameter tuning** (`scripts/tune_hyperparams.py`):
  - Grid search over learning rate, n_steps, batch_size, epochs, entropy coef, gamma
  - Automatic baseline comparison per trial
  - Best model saving, top-5 results summary

**Key outcomes**:
- Full training pipeline for all 4 agent types (firm, household, government, bank)
- Multi-agent training with sequential and simultaneous modes
- Policy comparison across 6 shock scenarios with statistical aggregation
- Test suite: 24 new tests for wrappers and training infrastructure
- Total tests: 367 passing

---

## Phase 4 — Advanced Economic Extensions ✅

**Status**: Complete

**What was built** (`econosim/extensions/`):
- **Multi-sector production** (`multi_sector.py`):
  - `Good`: Typed commodities (consumption, intermediate, capital) with depreciation
  - `SectorInventory`: Multi-good inventory with weighted-average costing
  - `InputOutputMatrix`: Leontief I-O matrix with technical coefficients
    - `inputs_required()`: Calculate intermediate inputs per output unit
    - `labor_required()`: Labor coefficients per sector
    - `leontief_inverse()`: (I-A)^(-1) for total requirements analysis
    - `total_requirements()`: Multiplier-based demand propagation
    - `is_productive()`: Hawkins-Simon condition validation
  - `Sector`: Sector aggregation with price computation and statistics
  - `create_default_sectors()`: 3-sector economy (agriculture, manufacturing, services)
- **Labor skill differentiation** (`skilled_labor.py`):
  - `SkillLevel`: 4-tier enum (Unskilled → Highly Skilled) with productivity/wage multipliers
  - `SkilledHousehold`: Skill-based productivity, experience accumulation, training/upgrade
  - `SkilledFirm`: Skill-differentiated hiring, per-skill wage tracking
  - `SkilledLaborMarket`: Skill-based matching with priority ordering
  - `SkillDistribution`: Labor force skill composition tracking
  - `SkillRequirement`: Job posting skill requirements with custom wages
  - Wage dispersion coefficient of variation metric
- **Bond markets** (`bonds.py`):
  - `Bond`: Fixed-income security with coupon rate, maturity, fair value (DCF)
  - `BondMarket`: Primary issuance and secondary trading
    - Coupon processing and maturity handling
    - Yield curve construction from outstanding bonds
  - `GovernmentDebtManager`: Bond-financed government spending
    - Debt issuance with configurable maturity and coupon rates
    - Debt service (coupons + principal redemption)
    - Debt-to-GDP ratio tracking and limits
- **Adaptive expectations** (`expectations.py`):
  - `AdaptiveExpectations`: Exponential smoothing with configurable alpha
  - `RollingExpectations`: Rolling window average with optional trend extrapolation
  - `WeightedExpectations`: Combines multiple signals with configurable weights
  - `AgentExpectations`: Container for price, wage, demand, inflation forecasts
  - All models: multi-step forecasting, error tracking, state serialization
- **Network effects** (`networks.py`):
  - `EconomicNetwork`: Base directed weighted graph with network metrics
    - Degree/weighted centrality, density, HHI concentration
    - Local/average clustering coefficients
    - Connected components detection
    - Edge decay for temporal dynamics
  - `TradeNetwork`: Goods trade relationships, seller concentration, buyer diversity
  - `CreditNetwork`: Lending relationships, exposure tracking
    - Contagion risk analysis (first-order default impact)
    - Systemic risk scoring (density × concentration × exposure)

**Key outcomes**:
- 5 modular extension modules ready for integration with core simulation
- All extensions follow the existing codebase patterns (observations, state management)
- 135 new tests for all Phase 4 extensions
- Total tests: 367 passing, 0 warnings

---

## Phase 5 — Platform Features ✅

**Status**: Complete

**What was built**:
- **Modern Next.js dashboard**: React + TypeScript + Tailwind CSS frontend (`web/`)
  - Dark theme with glass morphism, gradient accents, animated transitions
  - 6 KPI cards with sparkline mini-charts and trend deltas
  - 5 tabbed views: Macro, Labor & Production, Government, Money & Credit, Data
  - Interactive recharts-based charts with gradient fills, legends, and CI bands
  - Parameter control sidebar with scenario presets (Baseline, High Growth, Recession, Tight Money)
  - Collapsible sidebar with reset-to-defaults functionality
  - Loading skeleton states with shimmer animations
  - Animated welcome screen with feature cards
  - Column-selectable data table with CSV/JSON export
  - Responsive layout with staggered animations
- **FastAPI backend** (`api/main.py`): RESTful API serving simulation data
  - `POST /api/simulate` — run simulation with custom config
  - `GET /api/defaults` — default config
  - `GET /api/health` — health check
  - CORS-enabled for cross-origin frontend
- **Vercel deployment config**: `vercel.json` + `api/requirements.txt` for free deployment
- **Bug fixes**:
  - Fixed `inventory_asset` balance sheet account (was never updated, now syncs after production and sales)
  - Fixed delinquency threshold (was marking loans delinquent after 1 missed payment instead of using configurable threshold)
  - Firm initial equity now includes inventory value
  - Fixed Vercel deployment: requirements.txt now in api/ directory where @vercel/python expects it
- **Test coverage expanded**: Agent and market tests added (78 new tests)
- **Phase 4 extension integration** into core simulation engine:
  - Expectations: `AgentExpectations` per firm, updated each period with realized price/wage/demand/inflation
  - Networks: `TradeNetwork` records firm sales, `CreditNetwork` records loan issuance, edge decay per period
  - Bonds: `GovernmentDebtManager` issues bonds when government faces fiscal shortfall before sovereign money creation
  - All extensions gated by feature flags in `ExtensionsConfig` (off by default)
  - 14 new extension metrics added to `compute_period_metrics()`
  - 18 new integration tests verifying extensions work with core simulation

- **Data persistence**: Added SQLite + SQLAlchemy storage for saving simulation runs, with FastAPI endpoints `/api/runs`.
- **Scenario comparison UI**: Built an interactive multi-line comparison view (`CompareTab.tsx`) overlaying multiple stored runs via Recharts.
- **Collaboration**: Re-evaluated and removed complexity for MVP scope. Database IDs can be shared securely inside a private environment.
- **Vercel deployment**: Configured Next.js frontend and Python backend serverless rewrites inside `vercel.json` with instructions on connecting an external Postgres database like Supabase.

---

## Current Status Summary

- **Phases 0-3c**: ✅ Complete (full RL training pipeline)
- **Phase 4**: ✅ Complete (advanced economic extensions)
- **Phase 5 (Platform)**: ✅ Complete (Next.js UI + FastAPI backend + DB persistence + Comparisons)
- **Tests**: 389 passing, 0 warnings
- **Dashboard (legacy)**: `streamlit run dashboard.py` at `http://localhost:8501`
- **Dashboard (modern)**: `cd web && npm run dev` at `http://localhost:3000`
- **API**: `cd api && uvicorn main:app` at `http://localhost:8000`
- **RL**: Ready for training (`scripts/train_firm_rl.py`)
- **Next immediate steps**: Start running multi-agent RL evaluations locally or deploy to a live Vercel endpoint.

---

## Quick Start Commands

```bash
# Install Python package
pip install -e ".[dev,rl]"

# Run modern Next.js dashboard
cd web && npm install && npm run dev  # http://localhost:3000

# Run FastAPI backend (needed for dashboard)
pip install fastapi uvicorn
cd api && uvicorn main:app --reload  # http://localhost:8000

# Run legacy Streamlit dashboard
streamlit run dashboard.py  # http://localhost:8501

# Run simulation CLI
python -m econosim --scenario scenarios/baseline.yaml --periods 120

# Run RL training (single agent — any of: firm, household, government, bank)
python scripts/train_agent.py --agent firm --timesteps 50000 --reward profit
python scripts/train_agent.py --agent government --timesteps 50000 --reward welfare --normalize

# Run multi-agent training
python scripts/train_multiagent.py --timesteps 50000 --mode sequential

# Compare RL vs baseline across scenarios
python scripts/compare_policies.py --agent firm --model outputs/rl/firm/final_model

# Hyperparameter tuning
python scripts/tune_hyperparams.py --agent firm --timesteps 20000

# Run tests
pytest tests/
```

---

## Architecture Overview

```
Core Accounting → Agents → Markets → Engine → Metrics → Config → Experiments → RL
                                                          ↓
                                              FastAPI Backend (api/)
                                                          ↓
                                              Next.js Frontend (web/)
```

**Key files**:
- `web/` — Next.js + React + Tailwind CSS frontend
- `api/main.py` — FastAPI backend
- `dashboard.py` — Streamlit UI (legacy)
- `scripts/train_firm_rl.py` — RL training
- `econosim/rl/firm_env.py` — Gymnasium environment
- `PROJECT_LOG.md` — Detailed implementation log
- `vercel.json` — Vercel deployment config
