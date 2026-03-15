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

## Phase 3c — RL Training & Evaluation 🚧

**Status**: Not started

**Planned work**:
- **Single-agent training**: Run PPO for each env, compare vs rule-based baseline
- **Multi-agent training**: PettingZoo + RLlib or independent learners
- **Policy comparison**: RL vs rule-based agents across scenarios
- **Hyperparameter tuning**: Reward shaping, observation normalization

**Key challenges**:
- Coordination between multiple RL agents
- Training stability with simultaneous learners
- Meaningful reward design for macro outcomes

---

## Phase 4 — Advanced Economic Extensions 🚧

**Status**: Not started

**Planned work**:
- **Multiple goods/sectors**: Multi-sector production, input-output matrices
- **Labor skill differentiation**: Skill levels, wage dispersion
- **Bond markets**: Government debt issuance, yield curves
- **Expectations**: Adaptive expectations, learning dynamics
- **Network effects**: Trade/credit graphs, contagion

**Key challenges**:
- Model complexity vs interpretability
- Computational performance with larger systems
- Calibration to real-world data

---

## Phase 5 — Platform Features ✅ (Partial)

**Status**: In progress

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

**Remaining work**:
- **Data persistence**: Database storage for long runs
- **Collaboration**: Shared scenarios, result sharing
- **Scenario comparison UI**: Multi-run overlay charts

**Key challenges**:
- Scalability for concurrent users
- Data model design for persistence

---

## Current Status Summary

- **Phases 0-3b**: ✅ Complete
- **Phase 3c**: 🚧 Environments ready, training not yet run
- **Phase 5 (Platform)**: ✅ Partial (Next.js UI + FastAPI backend)
- **Tests**: 208 passing, 0 warnings
- **Dashboard (legacy)**: `streamlit run dashboard.py` at `http://localhost:8501`
- **Dashboard (modern)**: `cd web && npm run dev` at `http://localhost:3000`
- **API**: `cd api && uvicorn main:app` at `http://localhost:8000`
- **RL**: Ready for training (`scripts/train_firm_rl.py`)
- **Next immediate steps**: Run RL training, deploy to Vercel, add scenario comparison

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

# Run RL training
python scripts/train_firm_rl.py --timesteps 50000 --reward profit

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
