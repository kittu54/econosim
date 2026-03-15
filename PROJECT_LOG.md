# EconoSim — Project Log

> Canonical working memory for the EconoSim multi-agent economic simulation platform.
> Another engineer or coding agent should be able to resume work immediately from this document.

---

## 1. Project Overview

**EconoSim** is a multi-agent AI economic simulation platform where households, firms, banks, and government interact in a closed economy. Macroeconomic behavior emerges from micro-level incentives, accounting rules, market interactions, policy, and shocks.

**Goals**: Research-capable simulation platform, portfolio project, future paper/demo candidate, possible startup-grade sandbox for economic experimentation.

**Core principles** (in priority order):
1. Accounting integrity (stock-flow consistency)
2. Coherent simulation dynamics
3. Reproducibility
4. Modularity
5. Experimentability
6. Future RL compatibility

---

## 2. Current Architecture Summary

```
econosim/
├── core/           # Accounting primitives, contracts, goods tracking
│   ├── accounting.py   # Account, BalanceSheet, Ledger, Transaction
│   ├── contracts.py    # LoanContract, LoanBook, LoanStatus
│   └── goods.py        # Inventory (production, sales, COGS)
├── agents/         # Economic agent models
│   ├── base.py         # BaseAgent ABC with balance-sheet integration
│   ├── household.py    # Labor supply, consumption, savings
│   ├── firm.py         # Hiring, production, pricing, inventory
│   ├── bank.py         # Lending, reserves, capital adequacy, defaults
│   └── government.py   # Taxes, transfers, spending
├── markets/        # Market clearing mechanisms
│   ├── labor.py        # Random matching, wage payment
│   ├── goods.py        # Proportional rationing, consumption
│   └── credit.py       # Firm loan applications, bank approval
├── engine/         # Simulation orchestrator
│   └── simulation.py   # build_simulation, step, run_simulation, shocks
├── metrics/        # Metrics collection and analysis
│   └── collector.py    # DataFrame conversion, inflation, GDP growth, export
├── config/         # Pydantic configuration schemas
│   └── schema.py       # SimulationConfig, agent configs, ShockSpec
├── experiments/    # Experiment runner
│   └── runner.py       # run_experiment, run_batch, YAML loading
└── rl/             # RL interface scaffold (Phase 3)
    └── env.py          # EconEnvInterface ABC, observation/action specs
```

**Key architectural decisions**:
- All monetary flows go through the central `Ledger` via double-entry transactions
- `transfer_deposits()` uses two within-entity double-entry postings (sender: debit equity/credit deposits; receiver: debit deposits/credit equity) to preserve A-L=E on every transfer
- Loan issuance creates money (endogenous money), repayment destroys it
- Write-offs transfer losses to bank equity and relieve borrower debt
- Agent decision logic is rule-based (simple, interpretable, debuggable)
- Domain logic is fully decoupled from any framework (no Mesa dependency)

---

## 3. What Has Been Implemented

### Phase 0 — Complete ✅
- [x] Project structure and packaging (`pyproject.toml`, editable install)
- [x] Core accounting: `Account`, `AccountType`, `BalanceSheet`, `Ledger`, `Transaction`
- [x] Loan contracts: `LoanContract`, `LoanBook`, `LoanStatus`
- [x] Goods/inventory: `Inventory` with weighted-average costing
- [x] Agent models: `Household`, `Firm`, `Bank`, `Government`
- [x] Market modules: `LaborMarket`, `GoodsMarket`, `CreditMarket`
- [x] Simulation engine: `build_simulation()`, `step()`, `run_simulation()`
- [x] Shock system: supply, demand, credit, fiscal shocks via config
- [x] Metrics engine: GDP, unemployment, inflation, Gini, credit metrics
- [x] Config system: Pydantic schemas, YAML scenario files
- [x] Experiment runner: single runs, batch runs with multiple seeds
- [x] RL interface scaffold: `EconEnvInterface` ABC, observation/action specs
- [x] Test suite: **55 tests passing** (accounting, contracts, goods, simulation integration)
- [x] Reproducibility: seeded runs produce identical results
- [x] Accounting invariants: all balance sheets balanced after every step

### Phase 1 — Complete ✅
- [x] Rule-based MVP simulation engine runs end-to-end
- [x] Economy dynamics stabilization (sovereign money creation, revenue-based hiring)
- [x] CLI entry point (`python -m econosim`)
- [x] Mild deflation identified as calibration issue, acceptable for MVP

### Phase 2 — Complete ✅
- [x] Enhanced batch runner with cross-run aggregation and 95% CI bands
- [x] Parameter sweep tooling (`run_parameter_sweep` with dotted param paths)
- [x] Enriched metrics: inflation rate, GDP growth, velocity of money
- [x] Scenario comparison (`compare_scenarios` long-form DataFrame)
- [x] Streamlit dashboard (`dashboard.py`) with interactive config, Plotly charts, CI bands
- [x] Test suite expanded to **70 tests passing**

### Phase 3 — Complete ✅
- [x] Gymnasium-compatible `FirmEnv` environment (`econosim/rl/firm_env.py`)
- [x] RL agent controls single firm’s pricing, wages, and hiring via monkey-patched methods
- [x] 3 reward functions: `profit`, `gdp`, `balanced`
- [x] 14-dim continuous observation, 3-dim continuous action space
- [x] Gymnasium registration (`EconoSim-Firm-v0`)
- [x] SB3 training script (`scripts/train_firm_rl.py`) with PPO, eval callback, baseline comparison
- [x] Dashboard redesigned: dark sidebar, styled KPI cards with deltas, 5 tabs (Macro, Labor, Government, Money, Data), stacked area charts, column selector
- [x] Government fiscal metrics added to simulation output (tax revenue, transfers, spending, money creation)
- [x] Test suite expanded to **86 tests passing** (16 new RL env tests)

### Phase 3b — Complete ✅
- [x] `HouseholdEnv`: 12-dim obs, 2-dim action (consumption fraction, reservation wage mult), 3 reward modes
- [x] `GovernmentEnv`: 12-dim obs, 3-dim action (tax rate, transfer mult, spending mult), 4 reward modes
- [x] `BankEnv`: 12-dim obs, 2-dim action (base interest rate, capital adequacy ratio), 3 reward modes
- [x] `EconoSimMultiAgentEnv`: PettingZoo parallel env with all 4 agent types acting simultaneously
- [x] All envs registered with Gymnasium (`EconoSim-{Firm,Household,Government,Bank}-v0`)
- [x] Test suite expanded to **130 tests passing** (44 new RL env tests)

---

### Phase 5 — In Progress (Platform)
- [x] Modern Next.js dashboard (`web/`) with React, TypeScript, Tailwind CSS
- [x] FastAPI backend (`api/main.py`) for serving simulation data
- [x] Vercel deployment configuration
- [x] Fixed `inventory_asset` balance sheet bug
- [x] Fixed delinquency threshold bug
- [x] Added 78 new tests for agents and markets (208 total)
- [ ] Data persistence layer
- [ ] Scenario comparison UI

---

## 4. What Is Currently In Progress

- Phase 5 platform features: Modern UI deployed, backend API ready
- Ready for RL training runs (Phase 3c) and advanced extensions (Phase 4)

---

## 5. What Is Planned Next

### Near-term
- Run RL training and compare agent vs baseline
- Longer-run calibration to reduce deflation bias
- More scenario YAML files (demand shock, credit crunch, fiscal austerity)

### Phase 3c: RL training and evaluation
- Run single-agent training for each env
- Multi-agent training via PettingZoo/RLlib
- Policy comparison: RL vs rule-based agents across scenarios

### Phase 4: Advanced economic extensions
- Multiple goods / sectors
- Labor skill differentiation
- Bond markets / government debt
- Expectations and adaptive behavior
- Network effects (trade/credit graphs)

### Phase 5: Dashboard / API / platform
- Streamlit interactive dashboard
- FastAPI backend
- React frontend (if needed)

---

## 6. File/Folder Map

| Path | Purpose |
|------|---------|
| `econosim/core/accounting.py` | Account, BalanceSheet, Ledger, Transaction |
| `econosim/core/contracts.py` | LoanContract, LoanBook |
| `econosim/core/goods.py` | Inventory tracking |
| `econosim/agents/base.py` | BaseAgent ABC |
| `econosim/agents/household.py` | Household agent |
| `econosim/agents/firm.py` | Firm agent |
| `econosim/agents/bank.py` | Bank agent |
| `econosim/agents/government.py` | Government agent |
| `econosim/markets/labor.py` | Labor market clearing |
| `econosim/markets/goods.py` | Goods market clearing |
| `econosim/markets/credit.py` | Credit market clearing |
| `econosim/engine/simulation.py` | Simulation loop orchestrator |
| `econosim/metrics/collector.py` | Metrics utilities |
| `econosim/config/schema.py` | Pydantic config schemas |
| `econosim/experiments/runner.py` | Experiment execution |
| `econosim/rl/env.py` | RL environment interface |
| `scenarios/baseline.yaml` | Default baseline scenario |
| `scenarios/supply_shock.yaml` | Supply shock scenario |
| `dashboard.py` | Streamlit interactive dashboard |
| `econosim/rl/firm_env.py` | Gymnasium FirmEnv for RL |
| `econosim/rl/registration.py` | Gymnasium env registration |
| `scripts/train_firm_rl.py` | SB3 PPO training script |
| `tests/test_core/` | Core accounting/contract/goods tests |
| `tests/test_integration/` | Full simulation integration tests |
| `tests/test_experiments/` | Experiment runner / metrics tests |
| `tests/test_rl/` | RL environment tests |
| `web/` | Next.js + React + Tailwind CSS frontend |
| `web/src/app/page.tsx` | Main dashboard page component |
| `web/src/components/` | KPI cards, charts, sidebar, tab views |
| `web/src/lib/` | API client, types, formatting utilities |
| `api/main.py` | FastAPI backend serving simulation data |
| `vercel.json` | Vercel deployment configuration |
| `requirements.txt` | Python dependencies for deployment |

---

## 7. Major Design Decisions and Rationale

1. **Double-entry accounting via central Ledger**: Every monetary flow is recorded as a transaction. No silent creation/destruction of money except via explicit loan issuance (endogenous money creation) and write-offs.

2. **Within-entity double-entry for transfers**: `transfer_deposits()` posts two within-entity transactions rather than one cross-entity transaction. This ensures A-L=E is preserved on every entity after every transfer, regardless of whether the entity is a bank (deposits=LIABILITY) or non-bank (deposits=ASSET).

3. **No Mesa dependency**: Domain logic is pure Python. The simulation loop is a simple function that calls sub-steps in order. This keeps the system testable, debuggable, and framework-independent.

4. **Pydantic for config**: Typed, validated configuration with sensible defaults. YAML scenario files for experiment definition.

5. **Rule-based agents first**: Simple, interpretable decision rules (consumption propensity, inventory-target pricing, capital adequacy lending) that can be debugged and compared against future RL policies.

6. **Bank equity = initial reserves**: At inception, the bank's only asset is reserves. Equity must equal net assets (reserves - 0 liabilities = reserves). The `initial_equity` config param was misleading and was fixed.

7. **Sovereign money creation (MMT/SFC approach)**: The government is a currency issuer. When fiscal spending would exceed deposits, it creates money via `ensure_solvency()` (debit deposits/credit equity). This prevents government deposit depletion from killing the economy. Tracked via `money_created` and `cumulative_money_created`.

8. **Buffer-stock consumption (SFC standard)**: Household consumption = α1 × disposable_income + α2 × wealth. The wealth term (α2=0.4) ensures spending continues when income drops, preventing deflationary death spirals.

9. **Revenue-based hiring**: Firms estimate demand using `max(prev_units_sold, prev_revenue / price)`. This ensures government service contracts (which provide revenue but not goods sales) still signal hiring demand.

10. **Government spending as fiscal stabilizer**: Government spending is injected as service contracts even when firms have zero inventory. This prevents a doom loop where zero inventory → zero spending → zero revenue → zero hiring.

---

## 8. Known Bugs / Limitations / Technical Debt

- **Firm `inventory_asset` account** exists on balance sheet but is never updated to reflect inventory value changes. This is cosmetic (doesn't break accounting since production costs flow through deposits/equity).
- **Mild deflation**: With current defaults, prices decline slowly over long runs. This is a calibration issue — the goods market is supply-constrained and household savings accumulate. Acceptable for MVP.
- **No household borrowing**: Credit market only serves firms currently.
- **Single good, single labor type**: MVP constraint, by design.
- **No firm entry/exit**: Fixed number of firms throughout simulation.
- **Bank reserves**: Not actively managed or constrained. Reserves account exists but isn't depleted by lending (endogenous money creation doesn't require reserves in current model).

---

## 9. Open Questions

- Should firm inventory be tracked on the balance sheet as an asset (with COGS adjustments), or is the current off-balance-sheet `Inventory` object sufficient for MVP?
- What is the right calibration for agent parameters to produce plausible 120-period dynamics?
- How to handle firm bankruptcy / exit in later phases?

---

## 10. Experiment Notes

### Baseline (20 periods, default params)
- Full employment (U=0%) sustained through period 19
- GDP ~7,000-8,000 (nominal)
- Mild deflation (~3% per period) — prices drift from 10.0 → 7.1
- Government money creation kicks in around period 34 when initial deposits deplete
- Accounting invariants hold throughout

---

## 11. Current Working State

- **All 208 tests pass** with 0 warnings
- Package installs via `pip install -e ".[dev]"`
- Python 3.11+ required
- Simulation builds, runs, and produces metrics
- Accounting invariants verified after every step (including inventory asset)
- Seeded reproducibility confirmed
- Supply and demand shocks produce expected directional responses
- Modern Next.js dashboard at `web/` (React + TypeScript + Tailwind CSS)
- FastAPI backend at `api/` for serving simulation data
- Vercel deployment configuration ready (`vercel.json`)
- Legacy Streamlit dashboard still functional (`dashboard.py`)

---

## 12. Changelog

### Session 1 — 2025-03-15
- Created project structure and packaging
- Implemented core accounting layer (Account, BalanceSheet, Ledger, Transaction)
- Implemented loan contracts (LoanContract, LoanBook)
- Implemented inventory tracking (Inventory)
- Implemented all 4 agent types (Household, Firm, Bank, Government)
- Implemented all 3 market modules (Labor, Goods, Credit)
- Implemented simulation engine with 10-step loop
- Implemented shock system (supply, demand, credit, fiscal)
- Implemented metrics engine (GDP, unemployment, inflation, Gini, credit)
- Implemented config system with Pydantic schemas
- Implemented experiment runner with batch support
- Created RL interface scaffold
- Created baseline and supply_shock scenario YAML files
- **Bug fix**: Bank initial equity was set to `equity + reserves` but only reserves existed as an asset. Fixed to `equity = reserves`.
- **Bug fix**: `transfer_deposits()` only moved deposit assets without adjusting equity, breaking A-L=E invariant. Rewrote as two within-entity double-entry postings that adjust both deposits and equity on each side.
- **Bug fix**: Bank interest payment handler double-counted equity adjustment. Removed redundant manual post after `transfer_deposits` was fixed.
- Fixed Pydantic v2 deprecation warning (class Config → model_config dict)
- Wrote 55 tests covering accounting, contracts, goods, simulation integration
- All tests passing, all accounting invariants verified

### Session 2 — 2025-03-15 (dynamics fixes)
- **Bug fix**: Economy collapsed to GDP=0 and 100% unemployment by period 21. Root causes:
  - Government spending was gated on firm inventory → no money injection when inventory=0
  - Price adjustment raised prices infinitely when sales=0 (dead market ≠ excess demand)
  - Firm hiring based only on `prev_units_sold`, missing government service contract revenue
  - Initial demand estimate (`prev_units_sold`) overwritten by `reset_period_state()` at t=0
  - Government ran out of finite deposits → economy lost its fiscal stabilizer
- **Fix**: Sovereign money creation via `Government.ensure_solvency()` (MMT/SFC currency issuer)
- **Fix**: Government spending now injects as service contracts even with zero inventory
- **Fix**: Price adjustment only raises prices when there were actual sales (`prev_units_sold > 0.1`)
- **Fix**: Revenue-based hiring: `demand_estimate = max(prev_units_sold, prev_revenue / price)`
- **Fix**: Minimum hiring floor (always try 1 worker if affordable)
- **Fix**: Initial demand estimate now sets `units_sold` and `revenue` (not `prev_units_sold`) so `reset_period_state()` propagates correctly
- **Fix**: Buffer-stock consumption rule: C = α1 × income + α2 × wealth (SFC standard)
- Recalibrated defaults: wage=60, productivity=8, wealth_propensity=0.4, firm_deposits=15000
- Added CLI entry point (`python -m econosim`)
- Created README.md
- All 55 tests still passing

### Session 3 — 2025-03-15 (Phase 2: experiments + dashboard)
- Enhanced `run_batch()` to return aggregated DataFrame with 95% CI bands via `aggregate_runs()`
- Added `run_parameter_sweep()` for systematic parameter exploration (dotted paths, cartesian product)
- Added `enrich_dataframe()` (inflation, GDP growth, velocity), `compare_scenarios()` (long-form comparison)
- Fixed `model_copy` bug: parameter sweep used `deep=True` + `setattr` instead of `update=` which replaced sub-models with dicts
- Built Streamlit dashboard (`dashboard.py`) with:
  - Full sidebar config for all agent/market parameters
  - KPI cards (GDP, unemployment, price, Gini, loans)
  - Tabbed layout: Macro, Labor & Firms, Money & Credit, Data
  - Plotly charts with CI bands from batch runs
  - Raw data table + CSV export
- Added 15 new tests for experiments/metrics (70 total, all passing)
- Added `scipy>=1.11` to dependencies
- Updated `pyproject.toml`, `PROJECT_LOG.md`

### Session 4 — 2025-03-15 (Phase 3: RL + UI overhaul)
- Built `FirmEnv` Gymnasium environment (`econosim/rl/firm_env.py`):
  - 14-dim obs: firm state + macro indicators
  - 3-dim action: price multiplier (0.8-1.2), wage multiplier (0.9-1.1), vacancy fraction (0-1)
  - Monkey-patches firm’s `decide_vacancies`, `adjust_price`, `adjust_wage` per step
  - 3 reward modes: profit, GDP, balanced
- Added module-level RL helper functions and Gymnasium registration
- Created SB3 PPO training script (`scripts/train_firm_rl.py`) with:
  - Eval callback, baseline comparison, TensorBoard logging
  - Model save/load, JSON results export
- Redesigned Streamlit dashboard:
  - Dark sidebar with gradient background
  - Styled KPI cards with period-over-period deltas
  - 5 tabs: Macro, Labor & Production, Government, Money & Credit, Data
  - Government tab with fiscal KPIs, budget balance, sovereign money creation charts
  - Stacked area chart for deposit distribution
  - Column selector + formatted data tables in Data tab
  - Consistent Tailwind-inspired color palette
- Added government fiscal metrics to simulation output (tax revenue, transfers, spending, money creation, avg_wage, vacancies)
- Fixed DataFrame fragmentation warning in `aggregate_runs`
- Added 16 new RL environment tests (86 total, all passing)
- Installed gymnasium, updated pyproject.toml

### Session 5 — 2025-03-15 (Phase 3b: multi-agent RL)
- Built `HouseholdEnv` (12-dim obs, 2-dim action: consumption fraction + reservation wage mult)
- Built `GovernmentEnv` (12-dim obs, 3-dim action: tax rate + transfer mult + spending mult)
- Built `BankEnv` (12-dim obs, 2-dim action: base interest rate + capital adequacy ratio)
- Built `EconoSimMultiAgentEnv` (PettingZoo parallel, all 4 agents act simultaneously)
- Each env has multiple reward functions (utility, welfare, profit, stability, etc.)
- Registered all envs with Gymnasium
- Fixed household ID format (`hh_0000` not `hh_000`)
- Added 44 new RL tests (130 total, all passing)
- Installed pettingzoo

### Session 6 — 2026-03-15 (Phase 5: Modern UI + Bug Fixes)

**Next.js Dashboard (`web/`)**:
- Created Next.js 16 + TypeScript + Tailwind CSS v4 frontend
- Dark theme with gradient accents, smooth animations (fadeIn, pulse-glow, shimmer)
- 6 KPI cards with period-over-period trend deltas and directional arrows
- 5 tabbed dashboard views:
  - **Macro**: GDP, price level, unemployment, inflation, Gini, GDP growth
  - **Labor & Production**: employment, vacancies, wages, production, inventory, consumption
  - **Government**: fiscal KPI cards, fiscal flows, budget balance, deposits, sovereign money creation
  - **Money & Credit**: stacked deposit distribution, sector deposits, loans, bank equity, CAR, velocity
  - **Data**: column-selectable data table, summary statistics, CSV/JSON export
- Interactive sidebar with collapsible parameter sections (Simulation, Households, Firms, Government, Banking)
- Recharts-based charts with CI band support for batch runs
- Responsive layout optimized for desktop and tablet

**FastAPI Backend (`api/`)**:
- RESTful API serving simulation results
- `POST /api/simulate` — configurable simulation runs
- `GET /api/defaults` — default config values
- `GET /api/health` — health check endpoint
- CORS middleware for cross-origin frontend access

**Vercel Deployment**:
- `vercel.json` with Python + Next.js build configuration
- `requirements.txt` for Python serverless functions
- `.env.local` for development API URL

**Bug Fixes**:
- **CRITICAL**: Fixed `inventory_asset` balance sheet account — was created with 0.0 balance and never updated. Now:
  - Initial value set to `initial_inventory * unit_cost` at firm creation
  - `_sync_inventory_asset()` method adjusts balance after production and sales
  - Called after `produce()`, goods market clearing, and government purchases
  - Firm equity now correctly includes inventory value
- **Fixed delinquency threshold**: Was marking loans delinquent after just 1 missed payment. Now uses `max(1, default_threshold_periods // 2)` before flagging delinquency
- Balance sheet invariant now holds including inventory asset tracking

**Test Coverage Expansion** (130 → 208 tests):
- Added `tests/test_agents/test_household.py` — 14 tests (init, decisions, period state, observations)
- Added `tests/test_agents/test_firm.py` — 20 tests (init, decisions, production, inventory sync, borrowing)
- Added `tests/test_agents/test_government.py` — 12 tests (init, tax, transfers, sovereign money, budget)
- Added `tests/test_agents/test_bank.py` — 10 tests (init, lending, defaults, observations)
- Added `tests/test_markets/test_labor.py` — 9 tests (matching, wages, balance sheets, edge cases)
- Added `tests/test_markets/test_goods.py` — 7 tests (sales, inventory, deposits, balance sheets)
- Added `tests/test_markets/test_credit.py` — 4 tests (applications, loans, balance sheets)
- All 208 tests passing

**New Files Added**:
- `web/` — Complete Next.js frontend (pages, components, types, API client)
- `api/main.py` — FastAPI backend
- `vercel.json` — Vercel deployment config
- `requirements.txt` — Python dependencies for deployment
- 7 new test files in `tests/test_agents/` and `tests/test_markets/`

---

### Session 7 — Radical UI Overhaul & Deployment Fix

**Vercel Deployment Fix**:
- Created `api/requirements.txt` — Vercel's `@vercel/python` builder looks for requirements relative to the handler file, not the project root
- Updated `vercel.json` with `maxLambdaSize: "50mb"` to accommodate numpy/pandas
- The `sys.path.insert` approach in `api/main.py` works because Vercel deploys all project files

**Radical UI Redesign** (all files in `web/src/`):

*Global Styles (`app/globals.css`)*:
- New color palette: deeper background (`#06080d`), richer surface colors, three accent colors (blue, violet, cyan)
- Subtle grid background pattern with radial gradient
- Glass morphism utilities (`.glass`, `.glass-strong`) with backdrop blur
- Blue glow effects (`.glow-blue`, `.glow-blue-sm`)
- New animations: `fadeInScale`, `slideInRight`, `float`, `gradient-shift`, `count-up`
- Staggered animation delays for child elements (`.stagger-children`)
- Custom range input styling with glowing thumb
- Chart container with gradient border effect (`.chart-container`)

*KPI Cards (`components/KpiCard.tsx`)*:
- Added inline SVG sparkline mini-charts showing metric trend over time
- Glass morphism background with backdrop blur
- Smoother hover transitions with group hover effects
- Count-up animation on values
- Accepts `sparklineData` and `sparklineColor` props

*Sidebar (`components/controls/Sidebar.tsx`)*:
- **Scenario presets**: Baseline, High Growth, Recession, Tight Money — one-click configs
- **Reset button** to restore defaults
- **Collapsible mode**: sidebar collapses to thin icon strip
- Animated section expand/collapse with max-height transitions
- Slider values displayed in accent color with tabular-nums
- Glass morphism background

*Charts (`components/charts/MetricChart.tsx`)*:
- Gradient fills for area charts (defined via SVG `<linearGradient>`)
- Legends enabled for multi-metric charts
- Custom tooltip with glass morphism styling and drop shadow
- Active dot indicators on hover
- Vertical grid lines removed for cleaner look
- Chart wrapper component with optional subtitle
- Subtle gradient border via CSS pseudo-element

*Main Page (`app/page.tsx`)*:
- **Welcome screen redesign**: Animated floating hero icon, 8 feature cards in 4-column grid, staggered fade-in animations, architecture info bar (208 tests, Next.js+FastAPI, Gymnasium RL, SFC Accounting)
- **Loading skeleton**: Shows shimmer skeleton when simulation is running (first run)
- **Run indicator**: Shows run count, period count, and batch/single mode
- **Sparkline data**: Extracts sampled values for each KPI card's sparkline
- Tab content wrapped in animation container for smooth transitions

*Tab Components* (all tabs updated):
- Added `subtitle` descriptions to all charts explaining what each metric shows
- Consistent spacing and visual hierarchy

*New Components*:
- `components/LoadingSkeleton.tsx` — KPI, chart, and full dashboard skeleton states

**Code Quality**:
- Full codebase audit: 0 bugs found, 0 TODOs, 0 FIXMEs, 0 incomplete implementations
- All 208 tests passing
- Next.js build compiles successfully
- All accounting invariants verified
