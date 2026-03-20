# EconoSim

Multi-agent AI economic simulation platform with stock-flow-consistent accounting.

Households, firms, banks, and government interact in a closed economy. Macroeconomic behavior emerges from micro-level incentives, accounting rules, market interactions, policy, and shocks.

## Quick Start

```bash
# Install Python package
pip install -e ".[dev,rl]"

# Run modern dashboard (Next.js)
cd web && npm install && npm run dev    # http://localhost:3000

# Run API backend (required for dashboard)
pip install fastapi uvicorn
python -m uvicorn api.main:app --reload  # http://localhost:8000

# Run legacy Streamlit dashboard
pip install -e ".[viz]"
streamlit run dashboard.py               # http://localhost:8501

# Run tests (575+ tests)
pytest tests/

# Run a simulation from CLI
python -m econosim --scenario scenarios/baseline.yaml --periods 120

# Run RL training (single agent — any of: firm, household, government, bank)
python scripts/train_agent.py --agent firm --timesteps 50000 --reward profit
python scripts/train_agent.py --agent government --timesteps 50000 --reward welfare --normalize

# Run multi-agent training
python scripts/train_multiagent.py --timesteps 50000 --mode sequential

# Compare RL vs baseline across scenarios
python scripts/compare_policies.py --agent firm --model outputs/rl/firm/final_model
```

## Architecture

| Layer | Modules | Purpose |
|-------|---------|---------|
| **Core** | `accounting`, `contracts`, `goods` | Double-entry ledger, loan tracking, inventory |
| **Agents** | `household`, `firm`, `bank`, `government` | Rule-based economic agents with balance sheets |
| **Markets** | `labor`, `goods`, `credit` | Market clearing mechanisms |
| **Engine** | `simulation` | Simulation loop orchestrator |
| **Policies** | `interfaces`, `rule_based` | Swappable policy interfaces (Firm, Household, Bank, Government) |
| **Data** | `fred`, `bea`, `imf`, `pipelines`, `storage` | Real-world data ingestion (FRED, BEA, IMF) with versioned Parquet storage |
| **Measurement** | `national_accounts` | Simulation state → observable macro series (GDP, inflation, unemployment) |
| **Calibration** | `parameters`, `moments`, `engine` | SMM and Bayesian calibration to empirical moments |
| **Forecasting** | `engine`, `backtesting` | Probabilistic ensemble forecasting with fan charts and backtesting |
| **Learning** | `transformers` | Macro forecasting transformer (NumPy prototype) |
| **Metrics** | `collector` | GDP, unemployment, inflation, Gini, credit metrics |
| **Config** | `schema` | Pydantic configuration with YAML scenario support |
| **Experiments** | `runner` | Single/batch experiment execution |
| **RL** | `firm_env`, `household_env`, `government_env`, `bank_env`, `macro_env` | Gymnasium + PettingZoo + standalone RL environments |
| **Extensions**| `multi_sector`, `skilled_labor`, `bonds`, `expectations`, `networks` | Advanced macroeconomic functionality |
| **API** | `api/main.py` | FastAPI backend (simulate, calibrate, forecast, backtest, data) |
| **Frontend** | `web/` | Next.js + React + TypeScript + Tailwind CSS dashboard |

## Dashboard

The modern dashboard provides:

- **6 KPI cards** with sparkline mini-charts and trend deltas
- **7 tabbed views**: Macro, Labor & Production, Government, Money & Credit, Extensions, Compare, Data
- **Scenario presets**: Baseline, High Growth, Recession, Tight Money
- **Interactive controls** for all agent and market parameters with collapsible sidebar
- **Recharts-based charts** with gradient fills, legends, and CI bands for batch runs
- **Data export** to CSV and JSON
- **Glass morphism dark theme** with staggered animations and loading skeletons

## Key Design Principles

1. **Accounting integrity**: Every monetary flow is a double-entry transaction. A - L = E holds for every entity after every step.
2. **Endogenous money**: Bank lending creates deposits; repayment destroys them.
3. **No framework lock-in**: Domain logic is pure Python, decoupled from any simulation framework.
4. **Reproducibility**: Seeded RNG produces identical runs.
5. **RL-ready**: Full Gymnasium/PettingZoo environments for all agent types with multi-agent support.

## RL Environments & Training Pipeline

The platform provides a complete reinforcement learning suite:
- **Unified Training**: Single script `train_agent.py` to train any of the 4 agent types using PPO/A2C
- **Multi-Agent Training**: `train_multiagent.py` supporting sequential or simultaneous independent learners
- **Policy Evaluation**: `compare_policies.py` to benchmark RL vs rule-based agents across 6 shock scenarios
- **Wrappers**: Observation/reward normalization, action clipping, and metrics recording

| Environment | Observation | Action | Reward Modes |
|-------------|------------|--------|--------------|
| `EconoSim-Firm-v0` | 14-dim (firm state + macro) | 3-dim (price, wage, vacancies) | profit, gdp, balanced |
| `EconoSim-Household-v0` | 12-dim | 2-dim (consumption, reservation wage) | utility, consumption, balanced |
| `EconoSim-Government-v0` | 12-dim | 3-dim (tax, transfers, spending) | welfare, gdp, employment, balanced |
| `EconoSim-Bank-v0` | 12-dim | 2-dim (interest rate, CAR) | profit, stability, growth |
| `EconoSim-MultiAgent-v0`| PettingZoo Parallel Env | Combined Action Specs | Respective Reward Modes |

## Advanced Macroeconomic Extensions

Phase 4 extensions wire advanced economic realities into the core simulation loop (enabled via feature flags):
- **Multi-Sector Production**: Typed goods, Leontief I-O matrices, inter-sector coefficients
- **Labor Skill Differentiation**: 4-tier skill levels, wage premiums, experience bounds, matching priorities
- **Bond Markets**: Sovereign debt issuance, yield curves, primary/secondary trading
- **Adaptive Expectations**: Exponential smoothing, rolling windows, forecasted decision making
- **Network Effects**: Trade and credit topology, HHI, subsystem connections, contagion risk

## Deployment

Configured for **Vercel** free-tier deployment:

```bash
# Deploy to Vercel
vercel
```

## Macro World Model

EconoSim includes a forecasting-grade macro world model layer (Phases M0-M9):
- **Data Ingestion**: FRED, BEA, IMF clients with versioned Parquet storage
- **Measurement Model**: Maps simulation state to observable macro series (GDP, inflation, unemployment)
- **Policy Interfaces**: Swappable Firm/Household/Bank/Government policies with 13 action fields
- **Calibration**: SMM and Bayesian calibration to empirical moments (8 parameters, 10 target moments)
- **Probabilistic Forecasting**: Ensemble fan charts with event probabilities (recession, high inflation)
- **Backtesting**: Rolling-origin evaluation with CRPS, PIT calibration, skill scores vs 3 benchmarks
- **Transformer**: NumPy prototype for residual correction and simulation emulation

## Project Status

- **Phases 0-5**: Complete (core sim, RL, extensions, Next.js UI, 494 tests)
- **Phases M0-M7**: Complete (data, measurement, policies, calibration, forecasting, backtesting, transformer)
- **Phase M8 (Performance)**: Not started
- **Phase M9 (Platform)**: In progress — API endpoints, Streamlit fan charts, FRED pipelines
- **575+ tests passing**
- See [phases.md](phases.md) for macro roadmap and [project_log.md](project_log.md) for detailed changelog.

## License

MIT
