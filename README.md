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

# Run tests (208 tests)
pytest tests/

# Run a simulation from CLI
python -m econosim --scenario scenarios/baseline.yaml --periods 120
```

## Architecture

| Layer | Modules | Purpose |
|-------|---------|---------|
| **Core** | `accounting`, `contracts`, `goods` | Double-entry ledger, loan tracking, inventory |
| **Agents** | `household`, `firm`, `bank`, `government` | Rule-based economic agents with balance sheets |
| **Markets** | `labor`, `goods`, `credit` | Market clearing mechanisms |
| **Engine** | `simulation` | Simulation loop orchestrator |
| **Metrics** | `collector` | GDP, unemployment, inflation, Gini, credit metrics |
| **Config** | `schema` | Pydantic configuration with YAML scenario support |
| **Experiments** | `runner` | Single/batch experiment execution |
| **RL** | `firm_env`, `household_env`, `government_env`, `bank_env`, `multi_agent_env` | Gymnasium + PettingZoo environments |
| **API** | `api/main.py` | FastAPI backend for serving simulation data |
| **Frontend** | `web/` | Next.js + React + TypeScript + Tailwind CSS dashboard |

## Dashboard

The modern dashboard provides:

- **6 KPI cards** with sparkline mini-charts and trend deltas
- **5 tabbed views**: Macro, Labor & Production, Government, Money & Credit, Data
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
5. **RL-ready**: Full Gymnasium/PettingZoo environments for all agent types.

## RL Environments

| Environment | Observation | Action | Reward Modes |
|-------------|------------|--------|--------------|
| `EconoSim-Firm-v0` | 14-dim (firm state + macro) | 3-dim (price, wage, vacancies) | profit, gdp, balanced |
| `EconoSim-Household-v0` | 12-dim | 2-dim (consumption, reservation wage) | utility, consumption, balanced |
| `EconoSim-Government-v0` | 12-dim | 3-dim (tax, transfers, spending) | welfare, gdp, employment, balanced |
| `EconoSim-Bank-v0` | 12-dim | 2-dim (interest rate, CAR) | profit, stability, growth |

## Deployment

Configured for **Vercel** free-tier deployment:

```bash
# Deploy to Vercel
vercel
```

## Project Status

- **Phases 0-3b**: Complete (core sim, RL environments, 208 tests)
- **Phase 5**: Modern Next.js UI + FastAPI backend deployed
- See [PHASES.md](PHASES.md) for roadmap and [PROJECT_LOG.md](PROJECT_LOG.md) for detailed changelog.

## License

MIT
