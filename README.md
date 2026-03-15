# EconoSim

Multi-agent AI economic simulation platform with stock-flow-consistent accounting.

Households, firms, banks, and government interact in a closed economy. Macroeconomic behavior emerges from micro-level incentives, accounting rules, market interactions, policy, and shocks.

## Quick Start

```bash
# Create virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Run a simulation
python -m econosim.experiments.runner
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
| **RL** | `env` | Gymnasium-compatible interface scaffold |

## MVP Scope

- 100 households, 5 firms, 1 bank, 1 government
- 1 consumption good, 1 labor type
- Monthly time steps
- Rule-based agent decisions
- Supply, demand, credit, and fiscal shocks
- Seeded reproducibility
- Full accounting invariant enforcement

## Key Design Principles

1. **Accounting integrity**: Every monetary flow is a double-entry transaction. A - L = E holds for every entity after every step.
2. **Endogenous money**: Bank lending creates deposits; repayment destroys them.
3. **No framework lock-in**: Domain logic is pure Python, decoupled from any simulation framework.
4. **Reproducibility**: Seeded RNG produces identical runs.
5. **RL-ready**: Designed for future Gymnasium/PettingZoo integration.

## Running Scenarios

```python
from econosim.config import SimulationConfig
from econosim.engine import run_simulation

# Default baseline
state = run_simulation(SimulationConfig())

# From YAML
from econosim.experiments import load_config_from_yaml, run_experiment
config = load_config_from_yaml("scenarios/supply_shock.yaml")
result = run_experiment(config, output_dir="outputs")
```

## Tests

```bash
pytest tests/ -v          # all tests
pytest tests/test_core/   # accounting, contracts, goods
pytest tests/test_integration/  # full simulation
```

## Project Status

See [PROJECT_LOG.md](PROJECT_LOG.md) for detailed progress, architecture decisions, and changelog.

## License

MIT
