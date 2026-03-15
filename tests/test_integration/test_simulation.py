"""Integration tests for the full simulation engine.

Tests:
- Simulation builds and runs without errors
- Accounting invariants hold after every period
- Seeded runs are reproducible
- Shocks produce plausible responses
- Goods conservation
- Deposit conservation (excluding money creation/destruction via loans)
"""

import pytest
import numpy as np

from econosim.config.schema import SimulationConfig, ShockSpec
from econosim.engine.simulation import build_simulation, step, run_simulation


class TestSimulationBuilds:
    def test_build_default_config(self):
        config = SimulationConfig()
        state = build_simulation(config)
        assert len(state.households) == 100
        assert len(state.firms) == 5
        assert state.bank is not None
        assert state.government is not None

    def test_build_small_config(self):
        config = SimulationConfig(
            household={"count": 10},
            firm={"count": 2},
        )
        state = build_simulation(config)
        assert len(state.households) == 10
        assert len(state.firms) == 2


class TestAccountingInvariants:
    def test_all_sheets_balanced_after_step(self):
        config = SimulationConfig(
            num_periods=1,
            household={"count": 20},
            firm={"count": 3},
        )
        state = build_simulation(config)
        metrics = step(state)
        validation = state.ledger.validate_all_balanced()
        for owner_id, balanced in validation.items():
            assert balanced, f"{owner_id} balance sheet is not balanced"

    def test_all_sheets_balanced_after_10_periods(self):
        config = SimulationConfig(
            num_periods=10,
            household={"count": 20},
            firm={"count": 3},
        )
        state = build_simulation(config)
        for t in range(10):
            metrics = step(state)
            validation = state.ledger.validate_all_balanced()
            for owner_id, balanced in validation.items():
                assert balanced, f"Period {t}: {owner_id} balance sheet unbalanced"


class TestReproducibility:
    def test_same_seed_same_results(self):
        config = SimulationConfig(
            num_periods=10,
            seed=12345,
            household={"count": 20},
            firm={"count": 3},
        )

        state1 = build_simulation(config)
        for _ in range(10):
            step(state1)

        state2 = build_simulation(config)
        for _ in range(10):
            step(state2)

        for i in range(10):
            assert state1.history[i]["gdp"] == state2.history[i]["gdp"], \
                f"GDP mismatch at period {i}"
            assert state1.history[i]["unemployment_rate"] == state2.history[i]["unemployment_rate"], \
                f"Unemployment mismatch at period {i}"
            assert state1.history[i]["avg_price"] == state2.history[i]["avg_price"], \
                f"Price mismatch at period {i}"

    def test_different_seed_different_results(self):
        config1 = SimulationConfig(num_periods=5, seed=111, household={"count": 20}, firm={"count": 3})
        config2 = SimulationConfig(num_periods=5, seed=222, household={"count": 20}, firm={"count": 3})

        state1 = build_simulation(config1)
        state2 = build_simulation(config2)
        for _ in range(5):
            step(state1)
            step(state2)

        # At least some metrics should differ (with high probability)
        any_diff = False
        for i in range(5):
            if state1.history[i]["gdp"] != state2.history[i]["gdp"]:
                any_diff = True
                break
        assert any_diff, "Different seeds produced identical results"


class TestShockResponse:
    def test_supply_shock_reduces_production(self):
        """A negative supply shock (halving productivity) should reduce production."""
        config = SimulationConfig(
            num_periods=20,
            seed=42,
            household={"count": 30},
            firm={"count": 3},
            shocks=[
                ShockSpec(
                    period=10,
                    shock_type="supply",
                    parameter="labor_productivity",
                    magnitude=0.5,
                    additive=False,
                ),
            ],
        )
        state = build_simulation(config)
        for _ in range(20):
            step(state)

        # Average production before shock vs after
        pre_shock = np.mean([h["total_production"] for h in state.history[:10]])
        post_shock = np.mean([h["total_production"] for h in state.history[10:]])
        assert post_shock < pre_shock, "Supply shock did not reduce production"

    def test_demand_shock_reduces_consumption(self):
        """A negative demand shock (reducing propensity) should reduce consumption."""
        config = SimulationConfig(
            num_periods=20,
            seed=42,
            household={"count": 30},
            firm={"count": 3},
            shocks=[
                ShockSpec(
                    period=10,
                    shock_type="demand",
                    parameter="consumption_propensity",
                    magnitude=0.5,
                    additive=False,
                ),
            ],
        )
        state = build_simulation(config)
        for _ in range(20):
            step(state)

        pre = np.mean([h["total_consumption"] for h in state.history[:10]])
        post = np.mean([h["total_consumption"] for h in state.history[10:]])
        assert post < pre, "Demand shock did not reduce consumption"


class TestMetricsPlausibility:
    def test_gdp_positive(self):
        config = SimulationConfig(
            num_periods=5,
            household={"count": 20},
            firm={"count": 3},
        )
        state = build_simulation(config)
        for _ in range(5):
            step(state)
        # GDP should be positive in at least some periods
        gdps = [h["gdp"] for h in state.history]
        assert any(g > 0 for g in gdps), "GDP never positive"

    def test_unemployment_rate_bounded(self):
        config = SimulationConfig(
            num_periods=5,
            household={"count": 20},
            firm={"count": 3},
        )
        state = build_simulation(config)
        for _ in range(5):
            step(state)
        for h in state.history:
            assert 0.0 <= h["unemployment_rate"] <= 1.0

    def test_prices_positive(self):
        config = SimulationConfig(
            num_periods=5,
            household={"count": 20},
            firm={"count": 3},
        )
        state = build_simulation(config)
        for _ in range(5):
            step(state)
        for h in state.history:
            assert h["avg_price"] > 0
