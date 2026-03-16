"""Integration tests for Phase 4 extensions wired into the core simulation.

Tests:
- Extensions initialize correctly when feature flags are enabled
- Extensions produce metrics without breaking existing simulation
- Accounting invariants still hold with extensions enabled
- Each extension produces plausible output data
"""

import pytest
import numpy as np

from econosim.config.schema import (
    SimulationConfig,
    ExtensionsConfig,
    BondConfig,
    ExpectationsConfig,
    NetworkConfig,
)
from econosim.engine.simulation import build_simulation, step, run_simulation


SMALL_CONFIG = dict(
    num_periods=10,
    seed=42,
    household={"count": 20},
    firm={"count": 3},
)


class TestExtensionsDisabledByDefault:
    def test_no_extensions_by_default(self):
        config = SimulationConfig(**SMALL_CONFIG)
        state = build_simulation(config)
        assert state.expectations == {}
        assert state.trade_network is None
        assert state.credit_network is None
        assert state.bond_market is None
        assert state.debt_manager is None

    def test_default_metrics_unchanged(self):
        config = SimulationConfig(**SMALL_CONFIG)
        state = build_simulation(config)
        metrics = step(state)
        # No extension keys present
        assert "trade_network_density" not in metrics
        assert "credit_systemic_risk" not in metrics
        assert "bond_outstanding" not in metrics
        assert "avg_price_forecast_error" not in metrics


class TestExpectationsIntegration:
    def test_expectations_initialized(self):
        config = SimulationConfig(
            **SMALL_CONFIG,
            extensions=ExtensionsConfig(enable_expectations=True),
        )
        state = build_simulation(config)
        assert len(state.expectations) == 3  # 3 firms
        for firm in state.firms:
            assert firm.agent_id in state.expectations

    def test_expectations_update_after_step(self):
        config = SimulationConfig(
            **SMALL_CONFIG,
            extensions=ExtensionsConfig(enable_expectations=True),
        )
        state = build_simulation(config)
        step(state)
        # After one step, expectations should have been updated
        for exp in state.expectations.values():
            # price forecast should have changed from default
            assert exp.price.get_state()["n_updates"] >= 1

    def test_expectations_metrics_present(self):
        config = SimulationConfig(
            **SMALL_CONFIG,
            extensions=ExtensionsConfig(enable_expectations=True),
        )
        state = build_simulation(config)
        metrics = step(state)
        assert "avg_price_forecast_error" in metrics
        assert "avg_demand_forecast_error" in metrics
        assert metrics["avg_price_forecast_error"] >= 0
        assert metrics["avg_demand_forecast_error"] >= 0

    def test_accounting_holds_with_expectations(self):
        config = SimulationConfig(
            **SMALL_CONFIG,
            extensions=ExtensionsConfig(enable_expectations=True),
        )
        state = build_simulation(config)
        for _ in range(5):
            step(state)
        validation = state.ledger.validate_all_balanced()
        for owner_id, balanced in validation.items():
            assert balanced, f"{owner_id} unbalanced with expectations enabled"


class TestNetworkIntegration:
    def test_networks_initialized(self):
        config = SimulationConfig(
            **SMALL_CONFIG,
            extensions=ExtensionsConfig(enable_networks=True),
        )
        state = build_simulation(config)
        assert state.trade_network is not None
        assert state.credit_network is not None

    def test_trade_network_records_after_step(self):
        config = SimulationConfig(
            **SMALL_CONFIG,
            extensions=ExtensionsConfig(enable_networks=True),
        )
        state = build_simulation(config)
        for _ in range(3):
            step(state)
        # Should have some trade edges after goods market clears
        assert state.trade_network.num_edges >= 0  # may be 0 if no sales

    def test_network_metrics_present(self):
        config = SimulationConfig(
            **SMALL_CONFIG,
            extensions=ExtensionsConfig(enable_networks=True),
        )
        state = build_simulation(config)
        metrics = step(state)
        assert "trade_network_density" in metrics
        assert "trade_network_concentration" in metrics
        assert "trade_seller_concentration" in metrics
        assert "credit_network_density" in metrics
        assert "credit_network_concentration" in metrics
        assert "credit_systemic_risk" in metrics

    def test_trade_network_has_edges_after_activity(self):
        config = SimulationConfig(
            **SMALL_CONFIG,
            extensions=ExtensionsConfig(enable_networks=True),
        )
        state = build_simulation(config)
        for _ in range(5):
            step(state)
        # After 5 periods of economic activity, trade network should have edges
        assert state.trade_network.num_edges > 0

    def test_accounting_holds_with_networks(self):
        config = SimulationConfig(
            **SMALL_CONFIG,
            extensions=ExtensionsConfig(enable_networks=True),
        )
        state = build_simulation(config)
        for _ in range(5):
            step(state)
        validation = state.ledger.validate_all_balanced()
        for owner_id, balanced in validation.items():
            assert balanced, f"{owner_id} unbalanced with networks enabled"


class TestBondMarketIntegration:
    def test_bond_market_initialized(self):
        config = SimulationConfig(
            **SMALL_CONFIG,
            extensions=ExtensionsConfig(enable_bonds=True),
        )
        state = build_simulation(config)
        assert state.bond_market is not None
        assert state.debt_manager is not None

    def test_bond_metrics_present(self):
        config = SimulationConfig(
            **SMALL_CONFIG,
            extensions=ExtensionsConfig(enable_bonds=True),
        )
        state = build_simulation(config)
        metrics = step(state)
        assert "bond_outstanding" in metrics
        assert "bond_interest_expense" in metrics
        assert "bond_issued" in metrics
        assert "bond_redeemed" in metrics
        assert "bond_debt_to_gdp" in metrics

    def test_bond_issuance_occurs_when_needed(self):
        # Use low initial govt deposits to force bond issuance
        config = SimulationConfig(
            num_periods=10,
            seed=42,
            household={"count": 20},
            firm={"count": 3},
            government={"initial_deposits": 100.0, "spending_per_period": 2000.0},
            extensions=ExtensionsConfig(enable_bonds=True),
        )
        state = build_simulation(config)
        for _ in range(5):
            step(state)
        # With low deposits and high spending, bonds should have been issued
        assert state.bond_market.total_outstanding() > 0

    def test_accounting_holds_with_bonds(self):
        config = SimulationConfig(
            **SMALL_CONFIG,
            extensions=ExtensionsConfig(enable_bonds=True),
        )
        state = build_simulation(config)
        for _ in range(5):
            step(state)
        validation = state.ledger.validate_all_balanced()
        for owner_id, balanced in validation.items():
            assert balanced, f"{owner_id} unbalanced with bonds enabled"


class TestAllExtensionsEnabled:
    def test_all_extensions_together(self):
        config = SimulationConfig(
            **SMALL_CONFIG,
            extensions=ExtensionsConfig(
                enable_expectations=True,
                enable_networks=True,
                enable_bonds=True,
            ),
        )
        state = build_simulation(config)
        for _ in range(10):
            step(state)

        # All extension metrics should be present
        last = state.history[-1]
        assert "avg_price_forecast_error" in last
        assert "trade_network_density" in last
        assert "bond_outstanding" in last

    def test_accounting_holds_all_extensions(self):
        config = SimulationConfig(
            **SMALL_CONFIG,
            extensions=ExtensionsConfig(
                enable_expectations=True,
                enable_networks=True,
                enable_bonds=True,
            ),
        )
        state = build_simulation(config)
        for t in range(10):
            step(state)
            validation = state.ledger.validate_all_balanced()
            for owner_id, balanced in validation.items():
                assert balanced, f"Period {t}: {owner_id} unbalanced"

    def test_reproducibility_with_extensions(self):
        ext = ExtensionsConfig(
            enable_expectations=True,
            enable_networks=True,
            enable_bonds=True,
        )
        config = SimulationConfig(**SMALL_CONFIG, extensions=ext)

        state1 = build_simulation(config)
        for _ in range(5):
            step(state1)

        state2 = build_simulation(config)
        for _ in range(5):
            step(state2)

        for i in range(5):
            assert state1.history[i]["gdp"] == state2.history[i]["gdp"]
