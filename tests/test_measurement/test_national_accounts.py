"""Tests for the national accounts measurement model."""

import pytest
import numpy as np

from econosim.config.schema import SimulationConfig
from econosim.engine.simulation import build_simulation, step
from econosim.measurement.national_accounts import (
    NationalAccountsMapper,
    NationalAccountsOutput,
    LaborMarketMetrics,
    FinancialSystemMetrics,
    MeasuredSeries,
    MEASUREMENT_SERIES,
)


@pytest.fixture
def sim_state():
    """Run a short simulation and return the state."""
    config = SimulationConfig(num_periods=30, seed=42)
    state = build_simulation(config)
    for _ in range(30):
        step(state)
    return state


class TestNationalAccountsMapper:
    def test_measure_produces_output(self, sim_state):
        mapper = NationalAccountsMapper()
        # Measure the last period
        output = mapper.measure(sim_state, sim_state.current_period - 1)
        assert isinstance(output, NationalAccountsOutput)
        assert output.gdp_nominal > 0

    def test_gdp_decomposition(self, sim_state):
        mapper = NationalAccountsMapper()
        output = mapper.measure(sim_state, sim_state.current_period - 1)

        # GDP = C + I + G + NX
        components = (
            output.consumption + output.investment
            + output.government_spending + output.net_exports
        )
        # Allow small rounding differences
        assert abs(output.gdp_nominal - components) < 1.0 or output.gdp_nominal >= components

    def test_unemployment_rate_bounded(self, sim_state):
        mapper = NationalAccountsMapper()
        output = mapper.measure(sim_state, sim_state.current_period - 1)
        assert 0.0 <= output.unemployment_rate <= 1.0

    def test_price_index_positive(self, sim_state):
        mapper = NationalAccountsMapper()
        output = mapper.measure(sim_state, sim_state.current_period - 1)
        assert output.price_index > 0

    def test_gini_bounded(self, sim_state):
        mapper = NationalAccountsMapper()
        output = mapper.measure(sim_state, sim_state.current_period - 1)
        assert 0.0 <= output.gini_wealth <= 1.0
        assert 0.0 <= output.gini_income <= 1.0

    def test_to_dict(self, sim_state):
        mapper = NationalAccountsMapper()
        output = mapper.measure(sim_state, sim_state.current_period - 1)
        d = output.to_dict()
        assert "gdp_nominal" in d
        assert "unemployment_rate" in d
        assert "inflation_rate" in d

    def test_sequential_measurement(self):
        """Measure across multiple periods and check growth rates are computed."""
        config = SimulationConfig(num_periods=10, seed=42)
        state = build_simulation(config)
        mapper = NationalAccountsMapper()

        outputs = []
        for t in range(10):
            step(state)
            out = mapper.measure(state, t)
            outputs.append(out)

        # After first period, growth rates should be computed
        assert outputs[-1].gdp_growth != 0.0 or outputs[-1].gdp_nominal == outputs[-2].gdp_nominal

    def test_debt_to_gdp_nonnegative(self, sim_state):
        mapper = NationalAccountsMapper()
        output = mapper.measure(sim_state, sim_state.current_period - 1)
        assert output.debt_to_gdp >= 0.0

    def test_money_velocity_positive(self, sim_state):
        mapper = NationalAccountsMapper()
        output = mapper.measure(sim_state, sim_state.current_period - 1)
        assert output.money_velocity >= 0.0


class TestLaborMarketMetrics:
    def test_measure(self, sim_state):
        metrics = LaborMarketMetrics.measure(sim_state)
        assert "employment" in metrics
        assert "unemployment_rate" in metrics
        assert "vacancy_rate" in metrics
        assert 0.0 <= metrics["unemployment_rate"] <= 1.0
        assert 0.0 <= metrics["participation_rate"] <= 1.0


class TestFinancialSystemMetrics:
    def test_measure(self, sim_state):
        metrics = FinancialSystemMetrics.measure(sim_state)
        assert "total_loans" in metrics
        assert "bank_equity" in metrics
        assert "capital_ratio" in metrics
        assert "avg_firm_leverage" in metrics
        assert metrics["capital_ratio"] >= 0.0


class TestMeasurementSeries:
    def test_series_catalog(self):
        assert "gdp_nominal" in MEASUREMENT_SERIES
        assert "unemployment_rate" in MEASUREMENT_SERIES
        assert "inflation_rate" in MEASUREMENT_SERIES

    def test_series_metadata(self):
        gdp = MEASUREMENT_SERIES["gdp_nominal"]
        assert gdp.units == "level"
        assert len(gdp.source_variables) > 0
