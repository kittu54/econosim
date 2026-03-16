"""Tests for adaptive expectations and learning dynamics."""

from __future__ import annotations

import numpy as np
import pytest

from econosim.extensions.expectations import (
    AdaptiveExpectations,
    RollingExpectations,
    WeightedExpectations,
    AgentExpectations,
)


class TestAdaptiveExpectations:
    def test_initial_forecast(self):
        model = AdaptiveExpectations(initial_value=10.0)
        assert model.forecast() == 10.0

    def test_update_moves_toward_actual(self):
        model = AdaptiveExpectations(initial_value=10.0, alpha=0.5)
        model.update(20.0)
        # Expected: 0.5 * 20 + 0.5 * 10 = 15
        assert model.forecast() == 15.0

    def test_high_alpha_adapts_fast(self):
        model = AdaptiveExpectations(initial_value=10.0, alpha=0.9)
        model.update(20.0)
        assert model.forecast() > 18.0  # Close to 20

    def test_low_alpha_adapts_slow(self):
        model = AdaptiveExpectations(initial_value=10.0, alpha=0.1)
        model.update(20.0)
        assert model.forecast() < 12.0  # Close to 10

    def test_forecast_error(self):
        model = AdaptiveExpectations(initial_value=10.0)
        model.update(15.0)
        assert model.forecast_error() == 5.0  # 15 - 10

    def test_flat_forecast(self):
        model = AdaptiveExpectations(initial_value=10.0)
        forecasts = model.forecast_n(5)
        assert len(forecasts) == 5
        assert all(f == 10.0 for f in forecasts)

    def test_mean_absolute_error(self):
        model = AdaptiveExpectations(initial_value=10.0, alpha=0.5)
        model.update(20.0)  # error = 10
        model.update(15.0)  # error = 15 - 15 = 0
        mae = model.mean_absolute_error
        assert mae == 5.0  # (10 + 0) / 2

    def test_get_state(self):
        model = AdaptiveExpectations(initial_value=10.0, alpha=0.3, name="price")
        model.update(12.0)
        state = model.get_state()
        assert state["name"] == "price"
        assert state["type"] == "adaptive"
        assert state["alpha"] == 0.3
        assert state["n_updates"] == 1


class TestRollingExpectations:
    def test_initial_forecast(self):
        model = RollingExpectations(initial_value=10.0, window=4)
        assert model.forecast() == 10.0

    def test_rolling_average(self):
        model = RollingExpectations(initial_value=10.0, window=3)
        model.update(20.0)
        model.update(30.0)
        # History: [10, 20, 30] -> mean = 20
        assert model.forecast() == 20.0

    def test_window_rolling(self):
        model = RollingExpectations(initial_value=10.0, window=3)
        model.update(20.0)
        model.update(30.0)
        model.update(40.0)
        # Window: [20, 30, 40] -> mean = 30
        assert model.forecast() == 30.0

    def test_trend_extrapolation(self):
        model = RollingExpectations(initial_value=10.0, window=5, use_trend=True)
        model.update(12.0)
        model.update(14.0)
        model.update(16.0)
        # Clear upward trend, forecast should exceed current mean
        forecast = model.forecast()
        mean_val = np.mean([10.0, 12.0, 14.0, 16.0])
        assert forecast > mean_val

    def test_forecast_n_no_trend(self):
        model = RollingExpectations(initial_value=10.0, window=3)
        model.update(10.0)
        model.update(10.0)
        forecasts = model.forecast_n(3)
        assert all(f == 10.0 for f in forecasts)

    def test_forecast_n_with_trend(self):
        model = RollingExpectations(initial_value=10.0, window=5, use_trend=True)
        for v in [12.0, 14.0, 16.0]:
            model.update(v)
        forecasts = model.forecast_n(3)
        # Should show increasing values
        assert forecasts[2] > forecasts[0]

    def test_forecast_error(self):
        model = RollingExpectations(initial_value=10.0, window=3)
        model.update(15.0)
        assert model.forecast_error() == 5.0  # 15 - 10

    def test_get_state(self):
        model = RollingExpectations(initial_value=10.0, name="demand")
        state = model.get_state()
        assert state["name"] == "demand"
        assert state["type"] == "rolling"


class TestWeightedExpectations:
    def test_weighted_forecast(self):
        m1 = AdaptiveExpectations(initial_value=10.0)
        m2 = AdaptiveExpectations(initial_value=20.0)
        weighted = WeightedExpectations([(m1, 1.0), (m2, 1.0)])
        # (10 + 20) / 2 = 15
        assert weighted.forecast() == 15.0

    def test_unequal_weights(self):
        m1 = AdaptiveExpectations(initial_value=10.0)
        m2 = AdaptiveExpectations(initial_value=20.0)
        weighted = WeightedExpectations([(m1, 3.0), (m2, 1.0)])
        # (10*3 + 20*1) / 4 = 12.5
        assert weighted.forecast() == 12.5

    def test_update_propagates(self):
        m1 = AdaptiveExpectations(initial_value=10.0, alpha=0.5)
        m2 = AdaptiveExpectations(initial_value=20.0, alpha=0.5)
        weighted = WeightedExpectations([(m1, 1.0), (m2, 1.0)])
        weighted.update(30.0)
        # m1: 0.5*30 + 0.5*10 = 20
        # m2: 0.5*30 + 0.5*20 = 25
        # weighted: (20 + 25) / 2 = 22.5
        assert weighted.forecast() == 22.5

    def test_forecast_n(self):
        m1 = AdaptiveExpectations(initial_value=10.0)
        m2 = RollingExpectations(initial_value=20.0)
        weighted = WeightedExpectations([(m1, 1.0), (m2, 1.0)])
        forecasts = weighted.forecast_n(3)
        assert len(forecasts) == 3

    def test_get_state(self):
        m1 = AdaptiveExpectations(initial_value=10.0, name="exp")
        weighted = WeightedExpectations([(m1, 1.0)], name="combined")
        state = weighted.get_state()
        assert state["name"] == "combined"
        assert len(state["components"]) == 1


class TestAgentExpectations:
    def test_creation(self):
        exp = AgentExpectations("firm_0")
        assert exp.agent_id == "firm_0"
        assert exp.price.forecast() == 10.0
        assert exp.wage.forecast() == 60.0

    def test_update_all(self):
        exp = AgentExpectations("firm_0")
        exp.update_all(actual_price=12.0, actual_wage=65.0, actual_demand=120.0)
        assert exp.price.forecast() != 10.0
        assert exp.wage.forecast() != 60.0

    def test_selective_update(self):
        exp = AgentExpectations("firm_0")
        original_wage = exp.wage.forecast()
        exp.update_all(actual_price=12.0)
        # Wage should not change
        assert exp.wage.forecast() == original_wage

    def test_get_state(self):
        exp = AgentExpectations("firm_0")
        state = exp.get_state()
        assert state["agent_id"] == "firm_0"
        assert "price" in state
        assert "wage" in state
        assert "demand" in state
        assert "inflation" in state
