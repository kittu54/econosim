"""
Adaptive expectations and learning dynamics.

Implements expectation formation rules that agents use to forecast
future prices, demand, wages, and other economic variables.

Key concepts:
- ExpectationModel: Abstract base for expectation formation
- AdaptiveExpectations: Simple adaptive (exponential smoothing) model
- RollingExpectations: Rolling window average expectations
- WeightedExpectations: Combines multiple signals with configurable weights
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from econosim.core.accounting import round_money


class ExpectationModel(ABC):
    """Abstract base class for expectation formation."""

    @abstractmethod
    def update(self, actual: float) -> None:
        """Update the model with a new realized value."""

    @abstractmethod
    def forecast(self) -> float:
        """Generate a forecast for the next period."""

    @abstractmethod
    def forecast_n(self, n: int) -> list[float]:
        """Generate forecasts for the next n periods."""

    @abstractmethod
    def forecast_error(self) -> float:
        """Return the most recent forecast error."""

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Return the model state for logging."""


class AdaptiveExpectations(ExpectationModel):
    """Simple adaptive expectations (exponential smoothing).

    x_expected(t+1) = alpha * x_actual(t) + (1 - alpha) * x_expected(t)

    Higher alpha means faster adaptation to new information.
    Lower alpha means more weight on past expectations (inertia).
    """

    def __init__(
        self,
        initial_value: float = 0.0,
        alpha: float = 0.3,
        name: str = "adaptive",
    ) -> None:
        self.name = name
        self.alpha = alpha
        self._expected = initial_value
        self._last_actual = initial_value
        self._last_error = 0.0
        self._n_updates = 0
        self._cumulative_error = 0.0

    def update(self, actual: float) -> None:
        self._last_error = actual - self._expected
        self._cumulative_error += abs(self._last_error)
        self._expected = self.alpha * actual + (1 - self.alpha) * self._expected
        self._last_actual = actual
        self._n_updates += 1

    def forecast(self) -> float:
        return round_money(self._expected)

    def forecast_n(self, n: int) -> list[float]:
        # Adaptive expectations produce flat forecasts
        return [self.forecast()] * n

    def forecast_error(self) -> float:
        return self._last_error

    @property
    def mean_absolute_error(self) -> float:
        if self._n_updates == 0:
            return 0.0
        return self._cumulative_error / self._n_updates

    def get_state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "adaptive",
            "expected": self._expected,
            "last_actual": self._last_actual,
            "last_error": self._last_error,
            "alpha": self.alpha,
            "n_updates": self._n_updates,
            "mean_abs_error": self.mean_absolute_error,
        }


class RollingExpectations(ExpectationModel):
    """Rolling window average expectations.

    Forecast = mean of the last `window` observed values.
    Optionally adds trend extrapolation.
    """

    def __init__(
        self,
        initial_value: float = 0.0,
        window: int = 6,
        use_trend: bool = False,
        name: str = "rolling",
    ) -> None:
        self.name = name
        self.window = window
        self.use_trend = use_trend
        self._history: deque[float] = deque(maxlen=window)
        self._history.append(initial_value)
        self._last_error = 0.0

    def update(self, actual: float) -> None:
        self._last_error = actual - self.forecast()
        self._history.append(actual)

    def forecast(self) -> float:
        if not self._history:
            return 0.0
        base = float(np.mean(self._history))
        if self.use_trend and len(self._history) >= 3:
            trend = self._compute_trend()
            base += trend
        return round_money(base)

    def forecast_n(self, n: int) -> list[float]:
        if not self._history:
            return [0.0] * n
        base = float(np.mean(self._history))
        if self.use_trend and len(self._history) >= 3:
            trend = self._compute_trend()
            return [round_money(base + trend * (i + 1)) for i in range(n)]
        return [round_money(base)] * n

    def _compute_trend(self) -> float:
        """Simple linear trend from history."""
        values = list(self._history)
        n = len(values)
        if n < 2:
            return 0.0
        x = np.arange(n)
        slope = float(np.polyfit(x, values, 1)[0])
        return slope

    def forecast_error(self) -> float:
        return self._last_error

    def get_state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "rolling",
            "forecast": self.forecast(),
            "window": self.window,
            "use_trend": self.use_trend,
            "history_length": len(self._history),
            "last_error": self._last_error,
        }


class WeightedExpectations(ExpectationModel):
    """Combines multiple expectation models with configurable weights.

    Useful for agents that consider multiple signals:
    - Past experience (adaptive/rolling)
    - Announced policy (e.g., central bank target)
    - Peer behavior (e.g., other firms' prices)

    Forecast = sum(weight_i * model_i.forecast()) / sum(weights)
    """

    def __init__(
        self,
        models: list[tuple[ExpectationModel, float]],  # (model, weight) pairs
        name: str = "weighted",
    ) -> None:
        self.name = name
        self.models = models
        self._last_error = 0.0

    def update(self, actual: float) -> None:
        self._last_error = actual - self.forecast()
        for model, _ in self.models:
            model.update(actual)

    def forecast(self) -> float:
        total_weight = sum(w for _, w in self.models)
        if total_weight <= 0:
            return 0.0
        weighted_sum = sum(m.forecast() * w for m, w in self.models)
        return round_money(weighted_sum / total_weight)

    def forecast_n(self, n: int) -> list[float]:
        total_weight = sum(w for _, w in self.models)
        if total_weight <= 0:
            return [0.0] * n
        forecasts = [m.forecast_n(n) for m, _ in self.models]
        weights = [w for _, w in self.models]
        result = []
        for i in range(n):
            weighted = sum(f[i] * w for f, w in zip(forecasts, weights))
            result.append(round_money(weighted / total_weight))
        return result

    def forecast_error(self) -> float:
        return self._last_error

    def get_state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "weighted",
            "forecast": self.forecast(),
            "last_error": self._last_error,
            "components": [
                {"model": m.get_state(), "weight": w}
                for m, w in self.models
            ],
        }


@dataclass
class AgentExpectations:
    """Container for all expectations held by an agent.

    Provides a structured way for agents to form expectations about
    multiple economic variables simultaneously.
    """

    agent_id: str
    price: ExpectationModel = field(default_factory=lambda: AdaptiveExpectations(10.0, 0.3, "price"))
    wage: ExpectationModel = field(default_factory=lambda: AdaptiveExpectations(60.0, 0.2, "wage"))
    demand: ExpectationModel = field(default_factory=lambda: RollingExpectations(100.0, 4, True, "demand"))
    inflation: ExpectationModel = field(default_factory=lambda: AdaptiveExpectations(0.0, 0.5, "inflation"))

    def update_all(
        self,
        actual_price: float | None = None,
        actual_wage: float | None = None,
        actual_demand: float | None = None,
        actual_inflation: float | None = None,
    ) -> None:
        """Update all expectation models with realized values."""
        if actual_price is not None:
            self.price.update(actual_price)
        if actual_wage is not None:
            self.wage.update(actual_wage)
        if actual_demand is not None:
            self.demand.update(actual_demand)
        if actual_inflation is not None:
            self.inflation.update(actual_inflation)

    def get_state(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "price": self.price.get_state(),
            "wage": self.wage.get_state(),
            "demand": self.demand.get_state(),
            "inflation": self.inflation.get_state(),
        }
