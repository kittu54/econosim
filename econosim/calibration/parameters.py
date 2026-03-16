"""Parameter registry with priors, bounds, and calibration metadata.

Separates parameters into:
- Fixed: set by the modeler, not estimated
- Calibrated: estimated from data via SMM or Bayesian methods
- Derived: computed from other parameters

Each calibrated parameter has:
- Prior distribution (for Bayesian methods)
- Bounds (for optimization)
- Transform (for unconstrained optimization)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field


class ParameterStatus(Enum):
    FIXED = auto()
    CALIBRATED = auto()
    DERIVED = auto()


class PriorType(Enum):
    UNIFORM = auto()
    NORMAL = auto()
    LOGNORMAL = auto()
    BETA = auto()
    GAMMA = auto()
    FIXED = auto()


@dataclass
class Prior:
    """Prior distribution specification for Bayesian estimation."""

    prior_type: PriorType = PriorType.UNIFORM
    params: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def uniform(low: float, high: float) -> Prior:
        return Prior(PriorType.UNIFORM, {"low": low, "high": high})

    @staticmethod
    def normal(mean: float, std: float) -> Prior:
        return Prior(PriorType.NORMAL, {"mean": mean, "std": std})

    @staticmethod
    def lognormal(mu: float, sigma: float) -> Prior:
        return Prior(PriorType.LOGNORMAL, {"mu": mu, "sigma": sigma})

    @staticmethod
    def beta(a: float, b: float) -> Prior:
        return Prior(PriorType.BETA, {"a": a, "b": b})

    @staticmethod
    def gamma(shape: float, scale: float) -> Prior:
        return Prior(PriorType.GAMMA, {"shape": shape, "scale": scale})

    def log_pdf(self, x: float) -> float:
        """Compute log prior density at x."""
        from scipy import stats as sp

        if self.prior_type == PriorType.UNIFORM:
            low, high = self.params["low"], self.params["high"]
            if low <= x <= high:
                return -math.log(high - low)
            return -np.inf
        elif self.prior_type == PriorType.NORMAL:
            return float(sp.norm.logpdf(x, self.params["mean"], self.params["std"]))
        elif self.prior_type == PriorType.LOGNORMAL:
            return float(sp.lognorm.logpdf(x, self.params["sigma"], scale=math.exp(self.params["mu"])))
        elif self.prior_type == PriorType.BETA:
            return float(sp.beta.logpdf(x, self.params["a"], self.params["b"]))
        elif self.prior_type == PriorType.GAMMA:
            return float(sp.gamma.logpdf(x, self.params["shape"], scale=self.params["scale"]))
        elif self.prior_type == PriorType.FIXED:
            return 0.0
        return 0.0

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        """Draw samples from the prior."""
        if self.prior_type == PriorType.UNIFORM:
            return rng.uniform(self.params["low"], self.params["high"], n)
        elif self.prior_type == PriorType.NORMAL:
            return rng.normal(self.params["mean"], self.params["std"], n)
        elif self.prior_type == PriorType.LOGNORMAL:
            return rng.lognormal(self.params["mu"], self.params["sigma"], n)
        elif self.prior_type == PriorType.BETA:
            return rng.beta(self.params["a"], self.params["b"], n)
        elif self.prior_type == PriorType.GAMMA:
            return rng.gamma(self.params["shape"], n) * self.params["scale"]
        return np.full(n, self.params.get("value", 0.0))


@dataclass
class ParameterSpec:
    """Specification for a single model parameter."""

    name: str
    config_path: str  # dotted path in SimulationConfig, e.g. "firm.labor_productivity"
    status: ParameterStatus = ParameterStatus.FIXED
    default_value: float = 0.0
    lower_bound: float = -np.inf
    upper_bound: float = np.inf
    prior: Prior = field(default_factory=lambda: Prior.uniform(0, 1))
    description: str = ""
    transform: str = "identity"  # "identity", "log", "logit"

    def to_unconstrained(self, x: float) -> float:
        """Transform to unconstrained space for optimization."""
        if self.transform == "log":
            return math.log(max(x, 1e-10))
        elif self.transform == "logit":
            x_clipped = max(min(x, 1 - 1e-10), 1e-10)
            return math.log(x_clipped / (1 - x_clipped))
        return x

    def from_unconstrained(self, y: float) -> float:
        """Transform back from unconstrained space."""
        if self.transform == "log":
            x = math.exp(y)
        elif self.transform == "logit":
            x = 1.0 / (1.0 + math.exp(-y))
        else:
            x = y
        return max(self.lower_bound, min(self.upper_bound, x))


class ParameterRegistry:
    """Registry of all model parameters with metadata."""

    def __init__(self) -> None:
        self._params: dict[str, ParameterSpec] = {}

    def register(self, spec: ParameterSpec) -> None:
        self._params[spec.name] = spec

    def get(self, name: str) -> ParameterSpec:
        if name not in self._params:
            raise KeyError(f"Parameter '{name}' not registered")
        return self._params[name]

    @property
    def calibrated(self) -> list[ParameterSpec]:
        return [p for p in self._params.values() if p.status == ParameterStatus.CALIBRATED]

    @property
    def fixed(self) -> list[ParameterSpec]:
        return [p for p in self._params.values() if p.status == ParameterStatus.FIXED]

    @property
    def all_params(self) -> list[ParameterSpec]:
        return list(self._params.values())

    def calibrated_names(self) -> list[str]:
        return [p.name for p in self.calibrated]

    def calibrated_bounds(self) -> list[tuple[float, float]]:
        return [(p.lower_bound, p.upper_bound) for p in self.calibrated]

    def calibrated_defaults(self) -> np.ndarray:
        return np.array([p.default_value for p in self.calibrated])

    def to_vector(self, param_dict: dict[str, float]) -> np.ndarray:
        """Convert named parameter dict to vector (calibrated params only)."""
        return np.array([param_dict[p.name] for p in self.calibrated])

    def from_vector(self, vec: np.ndarray) -> dict[str, float]:
        """Convert vector to named parameter dict (calibrated params only)."""
        params = self.calibrated
        return {p.name: float(vec[i]) for i, p in enumerate(params)}

    def log_prior(self, param_dict: dict[str, float]) -> float:
        """Compute log prior density for calibrated parameters."""
        log_p = 0.0
        for p in self.calibrated:
            log_p += p.prior.log_pdf(param_dict[p.name])
        return log_p


def default_macro_registry() -> ParameterRegistry:
    """Create the default parameter registry for US macro calibration.

    Parameters chosen based on standard macro calibration targets.
    """
    reg = ParameterRegistry()

    # --- Household parameters ---
    reg.register(ParameterSpec(
        name="consumption_propensity",
        config_path="household.consumption_propensity",
        status=ParameterStatus.CALIBRATED,
        default_value=0.8,
        lower_bound=0.3, upper_bound=0.99,
        prior=Prior.beta(8, 2),
        transform="logit",
        description="Marginal propensity to consume out of income",
    ))
    reg.register(ParameterSpec(
        name="wealth_propensity",
        config_path="household.wealth_propensity",
        status=ParameterStatus.CALIBRATED,
        default_value=0.4,
        lower_bound=0.01, upper_bound=0.8,
        prior=Prior.beta(4, 6),
        transform="logit",
        description="Marginal propensity to consume out of wealth",
    ))

    # --- Firm parameters ---
    reg.register(ParameterSpec(
        name="labor_productivity",
        config_path="firm.labor_productivity",
        status=ParameterStatus.CALIBRATED,
        default_value=8.0,
        lower_bound=1.0, upper_bound=50.0,
        prior=Prior.lognormal(2.0, 0.5),
        transform="log",
        description="Output per worker per period",
    ))
    reg.register(ParameterSpec(
        name="price_adjustment_speed",
        config_path="firm.price_adjustment_speed",
        status=ParameterStatus.CALIBRATED,
        default_value=0.03,
        lower_bound=0.001, upper_bound=0.3,
        prior=Prior.beta(3, 97),
        transform="logit",
        description="Price adjustment speed (Calvo-like stickiness inverse)",
    ))
    reg.register(ParameterSpec(
        name="wage_adjustment_speed",
        config_path="firm.wage_adjustment_speed",
        status=ParameterStatus.CALIBRATED,
        default_value=0.02,
        lower_bound=0.001, upper_bound=0.2,
        prior=Prior.beta(2, 98),
        transform="logit",
        description="Wage adjustment speed",
    ))
    reg.register(ParameterSpec(
        name="target_inventory_ratio",
        config_path="firm.target_inventory_ratio",
        status=ParameterStatus.CALIBRATED,
        default_value=0.2,
        lower_bound=0.01, upper_bound=1.0,
        prior=Prior.beta(2, 8),
        transform="logit",
        description="Target inventory-to-sales ratio",
    ))

    # --- Bank parameters ---
    reg.register(ParameterSpec(
        name="base_interest_rate",
        config_path="bank.base_interest_rate",
        status=ParameterStatus.CALIBRATED,
        default_value=0.005,
        lower_bound=0.0001, upper_bound=0.05,
        prior=Prior.beta(1, 99),
        transform="logit",
        description="Bank base lending rate (per period)",
    ))
    reg.register(ParameterSpec(
        name="risk_premium",
        config_path="bank.risk_premium",
        status=ParameterStatus.FIXED,
        default_value=0.002,
        description="Bank risk premium over base rate",
    ))

    # --- Government parameters ---
    reg.register(ParameterSpec(
        name="income_tax_rate",
        config_path="government.income_tax_rate",
        status=ParameterStatus.FIXED,
        default_value=0.2,
        description="Flat income tax rate",
    ))
    reg.register(ParameterSpec(
        name="govt_spending",
        config_path="government.spending_per_period",
        status=ParameterStatus.CALIBRATED,
        default_value=2000.0,
        lower_bound=100.0, upper_bound=20000.0,
        prior=Prior.lognormal(7.5, 0.5),
        transform="log",
        description="Government spending per period",
    ))

    return reg
