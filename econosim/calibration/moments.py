"""Moment definitions for calibration targets.

Moments are summary statistics computed from simulation output
that are matched against their empirical counterparts from data.

Examples: GDP volatility, inflation persistence, unemployment mean,
credit-to-GDP ratio, Gini coefficient, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class MomentDefinition:
    """Definition of a single moment to match in calibration."""

    name: str
    description: str
    empirical_value: float  # target value from data
    empirical_std: float = 0.0  # uncertainty in empirical value (for weighting)
    compute_fn: Callable[[pd.DataFrame], float] | None = None
    series_name: str = ""  # column name in simulation output
    statistic: str = "mean"  # "mean", "std", "autocorr", "ratio", "quantile"
    burn_in: int = 20  # periods to discard before computing

    def compute(self, df: pd.DataFrame) -> float:
        """Compute this moment from simulation output DataFrame."""
        if self.compute_fn is not None:
            return self.compute_fn(df)

        if not self.series_name or self.series_name not in df.columns:
            return np.nan

        series = df[self.series_name].iloc[self.burn_in:]
        if len(series) == 0:
            return np.nan

        if self.statistic == "mean":
            return float(series.mean())
        elif self.statistic == "std":
            return float(series.std())
        elif self.statistic == "cv":
            return float(series.std() / max(series.mean(), 1e-10))
        elif self.statistic == "autocorr":
            return float(series.autocorr(lag=1)) if len(series) > 1 else 0.0
        elif self.statistic == "ratio":
            return float(series.mean())  # for pre-computed ratios
        elif self.statistic == "min":
            return float(series.min())
        elif self.statistic == "max":
            return float(series.max())
        elif self.statistic == "median":
            return float(series.median())
        elif self.statistic == "final":
            return float(series.iloc[-1])
        else:
            return float(series.mean())


class MomentSet:
    """A collection of moments to match during calibration."""

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._moments: list[MomentDefinition] = []

    def add(self, moment: MomentDefinition) -> None:
        self._moments.append(moment)

    @property
    def moments(self) -> list[MomentDefinition]:
        return list(self._moments)

    @property
    def names(self) -> list[str]:
        return [m.name for m in self._moments]

    @property
    def empirical_values(self) -> np.ndarray:
        return np.array([m.empirical_value for m in self._moments])

    @property
    def empirical_stds(self) -> np.ndarray:
        return np.array([m.empirical_std if m.empirical_std > 0 else 1.0 for m in self._moments])

    def compute_all(self, df: pd.DataFrame) -> np.ndarray:
        """Compute all moments from simulation output."""
        return np.array([m.compute(df) for m in self._moments])

    def compute_dict(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute all moments and return as named dict."""
        return {m.name: m.compute(df) for m in self._moments}

    def weighting_matrix(self, method: str = "inverse_variance") -> np.ndarray:
        """Compute the weighting matrix for SMM.

        Args:
            method: "identity", "inverse_variance", or "diagonal"
        """
        n = len(self._moments)
        if method == "identity":
            return np.eye(n)
        elif method == "inverse_variance":
            stds = self.empirical_stds
            return np.diag(1.0 / (stds ** 2))
        elif method == "diagonal":
            stds = self.empirical_stds
            return np.diag(1.0 / stds)
        return np.eye(n)

    def __len__(self) -> int:
        return len(self._moments)


def default_us_moments() -> MomentSet:
    """Default moment set for US macroeconomic calibration.

    Empirical values are approximate US averages (stylized facts).
    """
    ms = MomentSet("us_macro")

    # GDP
    ms.add(MomentDefinition(
        name="gdp_growth_mean",
        description="Average GDP growth rate (quarterly, annualized ~2-3%)",
        empirical_value=0.007,  # ~0.7% per quarter
        empirical_std=0.002,
        series_name="gdp_growth",
        statistic="mean",
    ))
    ms.add(MomentDefinition(
        name="gdp_growth_vol",
        description="GDP growth volatility",
        empirical_value=0.008,
        empirical_std=0.003,
        series_name="gdp_growth",
        statistic="std",
    ))
    ms.add(MomentDefinition(
        name="gdp_growth_autocorr",
        description="GDP growth persistence",
        empirical_value=0.3,
        empirical_std=0.1,
        series_name="gdp_growth",
        statistic="autocorr",
    ))

    # Unemployment
    ms.add(MomentDefinition(
        name="unemployment_mean",
        description="Average unemployment rate",
        empirical_value=0.055,
        empirical_std=0.01,
        series_name="unemployment_rate",
        statistic="mean",
    ))
    ms.add(MomentDefinition(
        name="unemployment_vol",
        description="Unemployment rate volatility",
        empirical_value=0.015,
        empirical_std=0.005,
        series_name="unemployment_rate",
        statistic="std",
    ))

    # Inflation
    ms.add(MomentDefinition(
        name="inflation_mean",
        description="Average inflation rate (per period)",
        empirical_value=0.005,  # ~0.5% per quarter
        empirical_std=0.003,
        series_name="inflation_rate",
        statistic="mean",
    ))
    ms.add(MomentDefinition(
        name="inflation_persistence",
        description="Inflation autocorrelation",
        empirical_value=0.6,
        empirical_std=0.1,
        series_name="inflation_rate",
        statistic="autocorr",
    ))

    # Credit
    ms.add(MomentDefinition(
        name="credit_to_gdp",
        description="Average credit-to-GDP ratio",
        empirical_value=0.5,
        empirical_std=0.2,
        compute_fn=lambda df: float(
            (df["total_loans_outstanding"].iloc[20:] / df["gdp"].iloc[20:].replace(0, np.nan)).mean()
        ) if "total_loans_outstanding" in df.columns and "gdp" in df.columns else np.nan,
    ))

    # Inequality
    ms.add(MomentDefinition(
        name="gini_wealth",
        description="Wealth Gini coefficient",
        empirical_value=0.4,
        empirical_std=0.1,
        series_name="gini_deposits",
        statistic="mean",
    ))

    # Inventory
    ms.add(MomentDefinition(
        name="inventory_output_ratio",
        description="Inventory-to-output ratio",
        empirical_value=0.15,
        empirical_std=0.05,
        compute_fn=lambda df: float(
            (df["total_inventory"].iloc[20:] / df["total_production"].iloc[20:].replace(0, np.nan)).mean()
        ) if "total_inventory" in df.columns and "total_production" in df.columns else np.nan,
    ))

    return ms
