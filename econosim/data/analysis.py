"""Empirical data analysis pipeline for EconoSim.

Loads real-world economic data and produces structured analysis that
can drive simulations, calibrate parameters, and generate reports.
Bridges the gap between raw FRED/BEA data and simulation inputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EmpiricalAnalysis:
    """Structured analysis of empirical economic data."""

    data: pd.DataFrame
    moments: dict[str, float] = field(default_factory=dict)
    trends: dict[str, str] = field(default_factory=dict)
    regime: str = "normal"  # normal, recession, expansion, crisis
    key_events: list[str] = field(default_factory=list)
    summary_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    correlations: dict[str, float] = field(default_factory=dict)

    def to_narrative(self) -> str:
        """Convert analysis to natural language narrative."""
        lines = [f"Economic Regime: {self.regime.upper()}"]

        if self.trends:
            lines.append("\nTrends:")
            for var, trend in self.trends.items():
                lines.append(f"  - {var}: {trend}")

        if self.key_events:
            lines.append("\nKey Events:")
            for event in self.key_events:
                lines.append(f"  - {event}")

        if self.moments:
            lines.append("\nKey Statistics:")
            for name, val in self.moments.items():
                lines.append(f"  - {name}: {val:.4f}")

        return "\n".join(lines)


def analyze_simulation_data(
    df: pd.DataFrame,
    burn_in: int = 5,
) -> EmpiricalAnalysis:
    """Analyze simulation output DataFrame.

    Args:
        df: Simulation results DataFrame with standard metric columns.
        burn_in: Initial periods to skip for statistics.

    Returns:
        EmpiricalAnalysis with computed statistics, trends, and regime.
    """
    data = df.iloc[burn_in:].copy() if len(df) > burn_in else df.copy()

    analysis = EmpiricalAnalysis(data=data)

    # Summary statistics for each numeric column
    for col in data.select_dtypes(include=[np.number]).columns:
        series = data[col].dropna()
        if len(series) > 0:
            analysis.summary_stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "last": float(series.iloc[-1]),
            }

    # Compute key moments
    analysis.moments = _compute_moments(data)

    # Detect trends
    analysis.trends = _detect_trends(data)

    # Classify economic regime
    analysis.regime = _classify_regime(data)

    # Identify key events
    analysis.key_events = _detect_events(data)

    # Cross-variable correlations
    analysis.correlations = _compute_correlations(data)

    return analysis


def _compute_moments(df: pd.DataFrame) -> dict[str, float]:
    """Compute standard macro moments from simulation data."""
    moments: dict[str, float] = {}

    if "gdp" in df.columns:
        gdp = df["gdp"].dropna()
        if len(gdp) > 1:
            growth = gdp.pct_change().dropna()
            moments["mean_gdp_growth"] = float(growth.mean())
            moments["std_gdp_growth"] = float(growth.std())
            moments["gdp_final"] = float(gdp.iloc[-1])
            if len(growth) > 1:
                moments["gdp_growth_autocorr"] = float(growth.autocorr(lag=1))

    if "unemployment_rate" in df.columns:
        unemp = df["unemployment_rate"].dropna()
        if len(unemp) > 0:
            moments["mean_unemployment"] = float(unemp.mean())
            moments["std_unemployment"] = float(unemp.std())

    if "avg_price" in df.columns:
        prices = df["avg_price"].dropna()
        if len(prices) > 1:
            inflation = prices.pct_change().dropna()
            moments["mean_inflation"] = float(inflation.mean())
            moments["std_inflation"] = float(inflation.std())

    if "gini_coefficient" in df.columns:
        gini = df["gini_coefficient"].dropna()
        if len(gini) > 0:
            moments["mean_gini"] = float(gini.mean())

    if "total_credit" in df.columns and "gdp" in df.columns:
        credit = df["total_credit"].dropna()
        gdp = df["gdp"].dropna()
        aligned = pd.DataFrame({"c": credit, "g": gdp}).dropna()
        if len(aligned) > 0 and aligned["g"].mean() > 0:
            moments["credit_gdp_ratio"] = float((aligned["c"] / aligned["g"]).mean())

    return moments


def _detect_trends(df: pd.DataFrame) -> dict[str, str]:
    """Detect directional trends in key variables."""
    trends: dict[str, str] = {}

    key_vars = ["gdp", "unemployment_rate", "avg_price", "gini_coefficient", "total_credit"]

    for var in key_vars:
        if var not in df.columns:
            continue
        series = df[var].dropna()
        if len(series) < 4:
            continue

        # Compare first quarter to last quarter
        n = max(1, len(series) // 4)
        early = series.iloc[:n].mean()
        late = series.iloc[-n:].mean()

        if early == 0:
            trends[var] = "stable (from zero)"
            continue

        pct_change = (late - early) / abs(early)

        if pct_change > 0.1:
            trends[var] = f"rising (+{pct_change:.1%})"
        elif pct_change < -0.1:
            trends[var] = f"falling ({pct_change:.1%})"
        else:
            trends[var] = f"stable ({pct_change:+.1%})"

    return trends


def _classify_regime(df: pd.DataFrame) -> str:
    """Classify the current economic regime."""
    if "gdp" not in df.columns or len(df) < 4:
        return "normal"

    gdp = df["gdp"].dropna()
    if len(gdp) < 4:
        return "normal"

    recent_growth = gdp.iloc[-4:].pct_change().dropna()
    avg_growth = float(recent_growth.mean()) if len(recent_growth) > 0 else 0

    unemp = 0.0
    if "unemployment_rate" in df.columns:
        unemp_series = df["unemployment_rate"].dropna()
        if len(unemp_series) > 0:
            unemp = float(unemp_series.iloc[-1])

    if avg_growth < -0.02 and unemp > 0.1:
        return "crisis"
    elif avg_growth < -0.005 or unemp > 0.08:
        return "recession"
    elif avg_growth > 0.03:
        return "expansion"
    else:
        return "normal"


def _detect_events(df: pd.DataFrame) -> list[str]:
    """Detect notable economic events in the data."""
    events: list[str] = []

    if "gdp" in df.columns and len(df) > 2:
        gdp = df["gdp"].dropna()
        if len(gdp) > 2:
            growth = gdp.pct_change().dropna()
            # GDP contractions
            contractions = growth[growth < -0.02]
            if len(contractions) > 0:
                events.append(
                    f"GDP contraction detected in {len(contractions)} period(s), "
                    f"worst: {float(contractions.min()):.1%}"
                )

    if "unemployment_rate" in df.columns:
        unemp = df["unemployment_rate"].dropna()
        if len(unemp) > 0:
            max_unemp = float(unemp.max())
            if max_unemp > 0.15:
                events.append(f"High unemployment spike: {max_unemp:.1%}")

    if "avg_price" in df.columns and len(df) > 2:
        prices = df["avg_price"].dropna()
        if len(prices) > 2:
            inflation = prices.pct_change().dropna()
            max_inflation = float(inflation.max()) if len(inflation) > 0 else 0
            if max_inflation > 0.05:
                events.append(f"Inflation spike: {max_inflation:.1%} per period")
            min_inflation = float(inflation.min()) if len(inflation) > 0 else 0
            if min_inflation < -0.05:
                events.append(f"Deflation episode: {min_inflation:.1%} per period")

    return events


def _compute_correlations(df: pd.DataFrame) -> dict[str, float]:
    """Compute key cross-variable correlations."""
    pairs = [
        ("gdp", "unemployment_rate"),
        ("gdp", "avg_price"),
        ("gdp", "total_credit"),
        ("unemployment_rate", "gini_coefficient"),
    ]

    correlations: dict[str, float] = {}
    for var1, var2 in pairs:
        if var1 in df.columns and var2 in df.columns:
            s1 = df[var1].dropna()
            s2 = df[var2].dropna()
            aligned = pd.DataFrame({"a": s1, "b": s2}).dropna()
            if len(aligned) > 3:
                corr = float(aligned["a"].corr(aligned["b"]))
                correlations[f"{var1}_vs_{var2}"] = corr

    return correlations


def load_and_analyze(
    source: str = "simulation",
    config: Any | None = None,
    **kwargs: Any,
) -> EmpiricalAnalysis:
    """High-level entry point: load data and produce analysis.

    Args:
        source: "simulation" to run a new sim, "csv" to load from file,
                "fred" to pull from FRED.
        config: SimulationConfig for simulation source.
        **kwargs: Additional arguments (file_path for csv, etc.)

    Returns:
        EmpiricalAnalysis with full analysis results.
    """
    if source == "simulation":
        from econosim.config.schema import SimulationConfig
        from econosim.experiments.runner import run_experiment

        sim_config = config or SimulationConfig()
        result = run_experiment(sim_config)
        df = result["dataframe"]
        return analyze_simulation_data(df)

    elif source == "csv":
        file_path = kwargs.get("file_path", "data.csv")
        df = pd.read_csv(file_path)
        return analyze_simulation_data(df, burn_in=0)

    elif source == "fred":
        from econosim.data.pipelines import pull_us_macro_baseline, compute_calibration_moments

        df = pull_us_macro_baseline(**kwargs)
        analysis = EmpiricalAnalysis(data=df)
        if not df.empty:
            analysis.moments = compute_calibration_moments(df)
            analysis.trends = _detect_trends(df)
        return analysis

    else:
        raise ValueError(f"Unknown source: {source}. Use 'simulation', 'csv', or 'fred'.")
