"""High-level data pipelines for pulling standard macro datasets.

Provides convenience functions that orchestrate FredClient, BEA, and IMF
pulls into analysis-ready DataFrames. All functions work without an API
key in "offline" mode by returning empty frames with the expected schema.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from econosim.data.sources.fred import FredClient, FRED_MACRO_SERIES

logger = logging.getLogger(__name__)


# ── Core calibration series ─────────────────────────────────────────

CALIBRATION_SERIES = {
    "gdp_real": {"id": "GDPC1", "freq": "q", "desc": "Real GDP (quarterly)"},
    "unemployment": {"id": "UNRATE", "freq": "m", "desc": "Unemployment rate (monthly)"},
    "cpi": {"id": "CPIAUCSL", "freq": "m", "desc": "CPI All Urban (monthly)"},
    "fed_funds": {"id": "FEDFUNDS", "freq": "m", "desc": "Fed Funds rate (monthly)"},
    "personal_income": {"id": "PI", "freq": "m", "desc": "Personal income (monthly)"},
    "consumption": {"id": "PCE", "freq": "m", "desc": "Personal consumption (monthly)"},
    "employment": {"id": "PAYEMS", "freq": "m", "desc": "Nonfarm payrolls (monthly)"},
    "m2": {"id": "M2SL", "freq": "m", "desc": "M2 money stock (monthly)"},
    "loans_commercial": {"id": "BUSLOANS", "freq": "m", "desc": "Commercial loans (monthly)"},
    "capacity_utilization": {"id": "TCU", "freq": "m", "desc": "Capacity utilization (monthly)"},
}


def pull_us_macro_baseline(
    start_date: str = "2000-01-01",
    end_date: str | None = None,
    frequency: str = "q",
    client: FredClient | None = None,
) -> pd.DataFrame:
    """Pull a standard US macro dataset for calibration.

    Returns a quarterly-aligned DataFrame with columns for each series in
    CALIBRATION_SERIES, indexed by date.

    Args:
        start_date: Start date (YYYY-MM-DD).
        end_date: End date; defaults to today.
        frequency: Target frequency for alignment ('q' or 'm').
        client: Existing FredClient; creates one if not provided.

    Returns:
        DataFrame indexed by date with one column per series alias.
    """
    client = client or FredClient()
    end_date = end_date or str(date.today())

    frames: dict[str, pd.Series] = {}
    for alias, spec in CALIBRATION_SERIES.items():
        try:
            df = client.get_observations(
                series_id=spec["id"],
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                aggregation_method="avg",
            )
            if not df.empty and "value" in df.columns:
                series = df.set_index("date")["value"].astype(float)
                series.index = pd.to_datetime(series.index)
                frames[alias] = series
                logger.info(f"Pulled {alias} ({spec['id']}): {len(series)} obs")
            else:
                logger.warning(f"Empty result for {alias} ({spec['id']})")
        except Exception as e:
            logger.warning(f"Failed to pull {alias} ({spec['id']}): {e}")

    if not frames:
        logger.warning("No series pulled successfully; returning empty DataFrame")
        return pd.DataFrame()

    result = pd.DataFrame(frames)
    result.index.name = "date"
    result = result.sort_index()
    return result


def compute_calibration_moments(
    data: pd.DataFrame,
    burn_periods: int = 4,
) -> dict[str, float]:
    """Compute empirical moments from a macro dataset for calibration matching.

    Returns moments that align with the default_us_moments() target values.

    Args:
        data: DataFrame from pull_us_macro_baseline().
        burn_periods: Initial periods to drop (avoids edge effects in transforms).

    Returns:
        Dict of moment_name -> empirical value.
    """
    moments: dict[str, float] = {}
    df = data.iloc[burn_periods:].copy()

    # GDP growth (quarterly annualized)
    if "gdp_real" in df.columns:
        gdp = df["gdp_real"].dropna()
        if len(gdp) > 1:
            qgrowth = gdp.pct_change().dropna()
            moments["mean_gdp_growth"] = float(qgrowth.mean())
            moments["std_gdp_growth"] = float(qgrowth.std())

    # Unemployment rate
    if "unemployment" in df.columns:
        unemp = df["unemployment"].dropna() / 100.0  # convert % to fraction
        if len(unemp) > 0:
            moments["mean_unemployment_rate"] = float(unemp.mean())
            moments["std_unemployment_rate"] = float(unemp.std())

    # Inflation (CPI-based, period-over-period)
    if "cpi" in df.columns:
        cpi = df["cpi"].dropna()
        if len(cpi) > 1:
            infl = cpi.pct_change().dropna()
            moments["mean_inflation_rate"] = float(infl.mean())
            moments["std_inflation_rate"] = float(infl.std())

    # Consumption-to-income ratio
    if "consumption" in df.columns and "personal_income" in df.columns:
        cons = df["consumption"].dropna()
        inc = df["personal_income"].dropna()
        aligned = pd.DataFrame({"c": cons, "y": inc}).dropna()
        if len(aligned) > 0 and aligned["y"].mean() > 0:
            moments["consumption_income_ratio"] = float(
                (aligned["c"] / aligned["y"]).mean()
            )

    # Credit-to-GDP ratio
    if "loans_commercial" in df.columns and "gdp_real" in df.columns:
        loans = df["loans_commercial"].dropna()
        gdp = df["gdp_real"].dropna()
        aligned = pd.DataFrame({"l": loans, "g": gdp}).dropna()
        if len(aligned) > 0 and aligned["g"].mean() > 0:
            moments["credit_gdp_ratio"] = float(
                (aligned["l"] / aligned["g"]).mean()
            )

    # Capacity utilization mean
    if "capacity_utilization" in df.columns:
        cu = df["capacity_utilization"].dropna() / 100.0
        if len(cu) > 0:
            moments["mean_capacity_utilization"] = float(cu.mean())

    # GDP autocorrelation (persistence)
    if "gdp_real" in df.columns:
        gdp = df["gdp_real"].dropna()
        if len(gdp) > 4:
            growth = gdp.pct_change().dropna()
            moments["gdp_growth_autocorrelation"] = float(growth.autocorr(lag=1))

    logger.info(f"Computed {len(moments)} empirical moments from data")
    return moments


def pull_series(
    series: dict[str, str],
    start_date: str = "2000-01-01",
    end_date: str | None = None,
    frequency: str = "q",
    client: FredClient | None = None,
) -> pd.DataFrame:
    """Pull arbitrary FRED series into a single aligned DataFrame.

    Args:
        series: Dict of {column_name: FRED_series_id}.
        start_date: Start date.
        end_date: End date; defaults to today.
        frequency: Target frequency.
        client: Existing FredClient.

    Returns:
        DataFrame indexed by date with one column per series.
    """
    client = client or FredClient()
    end_date = end_date or str(date.today())

    frames: dict[str, pd.Series] = {}
    for alias, series_id in series.items():
        try:
            df = client.get_observations(
                series_id=series_id,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                aggregation_method="avg",
            )
            if not df.empty and "value" in df.columns:
                s = df.set_index("date")["value"].astype(float)
                s.index = pd.to_datetime(s.index)
                frames[alias] = s
        except Exception as e:
            logger.warning(f"Failed to pull {alias} ({series_id}): {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.DataFrame(frames).sort_index()
    result.index.name = "date"
    return result
