"""FRED (Federal Reserve Economic Data) client.

Supports:
- Observation retrieval with date ranges
- Vintage/real-time data via realtime_start/realtime_end
- Series metadata queries
- Frequency aggregation
- Rate-limit handling with retries/backoff
- Raw response caching
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".econosim" / "cache" / "fred"
_FRED_BASE_URL = "https://api.stlouisfed.org/fred"


class FredClient:
    """Client for the FRED API with caching, retries, and vintage support."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        rate_limit_delay: float = 0.5,
    ) -> None:
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: float = 0.0

    def _cache_key(self, endpoint: str, params: dict[str, Any]) -> str:
        """Generate a deterministic cache key from endpoint and params."""
        key_str = endpoint + "|" + json.dumps(params, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _cached_get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """Check cache for a previous response."""
        key = self._cache_key(endpoint, params)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    cached = json.load(f)
                logger.debug(f"Cache hit: {endpoint} {params}")
                return cached
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _save_cache(self, endpoint: str, params: dict[str, Any], data: dict[str, Any]) -> None:
        """Save response to cache."""
        key = self._cache_key(endpoint, params)
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except OSError as e:
            logger.warning(f"Failed to cache response: {e}")

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

    def _request(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        """Make a FRED API request with caching and retries.

        Returns parsed JSON response dict.
        """
        cached = self._cached_get(endpoint, params)
        if cached is not None:
            return cached

        if not self.api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY env var or pass api_key to FredClient."
            )

        import urllib.request
        import urllib.parse
        import urllib.error

        full_params = {**params, "api_key": self.api_key, "file_type": "json"}
        url = f"{_FRED_BASE_URL}/{endpoint}?" + urllib.parse.urlencode(full_params)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                self._last_request_time = time.monotonic()

                req = urllib.request.Request(url, headers={"User-Agent": "econosim/0.1"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())

                self._save_cache(endpoint, params, data)
                return data

            except urllib.error.HTTPError as e:
                last_error = e
                if e.code == 429:  # Rate limited
                    wait = self.backoff_base * (2 ** attempt)
                    logger.warning(f"Rate limited, waiting {wait:.1f}s")
                    time.sleep(wait)
                elif e.code >= 500:
                    wait = self.backoff_base * (2 ** attempt)
                    logger.warning(f"Server error {e.code}, retrying in {wait:.1f}s")
                    time.sleep(wait)
                else:
                    raise
            except (urllib.error.URLError, OSError) as e:
                last_error = e
                wait = self.backoff_base * (2 ** attempt)
                logger.warning(f"Request failed: {e}, retrying in {wait:.1f}s")
                time.sleep(wait)

        raise ConnectionError(
            f"FRED request failed after {self.max_retries} attempts: {last_error}"
        )

    def get_series_info(self, series_id: str) -> dict[str, Any]:
        """Get metadata for a FRED series."""
        data = self._request("series", {"series_id": series_id})
        seriess = data.get("seriess", [])
        if not seriess:
            raise ValueError(f"Series '{series_id}' not found")
        return seriess[0]

    def get_observations(
        self,
        series_id: str,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
        frequency: str | None = None,
        aggregation_method: str | None = None,
        units: str | None = None,
        realtime_start: str | date | None = None,
        realtime_end: str | date | None = None,
    ) -> pd.DataFrame:
        """Fetch observations for a FRED series.

        Args:
            series_id: FRED series identifier (e.g. 'GDP', 'CPIAUCSL', 'UNRATE')
            start_date: Observation start date (YYYY-MM-DD)
            end_date: Observation end date (YYYY-MM-DD)
            frequency: Frequency aggregation ('m', 'q', 'a')
            aggregation_method: How to aggregate ('avg', 'sum', 'eop')
            units: Transformation ('lin', 'chg', 'ch1', 'pch', 'pc1', 'pca', 'cch', 'cca', 'log')
            realtime_start: Real-time period start (for vintage data)
            realtime_end: Real-time period end (for vintage data)

        Returns:
            DataFrame with columns: date, value, realtime_start, realtime_end
        """
        params: dict[str, Any] = {"series_id": series_id}
        if start_date:
            params["observation_start"] = str(start_date)
        if end_date:
            params["observation_end"] = str(end_date)
        if frequency:
            params["frequency"] = frequency
        if aggregation_method:
            params["aggregation_method"] = aggregation_method
        if units:
            params["units"] = units
        if realtime_start:
            params["realtime_start"] = str(realtime_start)
        if realtime_end:
            params["realtime_end"] = str(realtime_end)

        data = self._request("series/observations", params)
        observations = data.get("observations", [])

        if not observations:
            return pd.DataFrame(columns=["date", "value", "realtime_start", "realtime_end"])

        rows = []
        for obs in observations:
            val = obs.get("value", ".")
            rows.append({
                "date": pd.Timestamp(obs["date"]),
                "value": float(val) if val != "." else np.nan,
                "realtime_start": obs.get("realtime_start", ""),
                "realtime_end": obs.get("realtime_end", ""),
            })

        df = pd.DataFrame(rows)
        df = df.set_index("date").sort_index()
        return df

    def get_vintage_dates(self, series_id: str) -> list[str]:
        """Get all vintage dates for a series (for revision-aware analysis)."""
        data = self._request("series/vintagedates", {"series_id": series_id})
        return data.get("vintage_dates", [])

    def get_vintage(
        self,
        series_id: str,
        vintage_date: str | date,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
    ) -> pd.DataFrame:
        """Get observations as they were known on a specific vintage date."""
        return self.get_observations(
            series_id=series_id,
            start_date=start_date,
            end_date=end_date,
            realtime_start=str(vintage_date),
            realtime_end=str(vintage_date),
        )

    def search_series(self, search_text: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search FRED for series matching a text query."""
        data = self._request(
            "series/search",
            {"search_text": search_text, "limit": limit},
        )
        return data.get("seriess", [])


# Standard macro series used for calibration / forecasting
FRED_MACRO_SERIES = {
    "gdp": "GDP",                        # Gross Domestic Product (quarterly, billions)
    "gdp_real": "GDPC1",                 # Real GDP (quarterly, billions, chained 2017$)
    "cpi": "CPIAUCSL",                   # CPI for All Urban Consumers (monthly)
    "pce_deflator": "PCEPI",             # PCE Price Index (monthly)
    "unemployment": "UNRATE",            # Unemployment Rate (monthly, %)
    "employment": "PAYEMS",              # Total Nonfarm Payrolls (monthly, thousands)
    "fed_funds": "FEDFUNDS",             # Federal Funds Rate (monthly, %)
    "m2": "M2SL",                        # M2 Money Stock (monthly, billions)
    "loans_commercial": "BUSLOANS",      # Commercial and Industrial Loans (weekly→monthly)
    "govt_debt": "GFDEBTN",             # Federal Debt: Total Public Debt (quarterly)
    "govt_surplus": "FYFSD",             # Federal Surplus or Deficit (annual)
    "personal_income": "PI",             # Personal Income (monthly, billions)
    "consumption": "PCE",                # Personal Consumption Expenditures (monthly, billions)
    "investment": "GPDI",                # Gross Private Domestic Investment (quarterly, billions)
    "govt_spending": "GCE",              # Government Consumption & Investment (quarterly, billions)
    "industrial_production": "INDPRO",   # Industrial Production Index (monthly)
    "capacity_utilization": "TCU",       # Total Capacity Utilization (monthly, %)
    "housing_starts": "HOUST",           # Housing Starts (monthly, thousands)
    "sp500": "SP500",                    # S&P 500 Index (daily→monthly)
    "yield_10y": "GS10",                # 10-Year Treasury Yield (monthly, %)
    "yield_3m": "TB3MS",                # 3-Month Treasury Bill Rate (monthly, %)
    "bank_credit": "TOTBKCR",           # Bank Credit, All Commercial Banks
    "charge_off_rate": "DRALACBS",      # Delinquency Rate on All Loans
    "gini_income": "SIPOVGINIUSA",      # Gini Index for the US (annual)
}
