"""BEA (Bureau of Economic Analysis) client.

Supports:
- NIPA table retrieval
- Dataset/table/parameter discovery
- JSON response parsing into normalized long-form tables
- Rate-limit handling with retries
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".econosim" / "cache" / "bea"
_BEA_BASE_URL = "https://apps.bea.gov/api/data"


class BeaClient:
    """Client for the BEA API with caching and retries."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
        max_retries: int = 3,
        backoff_base: float = 1.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("BEA_API_KEY", "")
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def _cache_key(self, params: dict[str, Any]) -> str:
        key_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        """Make a BEA API request with caching and retries."""
        cache_file = self.cache_dir / f"{self._cache_key(params)}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.debug(f"BEA cache read failed: {e}")

        if not self.api_key:
            raise ValueError(
                "BEA API key required. Set BEA_API_KEY env var or pass api_key to BeaClient."
            )

        import urllib.request
        import urllib.parse
        import urllib.error

        full_params = {**params, "UserID": self.api_key, "ResultFormat": "JSON"}
        url = f"{_BEA_BASE_URL}?" + urllib.parse.urlencode(full_params)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "econosim/0.1"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())

                with open(cache_file, "w") as f:
                    json.dump(data, f)
                return data

            except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
                last_error = e
                wait = self.backoff_base * (2 ** attempt)
                logger.warning(f"BEA request failed: {e}, retrying in {wait:.1f}s")
                time.sleep(wait)

        raise ConnectionError(f"BEA request failed after {self.max_retries} attempts: {last_error}")

    def get_dataset_list(self) -> list[dict[str, Any]]:
        """Get list of available BEA datasets."""
        data = self._request({"method": "GetDataSetList"})
        results = data.get("BEAAPI", {}).get("Results", {})
        return results.get("Dataset", [])

    def get_parameter_list(self, dataset_name: str) -> list[dict[str, Any]]:
        """Get parameters for a BEA dataset."""
        data = self._request({
            "method": "GetParameterList",
            "DatasetName": dataset_name,
        })
        results = data.get("BEAAPI", {}).get("Results", {})
        return results.get("Parameter", [])

    def get_parameter_values(
        self, dataset_name: str, parameter_name: str
    ) -> list[dict[str, Any]]:
        """Get valid values for a dataset parameter."""
        data = self._request({
            "method": "GetParameterValues",
            "DatasetName": dataset_name,
            "ParameterName": parameter_name,
        })
        results = data.get("BEAAPI", {}).get("Results", {})
        return results.get("ParamValue", [])

    def get_nipa_table(
        self,
        table_name: str,
        frequency: str = "Q",
        year: str | list[str] = "X",
    ) -> pd.DataFrame:
        """Fetch a NIPA table and return as normalized DataFrame.

        Args:
            table_name: NIPA table identifier (e.g. 'T10101' for GDP)
            frequency: 'A' (annual), 'Q' (quarterly), 'M' (monthly)
            year: Year(s) to fetch, 'X' for all available

        Returns:
            Long-form DataFrame with columns: table, line, description, period, value
        """
        if isinstance(year, list):
            year = ",".join(str(y) for y in year)

        data = self._request({
            "method": "GetData",
            "DatasetName": "NIPA",
            "TableName": table_name,
            "Frequency": frequency,
            "Year": year,
        })

        results = data.get("BEAAPI", {}).get("Results", {})
        raw_data = results.get("Data", [])

        if not raw_data:
            return pd.DataFrame(columns=["table", "line", "description", "period", "value"])

        rows = []
        for item in raw_data:
            val_str = item.get("DataValue", "").replace(",", "")
            try:
                value = float(val_str) if val_str else None
            except ValueError:
                value = None

            rows.append({
                "table": table_name,
                "line": item.get("LineNumber", ""),
                "description": item.get("LineDescription", ""),
                "period": item.get("TimePeriod", ""),
                "value": value,
                "units": item.get("UNIT_MULT", ""),
                "metric": item.get("METRIC_NAME", ""),
            })

        return pd.DataFrame(rows)

    def get_gdp_components(
        self,
        frequency: str = "Q",
        year: str = "X",
    ) -> pd.DataFrame:
        """Convenience: fetch GDP and components (NIPA Table 1.1.1)."""
        return self.get_nipa_table("T10101", frequency=frequency, year=year)

    def get_price_indices(
        self,
        frequency: str = "Q",
        year: str = "X",
    ) -> pd.DataFrame:
        """Convenience: fetch GDP price indices (NIPA Table 1.1.4)."""
        return self.get_nipa_table("T10104", frequency=frequency, year=year)


# Key NIPA tables for macro calibration
BEA_NIPA_TABLES = {
    "gdp_components": "T10101",        # GDP and major components (nominal)
    "gdp_real": "T10106",              # Real GDP and components (chained dollars)
    "gdp_deflator": "T10104",          # Price indices for GDP components
    "personal_income": "T20100",       # Personal income and its disposition
    "govt_receipts": "T30100",         # Government current receipts and expenditures
    "saving_investment": "T50100",     # Saving and investment
}
