"""IMF SDMX client for cross-country macroeconomic data.

Supports:
- SDMX 2.1 REST API
- Dataset structure discovery (dataflows, dimensions)
- Panel/country data ingestion
- Normalization to internal tabular schema
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".econosim" / "cache" / "imf"
_IMF_BASE_URL = "http://dataservices.imf.org/REST/SDMX_JSON.svc"


class ImfSdmxClient:
    """Client for the IMF SDMX JSON API with caching and retries."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        rate_limit_delay: float = 1.0,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: float = 0.0

    def _cache_key(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

    def _request(self, path: str) -> dict[str, Any]:
        """Make an IMF SDMX request with caching and retries."""
        url = f"{_IMF_BASE_URL}/{path}"
        cache_file = self.cache_dir / f"{self._cache_key(url)}.json"

        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        import urllib.request
        import urllib.error

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                self._last_request_time = time.monotonic()

                req = urllib.request.Request(url, headers={"User-Agent": "econosim/0.1"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode())

                with open(cache_file, "w") as f:
                    json.dump(data, f)
                return data

            except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
                last_error = e
                wait = self.backoff_base * (2 ** attempt)
                logger.warning(f"IMF request failed: {e}, retrying in {wait:.1f}s")
                time.sleep(wait)

        raise ConnectionError(
            f"IMF request failed after {self.max_retries} attempts: {last_error}"
        )

    def get_dataflows(self) -> list[dict[str, str]]:
        """Get list of available IMF datasets (dataflows)."""
        data = self._request("Dataflow")
        structure = data.get("Structure", {})
        dataflows = structure.get("Dataflows", {}).get("Dataflow", [])
        results = []
        for df in dataflows:
            name = df.get("Name", {})
            if isinstance(name, dict):
                name = name.get("#text", str(name))
            results.append({
                "id": df.get("@id", ""),
                "name": str(name),
            })
        return results

    def get_data_structure(self, dataset_id: str) -> dict[str, Any]:
        """Get the structure (dimensions, codes) of a dataset."""
        data = self._request(f"DataStructure/{dataset_id}")
        return data.get("Structure", {})

    def get_dimension_codes(self, dataset_id: str, dimension: str) -> list[dict[str, str]]:
        """Get code list for a specific dimension of a dataset."""
        data = self._request(f"CodeList/{dataset_id}_{dimension}")
        structure = data.get("Structure", {})
        code_lists = structure.get("CodeLists", {}).get("CodeList", {})
        codes = code_lists.get("Code", [])
        if isinstance(codes, dict):
            codes = [codes]
        results = []
        for code in codes:
            desc = code.get("Description", {})
            if isinstance(desc, dict):
                desc = desc.get("#text", str(desc))
            results.append({
                "code": code.get("@value", ""),
                "description": str(desc),
            })
        return results

    def get_series(
        self,
        dataset_id: str,
        dimensions: dict[str, str | list[str]],
        start_period: str | None = None,
        end_period: str | None = None,
    ) -> pd.DataFrame:
        """Fetch time series data from an IMF dataset.

        Args:
            dataset_id: IMF dataset identifier (e.g. 'IFS' for International Financial Statistics)
            dimensions: Dimension filters, e.g. {'CL_FREQ': 'A', 'CL_AREA_IFS': 'US'}
            start_period: Start period (e.g. '2000')
            end_period: End period (e.g. '2023')

        Returns:
            DataFrame with columns: country, indicator, period, value
        """
        # Build dimension key string
        dim_parts = []
        for key, val in dimensions.items():
            if isinstance(val, list):
                val = "+".join(val)
            dim_parts.append(val)
        dim_key = ".".join(dim_parts)

        path = f"CompactData/{dataset_id}/{dim_key}"
        if start_period or end_period:
            params = []
            if start_period:
                params.append(f"startPeriod={start_period}")
            if end_period:
                params.append(f"endPeriod={end_period}")
            path += "?" + "&".join(params)

        data = self._request(path)
        dataset = data.get("CompactData", {}).get("DataSet", {})
        series_list = dataset.get("Series", [])
        if isinstance(series_list, dict):
            series_list = [series_list]

        rows = []
        for series in series_list:
            country = series.get("@REF_AREA", "")
            indicator = series.get("@INDICATOR", "")
            obs = series.get("Obs", [])
            if isinstance(obs, dict):
                obs = [obs]
            for ob in obs:
                val_str = ob.get("@OBS_VALUE", "")
                try:
                    value = float(val_str) if val_str else np.nan
                except ValueError:
                    value = np.nan
                rows.append({
                    "country": country,
                    "indicator": indicator,
                    "period": ob.get("@TIME_PERIOD", ""),
                    "value": value,
                })

        return pd.DataFrame(rows)

    def get_ifs_series(
        self,
        country: str | list[str],
        indicator: str | list[str],
        frequency: str = "Q",
        start_period: str | None = None,
        end_period: str | None = None,
    ) -> pd.DataFrame:
        """Convenience: fetch from International Financial Statistics (IFS).

        Args:
            country: ISO country code(s) (e.g. 'US', ['US', 'GB', 'JP'])
            indicator: IFS indicator code(s)
            frequency: 'A' (annual), 'Q' (quarterly), 'M' (monthly)
        """
        return self.get_series(
            dataset_id="IFS",
            dimensions={
                "CL_FREQ": frequency,
                "CL_AREA_IFS": country if isinstance(country, str) else "+".join(country),
                "CL_INDICATOR_IFS": (
                    indicator if isinstance(indicator, str) else "+".join(indicator)
                ),
            },
            start_period=start_period,
            end_period=end_period,
        )


# Key IMF IFS indicators
IMF_IFS_INDICATORS = {
    "gdp_nominal": "NGDP_XDC",          # GDP, nominal, domestic currency
    "gdp_real": "NGDP_R_XDC",           # GDP, real, domestic currency
    "cpi": "PCPI_IX",                    # Consumer Price Index
    "unemployment": "LUR_PT",           # Unemployment Rate
    "govt_revenue": "GGR_G01_GDP_PT",   # Government Revenue (% GDP)
    "govt_expenditure": "GGX_G01_GDP_PT",  # Government Expenditure (% GDP)
    "govt_debt": "GGXWDG_GDP_PT",       # Government Gross Debt (% GDP)
    "current_account": "BCA_GDP_BP6",   # Current Account Balance (% GDP)
    "broad_money": "FMB_XDC",           # Broad Money
    "policy_rate": "FPOLM_PA",          # Policy-related Interest Rate
}
