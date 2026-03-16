"""Tests for data source clients (FRED, BEA, IMF).

Tests client initialization, caching, and schema definitions
without making actual API calls.
"""

import json
import tempfile
from pathlib import Path

import pytest

from econosim.data.sources.fred import FredClient, FRED_MACRO_SERIES
from econosim.data.sources.bea import BeaClient, BEA_NIPA_TABLES
from econosim.data.sources.imf import ImfSdmxClient, IMF_IFS_INDICATORS


class TestFredClient:
    def test_init_default(self):
        client = FredClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.max_retries == 3

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "env_key")
        client = FredClient()
        assert client.api_key == "env_key"

    def test_cache_key_deterministic(self):
        client = FredClient(api_key="test")
        key1 = client._cache_key("series/observations", {"series_id": "GDP"})
        key2 = client._cache_key("series/observations", {"series_id": "GDP"})
        assert key1 == key2

    def test_cache_key_different_params(self):
        client = FredClient(api_key="test")
        key1 = client._cache_key("series/observations", {"series_id": "GDP"})
        key2 = client._cache_key("series/observations", {"series_id": "CPIAUCSL"})
        assert key1 != key2

    def test_cache_dir_created(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "fred_cache"
            client = FredClient(api_key="test", cache_dir=cache_dir)
            assert cache_dir.exists()

    def test_cache_save_and_load(self):
        with tempfile.TemporaryDirectory() as td:
            client = FredClient(api_key="test", cache_dir=td)
            data = {"observations": [{"date": "2020-01-01", "value": "100"}]}
            client._save_cache("test_endpoint", {"param": "val"}, data)
            loaded = client._cached_get("test_endpoint", {"param": "val"})
            assert loaded == data

    def test_cache_miss(self):
        with tempfile.TemporaryDirectory() as td:
            client = FredClient(api_key="test", cache_dir=td)
            loaded = client._cached_get("nonexistent", {})
            assert loaded is None

    def test_no_api_key_raises(self):
        client = FredClient(api_key="")
        with pytest.raises(ValueError, match="FRED API key required"):
            client._request("series", {"series_id": "GDP"})

    def test_macro_series_catalog(self):
        """Verify FRED_MACRO_SERIES has expected entries."""
        assert "gdp" in FRED_MACRO_SERIES
        assert "cpi" in FRED_MACRO_SERIES
        assert "unemployment" in FRED_MACRO_SERIES
        assert FRED_MACRO_SERIES["gdp"] == "GDP"
        assert FRED_MACRO_SERIES["unemployment"] == "UNRATE"
        assert len(FRED_MACRO_SERIES) >= 20


class TestBeaClient:
    def test_init_default(self):
        client = BeaClient(api_key="test_key")
        assert client.api_key == "test_key"

    def test_no_api_key_raises(self):
        client = BeaClient(api_key="")
        with pytest.raises(ValueError, match="BEA API key required"):
            client._request({"method": "GetDataSetList"})

    def test_nipa_tables_catalog(self):
        assert "gdp_components" in BEA_NIPA_TABLES
        assert BEA_NIPA_TABLES["gdp_components"] == "T10101"
        assert len(BEA_NIPA_TABLES) >= 5


class TestImfClient:
    def test_init_default(self):
        client = ImfSdmxClient()
        assert client.max_retries == 3
        assert client.rate_limit_delay == 1.0

    def test_cache_dir_created(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "imf_cache"
            client = ImfSdmxClient(cache_dir=cache_dir)
            assert cache_dir.exists()

    def test_ifs_indicators_catalog(self):
        assert "gdp_nominal" in IMF_IFS_INDICATORS
        assert "cpi" in IMF_IFS_INDICATORS
        assert "unemployment" in IMF_IFS_INDICATORS
        assert len(IMF_IFS_INDICATORS) >= 8
