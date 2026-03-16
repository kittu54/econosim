"""Tests for high-level data pipelines.

These tests run offline — they mock the FredClient to avoid needing an API key.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from econosim.data.pipelines import (
    CALIBRATION_SERIES,
    pull_us_macro_baseline,
    compute_calibration_moments,
    pull_series,
)


def _mock_fred_observations(series_id: str, **kwargs) -> pd.DataFrame:
    """Generate synthetic FRED data for testing."""
    n = 80
    dates = pd.date_range("2000-01-01", periods=n, freq="QS")
    rng = np.random.default_rng(hash(series_id) % 2**32)

    # Generate plausible values based on series
    if series_id in ("GDP", "GDPC1"):
        values = 15000 + np.cumsum(rng.normal(50, 20, n))
    elif series_id == "UNRATE":
        values = 5.0 + rng.normal(0, 0.5, n)
    elif series_id == "CPIAUCSL":
        values = 200 + np.cumsum(rng.normal(0.5, 0.2, n))
    elif series_id == "FEDFUNDS":
        values = 2.0 + rng.normal(0, 0.3, n)
    elif series_id in ("PI", "PCE"):
        values = 10000 + np.cumsum(rng.normal(30, 10, n))
    elif series_id == "PAYEMS":
        values = 140000 + np.cumsum(rng.normal(50, 30, n))
    elif series_id == "M2SL":
        values = 10000 + np.cumsum(rng.normal(40, 15, n))
    elif series_id == "BUSLOANS":
        values = 2000 + np.cumsum(rng.normal(10, 5, n))
    elif series_id == "TCU":
        values = 75.0 + rng.normal(0, 2, n)
    else:
        values = 100 + rng.normal(0, 5, n)

    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "value": values,
    })


@pytest.fixture
def mock_client():
    """Create a mock FredClient."""
    client = MagicMock()
    client.get_observations = MagicMock(side_effect=_mock_fred_observations)
    return client


class TestPullUsMacroBaseline:
    def test_returns_dataframe_with_expected_columns(self, mock_client):
        result = pull_us_macro_baseline(client=mock_client)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        # Should have columns matching CALIBRATION_SERIES keys
        for alias in CALIBRATION_SERIES:
            assert alias in result.columns, f"Missing column: {alias}"

    def test_index_is_datetime(self, mock_client):
        result = pull_us_macro_baseline(client=mock_client)
        assert pd.api.types.is_datetime64_any_dtype(result.index)

    def test_custom_date_range(self, mock_client):
        result = pull_us_macro_baseline(
            start_date="2010-01-01",
            end_date="2020-12-31",
            client=mock_client,
        )
        assert len(result) > 0

    def test_handles_empty_series(self):
        """If a series returns empty, it's skipped without error."""
        client = MagicMock()
        client.get_observations = MagicMock(return_value=pd.DataFrame())
        result = pull_us_macro_baseline(client=client)
        assert isinstance(result, pd.DataFrame)

    def test_handles_api_errors(self):
        """If all series fail, returns empty DataFrame."""
        client = MagicMock()
        client.get_observations = MagicMock(side_effect=Exception("API error"))
        result = pull_us_macro_baseline(client=client)
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestComputeCalibrationMoments:
    def test_computes_standard_moments(self, mock_client):
        data = pull_us_macro_baseline(client=mock_client)
        moments = compute_calibration_moments(data)

        expected_keys = [
            "mean_gdp_growth",
            "std_gdp_growth",
            "mean_unemployment_rate",
            "mean_inflation_rate",
            "consumption_income_ratio",
        ]
        for key in expected_keys:
            assert key in moments, f"Missing moment: {key}"
            assert np.isfinite(moments[key]), f"Non-finite moment: {key}={moments[key]}"

    def test_gdp_growth_is_reasonable(self, mock_client):
        data = pull_us_macro_baseline(client=mock_client)
        moments = compute_calibration_moments(data)
        # Quarterly GDP growth should be small (roughly -5% to +5%)
        assert -0.10 < moments["mean_gdp_growth"] < 0.10

    def test_unemployment_is_fraction(self, mock_client):
        data = pull_us_macro_baseline(client=mock_client)
        moments = compute_calibration_moments(data)
        # Should be a fraction (0 to 1), not percentage
        assert 0.0 < moments["mean_unemployment_rate"] < 1.0

    def test_handles_missing_columns(self):
        data = pd.DataFrame({"gdp_real": [100, 105, 110, 108, 112, 115]})
        moments = compute_calibration_moments(data, burn_periods=1)
        assert "mean_gdp_growth" in moments
        assert "mean_unemployment_rate" not in moments

    def test_handles_empty_data(self):
        moments = compute_calibration_moments(pd.DataFrame())
        assert isinstance(moments, dict)

    def test_autocorrelation_computed(self, mock_client):
        data = pull_us_macro_baseline(client=mock_client)
        moments = compute_calibration_moments(data)
        assert "gdp_growth_autocorrelation" in moments
        assert -1.0 <= moments["gdp_growth_autocorrelation"] <= 1.0


class TestPullSeries:
    def test_custom_series(self, mock_client):
        result = pull_series(
            {"gdp": "GDP", "cpi": "CPIAUCSL"},
            client=mock_client,
        )
        assert "gdp" in result.columns
        assert "cpi" in result.columns
        assert len(result) > 0

    def test_empty_on_failure(self):
        client = MagicMock()
        client.get_observations = MagicMock(side_effect=Exception("fail"))
        result = pull_series({"x": "INVALID"}, client=client)
        assert result.empty
