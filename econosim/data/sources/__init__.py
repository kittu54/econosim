"""Data source clients for FRED, BEA, and IMF."""

from econosim.data.sources.fred import FredClient
from econosim.data.sources.bea import BeaClient
from econosim.data.sources.imf import ImfSdmxClient

__all__ = ["FredClient", "BeaClient", "ImfSdmxClient"]
