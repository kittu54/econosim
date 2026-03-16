"""Tests for data storage and registry."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from econosim.data.storage.registry import DataStore, DatasetVersion, DataRegistry


class TestDataStore:
    def _make_store(self, tmp_path):
        return DataStore(store_dir=tmp_path)

    def test_save_and_load(self, tmp_path):
        store = self._make_store(tmp_path)
        df = pd.DataFrame({
            "gdp": [100, 105, 110],
            "inflation": [0.02, 0.03, 0.025],
        }, index=pd.date_range("2020-01-01", periods=3, freq="QE"))

        version = store.save("test_gdp", df, source="test")
        assert version.dataset_id == "test_gdp"
        assert version.num_observations == 3
        assert version.content_hash != ""

        loaded = store.load("test_gdp")
        assert len(loaded) == 3
        assert "gdp" in loaded.columns

    def test_load_specific_version(self, tmp_path):
        store = self._make_store(tmp_path)
        df1 = pd.DataFrame({"x": [1, 2, 3]})
        df2 = pd.DataFrame({"x": [4, 5, 6]})

        v1 = store.save("test", df1, source="test")
        v2 = store.save("test", df2, source="test")

        loaded_v1 = store.load("test", v1.version_id)
        assert loaded_v1["x"].iloc[0] == 1

        loaded_latest = store.load("test")
        assert loaded_latest["x"].iloc[0] == 4

    def test_list_datasets(self, tmp_path):
        store = self._make_store(tmp_path)
        df = pd.DataFrame({"x": [1]})
        store.save("ds1", df, source="test")
        store.save("ds2", df, source="test")

        datasets = store.list_datasets()
        assert "ds1" in datasets
        assert "ds2" in datasets

    def test_load_metadata(self, tmp_path):
        store = self._make_store(tmp_path)
        df = pd.DataFrame({"x": [1, 2]})
        version = store.save("test", df, source="fred", series_ids=["GDP"])

        meta = store.load_metadata("test")
        assert meta.source == "fred"
        assert meta.series_ids == ["GDP"]

    def test_nonexistent_dataset(self, tmp_path):
        store = self._make_store(tmp_path)
        with pytest.raises(FileNotFoundError):
            store.load("nonexistent")

    def test_content_hash_changes(self, tmp_path):
        store = self._make_store(tmp_path)
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"x": [3, 4]})

        h1 = store._content_hash(df1)
        h2 = store._content_hash(df2)
        assert h1 != h2


class TestDatasetVersion:
    def test_to_dict(self):
        v = DatasetVersion(
            dataset_id="test",
            version_id="v1",
            source="fred",
            series_ids=["GDP"],
            num_observations=100,
        )
        d = v.to_dict()
        assert d["dataset_id"] == "test"
        assert d["source"] == "fred"


class TestDataRegistry:
    def test_register_and_pull(self, tmp_path):
        store = DataStore(store_dir=tmp_path)
        registry = DataRegistry(store=store)

        def pull_fn():
            return pd.DataFrame({"gdp": [100, 105]})

        registry.register_pipeline("test_gdp", pull_fn, "Test GDP data")
        version = registry.pull("test_gdp")
        assert version.dataset_id == "test_gdp"

        loaded = registry.load("test_gdp")
        assert len(loaded) == 2

    def test_list_datasets(self, tmp_path):
        store = DataStore(store_dir=tmp_path)
        registry = DataRegistry(store=store)
        registry.register_pipeline("ds1", lambda: pd.DataFrame({"x": [1]}), "test")

        datasets = registry.list_datasets()
        assert "ds1" in datasets
