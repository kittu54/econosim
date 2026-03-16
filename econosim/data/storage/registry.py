"""Dataset registry and versioned Parquet storage.

Provides reproducible, versioned data management with provenance tracking.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_DEFAULT_STORE_DIR = Path.home() / ".econosim" / "data"


class DatasetVersion(BaseModel):
    """Metadata for a versioned dataset snapshot."""

    dataset_id: str
    version_id: str
    source: str  # "fred", "bea", "imf", "synthetic"
    series_ids: list[str] = Field(default_factory=list)
    pull_timestamp: str = ""
    start_date: str = ""
    end_date: str = ""
    frequency: str = ""  # "D", "M", "Q", "A"
    num_observations: int = 0
    num_series: int = 0
    content_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class DataStore:
    """Versioned Parquet + metadata storage for curated datasets."""

    def __init__(self, store_dir: str | Path | None = None) -> None:
        self.store_dir = Path(store_dir) if store_dir else _DEFAULT_STORE_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _dataset_dir(self, dataset_id: str) -> Path:
        d = self.store_dir / dataset_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _content_hash(df: pd.DataFrame) -> str:
        """Compute a deterministic hash of DataFrame contents."""
        # Use a stable representation
        buf = df.to_csv(index=True).encode("utf-8")
        return hashlib.sha256(buf).hexdigest()[:16]

    def save(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        source: str,
        series_ids: list[str] | None = None,
        frequency: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> DatasetVersion:
        """Save a DataFrame as a versioned Parquet dataset.

        Returns DatasetVersion with provenance metadata.
        """
        content_hash = self._content_hash(df)
        timestamp = datetime.utcnow().isoformat()
        version_id = f"{dataset_id}_{content_hash}_{timestamp[:10]}"

        ddir = self._dataset_dir(dataset_id)
        parquet_path = ddir / f"{version_id}.parquet"
        meta_path = ddir / f"{version_id}.meta.json"

        # Determine date range
        start_date = ""
        end_date = ""
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            start_date = str(df.index.min().date())
            end_date = str(df.index.max().date())
        elif "date" in df.columns and len(df) > 0:
            start_date = str(pd.Timestamp(df["date"].min()).date())
            end_date = str(pd.Timestamp(df["date"].max()).date())

        version = DatasetVersion(
            dataset_id=dataset_id,
            version_id=version_id,
            source=source,
            series_ids=series_ids or [],
            pull_timestamp=timestamp,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            num_observations=len(df),
            num_series=len(series_ids) if series_ids else df.shape[1],
            content_hash=content_hash,
            metadata=metadata or {},
        )

        # Save Parquet
        df.to_parquet(parquet_path, engine="pyarrow" if _has_pyarrow() else "fastparquet")
        logger.info(f"Saved dataset {version_id} ({len(df)} rows) to {parquet_path}")

        # Save metadata
        with open(meta_path, "w") as f:
            json.dump(version.to_dict(), f, indent=2)

        # Update latest pointer
        latest_path = ddir / "latest.json"
        with open(latest_path, "w") as f:
            json.dump({"version_id": version_id, "parquet": str(parquet_path)}, f, indent=2)

        return version

    def load(self, dataset_id: str, version_id: str | None = None) -> pd.DataFrame:
        """Load a dataset version from Parquet. If version_id is None, loads latest."""
        ddir = self._dataset_dir(dataset_id)

        if version_id is None:
            latest_path = ddir / "latest.json"
            if not latest_path.exists():
                raise FileNotFoundError(f"No dataset found for '{dataset_id}'")
            with open(latest_path) as f:
                latest = json.load(f)
            version_id = latest["version_id"]

        parquet_path = ddir / f"{version_id}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Dataset version not found: {parquet_path}")

        return pd.read_parquet(parquet_path)

    def load_metadata(self, dataset_id: str, version_id: str | None = None) -> DatasetVersion:
        """Load metadata for a dataset version."""
        ddir = self._dataset_dir(dataset_id)

        if version_id is None:
            latest_path = ddir / "latest.json"
            if not latest_path.exists():
                raise FileNotFoundError(f"No dataset found for '{dataset_id}'")
            with open(latest_path) as f:
                latest = json.load(f)
            version_id = latest["version_id"]

        meta_path = ddir / f"{version_id}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        with open(meta_path) as f:
            return DatasetVersion(**json.load(f))

    def list_datasets(self) -> list[str]:
        """List all dataset IDs in the store."""
        return [
            d.name for d in self.store_dir.iterdir()
            if d.is_dir() and (d / "latest.json").exists()
        ]

    def list_versions(self, dataset_id: str) -> list[str]:
        """List all version IDs for a dataset."""
        ddir = self._dataset_dir(dataset_id)
        return [
            p.stem.replace(".meta", "")
            for p in ddir.glob("*.meta.json")
        ]


class DataRegistry:
    """High-level registry for managing curated macro datasets.

    Combines data source clients with the DataStore for versioned,
    reproducible data pipelines.
    """

    def __init__(self, store: DataStore | None = None) -> None:
        self.store = store or DataStore()
        self._pipelines: dict[str, Any] = {}

    def register_pipeline(
        self,
        dataset_id: str,
        pull_fn: Any,
        description: str = "",
    ) -> None:
        """Register a data pull pipeline for a dataset.

        Args:
            dataset_id: Unique identifier for this dataset
            pull_fn: Callable that returns a DataFrame when invoked
            description: Human-readable description
        """
        self._pipelines[dataset_id] = {
            "pull_fn": pull_fn,
            "description": description,
        }

    def pull(self, dataset_id: str, **kwargs: Any) -> DatasetVersion:
        """Execute a registered pipeline and save the result."""
        if dataset_id not in self._pipelines:
            raise KeyError(f"No pipeline registered for '{dataset_id}'")

        pipeline = self._pipelines[dataset_id]
        df = pipeline["pull_fn"](**kwargs)

        return self.store.save(
            dataset_id=dataset_id,
            df=df,
            source=dataset_id.split("_")[0] if "_" in dataset_id else "custom",
            metadata={"description": pipeline["description"], **kwargs},
        )

    def load(self, dataset_id: str, version_id: str | None = None) -> pd.DataFrame:
        """Load a dataset from the store."""
        return self.store.load(dataset_id, version_id)

    def list_datasets(self) -> dict[str, str]:
        """List all registered pipelines and stored datasets."""
        result = {}
        for did, info in self._pipelines.items():
            result[did] = info["description"]
        for did in self.store.list_datasets():
            if did not in result:
                result[did] = "(stored, no pipeline)"
        return result


def _has_pyarrow() -> bool:
    try:
        import pyarrow
        return True
    except ImportError:
        return False
