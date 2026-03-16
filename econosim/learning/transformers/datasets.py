"""Dataset preparation for transformer training.

Handles:
- Time series windowing/patching
- Mixed-frequency alignment
- Simulation state summary encoding
- Train/val/test splitting with temporal awareness (no leakage)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TimeSeriesWindow:
    """A single windowed sample for transformer training."""

    context: np.ndarray  # (context_length, n_features)
    target: np.ndarray   # (horizon, n_targets)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeSeriesDatasetConfig:
    """Configuration for time series dataset construction."""

    context_length: int = 60  # look-back window
    horizon: int = 12          # forecast horizon
    stride: int = 1            # step between windows
    target_columns: list[str] = field(default_factory=lambda: [
        "gdp", "unemployment_rate", "avg_price",
    ])
    feature_columns: list[str] = field(default_factory=lambda: [
        "gdp", "unemployment_rate", "avg_price", "total_loans_outstanding",
        "inflation_rate", "gdp_growth", "gini_deposits", "bank_capital_ratio",
    ])
    normalize: bool = True
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    # test_fraction = 1 - train - val


class TimeSeriesDataset:
    """Prepare windowed time series data for transformer training."""

    def __init__(self, config: TimeSeriesDatasetConfig) -> None:
        self.config = config
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def prepare(self, df: pd.DataFrame) -> dict[str, list[TimeSeriesWindow]]:
        """Convert a DataFrame into train/val/test windowed samples.

        IMPORTANT: Splits are strictly temporal (no shuffling) to prevent leakage.
        """
        cfg = self.config

        # Select columns
        features = [c for c in cfg.feature_columns if c in df.columns]
        targets = [c for c in cfg.target_columns if c in df.columns]

        if not features or not targets:
            return {"train": [], "val": [], "test": []}

        feat_data = df[features].values.astype(np.float32)
        target_data = df[targets].values.astype(np.float32)

        # Handle NaN
        feat_data = np.nan_to_num(feat_data, nan=0.0)
        target_data = np.nan_to_num(target_data, nan=0.0)

        # Normalize using training data statistics
        n = len(feat_data)
        n_train = int(n * cfg.train_fraction)
        n_val = int(n * cfg.val_fraction)

        if cfg.normalize:
            self._mean = feat_data[:n_train].mean(axis=0)
            self._std = feat_data[:n_train].std(axis=0)
            self._std[self._std < 1e-8] = 1.0
            feat_data = (feat_data - self._mean) / self._std

            target_mean = target_data[:n_train].mean(axis=0)
            target_std = target_data[:n_train].std(axis=0)
            target_std[target_std < 1e-8] = 1.0
            target_data = (target_data - target_mean) / target_std

        # Create windows
        total_window = cfg.context_length + cfg.horizon
        windows = []
        for i in range(0, n - total_window + 1, cfg.stride):
            context = feat_data[i:i + cfg.context_length]
            target = target_data[i + cfg.context_length:i + total_window]
            windows.append(TimeSeriesWindow(
                context=context,
                target=target,
                metadata={"start_idx": i},
            ))

        # Split temporally
        n_windows = len(windows)
        n_train_w = int(n_windows * cfg.train_fraction)
        n_val_w = int(n_windows * cfg.val_fraction)

        return {
            "train": windows[:n_train_w],
            "val": windows[n_train_w:n_train_w + n_val_w],
            "test": windows[n_train_w + n_val_w:],
        }

    def prepare_from_simulations(
        self,
        sim_dfs: list[pd.DataFrame],
    ) -> dict[str, list[TimeSeriesWindow]]:
        """Prepare dataset from multiple simulation runs (synthetic data augmentation)."""
        all_windows: dict[str, list[TimeSeriesWindow]] = {"train": [], "val": [], "test": []}

        for df in sim_dfs:
            splits = self.prepare(df)
            for split in ["train", "val", "test"]:
                all_windows[split].extend(splits[split])

        return all_windows


class EmulatorDataset:
    """Dataset for training simulation emulators.

    Maps: (parameter_vector, initial_conditions, shock_path) → macro_trajectory
    """

    @staticmethod
    def prepare(
        param_vectors: np.ndarray,
        trajectories: list[pd.DataFrame],
        target_columns: list[str],
        horizon: int = 60,
    ) -> dict[str, np.ndarray]:
        """Prepare emulator training data.

        Args:
            param_vectors: (n_runs, n_params) parameter settings
            trajectories: list of simulation output DataFrames
            target_columns: columns to predict
            horizon: max trajectory length

        Returns:
            {"inputs": (n_runs, n_params), "targets": (n_runs, horizon, n_targets)}
        """
        n_runs = len(trajectories)
        n_targets = len(target_columns)
        targets = np.zeros((n_runs, horizon, n_targets))

        for i, df in enumerate(trajectories):
            for j, col in enumerate(target_columns):
                if col in df.columns:
                    vals = df[col].values[:horizon]
                    targets[i, :len(vals), j] = vals

        return {
            "inputs": param_vectors,
            "targets": targets,
        }
