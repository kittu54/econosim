"""Tests for transformer models and datasets."""

import numpy as np
import pandas as pd
import pytest

from econosim.learning.transformers.datasets import (
    TimeSeriesDataset,
    TimeSeriesDatasetConfig,
    TimeSeriesWindow,
    EmulatorDataset,
)
from econosim.learning.transformers.models import (
    TransformerConfig,
    MacroTransformer,
    ResidualForecaster,
    SimulationEmulator,
    NumpyTransformerBlock,
)
from econosim.learning.transformers.trainer import (
    NumpyTrainer,
    TrainingConfig,
    TransformerEvaluator,
)


class TestTimeSeriesDataset:
    def _make_df(self, n=200):
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "gdp": np.cumsum(rng.normal(10, 2, n)) + 1000,
            "unemployment_rate": 0.05 + 0.01 * rng.standard_normal(n),
            "avg_price": 10 + 0.5 * rng.standard_normal(n),
            "total_loans_outstanding": rng.uniform(100, 500, n),
            "inflation_rate": 0.02 + 0.005 * rng.standard_normal(n),
            "gdp_growth": 0.01 + 0.005 * rng.standard_normal(n),
            "gini_deposits": rng.beta(3, 7, n),
            "bank_capital_ratio": rng.beta(5, 50, n),
        })

    def test_prepare_creates_splits(self):
        config = TimeSeriesDatasetConfig(
            context_length=30, horizon=10, stride=5,
        )
        ds = TimeSeriesDataset(config)
        splits = ds.prepare(self._make_df())

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert len(splits["train"]) > 0

    def test_window_shapes(self):
        config = TimeSeriesDatasetConfig(
            context_length=20, horizon=5, stride=1,
            feature_columns=["gdp", "unemployment_rate"],
            target_columns=["gdp"],
        )
        ds = TimeSeriesDataset(config)
        splits = ds.prepare(self._make_df(100))

        if splits["train"]:
            w = splits["train"][0]
            assert w.context.shape == (20, 2)
            assert w.target.shape == (5, 1)

    def test_temporal_ordering(self):
        """Train windows should come before val/test windows."""
        config = TimeSeriesDatasetConfig(context_length=10, horizon=5, stride=1)
        ds = TimeSeriesDataset(config)
        splits = ds.prepare(self._make_df(100))

        if splits["train"] and splits["test"]:
            max_train = max(w.metadata["start_idx"] for w in splits["train"])
            min_test = min(w.metadata["start_idx"] for w in splits["test"])
            assert max_train <= min_test

    def test_multiple_simulations(self):
        config = TimeSeriesDatasetConfig(context_length=20, horizon=5, stride=5)
        ds = TimeSeriesDataset(config)
        dfs = [self._make_df(100) for _ in range(3)]
        splits = ds.prepare_from_simulations(dfs)
        assert len(splits["train"]) >= 3  # at least 1 per sim


class TestEmulatorDataset:
    def test_prepare(self):
        params = np.random.rand(10, 5)
        trajs = [pd.DataFrame({"gdp": np.random.rand(60), "urate": np.random.rand(60)}) for _ in range(10)]
        result = EmulatorDataset.prepare(params, trajs, ["gdp", "urate"], horizon=60)
        assert result["inputs"].shape == (10, 5)
        assert result["targets"].shape == (10, 60, 2)


class TestMacroTransformer:
    def test_forward_shape(self):
        config = TransformerConfig(
            n_features=4, context_length=40, horizon=10, n_targets=2,
            d_model=32, n_heads=2, n_layers=1, d_ff=64, patch_size=4,
        )
        model = MacroTransformer(config, seed=42)
        x = np.random.randn(40, 4).astype(np.float32)
        out = model.forward(x)
        assert out.shape == (10, 2)

    def test_deterministic(self):
        config = TransformerConfig(
            n_features=4, context_length=20, horizon=5, n_targets=2,
            d_model=16, n_heads=2, n_layers=1, d_ff=32, patch_size=4,
        )
        model = MacroTransformer(config, seed=42)
        x = np.random.randn(20, 4).astype(np.float32)
        out1 = model.forward(x)
        out2 = model.forward(x)
        np.testing.assert_array_equal(out1, out2)


class TestNumpyTransformerBlock:
    def test_forward_shape(self):
        rng = np.random.default_rng(42)
        block = NumpyTransformerBlock(d_model=16, n_heads=2, d_ff=32, rng=rng)
        x = np.random.randn(10, 16)
        out = block.forward(x)
        assert out.shape == (10, 16)


class TestResidualForecaster:
    def test_predict_residual(self):
        config = TransformerConfig(
            n_features=4, context_length=20, horizon=5, n_targets=2,
            d_model=16, n_heads=2, n_layers=1, d_ff=32, patch_size=4,
        )
        forecaster = ResidualForecaster(config, seed=42)
        history = np.random.randn(20, 4).astype(np.float32)
        sim_forecast = np.ones((5, 2)) * 100

        corrected = forecaster.predict_residual(history, sim_forecast)
        assert corrected.shape == (5, 2)
        # Correction should be close to original (residual is small/clipped)
        assert np.all(np.abs(corrected - sim_forecast) <= 0.5)


class TestSimulationEmulator:
    def test_emulate_shape(self):
        config = TransformerConfig(
            n_features=5, horizon=10, n_targets=3,
            d_model=16, d_ff=32,
        )
        emulator = SimulationEmulator(config, seed=42)
        params = np.array([0.8, 0.4, 8.0, 0.03, 0.02])
        out = emulator.emulate(params)
        assert out.shape == (10, 3)


class TestNumpyTrainer:
    def test_train_runs(self):
        config = TransformerConfig(
            n_features=4, context_length=20, horizon=5, n_targets=2,
            d_model=16, n_heads=2, n_layers=1, d_ff=32, patch_size=4,
        )
        model = MacroTransformer(config, seed=42)

        # Create dummy training data
        rng = np.random.default_rng(42)
        train_data = [
            TimeSeriesWindow(
                context=rng.standard_normal((20, 4)).astype(np.float32),
                target=rng.standard_normal((5, 2)).astype(np.float32),
            )
            for _ in range(10)
        ]
        val_data = train_data[:3]

        trainer = NumpyTrainer(
            model,
            TrainingConfig(max_epochs=5, batch_size=5, val_frequency=2),
        )
        result = trainer.train(train_data, val_data)

        assert result.num_epochs == 5
        assert len(result.train_losses) == 5
        assert result.elapsed_seconds > 0


class TestTransformerEvaluator:
    def test_evaluate(self):
        config = TransformerConfig(
            n_features=4, context_length=20, horizon=5, n_targets=2,
            d_model=16, n_heads=2, n_layers=1, d_ff=32, patch_size=4,
        )
        model = MacroTransformer(config, seed=42)

        rng = np.random.default_rng(42)
        test_data = [
            TimeSeriesWindow(
                context=rng.standard_normal((20, 4)).astype(np.float32),
                target=rng.standard_normal((5, 2)).astype(np.float32),
            )
            for _ in range(5)
        ]

        evaluator = TransformerEvaluator(model)
        metrics = evaluator.evaluate(test_data)

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert metrics["n_samples"] == 5
        assert metrics["mse"] > 0
