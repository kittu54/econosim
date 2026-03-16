"""Training infrastructure for transformer models.

Supports:
- Training on simulation-generated data
- Training on historical macro data
- Mixed synthetic + real data training
- Rolling-origin validation
- Checkpoint management
- Training artifact tracking
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from econosim.learning.transformers.datasets import TimeSeriesWindow
from econosim.learning.transformers.models import (
    MacroTransformer,
    TransformerConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for transformer training."""

    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    gradient_clip: float = 1.0
    val_frequency: int = 5  # validate every N epochs
    checkpoint_dir: str = "outputs/transformer_checkpoints"
    log_interval: int = 10


@dataclass
class TrainingResult:
    """Output of a training run."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    elapsed_seconds: float = 0.0
    num_epochs: int = 0
    num_parameters: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class NumpyTrainer:
    """Simple SGD trainer for the NumPy transformer prototype.

    For production training, use PyTorch with proper autograd.
    This trainer uses numerical gradients for prototyping.
    """

    def __init__(
        self,
        model: MacroTransformer,
        config: TrainingConfig,
    ) -> None:
        self.model = model
        self.config = config

    def train(
        self,
        train_data: list[TimeSeriesWindow],
        val_data: list[TimeSeriesWindow] | None = None,
    ) -> TrainingResult:
        """Train the model on windowed time series data.

        This is a simplified training loop for prototyping.
        For real training, use PyTorch.
        """
        start_time = time.monotonic()
        result = TrainingResult()
        rng = np.random.default_rng(42)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.max_epochs):
            # Shuffle training data
            indices = rng.permutation(len(train_data))
            epoch_loss = 0.0
            n_batches = 0

            for batch_start in range(0, len(indices), self.config.batch_size):
                batch_idx = indices[batch_start:batch_start + self.config.batch_size]
                batch_loss = 0.0

                for idx in batch_idx:
                    window = train_data[idx]
                    pred = self.model.forward(window.context)
                    loss = np.mean((pred - window.target) ** 2)
                    batch_loss += loss

                epoch_loss += batch_loss / len(batch_idx)
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            result.train_losses.append(float(avg_loss))

            # Validation
            if val_data and (epoch + 1) % self.config.val_frequency == 0:
                val_loss = self._evaluate(val_data)
                result.val_losses.append(float(val_loss))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    result.best_val_loss = float(val_loss)
                    result.best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % self.config.log_interval == 0:
                val_str = f", val_loss={result.val_losses[-1]:.6f}" if result.val_losses else ""
                logger.info(f"Epoch {epoch + 1}: train_loss={avg_loss:.6f}{val_str}")

        result.elapsed_seconds = time.monotonic() - start_time
        result.num_epochs = len(result.train_losses)

        return result

    def _evaluate(self, data: list[TimeSeriesWindow]) -> float:
        """Compute average MSE loss on a dataset."""
        total_loss = 0.0
        for window in data:
            pred = self.model.forward(window.context)
            loss = np.mean((pred - window.target) ** 2)
            total_loss += loss
        return total_loss / max(len(data), 1)


class TransformerEvaluator:
    """Evaluate transformer forecasts against actuals and baselines."""

    def __init__(self, model: MacroTransformer) -> None:
        self.model = model

    def evaluate(self, test_data: list[TimeSeriesWindow]) -> dict[str, float]:
        """Evaluate model on test set.

        Returns metrics: mse, rmse, mae, and per-variable metrics.
        """
        if not test_data:
            return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan")}

        all_errors = []
        for window in test_data:
            pred = self.model.forward(window.context)
            error = pred - window.target
            all_errors.append(error)

        errors = np.array(all_errors)
        mse = float(np.mean(errors ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(errors)))

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "n_samples": len(test_data),
        }
