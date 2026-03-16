"""Transformer model architectures for macro forecasting and emulation.

Provides lightweight transformer implementations that can be used without
heavy framework dependencies. Full PyTorch versions are optional.

Architecture choices:
- Patching: Input time series are divided into patches (like ViT) for efficiency
- Positional encoding: Learnable or sinusoidal
- Output heads: Separate heads for different tasks (forecasting, emulation)

Why transformers over simpler models:
- Can capture long-range dependencies in macro sequences
- Attention mechanism reveals which historical periods matter for forecasts
- Flexible conditioning on exogenous variables and scenarios
- Transfer learning potential across economies/calibrations

When NOT to use transformers:
- Short sequences (< 30 timesteps): AR/VAR is sufficient
- Pure noise / no signal: wasted compute
- When interpretability is paramount: use structural model directly
- Limited training data (< 100 sequences): overfitting risk
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TransformerConfig:
    """Configuration for the macro transformer model."""

    # Input
    n_features: int = 8       # number of input variables
    context_length: int = 60  # look-back window
    horizon: int = 12         # forecast horizon
    n_targets: int = 3        # number of target variables

    # Architecture
    d_model: int = 64         # model dimension
    n_heads: int = 4          # attention heads
    n_layers: int = 3         # transformer layers
    d_ff: int = 128           # feedforward dimension
    dropout: float = 0.1

    # Patching (PatchTST-style)
    patch_size: int = 4       # timesteps per patch
    patch_stride: int = 4     # stride between patches

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    weight_decay: float = 1e-5


class NumpyTransformerBlock:
    """Minimal transformer block implemented in pure NumPy.

    For production training, use the PyTorch version. This exists for:
    - Testing the architecture without torch dependency
    - Understanding the computation flow
    - Small-scale prototyping
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, rng: np.random.Generator) -> None:
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_ff = d_ff

        # Initialize weights (Xavier)
        scale = 1.0 / math.sqrt(d_model)
        self.W_q = rng.normal(0, scale, (d_model, d_model))
        self.W_k = rng.normal(0, scale, (d_model, d_model))
        self.W_v = rng.normal(0, scale, (d_model, d_model))
        self.W_o = rng.normal(0, scale, (d_model, d_model))

        ff_scale = 1.0 / math.sqrt(d_ff)
        self.W_ff1 = rng.normal(0, scale, (d_model, d_ff))
        self.b_ff1 = np.zeros(d_ff)
        self.W_ff2 = rng.normal(0, ff_scale, (d_ff, d_model))
        self.b_ff2 = np.zeros(d_model)

    def attention(self, x: np.ndarray) -> np.ndarray:
        """Multi-head self-attention. x: (seq_len, d_model)."""
        seq_len = x.shape[0]
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Split heads
        Q = Q.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)

        # Scaled dot-product attention
        scores = Q @ K.transpose(0, 2, 1) / math.sqrt(self.d_k)
        attn = _softmax(scores)
        out = attn @ V

        # Merge heads
        out = out.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        return out @ self.W_o

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: attention + FFN with residual connections."""
        # Self-attention + residual
        attn_out = self.attention(x)
        x = _layer_norm(x + attn_out)

        # FFN + residual
        ff_out = np.maximum(0, x @ self.W_ff1 + self.b_ff1)  # ReLU
        ff_out = ff_out @ self.W_ff2 + self.b_ff2
        x = _layer_norm(x + ff_out)

        return x


class MacroTransformer:
    """Lightweight transformer for macro time series forecasting.

    Designed for:
    - Residual forecasting over simulator outputs
    - Direct multi-step forecasting
    - Simulation emulation

    This is a NumPy prototype. For training, use PyTorch version.
    """

    def __init__(self, config: TransformerConfig, seed: int = 42) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)

        # Compute number of patches
        self.n_patches = (config.context_length - config.patch_size) // config.patch_stride + 1

        # Patch embedding
        scale = 1.0 / math.sqrt(config.d_model)
        self.patch_proj = self.rng.normal(
            0, scale, (config.n_features * config.patch_size, config.d_model)
        )

        # Positional encoding
        self.pos_encoding = self._sinusoidal_encoding(self.n_patches, config.d_model)

        # Transformer blocks
        self.blocks = [
            NumpyTransformerBlock(config.d_model, config.n_heads, config.d_ff, self.rng)
            for _ in range(config.n_layers)
        ]

        # Output head
        self.output_proj = self.rng.normal(
            0, scale, (config.d_model, config.horizon * config.n_targets)
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        Args:
            x: (context_length, n_features) input time series

        Returns:
            (horizon, n_targets) forecast
        """
        cfg = self.config

        # Patch embedding
        patches = []
        for i in range(self.n_patches):
            start = i * cfg.patch_stride
            patch = x[start:start + cfg.patch_size].flatten()
            patches.append(patch)

        patch_matrix = np.array(patches)  # (n_patches, patch_size * n_features)
        embedded = patch_matrix @ self.patch_proj  # (n_patches, d_model)

        # Add positional encoding
        embedded = embedded + self.pos_encoding[:len(embedded)]

        # Transformer blocks
        h = embedded
        for block in self.blocks:
            h = block.forward(h)

        # Pool (mean over patches) and project to output
        pooled = h.mean(axis=0)  # (d_model,)
        output = pooled @ self.output_proj  # (horizon * n_targets,)

        return output.reshape(cfg.horizon, cfg.n_targets)

    @staticmethod
    def _sinusoidal_encoding(max_len: int, d_model: int) -> np.ndarray:
        pe = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[:d_model // 2])
        return pe


class ResidualForecaster:
    """Transformer-based residual correction over simulator forecasts.

    Architecture:
    final_forecast = simulator_forecast + transformer_residual

    The transformer learns to correct systematic biases in the simulator
    without replacing the structural model.

    This preserves:
    - Accounting consistency (from simulator)
    - Causal structure (from simulator)
    - Pattern correction (from transformer)
    """

    def __init__(self, config: TransformerConfig, seed: int = 42) -> None:
        self.transformer = MacroTransformer(config, seed)
        self.config = config

    def predict_residual(
        self,
        history: np.ndarray,
        simulator_forecast: np.ndarray,
    ) -> np.ndarray:
        """Predict correction to simulator forecast.

        Args:
            history: (context_length, n_features) observed history
            simulator_forecast: (horizon, n_targets) simulator's forecast

        Returns:
            (horizon, n_targets) corrected forecast = simulator + residual
        """
        residual = self.transformer.forward(history)

        # Scale residual to prevent large corrections
        residual = np.clip(residual, -0.5, 0.5)

        return simulator_forecast + residual


class SimulationEmulator:
    """Transformer that approximates simulator outputs.

    Maps: parameter_vector → macro_trajectory

    Used to accelerate calibration by pre-screening parameter regions
    before running full (expensive) simulations.

    When to trust: initial screening, sensitivity analysis
    When to fall back to full sim: final calibration, validation, policy eval
    """

    def __init__(self, config: TransformerConfig, seed: int = 42) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)

        # Parameter embedding
        self.param_proj = self.rng.normal(
            0, 0.1, (config.n_features, config.d_model)
        )
        # Simple MLP emulator (transformer is overkill for param→trajectory)
        self.hidden = self.rng.normal(0, 0.1, (config.d_model, config.d_ff))
        self.output = self.rng.normal(
            0, 0.1, (config.d_ff, config.horizon * config.n_targets)
        )

    def emulate(self, param_vector: np.ndarray) -> np.ndarray:
        """Approximate simulation output from parameter vector.

        Args:
            param_vector: (n_params,) parameter values

        Returns:
            (horizon, n_targets) approximate macro trajectory
        """
        # Pad/truncate to expected size
        pv = np.zeros(self.config.n_features)
        pv[:min(len(param_vector), len(pv))] = param_vector[:len(pv)]

        h = pv @ self.param_proj
        h = np.maximum(0, h @ self.hidden)  # ReLU
        out = h @ self.output
        return out.reshape(self.config.horizon, self.config.n_targets)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)
