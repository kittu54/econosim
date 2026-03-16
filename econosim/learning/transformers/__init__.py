"""Transformer-based models for forecasting, emulation, and policy learning.

IMPORTANT: Transformers are auxiliary layers. They do NOT replace:
- The economic simulation core
- Stock-flow consistent accounting
- Explicit market mechanisms
- Structural calibration

They serve as:
1. Residual correction models over simulator forecasts
2. Simulation emulators for fast calibration screening
3. Optional policy architectures for agent decisions
"""
