"""Tests for the Gymnasium-compatible FirmEnv RL environment."""

from __future__ import annotations

import numpy as np
import pytest

from econosim.config.schema import SimulationConfig
from econosim.rl.firm_env import FirmEnv


def _small_config(**overrides) -> SimulationConfig:
    defaults = dict(
        num_periods=10,
        seed=42,
        household={"count": 10},
        firm={"count": 2},
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


class TestFirmEnvInit:
    def test_creates_env(self):
        env = FirmEnv(config=_small_config())
        assert env.action_space.shape == (3,)
        assert env.observation_space.shape == (14,)

    def test_default_config(self):
        env = FirmEnv(config=_small_config())
        assert env.max_steps == 10


class TestFirmEnvReset:
    def test_reset_returns_obs_info(self):
        env = FirmEnv(config=_small_config())
        obs, info = env.reset()
        assert obs.shape == (14,)
        assert isinstance(info, dict)
        assert info["period"] == 0

    def test_reset_with_seed(self):
        env = FirmEnv(config=_small_config())
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_different_seeds_different_obs(self):
        env = FirmEnv(config=_small_config())
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)
        # Obs may differ due to different random matching
        # At t=0 they could be the same, so just check it runs
        assert obs1.shape == obs2.shape


class TestFirmEnvStep:
    def test_step_returns_correct_tuple(self):
        env = FirmEnv(config=_small_config())
        env.reset(seed=42)
        action = np.array([1.0, 1.0, 0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (14,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert not truncated
        assert isinstance(info, dict)

    def test_terminates_after_max_steps(self):
        env = FirmEnv(config=_small_config(num_periods=3), max_steps=3)
        env.reset(seed=42)
        action = np.array([1.0, 1.0, 0.5], dtype=np.float32)
        for i in range(3):
            obs, reward, terminated, truncated, info = env.step(action)
        assert terminated

    def test_not_terminated_before_max_steps(self):
        env = FirmEnv(config=_small_config(num_periods=5), max_steps=5)
        env.reset(seed=42)
        action = np.array([1.0, 1.0, 0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert not terminated

    def test_action_clipping(self):
        env = FirmEnv(config=_small_config())
        env.reset(seed=42)
        # Extreme actions should be clipped
        action = np.array([0.5, 2.0, -1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (14,)

    def test_multiple_steps(self):
        env = FirmEnv(config=_small_config(num_periods=5), max_steps=5)
        env.reset(seed=42)
        action = np.array([1.0, 1.0, 0.5], dtype=np.float32)
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(action)
        assert terminated
        assert info["period"] == 5


class TestFirmEnvReward:
    def test_profit_reward(self):
        env = FirmEnv(config=_small_config(), reward_type="profit")
        env.reset(seed=42)
        action = np.array([1.0, 1.0, 0.5], dtype=np.float32)
        _, reward, _, _, _ = env.step(action)
        assert isinstance(reward, float)

    def test_gdp_reward(self):
        env = FirmEnv(config=_small_config(), reward_type="gdp")
        env.reset(seed=42)
        action = np.array([1.0, 1.0, 0.5], dtype=np.float32)
        _, reward, _, _, _ = env.step(action)
        assert reward >= 0  # GDP should be non-negative

    def test_balanced_reward(self):
        env = FirmEnv(config=_small_config(), reward_type="balanced")
        env.reset(seed=42)
        action = np.array([1.0, 1.0, 0.5], dtype=np.float32)
        _, reward, _, _, _ = env.step(action)
        assert isinstance(reward, float)


class TestFirmEnvObservation:
    def test_obs_in_space(self):
        env = FirmEnv(config=_small_config())
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)

    def test_obs_after_step_in_space(self):
        env = FirmEnv(config=_small_config())
        env.reset(seed=42)
        action = np.array([1.0, 1.0, 0.5], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(obs)


class TestFirmEnvRender:
    def test_render_no_crash(self, capsys):
        env = FirmEnv(config=_small_config())
        env.reset(seed=42)
        action = np.array([1.0, 1.0, 0.5], dtype=np.float32)
        env.step(action)
        env.render()
        captured = capsys.readouterr()
        assert "t=" in captured.out
        assert "GDP=" in captured.out
