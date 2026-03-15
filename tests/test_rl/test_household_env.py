"""Tests for the Gymnasium-compatible HouseholdEnv."""

from __future__ import annotations

import numpy as np
import pytest

from econosim.config.schema import SimulationConfig
from econosim.rl.household_env import HouseholdEnv


def _small_config(**overrides) -> SimulationConfig:
    defaults = dict(num_periods=10, seed=42, household={"count": 10}, firm={"count": 2})
    defaults.update(overrides)
    return SimulationConfig(**defaults)


class TestHouseholdEnvInit:
    def test_creates_env(self):
        env = HouseholdEnv(config=_small_config())
        assert env.action_space.shape == (2,)
        assert env.observation_space.shape == (12,)


class TestHouseholdEnvReset:
    def test_reset_returns_obs_info(self):
        env = HouseholdEnv(config=_small_config())
        obs, info = env.reset()
        assert obs.shape == (12,)
        assert info["period"] == 0

    def test_reset_with_seed(self):
        env = HouseholdEnv(config=_small_config())
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)


class TestHouseholdEnvStep:
    def test_step_returns_correct_tuple(self):
        env = HouseholdEnv(config=_small_config())
        env.reset(seed=42)
        action = np.array([0.5, 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (12,)
        assert isinstance(reward, float)
        assert not truncated

    def test_terminates_after_max_steps(self):
        env = HouseholdEnv(config=_small_config(num_periods=3), max_steps=3)
        env.reset(seed=42)
        action = np.array([0.5, 1.0], dtype=np.float32)
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(action)
        assert terminated

    def test_obs_in_space(self):
        env = HouseholdEnv(config=_small_config())
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        action = np.array([0.5, 1.0], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(obs)


class TestHouseholdEnvReward:
    def test_utility_reward(self):
        env = HouseholdEnv(config=_small_config(), reward_type="utility")
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(np.array([0.5, 1.0], dtype=np.float32))
        assert isinstance(reward, float)

    def test_consumption_reward(self):
        env = HouseholdEnv(config=_small_config(), reward_type="consumption")
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(np.array([0.5, 1.0], dtype=np.float32))
        assert reward >= 0

    def test_balanced_reward(self):
        env = HouseholdEnv(config=_small_config(), reward_type="balanced")
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(np.array([0.5, 1.0], dtype=np.float32))
        assert isinstance(reward, float)


class TestHouseholdEnvRender:
    def test_render_no_crash(self, capsys):
        env = HouseholdEnv(config=_small_config())
        env.reset(seed=42)
        env.step(np.array([0.5, 1.0], dtype=np.float32))
        env.render()
        captured = capsys.readouterr()
        assert "t=" in captured.out
