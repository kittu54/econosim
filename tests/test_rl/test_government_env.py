"""Tests for the Gymnasium-compatible GovernmentEnv."""

from __future__ import annotations

import numpy as np
import pytest

from econosim.config.schema import SimulationConfig
from econosim.rl.government_env import GovernmentEnv


def _small_config(**overrides) -> SimulationConfig:
    defaults = dict(num_periods=10, seed=42, household={"count": 10}, firm={"count": 2})
    defaults.update(overrides)
    return SimulationConfig(**defaults)


class TestGovernmentEnvInit:
    def test_creates_env(self):
        env = GovernmentEnv(config=_small_config())
        assert env.action_space.shape == (3,)
        assert env.observation_space.shape == (12,)


class TestGovernmentEnvReset:
    def test_reset_returns_obs_info(self):
        env = GovernmentEnv(config=_small_config())
        obs, info = env.reset()
        assert obs.shape == (12,)
        assert info["period"] == 0

    def test_reset_with_seed(self):
        env = GovernmentEnv(config=_small_config())
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)


class TestGovernmentEnvStep:
    def test_step_returns_correct_tuple(self):
        env = GovernmentEnv(config=_small_config())
        env.reset(seed=42)
        action = np.array([0.2, 1.0, 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (12,)
        assert isinstance(reward, float)
        assert not truncated

    def test_terminates_after_max_steps(self):
        env = GovernmentEnv(config=_small_config(num_periods=3), max_steps=3)
        env.reset(seed=42)
        action = np.array([0.2, 1.0, 1.0], dtype=np.float32)
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(action)
        assert terminated

    def test_policy_change_affects_state(self):
        env = GovernmentEnv(config=_small_config(num_periods=5), max_steps=5)
        env.reset(seed=42)
        # High tax, low spending
        action = np.array([0.4, 0.5, 0.5], dtype=np.float32)
        env.step(action)
        assert env._govt.income_tax_rate == pytest.approx(0.4)

    def test_obs_in_space(self):
        env = GovernmentEnv(config=_small_config())
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)


class TestGovernmentEnvReward:
    def test_welfare_reward(self):
        env = GovernmentEnv(config=_small_config(), reward_type="welfare")
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(np.array([0.2, 1.0, 1.0], dtype=np.float32))
        assert isinstance(reward, float)

    def test_gdp_reward(self):
        env = GovernmentEnv(config=_small_config(), reward_type="gdp")
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(np.array([0.2, 1.0, 1.0], dtype=np.float32))
        assert reward >= 0

    def test_employment_reward(self):
        env = GovernmentEnv(config=_small_config(), reward_type="employment")
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(np.array([0.2, 1.0, 1.0], dtype=np.float32))
        assert isinstance(reward, float)


class TestGovernmentEnvRender:
    def test_render_no_crash(self, capsys):
        env = GovernmentEnv(config=_small_config())
        env.reset(seed=42)
        env.step(np.array([0.2, 1.0, 1.0], dtype=np.float32))
        env.render()
        captured = capsys.readouterr()
        assert "t=" in captured.out
        assert "GDP=" in captured.out
