"""Tests for the Gymnasium-compatible BankEnv."""

from __future__ import annotations

import numpy as np
import pytest

from econosim.config.schema import SimulationConfig
from econosim.rl.bank_env import BankEnv


def _small_config(**overrides) -> SimulationConfig:
    defaults = dict(num_periods=10, seed=42, household={"count": 10}, firm={"count": 2})
    defaults.update(overrides)
    return SimulationConfig(**defaults)


class TestBankEnvInit:
    def test_creates_env(self):
        env = BankEnv(config=_small_config())
        assert env.action_space.shape == (2,)
        assert env.observation_space.shape == (12,)


class TestBankEnvReset:
    def test_reset_returns_obs_info(self):
        env = BankEnv(config=_small_config())
        obs, info = env.reset()
        assert obs.shape == (12,)
        assert info["period"] == 0

    def test_reset_with_seed(self):
        env = BankEnv(config=_small_config())
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)


class TestBankEnvStep:
    def test_step_returns_correct_tuple(self):
        env = BankEnv(config=_small_config())
        env.reset(seed=42)
        action = np.array([0.005, 0.08], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (12,)
        assert isinstance(reward, float)
        assert not truncated

    def test_terminates_after_max_steps(self):
        env = BankEnv(config=_small_config(num_periods=3), max_steps=3)
        env.reset(seed=42)
        action = np.array([0.005, 0.08], dtype=np.float32)
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(action)
        assert terminated

    def test_rate_change_affects_state(self):
        env = BankEnv(config=_small_config(num_periods=5), max_steps=5)
        env.reset(seed=42)
        action = np.array([0.02, 0.15], dtype=np.float32)
        env.step(action)
        assert env._bank.base_interest_rate == pytest.approx(0.02)
        assert env._bank.capital_adequacy_ratio == pytest.approx(0.15)

    def test_obs_in_space(self):
        env = BankEnv(config=_small_config())
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)


class TestBankEnvReward:
    def test_profit_reward(self):
        env = BankEnv(config=_small_config(), reward_type="profit")
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(np.array([0.005, 0.08], dtype=np.float32))
        assert isinstance(reward, float)

    def test_stability_reward(self):
        env = BankEnv(config=_small_config(), reward_type="stability")
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(np.array([0.005, 0.08], dtype=np.float32))
        assert isinstance(reward, float)

    def test_growth_reward(self):
        env = BankEnv(config=_small_config(), reward_type="growth")
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(np.array([0.005, 0.08], dtype=np.float32))
        assert reward >= 0


class TestBankEnvRender:
    def test_render_no_crash(self, capsys):
        env = BankEnv(config=_small_config())
        env.reset(seed=42)
        env.step(np.array([0.005, 0.08], dtype=np.float32))
        env.render()
        captured = capsys.readouterr()
        assert "t=" in captured.out
        assert "GDP=" in captured.out
