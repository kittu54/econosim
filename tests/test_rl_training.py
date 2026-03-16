"""
Tests for Phase 3c RL training infrastructure.

Tests wrappers, training script utilities, and multi-agent wrapper.
"""

from __future__ import annotations

import numpy as np
import pytest

from econosim.config.schema import SimulationConfig
from econosim.rl.firm_env import FirmEnv
from econosim.rl.household_env import HouseholdEnv
from econosim.rl.government_env import GovernmentEnv
from econosim.rl.bank_env import BankEnv
from econosim.rl.wrappers import (
    NormalizeObservation,
    NormalizeReward,
    ScaleReward,
    ClipAction,
    RecordEpisodeMetrics,
    RunningMeanStd,
)


SMALL_CONFIG = SimulationConfig(num_periods=10, seed=42,
                                 household={"count": 10},
                                 firm={"count": 2})


# ── RunningMeanStd tests ────────────────────────────────────


class TestRunningMeanStd:
    def test_initial_values(self):
        rms = RunningMeanStd(shape=(3,))
        assert rms.mean.shape == (3,)
        assert rms.var.shape == (3,)

    def test_update_shifts_mean(self):
        rms = RunningMeanStd(shape=(2,))
        data = np.array([10.0, 20.0])
        rms.update(data)
        # Mean should move toward the data
        assert rms.mean[0] > 0
        assert rms.mean[1] > 0

    def test_update_multiple(self):
        rms = RunningMeanStd(shape=())
        for val in [1.0, 2.0, 3.0, 4.0, 5.0]:
            rms.update(np.array([val]))
        # Mean should be roughly 3 (with small initial bias)
        assert 2.0 < rms.mean < 4.0


# ── NormalizeObservation tests ───────────────────────────────


class TestNormalizeObservation:
    def test_normalizes_observations(self):
        env = FirmEnv(config=SMALL_CONFIG, max_steps=5)
        wrapped = NormalizeObservation(env)

        obs, _ = wrapped.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32

        # Step and check normalized obs
        action = env.action_space.sample()
        obs2, _, _, _, _ = wrapped.step(action)
        assert obs2.shape == env.observation_space.shape

    def test_clips_observations(self):
        env = FirmEnv(config=SMALL_CONFIG, max_steps=5)
        wrapped = NormalizeObservation(env, clip_obs=5.0)

        obs, _ = wrapped.reset(seed=42)
        assert np.all(obs >= -5.0)
        assert np.all(obs <= 5.0)

    def test_training_mode(self):
        env = FirmEnv(config=SMALL_CONFIG, max_steps=5)
        wrapped = NormalizeObservation(env)
        wrapped.set_training(False)

        obs, _ = wrapped.reset(seed=42)
        count_before = wrapped.obs_rms.count
        wrapped.step(env.action_space.sample())
        # Count should not increase when not training
        assert wrapped.obs_rms.count == count_before


# ── NormalizeReward tests ────────────────────────────────────


class TestNormalizeReward:
    def test_normalizes_rewards(self):
        env = FirmEnv(config=SMALL_CONFIG, max_steps=5)
        wrapped = NormalizeReward(env)

        obs, _ = wrapped.reset(seed=42)
        action = env.action_space.sample()
        _, reward, _, _, _ = wrapped.step(action)
        assert isinstance(reward, float)

    def test_clips_rewards(self):
        env = FirmEnv(config=SMALL_CONFIG, max_steps=10)
        wrapped = NormalizeReward(env, clip_reward=5.0)

        obs, _ = wrapped.reset(seed=42)
        for _ in range(5):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = wrapped.step(action)
            assert -5.0 <= reward <= 5.0
            if terminated or truncated:
                break

    def test_reset_clears_returns(self):
        env = FirmEnv(config=SMALL_CONFIG, max_steps=5)
        wrapped = NormalizeReward(env)

        wrapped.reset(seed=42)
        wrapped.step(env.action_space.sample())
        assert wrapped.returns != 0.0

        wrapped.reset(seed=43)
        assert wrapped.returns == 0.0


# ── ScaleReward tests ────────────────────────────────────────


class TestScaleReward:
    def test_scales_reward(self):
        env = FirmEnv(config=SMALL_CONFIG, max_steps=5)
        raw_env = FirmEnv(config=SMALL_CONFIG, max_steps=5)
        wrapped = ScaleReward(env, scale=0.01)

        obs1, _ = raw_env.reset(seed=42)
        obs2, _ = wrapped.reset(seed=42)

        action = np.array([1.0, 1.0, 0.5], dtype=np.float32)
        _, raw_reward, _, _, _ = raw_env.step(action)
        _, scaled_reward, _, _, _ = wrapped.step(action)

        assert abs(scaled_reward - raw_reward * 0.01) < 0.01


# ── ClipAction tests ─────────────────────────────────────────


class TestClipAction:
    def test_clips_out_of_bounds_actions(self):
        env = FirmEnv(config=SMALL_CONFIG, max_steps=5)
        wrapped = ClipAction(env)

        obs, _ = wrapped.reset(seed=42)
        # Action outside bounds
        extreme_action = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        obs, _, _, _, _ = wrapped.step(extreme_action)
        # Should not crash; env handles clipping internally too
        assert obs.shape == env.observation_space.shape


# ── RecordEpisodeMetrics tests ───────────────────────────────


class TestRecordEpisodeMetrics:
    def test_records_metrics(self):
        env = FirmEnv(config=SMALL_CONFIG, max_steps=5)
        wrapped = RecordEpisodeMetrics(env)

        obs, _ = wrapped.reset(seed=42)
        done = False
        while not done:
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = wrapped.step(action)
            done = terminated or truncated

        assert len(wrapped.episode_metrics) == 1
        assert wrapped.episode_metrics[0]["length"] == 5
        assert "final" in wrapped.episode_metrics[0]

    def test_multiple_episodes(self):
        env = FirmEnv(config=SMALL_CONFIG, max_steps=3)
        wrapped = RecordEpisodeMetrics(env)

        for ep in range(3):
            obs, _ = wrapped.reset(seed=ep)
            done = False
            while not done:
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = wrapped.step(action)
                done = terminated or truncated

        assert len(wrapped.episode_metrics) == 3
        summaries = wrapped.get_episode_summaries()
        assert len(summaries) == 3
        assert all("final_gdp" in s for s in summaries)

    def test_episode_counter(self):
        env = FirmEnv(config=SMALL_CONFIG, max_steps=3)
        wrapped = RecordEpisodeMetrics(env)

        for ep in range(2):
            obs, _ = wrapped.reset(seed=ep)
            done = False
            while not done:
                obs, _, terminated, truncated, _ = wrapped.step(env.action_space.sample())
                done = terminated or truncated

        assert wrapped.episode_count == 2


# ── Wrapper composition tests ────────────────────────────────


class TestWrapperComposition:
    def test_normalize_obs_and_reward(self):
        """Test stacking NormalizeObservation and NormalizeReward."""
        env = FirmEnv(config=SMALL_CONFIG, max_steps=5)
        env = NormalizeObservation(env)
        env = NormalizeReward(env)

        obs, _ = env.reset(seed=42)
        assert obs.dtype == np.float32

        action = np.array([1.0, 1.0, 0.5], dtype=np.float32)
        obs, reward, _, _, _ = env.step(action)
        assert isinstance(reward, float)
        assert obs.dtype == np.float32

    def test_all_wrappers_stacked(self):
        """Test stacking all wrappers together."""
        env = FirmEnv(config=SMALL_CONFIG, max_steps=5)
        env = ClipAction(env)
        env = NormalizeObservation(env, clip_obs=10.0)
        env = NormalizeReward(env, clip_reward=10.0)
        env = RecordEpisodeMetrics(env)

        obs, _ = env.reset(seed=42)
        done = False
        while not done:
            action = np.array([1.0, 1.0, 0.5], dtype=np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            assert -10.0 <= reward <= 10.0
            assert np.all(obs >= -10.0)
            assert np.all(obs <= 10.0)

        assert env.episode_count == 1


# ── All envs with wrappers ───────────────────────────────────


@pytest.mark.parametrize("env_class,reward_type", [
    (FirmEnv, "profit"),
    (HouseholdEnv, "utility"),
    (GovernmentEnv, "welfare"),
    (BankEnv, "profit"),
])
class TestAllEnvsWithWrappers:
    def test_normalize_obs(self, env_class, reward_type):
        env = env_class(config=SMALL_CONFIG, max_steps=5, reward_type=reward_type)
        wrapped = NormalizeObservation(env)
        obs, _ = wrapped.reset(seed=42)
        assert obs.dtype == np.float32
        obs, _, _, _, _ = wrapped.step(env.action_space.sample())
        assert obs.dtype == np.float32

    def test_record_metrics(self, env_class, reward_type):
        env = env_class(config=SMALL_CONFIG, max_steps=3, reward_type=reward_type)
        wrapped = RecordEpisodeMetrics(env)
        obs, _ = wrapped.reset(seed=42)
        done = False
        while not done:
            obs, _, terminated, truncated, _ = wrapped.step(env.action_space.sample())
            done = terminated or truncated
        assert len(wrapped.episode_metrics) == 1
