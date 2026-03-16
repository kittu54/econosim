"""
Gymnasium wrappers for EconoSim RL environments.

Provides observation normalization, reward scaling, and action clipping
for more stable training.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RunningMeanStd:
    """Tracks running mean and standard deviation using Welford's algorithm."""

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # small value to prevent division by zero

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0) if x.ndim > 1 else x
        batch_var = np.var(x, axis=0) if x.ndim > 1 else np.zeros_like(x)
        batch_count = x.shape[0] if x.ndim > 1 else 1.0
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count


class NormalizeObservation(gym.ObservationWrapper):
    """Normalizes observations using running mean and variance.

    Clips normalized values to [-clip_obs, clip_obs] for stability.
    """

    def __init__(self, env: gym.Env, clip_obs: float = 10.0, epsilon: float = 1e-8):
        super().__init__(env)
        self.clip_obs = clip_obs
        self.epsilon = epsilon
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.training = True

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if self.training:
            self.obs_rms.update(observation)
        normalized = (observation - self.obs_rms.mean) / np.sqrt(
            self.obs_rms.var + self.epsilon
        )
        return np.clip(normalized, -self.clip_obs, self.clip_obs).astype(np.float32)

    def set_training(self, training: bool) -> None:
        self.training = training


class NormalizeReward(gym.RewardWrapper):
    """Normalizes rewards using running mean and variance.

    Uses a discounted return estimator for more stable normalization.
    """

    def __init__(
        self, env: gym.Env, gamma: float = 0.99,
        clip_reward: float = 10.0, epsilon: float = 1e-8,
    ):
        super().__init__(env)
        self.gamma = gamma
        self.clip_reward = clip_reward
        self.epsilon = epsilon
        self.return_rms = RunningMeanStd(shape=())
        self.returns = 0.0
        self.training = True

    def reward(self, reward: float) -> float:
        if self.training:
            self.returns = self.returns * self.gamma + reward
            self.return_rms.update(np.array([self.returns]))
        var = self.return_rms.var
        if isinstance(var, np.ndarray):
            var = var.item()
        std = float(np.sqrt(var + self.epsilon))
        normalized = reward / std
        return float(max(-self.clip_reward, min(self.clip_reward, normalized)))

    def reset(self, **kwargs):
        self.returns = 0.0
        return super().reset(**kwargs)

    def set_training(self, training: bool) -> None:
        self.training = training


class ScaleReward(gym.RewardWrapper):
    """Scales rewards by a fixed factor."""

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: float) -> float:
        return reward * self.scale


class ClipAction(gym.ActionWrapper):
    """Clips actions to the action space bounds (redundant safety layer)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, self.action_space.low, self.action_space.high)


class RecordEpisodeMetrics(gym.Wrapper):
    """Records per-episode macro metrics for analysis."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_metrics: list[dict] = []
        self._current_episode: list[dict] = []
        self.episode_count = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        metrics = info.get("metrics", {})
        self._current_episode.append(metrics)

        if terminated or truncated:
            self.episode_metrics.append({
                "episode": self.episode_count,
                "length": len(self._current_episode),
                "trajectory": self._current_episode,
                "final": self._current_episode[-1] if self._current_episode else {},
            })
            self.episode_count += 1
            self._current_episode = []

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._current_episode = []
        return self.env.reset(**kwargs)

    def get_episode_summaries(self) -> list[dict]:
        """Return summary statistics for all recorded episodes."""
        summaries = []
        for ep in self.episode_metrics:
            final = ep["final"]
            summaries.append({
                "episode": ep["episode"],
                "length": ep["length"],
                "final_gdp": final.get("gdp", 0.0),
                "final_unemployment": final.get("unemployment_rate", 0.0),
                "final_avg_price": final.get("avg_price", 0.0),
                "final_gini": final.get("gini_deposits", 0.0),
            })
        return summaries
