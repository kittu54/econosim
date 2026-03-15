"""Tests for the PettingZoo-compatible multi-agent environment."""

from __future__ import annotations

import numpy as np
import pytest

from econosim.config.schema import SimulationConfig

try:
    from econosim.rl.multi_agent_env import EconoSimMultiAgentEnv, HAS_PETTINGZOO
except ImportError:
    HAS_PETTINGZOO = False

pytestmark = pytest.mark.skipif(not HAS_PETTINGZOO, reason="pettingzoo not installed")


def _small_config(**overrides) -> SimulationConfig:
    defaults = dict(num_periods=5, seed=42, household={"count": 10}, firm={"count": 2})
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def _sample_actions(env) -> dict[str, np.ndarray]:
    return {agent: env.action_space(agent).sample() for agent in env.agents}


class TestMultiAgentEnvInit:
    def test_creates_env(self):
        env = EconoSimMultiAgentEnv(config=_small_config())
        assert set(env.possible_agents) == {"firm", "household", "government", "bank"}

    def test_action_spaces(self):
        env = EconoSimMultiAgentEnv(config=_small_config())
        assert env.action_space("firm").shape == (3,)
        assert env.action_space("household").shape == (2,)
        assert env.action_space("government").shape == (3,)
        assert env.action_space("bank").shape == (2,)

    def test_observation_spaces(self):
        env = EconoSimMultiAgentEnv(config=_small_config())
        assert env.observation_space("firm").shape == (14,)
        assert env.observation_space("household").shape == (12,)
        assert env.observation_space("government").shape == (12,)
        assert env.observation_space("bank").shape == (12,)


class TestMultiAgentEnvReset:
    def test_reset_returns_all_agents(self):
        env = EconoSimMultiAgentEnv(config=_small_config())
        obs, infos = env.reset()
        assert set(obs.keys()) == {"firm", "household", "government", "bank"}
        assert set(infos.keys()) == {"firm", "household", "government", "bank"}

    def test_obs_shapes(self):
        env = EconoSimMultiAgentEnv(config=_small_config())
        obs, _ = env.reset()
        assert obs["firm"].shape == (14,)
        assert obs["household"].shape == (12,)
        assert obs["government"].shape == (12,)
        assert obs["bank"].shape == (12,)

    def test_reset_with_seed(self):
        env = EconoSimMultiAgentEnv(config=_small_config())
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        for agent in env.possible_agents:
            np.testing.assert_array_equal(obs1[agent], obs2[agent])


class TestMultiAgentEnvStep:
    def test_step_returns_correct_keys(self):
        env = EconoSimMultiAgentEnv(config=_small_config())
        env.reset(seed=42)
        actions = _sample_actions(env)
        obs, rewards, terminations, truncations, infos = env.step(actions)
        for agent in env.possible_agents:
            assert agent in obs
            assert agent in rewards
            assert agent in terminations
            assert agent in truncations
            assert agent in infos

    def test_rewards_are_floats(self):
        env = EconoSimMultiAgentEnv(config=_small_config())
        env.reset(seed=42)
        actions = _sample_actions(env)
        _, rewards, _, _, _ = env.step(actions)
        for agent, r in rewards.items():
            assert isinstance(r, float), f"{agent} reward is not float"

    def test_terminates_after_max_steps(self):
        env = EconoSimMultiAgentEnv(config=_small_config(num_periods=3), max_steps=3)
        env.reset(seed=42)
        for i in range(3):
            actions = {agent: env.action_space(agent).sample() for agent in env.possible_agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)
        assert all(terminations.values())
        assert env.agents == []

    def test_not_terminated_before_max_steps(self):
        env = EconoSimMultiAgentEnv(config=_small_config(num_periods=5), max_steps=5)
        env.reset(seed=42)
        actions = _sample_actions(env)
        _, _, terminations, _, _ = env.step(actions)
        assert not any(terminations.values())

    def test_obs_in_spaces(self):
        env = EconoSimMultiAgentEnv(config=_small_config())
        obs, _ = env.reset(seed=42)
        for agent in env.possible_agents:
            assert env.observation_space(agent).contains(obs[agent])


class TestMultiAgentEnvRender:
    def test_render_no_crash(self, capsys):
        env = EconoSimMultiAgentEnv(config=_small_config())
        env.reset(seed=42)
        actions = _sample_actions(env)
        env.step(actions)
        env.render()
        captured = capsys.readouterr()
        assert "t=" in captured.out
        assert "GDP=" in captured.out
