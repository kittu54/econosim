#!/usr/bin/env python3
"""
Multi-agent RL training using independent learners approach.

Each agent type (firm, household, government, bank) trains its own PPO policy
independently while sharing the same simulation. Uses the PettingZoo parallel
environment with per-agent wrappers for SB3 compatibility.

Usage:
    python scripts/train_multiagent.py --timesteps 50000 --seed 42
    python scripts/train_multiagent.py --timesteps 100000 --mode sequential
    python scripts/train_multiagent.py --timesteps 50000 --mode simultaneous

Requires: pip install "econosim[rl]"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    print("stable-baselines3 and gymnasium are required. Install with: pip install 'econosim[rl]'")
    sys.exit(1)

from econosim.config.schema import SimulationConfig
from econosim.rl.multi_agent_env import EconoSimMultiAgentEnv


class SingleAgentWrapper(gym.Env):
    """Wraps one agent's view of the multi-agent environment for SB3 training.

    Other agents use fixed policies (either trained models or neutral actions).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        multi_env: EconoSimMultiAgentEnv,
        agent_id: str,
        other_policies: dict[str, any] | None = None,
    ) -> None:
        super().__init__()
        self.multi_env = multi_env
        self.agent_id = agent_id
        self.other_policies = other_policies or {}

        self.action_space = multi_env.action_space(agent_id)
        self.observation_space = multi_env.observation_space(agent_id)

        self._all_agents = list(multi_env.possible_agents)
        self._current_obs = None

        # Neutral actions for agents without trained policies
        self._neutral_actions = {
            "firm": np.array([1.0, 1.0, 0.5], dtype=np.float32),
            "household": np.array([0.5, 1.0], dtype=np.float32),
            "government": np.array([0.2, 1.0, 1.0], dtype=np.float32),
            "bank": np.array([0.005, 0.08], dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        all_obs, all_infos = self.multi_env.reset(seed=seed)
        self._current_obs = all_obs
        return all_obs[self.agent_id], all_infos.get(self.agent_id, {})

    def step(self, action):
        # Build actions for all agents
        actions = {}
        for agent in self._all_agents:
            if agent == self.agent_id:
                actions[agent] = action
            elif agent in self.other_policies:
                policy = self.other_policies[agent]
                obs = self._current_obs[agent]
                act, _ = policy.predict(obs, deterministic=True)
                actions[agent] = act
            else:
                actions[agent] = self._neutral_actions[agent]

        all_obs, all_rewards, all_terms, all_truncs, all_infos = self.multi_env.step(actions)

        # Handle terminal state where agents list becomes empty
        if self.agent_id in all_obs:
            self._current_obs = all_obs
            obs = all_obs[self.agent_id]
            reward = all_rewards[self.agent_id]
            terminated = all_terms[self.agent_id]
            truncated = all_truncs[self.agent_id]
            info = all_infos.get(self.agent_id, {})
        else:
            obs = self.observation_space.sample() * 0
            reward = 0.0
            terminated = True
            truncated = False
            info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        self.multi_env.render()


def train_sequential(args, out_dir: Path) -> dict:
    """Train agents one at a time, using previously trained agents as fixed policies."""
    agent_order = ["firm", "household", "bank", "government"]
    trained_policies = {}
    all_results = {}

    for agent_id in agent_order:
        print(f"\n{'=' * 60}")
        print(f"Training {agent_id.upper()} agent (sequential mode)")
        print(f"{'=' * 60}")

        config = SimulationConfig(num_periods=args.max_steps, seed=args.seed)
        multi_env = EconoSimMultiAgentEnv(config=config, max_steps=args.max_steps)

        # Wrap for this agent, using previously trained policies
        train_env = Monitor(
            SingleAgentWrapper(multi_env, agent_id, dict(trained_policies))
        )

        eval_multi = EconoSimMultiAgentEnv(
            config=SimulationConfig(num_periods=args.max_steps, seed=args.seed + 100),
            max_steps=args.max_steps,
        )
        eval_env = Monitor(
            SingleAgentWrapper(eval_multi, agent_id, dict(trained_policies))
        )

        model = PPO(
            "MlpPolicy", train_env, verbose=1, seed=args.seed,
            learning_rate=3e-4, n_steps=256, batch_size=64,
            n_epochs=10, gamma=0.99, ent_coef=0.01,
            tensorboard_log=str(out_dir / agent_id / "tb_logs"),
        )

        agent_dir = out_dir / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(agent_dir / "best_model"),
            log_path=str(agent_dir / "eval_logs"),
            eval_freq=max(5000, args.timesteps // 10),
            n_eval_episodes=3,
            deterministic=True,
        )

        model.learn(total_timesteps=args.timesteps, callback=eval_callback)
        model.save(str(agent_dir / "final_model"))

        trained_policies[agent_id] = model

        # Evaluate
        raw_multi = EconoSimMultiAgentEnv(
            config=SimulationConfig(num_periods=args.max_steps, seed=args.seed),
            max_steps=args.max_steps,
        )
        raw_env = SingleAgentWrapper(raw_multi, agent_id, trained_policies)
        results = _evaluate(model, raw_env)
        all_results[agent_id] = results

        print(f"\n{agent_id.upper()} results: reward={results['mean_reward']:.2f} "
              f"GDP={results['mean_final_gdp']:.0f} "
              f"U={results['mean_final_unemployment']:.2%}")

        train_env.close()
        eval_env.close()

    return all_results


def train_simultaneous(args, out_dir: Path) -> dict:
    """Train all agents simultaneously with round-robin updates."""
    agent_ids = ["firm", "household", "government", "bank"]
    models = {}
    all_results = {}

    # Initialize all models
    for agent_id in agent_ids:
        config = SimulationConfig(num_periods=args.max_steps, seed=args.seed)
        multi_env = EconoSimMultiAgentEnv(config=config, max_steps=args.max_steps)
        train_env = Monitor(SingleAgentWrapper(multi_env, agent_id))

        model = PPO(
            "MlpPolicy", train_env, verbose=0, seed=args.seed,
            learning_rate=3e-4, n_steps=256, batch_size=64,
            n_epochs=10, gamma=0.99, ent_coef=0.01,
            tensorboard_log=str(out_dir / agent_id / "tb_logs"),
        )
        models[agent_id] = model

    # Round-robin training
    steps_per_round = max(5000, args.timesteps // 20)
    rounds = args.timesteps // steps_per_round

    for round_num in range(rounds):
        print(f"\n--- Round {round_num + 1}/{rounds} ---")
        for agent_id in agent_ids:
            # Update other policies in the wrapper
            multi_env_new = EconoSimMultiAgentEnv(
                config=SimulationConfig(num_periods=args.max_steps, seed=args.seed),
                max_steps=args.max_steps,
            )
            other_policies = {k: v for k, v in models.items() if k != agent_id}
            wrapped = Monitor(SingleAgentWrapper(multi_env_new, agent_id, other_policies))
            models[agent_id].set_env(wrapped)
            models[agent_id].learn(total_timesteps=steps_per_round, reset_num_timesteps=False)
            print(f"  {agent_id}: trained {steps_per_round} steps")

    # Save and evaluate
    for agent_id in agent_ids:
        agent_dir = out_dir / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)
        models[agent_id].save(str(agent_dir / "final_model"))

        raw_multi = EconoSimMultiAgentEnv(
            config=SimulationConfig(num_periods=args.max_steps, seed=args.seed),
            max_steps=args.max_steps,
        )
        raw_env = SingleAgentWrapper(raw_multi, agent_id, models)
        results = _evaluate(models[agent_id], raw_env)
        all_results[agent_id] = results

    return all_results


def _evaluate(model, env, n_episodes: int = 5) -> dict:
    """Evaluate a trained model."""
    rewards, gdps, unemployment = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 1000)
        total_reward = 0.0
        done = False
        last_metrics = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            last_metrics = info.get("metrics", {})

        rewards.append(total_reward)
        gdps.append(last_metrics.get("gdp", 0.0))
        unemployment.append(last_metrics.get("unemployment_rate", 0.0))

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_final_gdp": float(np.mean(gdps)),
        "mean_final_unemployment": float(np.mean(unemployment)),
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-agent RL training")
    parser.add_argument("--timesteps", type=int, default=50_000,
                        help="Training timesteps per agent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=60, help="Episode length")
    parser.add_argument("--output", type=str, default="outputs/rl/multiagent",
                        help="Output directory")
    parser.add_argument("--mode", type=str, default="sequential",
                        choices=["sequential", "simultaneous"],
                        help="Training mode")
    args = parser.parse_args()

    out_dir = Path(args.output) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"EconoSim Multi-Agent RL Training")
    print(f"{'=' * 60}")
    print(f"  Mode:        {args.mode}")
    print(f"  Timesteps:   {args.timesteps:,} per agent")
    print(f"  Seed:        {args.seed}")
    print(f"  Max steps:   {args.max_steps}")
    print(f"  Output:      {out_dir}")
    print(f"{'=' * 60}\n")

    if args.mode == "sequential":
        results = train_sequential(args, out_dir)
    else:
        results = train_simultaneous(args, out_dir)

    # Print summary
    print(f"\n{'=' * 60}")
    print("Multi-Agent Training Summary")
    print(f"{'=' * 60}")
    print(f"{'Agent':<15} {'Mean Reward':>12} {'Final GDP':>12} {'Unemployment':>14}")
    print(f"{'-' * 55}")
    for agent_id, res in results.items():
        print(f"{agent_id:<15} {res['mean_reward']:>12.2f} "
              f"{res['mean_final_gdp']:>12.0f} "
              f"{res['mean_final_unemployment']:>13.2%}")
    print(f"{'=' * 60}")

    # Save combined results
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({"mode": args.mode, "args": vars(args), "results": results}, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
