#!/usr/bin/env python3
"""
Unified RL training script for all EconoSim agent types.

Supports training any of the 4 agent types (firm, household, government, bank)
with configurable hyperparameters, observation normalization, and baseline comparison.

Usage:
    python scripts/train_agent.py --agent firm --timesteps 50000 --reward profit
    python scripts/train_agent.py --agent household --timesteps 30000 --reward utility
    python scripts/train_agent.py --agent government --timesteps 50000 --reward welfare
    python scripts/train_agent.py --agent bank --timesteps 30000 --reward profit

Requires: pip install "econosim[rl]"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from stable_baselines3 import PPO, A2C, SAC
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    print("stable-baselines3 is required. Install with: pip install 'econosim[rl]'")
    sys.exit(1)

from econosim.config.schema import SimulationConfig
from econosim.rl.firm_env import FirmEnv
from econosim.rl.household_env import HouseholdEnv
from econosim.rl.government_env import GovernmentEnv
from econosim.rl.bank_env import BankEnv

# Agent type -> (env class, default reward, reward choices, neutral action)
AGENT_REGISTRY = {
    "firm": {
        "env_class": FirmEnv,
        "default_reward": "profit",
        "reward_choices": ["profit", "gdp", "balanced"],
        "neutral_action": np.array([1.0, 1.0, 0.5], dtype=np.float32),
    },
    "household": {
        "env_class": HouseholdEnv,
        "default_reward": "utility",
        "reward_choices": ["utility", "consumption", "balanced"],
        "neutral_action": np.array([0.5, 1.0], dtype=np.float32),
    },
    "government": {
        "env_class": GovernmentEnv,
        "default_reward": "welfare",
        "reward_choices": ["welfare", "gdp", "employment", "balanced"],
        "neutral_action": np.array([0.2, 1.0, 1.0], dtype=np.float32),
    },
    "bank": {
        "env_class": BankEnv,
        "default_reward": "profit",
        "reward_choices": ["profit", "stability", "growth"],
        "neutral_action": np.array([0.005, 0.08], dtype=np.float32),
    },
}

# Hyperparameter presets
HYPERPARAMS = {
    "default": {
        "learning_rate": 3e-4,
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "gae_lambda": 0.95,
    },
    "conservative": {
        "learning_rate": 1e-4,
        "n_steps": 512,
        "batch_size": 128,
        "n_epochs": 5,
        "gamma": 0.995,
        "ent_coef": 0.02,
        "clip_range": 0.1,
        "gae_lambda": 0.98,
    },
    "aggressive": {
        "learning_rate": 1e-3,
        "n_steps": 128,
        "batch_size": 32,
        "n_epochs": 15,
        "gamma": 0.98,
        "ent_coef": 0.005,
        "clip_range": 0.3,
        "gae_lambda": 0.9,
    },
}


def make_env(
    agent_type: str,
    seed: int = 42,
    max_steps: int = 60,
    reward_type: str | None = None,
):
    """Create an environment for the specified agent type."""
    info = AGENT_REGISTRY[agent_type]
    env_class = info["env_class"]
    reward = reward_type or info["default_reward"]
    config = SimulationConfig(num_periods=max_steps, seed=seed)
    return env_class(config=config, max_steps=max_steps, reward_type=reward)


def make_vec_env(
    agent_type: str,
    seed: int = 42,
    max_steps: int = 60,
    reward_type: str | None = None,
    normalize: bool = False,
    n_envs: int = 1,
):
    """Create a vectorized environment with optional observation normalization."""
    def _make():
        env = make_env(agent_type, seed, max_steps, reward_type)
        return Monitor(env)

    venv = DummyVecEnv([_make for _ in range(n_envs)])
    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return venv


def evaluate_agent(model, env, n_episodes: int = 5) -> dict:
    """Run the trained agent and collect performance metrics."""
    episode_rewards = []
    episode_gdps = []
    episode_unemployment = []

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

        episode_rewards.append(total_reward)
        episode_gdps.append(last_metrics.get("gdp", 0.0))
        episode_unemployment.append(last_metrics.get("unemployment_rate", 0.0))

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_final_gdp": float(np.mean(episode_gdps)),
        "mean_final_unemployment": float(np.mean(episode_unemployment)),
    }


def evaluate_baseline(env, neutral_action: np.ndarray, n_episodes: int = 5) -> dict:
    """Run the rule-based baseline (neutral actions) for comparison."""
    episode_rewards = []
    episode_gdps = []
    episode_unemployment = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 1000)
        total_reward = 0.0
        done = False
        last_metrics = {}

        while not done:
            obs, reward, terminated, truncated, info = env.step(neutral_action)
            total_reward += reward
            done = terminated or truncated
            last_metrics = info.get("metrics", {})

        episode_rewards.append(total_reward)
        episode_gdps.append(last_metrics.get("gdp", 0.0))
        episode_unemployment.append(last_metrics.get("unemployment_rate", 0.0))

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_final_gdp": float(np.mean(episode_gdps)),
        "mean_final_unemployment": float(np.mean(episode_unemployment)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train RL agent for EconoSim")
    parser.add_argument("--agent", type=str, required=True,
                        choices=list(AGENT_REGISTRY.keys()),
                        help="Agent type to train")
    parser.add_argument("--timesteps", type=int, default=50_000,
                        help="Training timesteps")
    parser.add_argument("--reward", type=str, default=None,
                        help="Reward function (agent-specific)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=60, help="Episode length")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--hyperparams", type=str, default="default",
                        choices=list(HYPERPARAMS.keys()),
                        help="Hyperparameter preset")
    parser.add_argument("--normalize", action="store_true",
                        help="Enable observation & reward normalization")
    parser.add_argument("--algorithm", type=str, default="PPO",
                        choices=["PPO", "A2C"],
                        help="RL algorithm")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="Path to saved model to evaluate (skip training)")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    args = parser.parse_args()

    agent_info = AGENT_REGISTRY[args.agent]
    reward_type = args.reward or agent_info["default_reward"]

    if reward_type not in agent_info["reward_choices"]:
        print(f"Invalid reward '{reward_type}' for {args.agent}. "
              f"Choices: {agent_info['reward_choices']}")
        sys.exit(1)

    out_dir = Path(args.output or f"outputs/rl/{args.agent}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"EconoSim RL Training — {args.agent.capitalize()} Agent")
    print(f"{'=' * 60}")
    print(f"  Algorithm:   {args.algorithm}")
    print(f"  Timesteps:   {args.timesteps:,}")
    print(f"  Reward:      {reward_type}")
    print(f"  Seed:        {args.seed}")
    print(f"  Max steps:   {args.max_steps}")
    print(f"  Hyperparams: {args.hyperparams}")
    print(f"  Normalize:   {args.normalize}")
    print(f"  Output:      {out_dir}")
    print(f"{'=' * 60}\n")

    # Create environments
    hp = HYPERPARAMS[args.hyperparams]

    if args.normalize:
        train_env = make_vec_env(args.agent, args.seed, args.max_steps,
                                 reward_type, normalize=True)
        eval_env = make_vec_env(args.agent, args.seed + 100, args.max_steps,
                                reward_type, normalize=True)
    else:
        train_env = Monitor(make_env(args.agent, args.seed, args.max_steps, reward_type))
        eval_env = Monitor(make_env(args.agent, args.seed + 100, args.max_steps, reward_type))

    algo_cls = {"PPO": PPO, "A2C": A2C}[args.algorithm]

    if args.eval_only:
        print(f"Loading model from {args.eval_only}...")
        model = algo_cls.load(args.eval_only)
    else:
        # Build algorithm kwargs (filter for algorithm compatibility)
        algo_kwargs = {
            "policy": "MlpPolicy",
            "env": train_env,
            "verbose": 1,
            "seed": args.seed,
            "learning_rate": hp["learning_rate"],
            "n_steps": hp["n_steps"],
            "batch_size": hp["batch_size"],
            "gamma": hp["gamma"],
            "ent_coef": hp["ent_coef"],
            "gae_lambda": hp["gae_lambda"],
            "tensorboard_log": str(out_dir / "tb_logs"),
        }
        if args.algorithm == "PPO":
            algo_kwargs["n_epochs"] = hp["n_epochs"]
            algo_kwargs["clip_range"] = hp["clip_range"]

        model = algo_cls(**algo_kwargs)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(out_dir / "best_model"),
            log_path=str(out_dir / "eval_logs"),
            eval_freq=max(5000, args.timesteps // 10),
            n_eval_episodes=3,
            deterministic=True,
        )

        print(f"Training {args.algorithm} agent...")
        model.learn(total_timesteps=args.timesteps, callback=eval_callback)

        model_path = out_dir / "final_model"
        model.save(str(model_path))
        print(f"\nModel saved to {model_path}")

        if args.normalize:
            train_env.save(str(out_dir / "vec_normalize.pkl"))

    # Evaluate
    print("\nEvaluating trained agent...")
    raw_env = make_env(args.agent, args.seed, args.max_steps, reward_type)
    rl_results = evaluate_agent(model, raw_env, args.eval_episodes)

    print("\nEvaluating rule-based baseline...")
    baseline_results = evaluate_baseline(raw_env, agent_info["neutral_action"],
                                          args.eval_episodes)

    print(f"\n{'=' * 60}")
    print("Results Comparison")
    print(f"{'=' * 60}")
    print(f"{'Metric':<30} {'RL Agent':>12} {'Baseline':>12} {'Delta':>10}")
    print(f"{'-' * 64}")
    for key in rl_results:
        rl_val = rl_results[key]
        bl_val = baseline_results[key]
        delta = rl_val - bl_val
        sign = "+" if delta >= 0 else ""
        print(f"{key:<30} {rl_val:>12.2f} {bl_val:>12.2f} {sign}{delta:>9.2f}")
    print(f"{'=' * 60}")

    # Save results
    results = {
        "agent_type": args.agent,
        "algorithm": args.algorithm,
        "hyperparams": args.hyperparams,
        "args": vars(args),
        "hyperparameters": hp,
        "rl_agent": rl_results,
        "baseline": baseline_results,
    }
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if not args.normalize:
        train_env.close()
        eval_env.close()
    else:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
