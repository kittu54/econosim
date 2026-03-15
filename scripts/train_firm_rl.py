#!/usr/bin/env python3
"""
Train an RL agent to control firm pricing, wages, and hiring.

Usage:
    python scripts/train_firm_rl.py [--timesteps 50000] [--reward profit] [--seed 42]

Requires: pip install "econosim[rl]"  (gymnasium, stable-baselines3, torch)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    print("stable-baselines3 is required. Install with: pip install 'econosim[rl]'")
    sys.exit(1)

from econosim.config.schema import SimulationConfig
from econosim.rl.firm_env import FirmEnv


def make_env(
    seed: int = 42,
    max_steps: int = 60,
    reward_type: str = "profit",
) -> FirmEnv:
    config = SimulationConfig(num_periods=max_steps, seed=seed)
    return FirmEnv(config=config, max_steps=max_steps, reward_type=reward_type)


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


def evaluate_baseline(env, n_episodes: int = 5) -> dict:
    """Run the rule-based baseline (neutral actions) for comparison."""
    episode_rewards = []
    episode_gdps = []
    episode_unemployment = []

    neutral_action = np.array([1.0, 1.0, 0.5], dtype=np.float32)

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
    parser = argparse.ArgumentParser(description="Train RL firm agent")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Training timesteps")
    parser.add_argument("--reward", type=str, default="profit",
                        choices=["profit", "gdp", "balanced"], help="Reward function")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=60, help="Episode length")
    parser.add_argument("--output", type=str, default="outputs/rl", help="Output directory")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="Path to saved model to evaluate (skip training)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"EconoSim RL Training — Firm Agent")
    print(f"{'='*60}")
    print(f"  Timesteps:   {args.timesteps:,}")
    print(f"  Reward:      {args.reward}")
    print(f"  Seed:        {args.seed}")
    print(f"  Max steps:   {args.max_steps}")
    print(f"  Output:      {out_dir}")
    print(f"{'='*60}\n")

    # Create environments
    train_env = Monitor(make_env(seed=args.seed, max_steps=args.max_steps,
                                  reward_type=args.reward))
    eval_env = Monitor(make_env(seed=args.seed + 100, max_steps=args.max_steps,
                                 reward_type=args.reward))

    if args.eval_only:
        print(f"Loading model from {args.eval_only}...")
        model = PPO.load(args.eval_only)
    else:
        # Train
        print("Training PPO agent...")
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            seed=args.seed,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,
            tensorboard_log=str(out_dir / "tb_logs"),
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(out_dir / "best_model"),
            log_path=str(out_dir / "eval_logs"),
            eval_freq=5000,
            n_eval_episodes=3,
            deterministic=True,
        )

        model.learn(total_timesteps=args.timesteps, callback=eval_callback)

        # Save final model
        model_path = out_dir / "final_model"
        model.save(str(model_path))
        print(f"\nModel saved to {model_path}")

    # Evaluate
    print("\nEvaluating trained agent...")
    raw_env = make_env(seed=args.seed, max_steps=args.max_steps, reward_type=args.reward)
    rl_results = evaluate_agent(model, raw_env)

    print("\nEvaluating rule-based baseline...")
    baseline_results = evaluate_baseline(raw_env)

    print(f"\n{'='*60}")
    print("Results Comparison")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'RL Agent':>12} {'Baseline':>12}")
    print(f"{'-'*54}")
    for key in rl_results:
        rl_val = rl_results[key]
        bl_val = baseline_results[key]
        print(f"{key:<30} {rl_val:>12.2f} {bl_val:>12.2f}")
    print(f"{'='*60}")

    # Save results
    results = {
        "args": vars(args),
        "rl_agent": rl_results,
        "baseline": baseline_results,
    }
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
