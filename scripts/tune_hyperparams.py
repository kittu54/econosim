#!/usr/bin/env python3
"""
Hyperparameter tuning for EconoSim RL agents.

Runs a grid search over key hyperparameters and reports the best configuration.
Supports parallel evaluation with multiple seeds for robust comparison.

Usage:
    python scripts/tune_hyperparams.py --agent firm --timesteps 20000
    python scripts/tune_hyperparams.py --agent government --timesteps 30000 --seeds 3

Requires: pip install "econosim[rl]"
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    print("stable-baselines3 is required. Install with: pip install 'econosim[rl]'")
    sys.exit(1)

from econosim.config.schema import SimulationConfig
from econosim.rl.firm_env import FirmEnv
from econosim.rl.household_env import HouseholdEnv
from econosim.rl.government_env import GovernmentEnv
from econosim.rl.bank_env import BankEnv

AGENT_ENVS = {
    "firm": (FirmEnv, "profit"),
    "household": (HouseholdEnv, "utility"),
    "government": (GovernmentEnv, "welfare"),
    "bank": (BankEnv, "profit"),
}

NEUTRAL_ACTIONS = {
    "firm": np.array([1.0, 1.0, 0.5], dtype=np.float32),
    "household": np.array([0.5, 1.0], dtype=np.float32),
    "government": np.array([0.2, 1.0, 1.0], dtype=np.float32),
    "bank": np.array([0.005, 0.08], dtype=np.float32),
}

# Hyperparameter search space
SEARCH_SPACE = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "n_steps": [128, 256, 512],
    "batch_size": [32, 64],
    "n_epochs": [5, 10],
    "ent_coef": [0.005, 0.01, 0.02],
    "gamma": [0.98, 0.99, 0.995],
}


def evaluate(model, env, neutral_action, n_episodes: int = 3) -> dict:
    """Quick evaluation of a model vs baseline."""
    rl_rewards = []
    bl_rewards = []

    for ep in range(n_episodes):
        # RL
        obs, _ = env.reset(seed=ep + 2000)
        total = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        rl_rewards.append(total)

        # Baseline
        obs, _ = env.reset(seed=ep + 2000)
        total = 0.0
        done = False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(neutral_action)
            total += reward
            done = terminated or truncated
        bl_rewards.append(total)

    rl_mean = float(np.mean(rl_rewards))
    bl_mean = float(np.mean(bl_rewards))

    return {
        "rl_reward": rl_mean,
        "baseline_reward": bl_mean,
        "improvement": rl_mean - bl_mean,
        "improvement_pct": (rl_mean - bl_mean) / max(abs(bl_mean), 1e-8) * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for RL agents")
    parser.add_argument("--agent", type=str, required=True,
                        choices=list(AGENT_ENVS.keys()))
    parser.add_argument("--timesteps", type=int, default=20_000,
                        help="Training timesteps per trial")
    parser.add_argument("--max-steps", type=int, default=60, help="Episode length")
    parser.add_argument("--seeds", type=int, default=2,
                        help="Number of seeds for evaluation")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Limit number of trials (default: all combinations)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    env_class, default_reward = AGENT_ENVS[args.agent]
    neutral_action = NEUTRAL_ACTIONS[args.agent]
    out_dir = Path(args.output or f"outputs/tuning/{args.agent}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate hyperparameter combinations
    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    combinations = list(itertools.product(*values))

    if args.max_trials and args.max_trials < len(combinations):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(combinations), size=args.max_trials, replace=False)
        combinations = [combinations[i] for i in sorted(indices)]

    print(f"{'=' * 60}")
    print(f"Hyperparameter Tuning — {args.agent.capitalize()} Agent")
    print(f"{'=' * 60}")
    print(f"  Trials:      {len(combinations)}")
    print(f"  Timesteps:   {args.timesteps:,} per trial")
    print(f"  Eval seeds:  {args.seeds}")
    print(f"{'=' * 60}\n")

    results = []
    best_improvement = -np.inf
    best_params = None

    for trial_idx, combo in enumerate(combinations):
        hp = dict(zip(keys, combo))

        # Validate batch_size <= n_steps
        if hp["batch_size"] > hp["n_steps"]:
            continue

        print(f"Trial {trial_idx + 1}/{len(combinations)}: {hp}")
        start_time = time.time()

        try:
            config = SimulationConfig(num_periods=args.max_steps, seed=42)
            env = Monitor(env_class(config=config, max_steps=args.max_steps,
                                     reward_type=default_reward))

            model = PPO(
                "MlpPolicy", env, verbose=0, seed=42,
                learning_rate=hp["learning_rate"],
                n_steps=hp["n_steps"],
                batch_size=hp["batch_size"],
                n_epochs=hp["n_epochs"],
                gamma=hp["gamma"],
                ent_coef=hp["ent_coef"],
            )

            model.learn(total_timesteps=args.timesteps)

            eval_env = env_class(
                config=SimulationConfig(num_periods=args.max_steps, seed=42),
                max_steps=args.max_steps,
                reward_type=default_reward,
            )
            eval_result = evaluate(model, eval_env, neutral_action, args.seeds)

            elapsed = time.time() - start_time
            trial_result = {
                "trial": trial_idx,
                "hyperparams": hp,
                "elapsed_seconds": round(elapsed, 1),
                **eval_result,
            }
            results.append(trial_result)

            if eval_result["improvement"] > best_improvement:
                best_improvement = eval_result["improvement"]
                best_params = hp
                model.save(str(out_dir / "best_model"))

            print(f"  RL={eval_result['rl_reward']:.1f} "
                  f"BL={eval_result['baseline_reward']:.1f} "
                  f"Imp={eval_result['improvement']:+.1f} "
                  f"({eval_result['improvement_pct']:+.1f}%) "
                  f"[{elapsed:.0f}s]"
                  f"{' *** BEST ***' if hp == best_params else ''}")

            env.close()

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "trial": trial_idx,
                "hyperparams": hp,
                "error": str(e),
            })

    # Summary
    print(f"\n{'=' * 60}")
    print("Tuning Summary")
    print(f"{'=' * 60}")
    if best_params:
        print(f"Best hyperparameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"Best improvement: {best_improvement:+.2f}")
    else:
        print("No successful trials.")

    # Top 5 results
    successful = [r for r in results if "improvement" in r]
    if successful:
        top5 = sorted(successful, key=lambda r: r["improvement"], reverse=True)[:5]
        print(f"\nTop 5 configurations:")
        for i, r in enumerate(top5, 1):
            print(f"  {i}. Imp={r['improvement']:+.1f} "
                  f"lr={r['hyperparams']['learning_rate']} "
                  f"ns={r['hyperparams']['n_steps']} "
                  f"bs={r['hyperparams']['batch_size']} "
                  f"ne={r['hyperparams']['n_epochs']} "
                  f"ec={r['hyperparams']['ent_coef']} "
                  f"g={r['hyperparams']['gamma']}")

    # Save results
    results_path = out_dir / "tuning_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "agent": args.agent,
            "args": vars(args),
            "best_params": best_params,
            "best_improvement": best_improvement,
            "trials": results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
