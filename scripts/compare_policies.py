#!/usr/bin/env python3
"""
Policy comparison: RL-trained agents vs rule-based baselines.

Runs both RL and rule-based agents across multiple scenarios and seeds,
collecting macro metrics for comparison. Generates comparison tables and
optionally exports data for visualization.

Usage:
    python scripts/compare_policies.py --agent firm --model outputs/rl/firm/final_model
    python scripts/compare_policies.py --agent firm --model outputs/rl/firm/final_model \
        --scenarios baseline supply_shock demand_shock
    python scripts/compare_policies.py --all-agents --model-dir outputs/rl

Requires: pip install "econosim[rl]"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from stable_baselines3 import PPO
except ImportError:
    print("stable-baselines3 is required. Install with: pip install 'econosim[rl]'")
    sys.exit(1)

from econosim.config.schema import SimulationConfig, ShockSpec
from econosim.rl.firm_env import FirmEnv
from econosim.rl.household_env import HouseholdEnv
from econosim.rl.government_env import GovernmentEnv
from econosim.rl.bank_env import BankEnv

AGENT_ENVS = {
    "firm": FirmEnv,
    "household": HouseholdEnv,
    "government": GovernmentEnv,
    "bank": BankEnv,
}

NEUTRAL_ACTIONS = {
    "firm": np.array([1.0, 1.0, 0.5], dtype=np.float32),
    "household": np.array([0.5, 1.0], dtype=np.float32),
    "government": np.array([0.2, 1.0, 1.0], dtype=np.float32),
    "bank": np.array([0.005, 0.08], dtype=np.float32),
}

# Pre-defined scenarios for comparison
SCENARIOS = {
    "baseline": SimulationConfig(name="baseline", num_periods=60, seed=42),
    "supply_shock": SimulationConfig(
        name="supply_shock", num_periods=60, seed=42,
        shocks=[ShockSpec(period=20, shock_type="supply",
                          parameter="labor_productivity", magnitude=0.7)],
    ),
    "demand_shock": SimulationConfig(
        name="demand_shock", num_periods=60, seed=42,
        shocks=[ShockSpec(period=20, shock_type="demand",
                          parameter="consumption_propensity", magnitude=0.6)],
    ),
    "credit_crunch": SimulationConfig(
        name="credit_crunch", num_periods=60, seed=42,
        shocks=[ShockSpec(period=20, shock_type="credit",
                          parameter="capital_adequacy_ratio", magnitude=2.0)],
    ),
    "tax_hike": SimulationConfig(
        name="tax_hike", num_periods=60, seed=42,
        shocks=[ShockSpec(period=20, shock_type="fiscal",
                          parameter="income_tax_rate", magnitude=1.5)],
    ),
    "stimulus": SimulationConfig(
        name="stimulus", num_periods=60, seed=42,
        shocks=[ShockSpec(period=20, shock_type="fiscal",
                          parameter="spending_per_period", magnitude=2.0)],
    ),
}

METRICS_TO_TRACK = [
    "gdp", "unemployment_rate", "avg_price", "gini_deposits",
    "total_consumption", "total_production", "total_loans_outstanding",
]


def run_episode(env, action_fn, seed: int) -> dict:
    """Run one episode with a given action function, collecting metrics over time."""
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    done = False
    trajectory = []

    while not done:
        action = action_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        metrics = info.get("metrics", {})
        trajectory.append({k: metrics.get(k, 0.0) for k in METRICS_TO_TRACK})

    return {
        "total_reward": total_reward,
        "trajectory": trajectory,
        "final_metrics": trajectory[-1] if trajectory else {},
    }


def compare_agent(
    agent_type: str,
    model_path: str,
    scenarios: list[str],
    seeds: list[int],
    reward_type: str = None,
) -> dict:
    """Compare RL agent vs baseline across scenarios and seeds."""

    env_class = AGENT_ENVS[agent_type]
    neutral_action = NEUTRAL_ACTIONS[agent_type]

    model = PPO.load(model_path)
    results = {}

    for scenario_name in scenarios:
        if scenario_name not in SCENARIOS:
            print(f"  Warning: Unknown scenario '{scenario_name}', skipping")
            continue

        config = SCENARIOS[scenario_name]
        rl_runs = []
        baseline_runs = []

        for seed in seeds:
            cfg = config.model_copy(update={"seed": seed})
            kwargs = {"config": cfg, "max_steps": cfg.num_periods}
            if reward_type:
                kwargs["reward_type"] = reward_type

            env = env_class(**kwargs)

            # RL agent
            rl_result = run_episode(
                env,
                lambda obs: model.predict(obs, deterministic=True)[0],
                seed,
            )
            rl_runs.append(rl_result)

            # Baseline
            baseline_result = run_episode(
                env,
                lambda obs: neutral_action,
                seed,
            )
            baseline_runs.append(baseline_result)

        results[scenario_name] = {
            "rl": _aggregate_runs(rl_runs),
            "baseline": _aggregate_runs(baseline_runs),
        }

    return results


def _aggregate_runs(runs: list[dict]) -> dict:
    """Aggregate metrics across multiple seeded runs."""
    rewards = [r["total_reward"] for r in runs]
    final_metrics = {}

    for key in METRICS_TO_TRACK:
        values = [r["final_metrics"].get(key, 0.0) for r in runs]
        final_metrics[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    # Trajectory means (averaged across runs, per timestep)
    max_len = max(len(r["trajectory"]) for r in runs)
    traj_means = {}
    for key in METRICS_TO_TRACK:
        per_step = []
        for t in range(max_len):
            step_vals = [
                r["trajectory"][t][key]
                for r in runs if t < len(r["trajectory"])
            ]
            per_step.append(float(np.mean(step_vals)))
        traj_means[key] = per_step

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "final_metrics": final_metrics,
        "trajectory_means": traj_means,
    }


def print_comparison(agent_type: str, results: dict) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'=' * 80}")
    print(f"Policy Comparison: {agent_type.upper()} — RL vs Rule-Based Baseline")
    print(f"{'=' * 80}")

    for scenario_name, data in results.items():
        rl = data["rl"]
        bl = data["baseline"]

        print(f"\n  Scenario: {scenario_name}")
        print(f"  {'Metric':<25} {'RL Mean':>10} {'BL Mean':>10} {'Delta':>10} {'RL Win?':>8}")
        print(f"  {'-' * 65}")

        print(f"  {'Total Reward':<25} {rl['reward_mean']:>10.1f} {bl['reward_mean']:>10.1f} "
              f"{rl['reward_mean'] - bl['reward_mean']:>+10.1f} "
              f"{'  YES' if rl['reward_mean'] > bl['reward_mean'] else '   NO':>8}")

        for key in METRICS_TO_TRACK:
            rl_val = rl["final_metrics"][key]["mean"]
            bl_val = bl["final_metrics"][key]["mean"]
            delta = rl_val - bl_val

            # For unemployment and gini, lower is better
            if key in ("unemployment_rate", "gini_deposits"):
                better = "  YES" if rl_val < bl_val else "   NO"
            else:
                better = "  YES" if rl_val > bl_val else "   NO"

            print(f"  {key:<25} {rl_val:>10.2f} {bl_val:>10.2f} {delta:>+10.2f} {better:>8}")


def main():
    parser = argparse.ArgumentParser(description="Compare RL vs rule-based policies")
    parser.add_argument("--agent", type=str, choices=list(AGENT_ENVS.keys()),
                        help="Agent type to compare")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--all-agents", action="store_true",
                        help="Compare all agent types")
    parser.add_argument("--model-dir", type=str, default="outputs/rl",
                        help="Base directory for trained models (with --all-agents)")
    parser.add_argument("--scenarios", nargs="+",
                        default=list(SCENARIOS.keys()),
                        help="Scenarios to compare")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 123, 456, 789, 1024],
                        help="Seeds for multiple runs")
    parser.add_argument("--reward", type=str, default=None,
                        help="Reward type for environments")
    parser.add_argument("--output", type=str, default="outputs/comparison",
                        help="Output directory for results")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.all_agents:
        for agent_type in AGENT_ENVS:
            model_path = Path(args.model_dir) / agent_type / "final_model.zip"
            if not model_path.exists():
                print(f"Skipping {agent_type}: model not found at {model_path}")
                continue
            print(f"\nComparing {agent_type}...")
            results = compare_agent(agent_type, str(model_path), args.scenarios, args.seeds)
            print_comparison(agent_type, results)
            all_results[agent_type] = results

    elif args.agent and args.model:
        results = compare_agent(args.agent, args.model, args.scenarios,
                                args.seeds, args.reward)
        print_comparison(args.agent, results)
        all_results[args.agent] = results

    else:
        parser.error("Specify --agent and --model, or use --all-agents")

    # Save results
    results_path = out_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
