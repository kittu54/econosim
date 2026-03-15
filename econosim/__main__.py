"""CLI entry point: python -m econosim [scenario.yaml] [--output-dir DIR] [--periods N] [--seed S]"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from econosim.config.schema import SimulationConfig
from econosim.engine.simulation import run_simulation
from econosim.metrics.collector import history_to_dataframe, export_results, summary_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description="EconoSim — multi-agent economic simulator")
    parser.add_argument("scenario", nargs="?", default=None, help="Path to scenario YAML file")
    parser.add_argument("--periods", type=int, default=None, help="Override number of periods")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--name", type=str, default=None, help="Override scenario name")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("econosim")

    if args.scenario:
        import yaml
        with open(args.scenario) as f:
            data = yaml.safe_load(f)
        config = SimulationConfig(**data)
    else:
        config = SimulationConfig()

    if args.periods is not None:
        config.num_periods = args.periods
    if args.seed is not None:
        config.seed = args.seed
    if args.name is not None:
        config.name = args.name
    config.output_dir = args.output_dir

    logger.info(f"Running scenario '{config.name}': {config.num_periods} periods, seed={config.seed}")
    state = run_simulation(config)

    df = history_to_dataframe(state.history)
    out_path = export_results(df, Path(config.output_dir) / config.name, config.name)
    logger.info(f"Metrics exported to {out_path}")

    stats = summary_statistics(df)
    print(f"\n{'='*60}")
    print(f"  Simulation Complete: {config.name}")
    print(f"  Periods: {config.num_periods} | Seed: {config.seed}")
    print(f"{'='*60}")
    if "gdp" in stats:
        print(f"  GDP        — mean: {stats['gdp']['mean']:>10.1f}  final: {stats['gdp']['final']:>10.1f}")
    if "unemployment_rate" in stats:
        print(f"  Unemp Rate — mean: {stats['unemployment_rate']['mean']:>10.4f}  final: {stats['unemployment_rate']['final']:>10.4f}")
    if "avg_price" in stats:
        print(f"  Avg Price  — mean: {stats['avg_price']['mean']:>10.4f}  final: {stats['avg_price']['final']:>10.4f}")
    if "gini_deposits" in stats:
        print(f"  Gini (dep) — mean: {stats['gini_deposits']['mean']:>10.4f}  final: {stats['gini_deposits']['final']:>10.4f}")
    if "total_loans_outstanding" in stats:
        print(f"  Loans Out  — mean: {stats['total_loans_outstanding']['mean']:>10.1f}  final: {stats['total_loans_outstanding']['final']:>10.1f}")
    print(f"{'='*60}")
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
