"""Calibration engines: SMM and Bayesian synthetic likelihood.

These engines estimate structural model parameters by matching
simulated moments to empirical targets from data.

SMM: Minimizes weighted distance between simulated and empirical moments.
Bayesian: Samples the posterior distribution using synthetic likelihood.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from econosim.calibration.moments import MomentSet
from econosim.calibration.parameters import ParameterRegistry
from econosim.config.schema import SimulationConfig

logger = logging.getLogger(__name__)


class CalibrationProfile(BaseModel):
    """Configuration for a calibration run."""

    name: str = "default"
    num_simulations: int = 5  # simulations per evaluation (for averaging)
    num_periods: int = 120
    burn_in: int = 20
    seed_base: int = 1000
    weighting_method: str = "inverse_variance"  # for SMM
    optimizer: str = "nelder-mead"  # "nelder-mead", "powell", "differential-evolution"
    max_iterations: int = 200
    tolerance: float = 1e-6
    use_common_random_numbers: bool = True
    verbose: bool = True


@dataclass
class CalibrationResult:
    """Output of a calibration run."""

    profile_name: str
    method: str
    estimated_params: dict[str, float]
    moment_fit: dict[str, dict[str, float]]  # moment_name -> {empirical, simulated, error}
    weighted_objective: float
    num_evaluations: int
    elapsed_seconds: float
    converged: bool
    seed: int
    config_hash: str = ""
    posterior_samples: np.ndarray | None = None  # for Bayesian
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Calibration Result: {self.profile_name} ({self.method})",
            f"  Converged: {self.converged}",
            f"  Objective: {self.weighted_objective:.6f}",
            f"  Evaluations: {self.num_evaluations}",
            f"  Time: {self.elapsed_seconds:.1f}s",
            "",
            "  Estimated Parameters:",
        ]
        for k, v in self.estimated_params.items():
            lines.append(f"    {k}: {v:.6f}")
        lines.append("")
        lines.append("  Moment Fit:")
        for m, fit in self.moment_fit.items():
            lines.append(
                f"    {m}: empirical={fit['empirical']:.4f}, "
                f"simulated={fit['simulated']:.4f}, "
                f"error={fit['error']:.4f}"
            )
        return "\n".join(lines)


class SimulationObjective:
    """Wraps the simulation into a callable objective function for calibration.

    Given parameter values, runs the simulation and computes moment distance.
    """

    def __init__(
        self,
        base_config: SimulationConfig,
        registry: ParameterRegistry,
        moments: MomentSet,
        profile: CalibrationProfile,
        sim_runner: Callable[[SimulationConfig], pd.DataFrame] | None = None,
        policies: dict[str, Any] | None = None,
    ) -> None:
        self.base_config = base_config
        self.registry = registry
        self.moments = moments
        self.profile = profile
        self.policies = policies or {}
        self._sim_runner = sim_runner or self._make_policy_runner()
        self._eval_count = 0
        self._W = moments.weighting_matrix(profile.weighting_method)

        # Override sim_runner if user provided one explicitly
        if sim_runner is not None:
            self._sim_runner = sim_runner

        # Pre-generate seeds for common random numbers
        if profile.use_common_random_numbers:
            rng = np.random.default_rng(profile.seed_base)
            self._seeds = rng.integers(0, 100000, size=profile.num_simulations).tolist()
        else:
            self._seeds = list(range(
                profile.seed_base,
                profile.seed_base + profile.num_simulations,
            ))

    def _make_policy_runner(self) -> Callable[[SimulationConfig], pd.DataFrame]:
        """Create a sim runner that passes policies to the simulation."""
        policies = self.policies

        def runner(config: SimulationConfig) -> pd.DataFrame:
            from econosim.engine.simulation import build_simulation, step
            from econosim.metrics.collector import history_to_dataframe, enrich_dataframe

            state = build_simulation(config)
            state.firm_policy = policies.get("firm_policy")
            state.household_policy = policies.get("household_policy")
            state.bank_policy = policies.get("bank_policy")
            state.government_policy = policies.get("government_policy")
            for _ in range(config.num_periods):
                step(state)
            return enrich_dataframe(history_to_dataframe(state.history))

        return runner

    def _apply_params(self, param_dict: dict[str, float]) -> SimulationConfig:
        """Apply calibrated parameters to base config."""
        config = self.base_config.model_copy(deep=True)
        for param in self.registry.calibrated:
            if param.name not in param_dict:
                continue
            value = param_dict[param.name]
            parts = param.config_path.split(".")
            if len(parts) == 2:
                section, key = parts
                sub_model = getattr(config, section)
                setattr(sub_model, key, value)
            else:
                setattr(config, parts[0], value)
        return config

    def evaluate(self, param_dict: dict[str, float]) -> tuple[float, np.ndarray]:
        """Evaluate objective: run simulations and compute moment distance.

        Returns (weighted_objective, simulated_moments_vector).
        """
        self._eval_count += 1
        config = self._apply_params(param_dict)

        # Run multiple simulations and average moments
        all_moments = []
        for seed in self._seeds:
            cfg = config.model_copy(update={
                "seed": seed,
                "num_periods": self.profile.num_periods,
            })
            try:
                df = self._sim_runner(cfg)
                sim_moments = self.moments.compute_all(df)
                all_moments.append(sim_moments)
            except Exception as e:
                logger.warning(f"Simulation failed with seed {seed}: {e}")
                continue

        if not all_moments:
            return np.inf, np.full(len(self.moments), np.nan)

        avg_moments = np.nanmean(all_moments, axis=0)

        # Compute weighted distance
        empirical = self.moments.empirical_values
        diff = avg_moments - empirical
        # Replace NaN with large penalty
        diff = np.where(np.isnan(diff), 10.0, diff)

        objective = float(diff @ self._W @ diff)

        if self.profile.verbose and self._eval_count % 10 == 0:
            logger.info(
                f"Eval {self._eval_count}: obj={objective:.6f}, "
                f"params={param_dict}"
            )

        return objective, avg_moments

    def __call__(self, param_vector: np.ndarray) -> float:
        """Callable interface for scipy optimizers (vector in, scalar out)."""
        param_dict = self.registry.from_vector(param_vector)
        # Clip to bounds
        for param in self.registry.calibrated:
            if param.name in param_dict:
                param_dict[param.name] = max(
                    param.lower_bound,
                    min(param.upper_bound, param_dict[param.name]),
                )
        obj, _ = self.evaluate(param_dict)
        return obj


class SmmCalibrator:
    """Simulated Method of Moments calibrator.

    Estimates parameters by minimizing the weighted distance between
    simulated and empirical moments using scipy optimization.
    """

    def __init__(
        self,
        objective: SimulationObjective,
        registry: ParameterRegistry,
        profile: CalibrationProfile,
    ) -> None:
        self.objective = objective
        self.registry = registry
        self.profile = profile

    def calibrate(self) -> CalibrationResult:
        """Run SMM calibration."""
        from scipy.optimize import minimize, differential_evolution

        start_time = time.monotonic()

        x0 = self.registry.calibrated_defaults()
        bounds = self.registry.calibrated_bounds()

        if self.profile.optimizer == "differential-evolution":
            result = differential_evolution(
                self.objective,
                bounds=bounds,
                maxiter=self.profile.max_iterations,
                tol=self.profile.tolerance,
                seed=self.profile.seed_base,
            )
        else:
            result = minimize(
                self.objective,
                x0=x0,
                method=self.profile.optimizer,
                bounds=bounds if self.profile.optimizer != "nelder-mead" else None,
                options={
                    "maxiter": self.profile.max_iterations,
                    "xatol": self.profile.tolerance,
                    "fatol": self.profile.tolerance,
                },
            )

        elapsed = time.monotonic() - start_time
        estimated = self.registry.from_vector(result.x)

        # Final evaluation for moment fit
        final_obj, sim_moments = self.objective.evaluate(estimated)
        emp_moments = self.objective.moments.empirical_values
        moment_fit = {}
        for i, m in enumerate(self.objective.moments.moments):
            moment_fit[m.name] = {
                "empirical": float(emp_moments[i]),
                "simulated": float(sim_moments[i]) if not np.isnan(sim_moments[i]) else None,
                "error": float(sim_moments[i] - emp_moments[i]) if not np.isnan(sim_moments[i]) else None,
            }

        return CalibrationResult(
            profile_name=self.profile.name,
            method="SMM",
            estimated_params=estimated,
            moment_fit=moment_fit,
            weighted_objective=final_obj,
            num_evaluations=self.objective._eval_count,
            elapsed_seconds=elapsed,
            converged=result.success if hasattr(result, "success") else True,
            seed=self.profile.seed_base,
        )


class BayesianCalibrator:
    """Bayesian calibration using synthetic likelihood / MCMC.

    Uses a random-walk Metropolis-Hastings sampler with
    synthetic Gaussian likelihood approximation.
    """

    def __init__(
        self,
        objective: SimulationObjective,
        registry: ParameterRegistry,
        profile: CalibrationProfile,
        num_samples: int = 1000,
        proposal_scale: float = 0.01,
    ) -> None:
        self.objective = objective
        self.registry = registry
        self.profile = profile
        self.num_samples = num_samples
        self.proposal_scale = proposal_scale

    def calibrate(self) -> CalibrationResult:
        """Run Bayesian calibration via MCMC."""
        start_time = time.monotonic()
        rng = np.random.default_rng(self.profile.seed_base)

        # Initialize at default values
        current = self.registry.calibrated_defaults()
        current_dict = self.registry.from_vector(current)
        current_obj, current_moments = self.objective.evaluate(current_dict)
        current_log_prior = self.registry.log_prior(current_dict)
        current_log_post = -0.5 * current_obj + current_log_prior

        samples = [current.copy()]
        objectives = [current_obj]
        accept_count = 0

        for i in range(1, self.num_samples):
            # Propose
            proposal = current + rng.normal(0, self.proposal_scale, size=len(current))

            # Clip to bounds
            for j, param in enumerate(self.registry.calibrated):
                proposal[j] = max(param.lower_bound, min(param.upper_bound, proposal[j]))

            prop_dict = self.registry.from_vector(proposal)
            prop_obj, prop_moments = self.objective.evaluate(prop_dict)
            prop_log_prior = self.registry.log_prior(prop_dict)
            prop_log_post = -0.5 * prop_obj + prop_log_prior

            # MH acceptance
            log_alpha = prop_log_post - current_log_post
            if np.log(rng.uniform()) < log_alpha:
                current = proposal
                current_dict = prop_dict
                current_obj = prop_obj
                current_log_post = prop_log_post
                accept_count += 1

            samples.append(current.copy())
            objectives.append(current_obj)

            if self.profile.verbose and (i + 1) % 100 == 0:
                accept_rate = accept_count / (i + 1)
                logger.info(
                    f"MCMC step {i+1}/{self.num_samples}: "
                    f"obj={current_obj:.6f}, accept_rate={accept_rate:.2f}"
                )

        elapsed = time.monotonic() - start_time
        samples_array = np.array(samples)

        # Use posterior mean as point estimate
        burn = min(self.num_samples // 4, 250)
        posterior_mean = np.mean(samples_array[burn:], axis=0)
        estimated = self.registry.from_vector(posterior_mean)

        # Final evaluation
        final_obj, sim_moments = self.objective.evaluate(estimated)
        emp_moments = self.objective.moments.empirical_values
        moment_fit = {}
        for i, m in enumerate(self.objective.moments.moments):
            moment_fit[m.name] = {
                "empirical": float(emp_moments[i]),
                "simulated": float(sim_moments[i]) if not np.isnan(sim_moments[i]) else None,
                "error": float(sim_moments[i] - emp_moments[i]) if not np.isnan(sim_moments[i]) else None,
            }

        return CalibrationResult(
            profile_name=self.profile.name,
            method="Bayesian",
            estimated_params=estimated,
            moment_fit=moment_fit,
            weighted_objective=final_obj,
            num_evaluations=self.objective._eval_count,
            elapsed_seconds=elapsed,
            converged=True,
            seed=self.profile.seed_base,
            posterior_samples=samples_array,
            metadata={
                "accept_rate": accept_count / self.num_samples,
                "burn_in": burn,
                "num_samples": self.num_samples,
            },
        )


def _default_sim_runner(config: SimulationConfig) -> pd.DataFrame:
    """Default simulation runner that returns enriched metrics DataFrame."""
    from econosim.engine.simulation import build_simulation, step
    from econosim.metrics.collector import history_to_dataframe, enrich_dataframe

    state = build_simulation(config)
    for _ in range(config.num_periods):
        step(state)
    df = enrich_dataframe(history_to_dataframe(state.history))
    return df
