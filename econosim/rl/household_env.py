"""
Gymnasium-compatible RL environment for a single household agent.

The RL agent controls one household's consumption and wage decisions.
All other agents run on their existing rule-based logic.

Observation: 12-dim continuous (household state + macro indicators)
Action: 2D continuous — consumption fraction, reservation wage multiplier
Reward: utility (consumption-based) with optional macro terms
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from econosim.config.schema import SimulationConfig
from econosim.engine.simulation import build_simulation, step as sim_step


def _rl_desired_consumption(hh, fraction: float) -> float:
    """RL-controlled consumption: fraction of available deposits."""
    desired = fraction * max(0.0, hh.deposits)
    return min(desired, max(0.0, hh.deposits))


def _rl_accept_wage(reservation: float, offered: float) -> bool:
    """RL-controlled wage acceptance."""
    return offered >= reservation


class HouseholdEnv(gym.Env):
    """Single-household RL environment wrapping the full EconoSim simulation.

    The RL agent replaces the rule-based decision logic for household_000.
    All other agents behave according to their rule-based defaults.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: SimulationConfig | None = None,
        max_steps: int | None = None,
        reward_type: str = "utility",
        controlled_hh_id: str = "hh_0000",
    ) -> None:
        super().__init__()

        self.config = config or SimulationConfig(num_periods=120)
        self.max_steps = max_steps or self.config.num_periods
        self.reward_type = reward_type
        self.controlled_hh_id = controlled_hh_id

        # Action space: [consumption_frac, reservation_wage_mult]
        # consumption_frac: fraction of deposits to consume (0.0 to 1.0)
        # reservation_wage_mult: multiplier on base reservation wage (0.5 to 2.0)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.5], dtype=np.float32),
            high=np.array([1.0, 2.0], dtype=np.float32),
        )

        self._obs_keys = [
            "deposits", "employed", "wage_income", "consumption_spending",
            "disposable_income", "total_debt", "taxes_paid", "transfers_received",
            "avg_price", "unemployment_rate", "gdp", "gini_deposits",
        ]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self._obs_keys),),
            dtype=np.float32,
        )

        self._state = None
        self._hh = None
        self._base_reservation_wage = None
        self._step_count = 0

    def _get_hh(self):
        for hh in self._state.households:
            if hh.agent_id == self.controlled_hh_id:
                return hh
        raise ValueError(f"Household {self.controlled_hh_id} not found")

    def _get_obs(self) -> np.ndarray:
        hh = self._hh
        m = self._state.history[-1] if self._state.history else {}
        return np.array([
            hh.deposits,
            float(hh.employed),
            hh.wage_income,
            hh.consumption_spending,
            hh.disposable_income,
            hh.total_debt,
            hh.taxes_paid,
            hh.transfers_received,
            m.get("avg_price", 10.0),
            m.get("unemployment_rate", 0.0),
            m.get("gdp", 0.0),
            m.get("gini_deposits", 0.0),
        ], dtype=np.float32)

    def _compute_reward(self, metrics: dict[str, Any]) -> float:
        hh = self._hh

        if self.reward_type == "utility":
            # Log utility of consumption + deposit safety buffer
            c = max(hh.consumption_spending, 0.01)
            return float(np.log(c) + 0.1 * np.log(max(hh.deposits, 1.0)))

        elif self.reward_type == "consumption":
            return float(hh.consumption_spending)

        elif self.reward_type == "balanced":
            c = max(hh.consumption_spending, 0.01)
            gdp = metrics.get("gdp", 0.0)
            return float(np.log(c) + 0.001 * gdp)

        return float(hh.consumption_spending)

    def _apply_actions(self, consumption_frac: float, reservation_wage_mult: float) -> None:
        hh = self._hh
        new_reservation = max(1.0, self._base_reservation_wage * reservation_wage_mult)

        hh.desired_consumption = lambda: _rl_desired_consumption(hh, consumption_frac)
        hh.accept_wage = lambda wage: _rl_accept_wage(new_reservation, wage)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            config = self.config.model_copy(update={"seed": seed})
        else:
            config = self.config

        self._state = build_simulation(config)
        self._hh = self._get_hh()
        self._base_reservation_wage = self._hh.reservation_wage
        self._step_count = 0

        return self._get_obs(), {"period": 0}

    def step(
        self, action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        consumption_frac = float(np.clip(action[0], 0.0, 1.0))
        reservation_wage_mult = float(np.clip(action[1], 0.5, 2.0))

        self._apply_actions(consumption_frac, reservation_wage_mult)

        metrics = sim_step(self._state)
        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(metrics)
        terminated = self._step_count >= self.max_steps
        truncated = False

        info = {
            "period": self._step_count,
            "metrics": metrics,
            "hh_obs": self._hh.get_observation(),
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self._state is None:
            return
        hh = self._hh
        m = self._state.history[-1] if self._state.history else {}
        print(
            f"t={self._step_count:3d} | "
            f"GDP={m.get('gdp', 0):8.0f} | "
            f"Emp={'Y' if hh.employed else 'N'} | "
            f"Dep={hh.deposits:.0f} | "
            f"Inc={hh.wage_income:.0f} C={hh.consumption_spending:.0f}"
        )
