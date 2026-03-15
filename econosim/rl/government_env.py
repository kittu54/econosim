"""
Gymnasium-compatible RL environment for the government agent.

The RL agent controls fiscal policy: tax rate, transfer amount, and spending.
All other agents run on their existing rule-based logic.

Observation: 12-dim continuous (fiscal state + macro indicators)
Action: 3D continuous — tax rate, transfer multiplier, spending multiplier
Reward: social welfare (GDP, employment) with optional deficit penalties
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from econosim.config.schema import SimulationConfig
from econosim.engine.simulation import build_simulation, step as sim_step


class GovernmentEnv(gym.Env):
    """Government RL environment wrapping the full EconoSim simulation.

    The RL agent sets fiscal policy parameters each period.
    All other agents behave according to their rule-based defaults.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: SimulationConfig | None = None,
        max_steps: int | None = None,
        reward_type: str = "welfare",
    ) -> None:
        super().__init__()

        self.config = config or SimulationConfig(num_periods=120)
        self.max_steps = max_steps or self.config.num_periods
        self.reward_type = reward_type

        # Action space: [tax_rate, transfer_mult, spending_mult]
        # tax_rate: absolute tax rate (0.0 to 0.5)
        # transfer_mult: multiplier on base transfer (0.0 to 3.0)
        # spending_mult: multiplier on base spending (0.0 to 3.0)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([0.5, 3.0, 3.0], dtype=np.float32),
        )

        self._obs_keys = [
            "deposits", "tax_revenue", "transfers_paid", "goods_spending",
            "budget_balance", "money_created", "cumulative_money_created",
            "unemployment_rate", "gdp", "avg_price", "gini_deposits",
            "total_hh_deposits",
        ]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self._obs_keys),),
            dtype=np.float32,
        )

        self._state = None
        self._govt = None
        self._base_transfer = None
        self._base_spending = None
        self._step_count = 0

    def _get_obs(self) -> np.ndarray:
        g = self._govt
        m = self._state.history[-1] if self._state.history else {}
        return np.array([
            g.deposits,
            g.tax_revenue,
            g.transfers_paid,
            g.goods_spending,
            g.budget_balance,
            g.money_created,
            g.cumulative_money_created,
            m.get("unemployment_rate", 0.0),
            m.get("gdp", 0.0),
            m.get("avg_price", 10.0),
            m.get("gini_deposits", 0.0),
            m.get("total_hh_deposits", 0.0),
        ], dtype=np.float32)

    def _compute_reward(self, metrics: dict[str, Any]) -> float:
        gdp = metrics.get("gdp", 0.0)
        unemployment = metrics.get("unemployment_rate", 0.0)
        gini = metrics.get("gini_deposits", 0.0)

        if self.reward_type == "welfare":
            # Maximize GDP, minimize unemployment and inequality
            return float(0.01 * gdp - 500.0 * unemployment - 100.0 * gini)

        elif self.reward_type == "gdp":
            return float(gdp)

        elif self.reward_type == "employment":
            return float(-unemployment)

        elif self.reward_type == "balanced":
            deficit = abs(self._govt.budget_balance)
            return float(0.01 * gdp - 500.0 * unemployment - 0.001 * deficit)

        return float(0.01 * gdp - 500.0 * unemployment)

    def _apply_actions(
        self, tax_rate: float, transfer_mult: float, spending_mult: float
    ) -> None:
        g = self._govt
        g.income_tax_rate = tax_rate
        g.transfer_per_unemployed = max(0.0, self._base_transfer * transfer_mult)
        g.spending_per_period = max(0.0, self._base_spending * spending_mult)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            config = self.config.model_copy(update={"seed": seed})
        else:
            config = self.config

        self._state = build_simulation(config)
        self._govt = self._state.government
        self._base_transfer = self._govt.transfer_per_unemployed
        self._base_spending = self._govt.spending_per_period
        self._step_count = 0

        return self._get_obs(), {"period": 0}

    def step(
        self, action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        tax_rate = float(np.clip(action[0], 0.0, 0.5))
        transfer_mult = float(np.clip(action[1], 0.0, 3.0))
        spending_mult = float(np.clip(action[2], 0.0, 3.0))

        self._apply_actions(tax_rate, transfer_mult, spending_mult)

        metrics = sim_step(self._state)
        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(metrics)
        terminated = self._step_count >= self.max_steps
        truncated = False

        info = {
            "period": self._step_count,
            "metrics": metrics,
            "govt_obs": self._govt.get_observation(),
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self._state is None:
            return
        g = self._govt
        m = self._state.history[-1] if self._state.history else {}
        print(
            f"t={self._step_count:3d} | "
            f"GDP={m.get('gdp', 0):8.0f} | "
            f"U={m.get('unemployment_rate', 0):5.1%} | "
            f"Tax={g.income_tax_rate:.0%} Tr={g.transfer_per_unemployed:.0f} "
            f"Sp={g.spending_per_period:.0f} | "
            f"Bal={g.budget_balance:.0f} MC={g.money_created:.0f}"
        )
