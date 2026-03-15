"""
Gymnasium-compatible RL environment for the bank agent.

The RL agent controls monetary policy: interest rates and capital adequacy.
All other agents run on their existing rule-based logic.

Observation: 12-dim continuous (bank state + macro indicators)
Action: 2D continuous — base interest rate, capital adequacy ratio
Reward: bank profit with optional stability penalties
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from econosim.config.schema import SimulationConfig
from econosim.engine.simulation import build_simulation, step as sim_step


class BankEnv(gym.Env):
    """Bank RL environment wrapping the full EconoSim simulation.

    The RL agent sets the bank's interest rate and capital adequacy target.
    All other agents behave according to their rule-based defaults.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: SimulationConfig | None = None,
        max_steps: int | None = None,
        reward_type: str = "profit",
    ) -> None:
        super().__init__()

        self.config = config or SimulationConfig(num_periods=120)
        self.max_steps = max_steps or self.config.num_periods
        self.reward_type = reward_type

        # Action space: [base_interest_rate, capital_adequacy_ratio]
        # base_interest_rate: (0.0 to 0.05)
        # capital_adequacy_ratio: (0.02 to 0.20)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.02], dtype=np.float32),
            high=np.array([0.05, 0.20], dtype=np.float32),
        )

        self._obs_keys = [
            "total_loans", "total_deposits_liability", "equity",
            "capital_ratio", "lending_rate", "interest_income",
            "default_losses", "active_loans_count",
            "unemployment_rate", "gdp", "avg_price", "loans_issued",
        ]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self._obs_keys),),
            dtype=np.float32,
        )

        self._state = None
        self._bank = None
        self._step_count = 0

    def _get_obs(self) -> np.ndarray:
        b = self._bank
        m = self._state.history[-1] if self._state.history else {}
        return np.array([
            b.total_loans,
            b.total_deposits_liability,
            b.equity_value,
            b.capital_ratio,
            b.lending_rate,
            b.interest_income,
            b.default_losses,
            float(len(b.loan_book.active_loans())),
            m.get("unemployment_rate", 0.0),
            m.get("gdp", 0.0),
            m.get("avg_price", 10.0),
            m.get("loans_issued", 0.0),
        ], dtype=np.float32)

    def _compute_reward(self, metrics: dict[str, Any]) -> float:
        b = self._bank

        if self.reward_type == "profit":
            return float(b.interest_income - b.default_losses)

        elif self.reward_type == "stability":
            profit = b.interest_income - b.default_losses
            gdp = metrics.get("gdp", 0.0)
            unemployment = metrics.get("unemployment_rate", 0.0)
            return float(profit + 0.005 * gdp - 200.0 * unemployment)

        elif self.reward_type == "growth":
            return float(metrics.get("gdp", 0.0))

        return float(b.interest_income - b.default_losses)

    def _apply_actions(self, base_rate: float, car: float) -> None:
        self._bank.base_interest_rate = base_rate
        self._bank.capital_adequacy_ratio = car

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            config = self.config.model_copy(update={"seed": seed})
        else:
            config = self.config

        self._state = build_simulation(config)
        self._bank = self._state.bank
        self._step_count = 0

        return self._get_obs(), {"period": 0}

    def step(
        self, action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        base_rate = float(np.clip(action[0], 0.0, 0.05))
        car = float(np.clip(action[1], 0.02, 0.20))

        self._apply_actions(base_rate, car)

        metrics = sim_step(self._state)
        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(metrics)
        terminated = self._step_count >= self.max_steps
        truncated = False

        info = {
            "period": self._step_count,
            "metrics": metrics,
            "bank_obs": self._bank.get_observation(),
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self._state is None:
            return
        b = self._bank
        m = self._state.history[-1] if self._state.history else {}
        print(
            f"t={self._step_count:3d} | "
            f"GDP={m.get('gdp', 0):8.0f} | "
            f"U={m.get('unemployment_rate', 0):5.1%} | "
            f"Rate={b.lending_rate:.3f} CAR={b.capital_ratio:.2f} | "
            f"Loans={b.total_loans:.0f} Eq={b.equity_value:.0f} | "
            f"IntInc={b.interest_income:.0f} Def={b.default_losses:.0f}"
        )
