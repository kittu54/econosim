"""
Gymnasium-compatible RL environment for a single firm agent.

The RL agent controls one firm's pricing, wage, and hiring decisions.
All other agents (households, other firms, bank, government) run on
their existing rule-based logic.

Observation: continuous vector of firm + macro state
Action: 3D continuous — price multiplier, wage multiplier, vacancy fraction
Reward: firm profit (revenue - wage_bill) with optional macro penalties
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from econosim.config.schema import SimulationConfig
from econosim.engine.simulation import build_simulation, step as sim_step


def _rl_decide_vacancies(firm, vacancies: int) -> int:
    """RL-controlled vacancy decision."""
    firm.vacancies = vacancies
    return vacancies


def _rl_adjust_price(firm, price: float) -> float:
    """RL-controlled price adjustment."""
    firm.price = price
    return price


def _rl_adjust_wage(firm, wage: float) -> float:
    """RL-controlled wage adjustment."""
    firm.posted_wage = wage
    return wage


class FirmEnv(gym.Env):
    """Single-firm RL environment wrapping the full EconoSim simulation.

    The RL agent replaces the rule-based decision logic for firm_000.
    All other agents behave according to their rule-based defaults.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: SimulationConfig | None = None,
        max_steps: int | None = None,
        reward_type: str = "profit",
        controlled_firm_id: str = "firm_000",
    ) -> None:
        super().__init__()

        self.config = config or SimulationConfig(num_periods=120)
        self.max_steps = max_steps or self.config.num_periods
        self.reward_type = reward_type
        self.controlled_firm_id = controlled_firm_id

        # Action space: [price_mult, wage_mult, vacancy_frac]
        # price_mult: multiplier on current price (0.8 to 1.2)
        # wage_mult: multiplier on current wage (0.9 to 1.1)
        # vacancy_frac: fraction of affordable workers to hire (0.0 to 1.0)
        self.action_space = spaces.Box(
            low=np.array([0.8, 0.9, 0.0], dtype=np.float32),
            high=np.array([1.2, 1.1, 1.0], dtype=np.float32),
        )

        # Observation space: 14 continuous features (normalized)
        self._obs_keys = [
            "deposits", "inventory", "price", "posted_wage", "workers_count",
            "revenue", "wage_bill", "units_sold", "total_debt",
            "avg_market_price", "unemployment_rate", "gdp",
            "prev_units_sold", "prev_revenue",
        ]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self._obs_keys),),
            dtype=np.float32,
        )

        self._state = None
        self._firm = None
        self._step_count = 0

    def _get_firm(self):
        for f in self._state.firms:
            if f.agent_id == self.controlled_firm_id:
                return f
        raise ValueError(f"Firm {self.controlled_firm_id} not found")

    def _get_obs(self) -> np.ndarray:
        f = self._firm
        last_metrics = self._state.history[-1] if self._state.history else {}

        obs = np.array([
            f.deposits,
            f.inventory.quantity,
            f.price,
            f.posted_wage,
            float(len(f.workers)),
            f.revenue,
            f.wage_bill,
            f.units_sold,
            f.total_debt,
            last_metrics.get("avg_price", f.price),
            last_metrics.get("unemployment_rate", 0.0),
            last_metrics.get("gdp", 0.0),
            f.prev_units_sold,
            f.prev_revenue,
        ], dtype=np.float32)

        return obs

    def _compute_reward(self, metrics: dict[str, Any]) -> float:
        f = self._firm

        if self.reward_type == "profit":
            return float(f.revenue - f.wage_bill)

        elif self.reward_type == "gdp":
            return float(metrics.get("gdp", 0.0))

        elif self.reward_type == "balanced":
            profit = f.revenue - f.wage_bill
            gdp = metrics.get("gdp", 0.0)
            unemployment = metrics.get("unemployment_rate", 0.0)
            # Profit + GDP bonus - unemployment penalty
            return float(profit + 0.01 * gdp - 1000.0 * unemployment)

        return float(f.revenue - f.wage_bill)

    def _apply_actions(
        self, price_mult: float, wage_mult: float, vacancy_frac: float
    ) -> None:
        """Override the controlled firm's rule-based decisions with RL actions.

        Monkey-patches decide_vacancies, adjust_price, and adjust_wage
        so the simulation step uses RL-chosen values.
        """
        f = self._firm
        from econosim.core.accounting import round_money

        # Price: multiply current price by the RL-chosen factor
        new_price = max(0.01, round_money(f.price * price_mult))

        # Wage: multiply current wage by the RL-chosen factor
        new_wage = max(1.0, round_money(f.posted_wage * wage_mult))

        # Vacancies: fraction of affordable workers
        affordable = int(f.deposits / max(new_wage, 1.0))
        new_vacancies = max(0, int(affordable * vacancy_frac))

        # Patch the methods for this step
        f.decide_vacancies = lambda: _rl_decide_vacancies(f, new_vacancies)
        f.adjust_price = lambda: _rl_adjust_price(f, new_price)
        f.adjust_wage = lambda: _rl_adjust_wage(f, new_wage)

    def _restore_methods(self) -> None:
        """Restore rule-based methods on the controlled firm."""
        from econosim.agents.firm import Firm
        f = self._firm
        f.decide_vacancies = Firm.decide_vacancies.__get__(f, Firm)
        f.adjust_price = Firm.adjust_price.__get__(f, Firm)
        f.adjust_wage = Firm.adjust_wage.__get__(f, Firm)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            config = self.config.model_copy(update={"seed": seed})
        else:
            config = self.config

        self._state = build_simulation(config)
        self._firm = self._get_firm()
        self._step_count = 0

        return self._get_obs(), {"period": 0}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        price_mult = float(np.clip(action[0], 0.8, 1.2))
        wage_mult = float(np.clip(action[1], 0.9, 1.1))
        vacancy_frac = float(np.clip(action[2], 0.0, 1.0))

        # Apply RL actions by overriding the firm's decision methods
        # before the simulation step runs.
        self._apply_actions(price_mult, wage_mult, vacancy_frac)

        metrics = sim_step(self._state)
        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(metrics)
        terminated = self._step_count >= self.max_steps
        truncated = False

        info = {
            "period": self._step_count,
            "metrics": metrics,
            "firm_obs": self._firm.get_observation(),
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self._state is None:
            return
        f = self._firm
        m = self._state.history[-1] if self._state.history else {}
        print(
            f"t={self._step_count:3d} | "
            f"GDP={m.get('gdp', 0):8.0f} | "
            f"U={m.get('unemployment_rate', 0):5.1%} | "
            f"P={f.price:.2f} W={f.posted_wage:.0f} | "
            f"Rev={f.revenue:.0f} WB={f.wage_bill:.0f} | "
            f"Inv={f.inventory.quantity:.0f} Workers={len(f.workers)}"
        )
