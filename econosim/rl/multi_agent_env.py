"""
PettingZoo-compatible parallel multi-agent RL environment.

All four agent types (firm, household, government, bank) act simultaneously
each period. The simulation advances one step per call to `step()`.

Usage:
    from econosim.rl.multi_agent_env import EconoSimMultiAgentEnv
    env = EconoSimMultiAgentEnv()
    observations, infos = env.reset()
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from pettingzoo import ParallelEnv
    from gymnasium import spaces
    HAS_PETTINGZOO = True
except ImportError:
    HAS_PETTINGZOO = False
    ParallelEnv = object

from econosim.config.schema import SimulationConfig
from econosim.core.accounting import round_money
from econosim.engine.simulation import build_simulation, step as sim_step
from econosim.rl.firm_env import _rl_decide_vacancies, _rl_adjust_price, _rl_adjust_wage


class EconoSimMultiAgentEnv(ParallelEnv):
    """Multi-agent parallel environment with firm, household, government, and bank agents.

    Each agent type has its own observation and action space.
    All agents act simultaneously; the simulation advances one period per step.
    """

    metadata = {"render_modes": ["human"], "name": "econosim_v0"}

    def __init__(
        self,
        config: SimulationConfig | None = None,
        max_steps: int | None = None,
        firm_id: str = "firm_000",
        household_id: str = "hh_0000",
    ) -> None:
        if not HAS_PETTINGZOO:
            raise ImportError("pettingzoo is required. Install with: pip install pettingzoo")

        self.config = config or SimulationConfig(num_periods=120)
        self.max_steps = max_steps or self.config.num_periods
        self.firm_id = firm_id
        self.household_id = household_id

        self.possible_agents = ["firm", "household", "government", "bank"]
        self.agents = list(self.possible_agents)

        # Action spaces per agent
        self._action_spaces = {
            "firm": spaces.Box(
                low=np.array([0.8, 0.9, 0.0], dtype=np.float32),
                high=np.array([1.2, 1.1, 1.0], dtype=np.float32),
            ),
            "household": spaces.Box(
                low=np.array([0.0, 0.5], dtype=np.float32),
                high=np.array([1.0, 2.0], dtype=np.float32),
            ),
            "government": spaces.Box(
                low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([0.5, 3.0, 3.0], dtype=np.float32),
            ),
            "bank": spaces.Box(
                low=np.array([0.0, 0.02], dtype=np.float32),
                high=np.array([0.05, 0.20], dtype=np.float32),
            ),
        }

        # Observation spaces per agent
        self._obs_sizes = {"firm": 14, "household": 12, "government": 12, "bank": 12}
        self._observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32)
            for agent, size in self._obs_sizes.items()
        }

        self._state = None
        self._firm = None
        self._hh = None
        self._govt = None
        self._bank = None
        self._base_reservation_wage = None
        self._base_transfer = None
        self._base_spending = None
        self._step_count = 0

    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_spaces[agent]

    def _get_firm_obs(self) -> np.ndarray:
        f = self._firm
        m = self._state.history[-1] if self._state.history else {}
        return np.array([
            f.deposits, f.inventory.quantity, f.price, f.posted_wage,
            float(len(f.workers)), f.revenue, f.wage_bill, f.units_sold,
            f.total_debt, m.get("avg_price", f.price),
            m.get("unemployment_rate", 0.0), m.get("gdp", 0.0),
            f.prev_units_sold, f.prev_revenue,
        ], dtype=np.float32)

    def _get_hh_obs(self) -> np.ndarray:
        hh = self._hh
        m = self._state.history[-1] if self._state.history else {}
        return np.array([
            hh.deposits, float(hh.employed), hh.wage_income,
            hh.consumption_spending, hh.disposable_income, hh.total_debt,
            hh.taxes_paid, hh.transfers_received,
            m.get("avg_price", 10.0), m.get("unemployment_rate", 0.0),
            m.get("gdp", 0.0), m.get("gini_deposits", 0.0),
        ], dtype=np.float32)

    def _get_govt_obs(self) -> np.ndarray:
        g = self._govt
        m = self._state.history[-1] if self._state.history else {}
        return np.array([
            g.deposits, g.tax_revenue, g.transfers_paid, g.goods_spending,
            g.budget_balance, g.money_created, g.cumulative_money_created,
            m.get("unemployment_rate", 0.0), m.get("gdp", 0.0),
            m.get("avg_price", 10.0), m.get("gini_deposits", 0.0),
            m.get("total_hh_deposits", 0.0),
        ], dtype=np.float32)

    def _get_bank_obs(self) -> np.ndarray:
        b = self._bank
        m = self._state.history[-1] if self._state.history else {}
        return np.array([
            b.total_loans, b.total_deposits_liability, b.equity_value,
            b.capital_ratio, b.lending_rate, b.interest_income,
            b.default_losses, float(len(b.loan_book.active_loans())),
            m.get("unemployment_rate", 0.0), m.get("gdp", 0.0),
            m.get("avg_price", 10.0), m.get("loans_issued", 0.0),
        ], dtype=np.float32)

    def _get_observations(self) -> dict[str, np.ndarray]:
        return {
            "firm": self._get_firm_obs(),
            "household": self._get_hh_obs(),
            "government": self._get_govt_obs(),
            "bank": self._get_bank_obs(),
        }

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        if seed is not None:
            config = self.config.model_copy(update={"seed": seed})
        else:
            config = self.config

        self._state = build_simulation(config)
        self.agents = list(self.possible_agents)

        # Locate controlled agents
        self._firm = next(f for f in self._state.firms if f.agent_id == self.firm_id)
        self._hh = next(h for h in self._state.households if h.agent_id == self.household_id)
        self._govt = self._state.government
        self._bank = self._state.bank

        self._base_reservation_wage = self._hh.reservation_wage
        self._base_transfer = self._govt.transfer_per_unemployed
        self._base_spending = self._govt.spending_per_period
        self._step_count = 0

        observations = self._get_observations()
        infos = {agent: {"period": 0} for agent in self.agents}
        return observations, infos

    def step(
        self, actions: dict[str, np.ndarray],
    ) -> tuple[dict, dict, dict, dict, dict]:
        # Apply firm actions
        if "firm" in actions:
            a = actions["firm"]
            price_mult = float(np.clip(a[0], 0.8, 1.2))
            wage_mult = float(np.clip(a[1], 0.9, 1.1))
            vacancy_frac = float(np.clip(a[2], 0.0, 1.0))

            f = self._firm
            new_price = max(0.01, round_money(f.price * price_mult))
            new_wage = max(1.0, round_money(f.posted_wage * wage_mult))
            affordable = int(f.deposits / max(new_wage, 1.0))
            new_vacancies = max(0, int(affordable * vacancy_frac))

            f.decide_vacancies = lambda: _rl_decide_vacancies(f, new_vacancies)
            f.adjust_price = lambda: _rl_adjust_price(f, new_price)
            f.adjust_wage = lambda: _rl_adjust_wage(f, new_wage)

        # Apply household actions
        if "household" in actions:
            a = actions["household"]
            consumption_frac = float(np.clip(a[0], 0.0, 1.0))
            res_wage_mult = float(np.clip(a[1], 0.5, 2.0))
            new_res_wage = max(1.0, self._base_reservation_wage * res_wage_mult)
            hh = self._hh
            hh.desired_consumption = lambda: min(
                consumption_frac * max(0.0, hh.deposits), max(0.0, hh.deposits)
            )
            hh.accept_wage = lambda wage: wage >= new_res_wage

        # Apply government actions
        if "government" in actions:
            a = actions["government"]
            self._govt.income_tax_rate = float(np.clip(a[0], 0.0, 0.5))
            self._govt.transfer_per_unemployed = max(
                0.0, self._base_transfer * float(np.clip(a[1], 0.0, 3.0))
            )
            self._govt.spending_per_period = max(
                0.0, self._base_spending * float(np.clip(a[2], 0.0, 3.0))
            )

        # Apply bank actions
        if "bank" in actions:
            a = actions["bank"]
            self._bank.base_interest_rate = float(np.clip(a[0], 0.0, 0.05))
            self._bank.capital_adequacy_ratio = float(np.clip(a[1], 0.02, 0.20))

        # Step simulation
        metrics = sim_step(self._state)
        self._step_count += 1

        observations = self._get_observations()
        terminated = self._step_count >= self.max_steps

        # Rewards
        f = self._firm
        hh = self._hh
        gdp = metrics.get("gdp", 0.0)
        unemployment = metrics.get("unemployment_rate", 0.0)
        gini = metrics.get("gini_deposits", 0.0)

        rewards = {
            "firm": float(f.revenue - f.wage_bill),
            "household": float(
                np.log(max(hh.consumption_spending, 0.01))
                + 0.1 * np.log(max(hh.deposits, 1.0))
            ),
            "government": float(0.01 * gdp - 500.0 * unemployment - 100.0 * gini),
            "bank": float(self._bank.interest_income - self._bank.default_losses),
        }

        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {"period": self._step_count, "metrics": metrics} for agent in self.agents}

        if terminated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        if self._state is None:
            return
        m = self._state.history[-1] if self._state.history else {}
        print(
            f"t={self._step_count:3d} | "
            f"GDP={m.get('gdp', 0):8.0f} | "
            f"U={m.get('unemployment_rate', 0):5.1%} | "
            f"P={m.get('avg_price', 0):.2f}"
        )
