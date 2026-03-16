"""Concrete RL environment wrapping the simulation with policy interfaces.

Provides a Gymnasium-compatible step/reset interface where the RL agent
controls one policy role (firm, bank, or government) while other roles
use rule-based defaults.

The environment translates between the RL action/observation dicts and
the typed policy interface dataclasses.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from econosim.config.schema import SimulationConfig
from econosim.engine.simulation import (
    build_simulation,
    build_macro_state,
    build_firm_state,
    build_bank_state,
    build_govt_state,
    build_household_state,
    step,
    SimulationState,
)
from econosim.policies.interfaces import (
    FirmPolicy, FirmState, FirmAction,
    BankPolicy, BankState, BankAction,
    GovernmentPolicy, GovernmentState, GovernmentAction,
    HouseholdPolicy, HouseholdState, HouseholdAction,
    MacroState,
)
from econosim.policies.rule_based import (
    RuleBasedFirmPolicy,
    RuleBasedBankPolicy,
    RuleBasedGovernmentPolicy,
    RuleBasedHouseholdPolicy,
)
from econosim.rl.env import EconEnvInterface  # env.py has no gymnasium dependency


class MacroEnv(EconEnvInterface):
    """Gymnasium-compatible RL environment for macroeconomic policy.

    The agent controls one role (e.g., 'government') and observes the full
    macro state plus role-specific state. Other roles use rule-based policies.

    Reward is configurable but defaults to GDP growth - unemployment penalty.
    """

    VALID_ROLES = ("government", "bank", "firm")

    def __init__(
        self,
        config: SimulationConfig | None = None,
        role: str = "government",
        max_steps: int = 100,
        reward_fn: Any | None = None,
    ) -> None:
        if role not in self.VALID_ROLES:
            raise ValueError(f"role must be one of {self.VALID_ROLES}, got '{role}'")

        self.config = config or SimulationConfig()
        self.role = role
        self.max_steps = max_steps
        self._reward_fn = reward_fn or _default_reward
        self._state: SimulationState | None = None
        self._step_count = 0

        # Default policies for non-controlled roles
        self._default_firm = RuleBasedFirmPolicy()
        self._default_bank = RuleBasedBankPolicy()
        self._default_govt = RuleBasedGovernmentPolicy()
        self._default_hh = RuleBasedHouseholdPolicy()

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment."""
        cfg = self.config.model_copy(deep=True)
        if seed is not None:
            cfg.seed = seed
        cfg.num_periods = self.max_steps + 10  # buffer

        self._state = build_simulation(cfg)
        self._step_count = 0

        # Set default policies for all non-controlled roles
        self._state.firm_policy = None if self.role == "firm" else self._default_firm
        self._state.bank_policy = None if self.role == "bank" else self._default_bank
        self._state.government_policy = None if self.role == "government" else self._default_govt
        self._state.household_policy = self._default_hh

        # Run one warm-up step to populate history
        step(self._state)
        self._step_count = 1

        obs = self._get_observation()
        return obs, {"period": self._state.current_period}

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Take one step with the given action."""
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")

        # Apply the RL agent's action as a one-step policy
        self._apply_action(action)

        # Step the simulation
        metrics = step(self._state)
        self._step_count += 1

        obs = self._get_observation()
        reward = self._reward_fn(metrics, self._state)
        terminated = False
        truncated = self._step_count >= self.max_steps

        info = {
            "period": self._state.current_period,
            "metrics": metrics,
        }

        return obs, reward, terminated, truncated, info

    def _apply_action(self, action: dict[str, Any]) -> None:
        """Convert RL action dict to a one-step policy and set it on the state."""
        if self.role == "government":
            policy = _OneStepGovtPolicy(action)
            self._state.government_policy = policy
        elif self.role == "bank":
            policy = _OneStepBankPolicy(action)
            self._state.bank_policy = policy
        elif self.role == "firm":
            policy = _OneStepFirmPolicy(action)
            self._state.firm_policy = policy

    def _get_observation(self) -> dict[str, Any]:
        """Build observation dict from current state."""
        macro = build_macro_state(self._state)
        obs: dict[str, Any] = {
            "macro": {
                "gdp": macro.gdp,
                "gdp_growth": macro.gdp_growth,
                "inflation_rate": macro.inflation_rate,
                "unemployment_rate": macro.unemployment_rate,
                "avg_price": macro.avg_price,
                "avg_wage": macro.avg_wage,
                "total_credit": macro.total_credit,
                "bank_capital_ratio": macro.bank_capital_ratio,
                "lending_rate": macro.lending_rate,
            },
        }

        if self.role == "government":
            gs = build_govt_state(self._state.government)
            obs["role"] = {
                "deposits": gs.deposits,
                "tax_revenue": gs.tax_revenue,
                "transfers_paid": gs.transfers_paid,
                "goods_spending": gs.goods_spending,
                "budget_balance": gs.budget_balance,
                "income_tax_rate": gs.income_tax_rate,
                "transfer_per_unemployed": gs.transfer_per_unemployed,
                "spending_per_period": gs.spending_per_period,
            }
        elif self.role == "bank":
            bs = build_bank_state(self._state.bank)
            obs["role"] = {
                "total_loans": bs.total_loans,
                "equity": bs.equity,
                "capital_ratio": bs.capital_ratio,
                "lending_rate": bs.lending_rate,
                "interest_income": bs.interest_income,
                "default_losses": bs.default_losses,
                "base_interest_rate": bs.base_interest_rate,
                "risk_premium": bs.risk_premium,
            }
        elif self.role == "firm":
            # Aggregate firm state
            firms = self._state.firms
            obs["role"] = {
                "avg_deposits": float(np.mean([f.deposits for f in firms])),
                "avg_inventory": float(np.mean([f.inventory.quantity for f in firms])),
                "avg_price": float(np.mean([f.price for f in firms])),
                "avg_wage": float(np.mean([f.posted_wage for f in firms])),
                "total_workers": sum(len(f.workers) for f in firms),
                "total_revenue": sum(f.revenue for f in firms),
            }

        return obs

    def observation_space_spec(self) -> dict[str, Any]:
        """Return observation space description."""
        return {
            "macro": {k: "float" for k in [
                "gdp", "gdp_growth", "inflation_rate", "unemployment_rate",
                "avg_price", "avg_wage", "total_credit", "bank_capital_ratio", "lending_rate",
            ]},
            "role": "dict[str, float] — role-specific state",
        }

    def action_space_spec(self) -> dict[str, Any]:
        """Return action space description."""
        if self.role == "government":
            return {
                "tax_rate": {"type": "float", "range": [0.0, 1.0]},
                "transfer_per_unemployed": {"type": "float", "range": [0.0, 500.0]},
                "spending_per_period": {"type": "float", "range": [0.0, 50000.0]},
            }
        elif self.role == "bank":
            return {
                "base_rate_adjustment": {"type": "float", "range": [-0.01, 0.01]},
                "capital_target_adjustment": {"type": "float", "range": [-0.02, 0.02]},
                "risk_premium_adjustment": {"type": "float", "range": [-0.005, 0.005]},
            }
        elif self.role == "firm":
            return {
                "vacancies": {"type": "int", "range": [0, 20]},
                "price_adjustment": {"type": "float", "range": [0.9, 1.1]},
            }
        return {}

    def to_flat_obs(self, obs: dict[str, Any]) -> np.ndarray:
        """Flatten observation dict to a numpy array for RL algorithms."""
        values = []
        for section in ["macro", "role"]:
            if section in obs and isinstance(obs[section], dict):
                values.extend(float(v) for v in obs[section].values())
        return np.array(values, dtype=np.float32)


# --- One-step policy wrappers ---

class _OneStepGovtPolicy(GovernmentPolicy):
    def __init__(self, action: dict[str, Any]) -> None:
        self._action = action

    def act(self, govt_state: GovernmentState, macro_state: MacroState) -> GovernmentAction:
        return GovernmentAction(
            tax_rate=self._action.get("tax_rate", govt_state.income_tax_rate),
            transfer_per_unemployed=self._action.get("transfer_per_unemployed", govt_state.transfer_per_unemployed),
            spending_per_period=self._action.get("spending_per_period", govt_state.spending_per_period),
        )


class _OneStepBankPolicy(BankPolicy):
    def __init__(self, action: dict[str, Any]) -> None:
        self._action = action

    def act(self, bank_state: BankState, macro_state: MacroState) -> BankAction:
        return BankAction(
            base_rate_adjustment=self._action.get("base_rate_adjustment", 0.0),
            capital_target_adjustment=self._action.get("capital_target_adjustment", 0.0),
            risk_premium_adjustment=self._action.get("risk_premium_adjustment", 0.0),
        )


class _OneStepFirmPolicy(FirmPolicy):
    def __init__(self, action: dict[str, Any]) -> None:
        self._action = action

    def act(self, firm_state: FirmState, macro_state: MacroState) -> FirmAction:
        return FirmAction(
            vacancies=self._action.get("vacancies", 1),
            price_adjustment=self._action.get("price_adjustment", 1.0),
        )


def _default_reward(metrics: dict[str, Any], state: SimulationState) -> float:
    """Default reward: GDP growth minus unemployment penalty."""
    history = state.history
    if len(history) < 2:
        return 0.0
    prev_gdp = history[-2]["gdp"]
    curr_gdp = history[-1]["gdp"]
    gdp_growth = (curr_gdp - prev_gdp) / max(prev_gdp, 1.0)
    unemp = history[-1]["unemployment_rate"]
    return float(gdp_growth - 0.5 * unemp)
