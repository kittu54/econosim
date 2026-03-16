"""National accounts measurement model.

Maps simulation state variables into observed macroeconomic aggregates
consistent with national accounting conventions.

Key principle: the measurement model is a deterministic function of simulation state.
It does NOT change the simulation — it only observes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from econosim.core.accounting import round_money


@dataclass
class MeasuredSeries:
    """Definition of a measured macroeconomic series."""

    name: str
    description: str
    units: str  # "level", "rate", "index", "pct", "ratio"
    frequency: str  # "period" (native sim step)
    source_variables: list[str]  # sim state variables used
    transformation: str = "identity"  # "identity", "pct_change", "log", "ratio", "index"
    base_period: int | None = None  # for index transformations


@dataclass
class NationalAccountsOutput:
    """Complete national accounts measurement for one simulation period."""

    period: int

    # GDP and components (expenditure approach)
    gdp_nominal: float = 0.0
    consumption: float = 0.0
    investment: float = 0.0  # inventory investment
    government_spending: float = 0.0
    net_exports: float = 0.0  # 0 in closed economy

    # GDP growth
    gdp_growth: float = 0.0
    gdp_real: float = 0.0  # deflated by price index

    # Price level
    price_index: float = 100.0
    inflation_rate: float = 0.0

    # Labor market
    employment: int = 0
    labor_force: int = 0
    unemployment_rate: float = 0.0
    avg_wage: float = 0.0
    total_wage_income: float = 0.0

    # Fiscal
    govt_revenue: float = 0.0
    govt_expenditure: float = 0.0
    govt_budget_balance: float = 0.0
    govt_debt: float = 0.0
    debt_to_gdp: float = 0.0

    # Financial
    total_bank_credit: float = 0.0
    credit_growth: float = 0.0
    bank_capital_ratio: float = 0.0
    default_rate: float = 0.0
    lending_rate: float = 0.0

    # Distribution
    gini_wealth: float = 0.0
    gini_income: float = 0.0

    # Sectoral
    total_production: float = 0.0
    total_inventory: float = 0.0
    inventory_to_output_ratio: float = 0.0
    capacity_utilization: float = 0.0

    # Money and deposits
    total_deposits: float = 0.0
    money_velocity: float = 0.0
    private_leverage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class NationalAccountsMapper:
    """Maps simulation state into national accounts aggregates.

    This is the bridge between the agent-based simulation and
    observable macroeconomic data series.
    """

    def __init__(self, base_price: float | None = None) -> None:
        self._base_price = base_price
        self._prev_gdp: float = 0.0
        self._prev_credit: float = 0.0
        self._prev_price_index: float = 100.0
        self._price_history: list[float] = []

    def measure(self, state: Any, period: int) -> NationalAccountsOutput:
        """Compute national accounts from current simulation state.

        Args:
            state: SimulationState object
            period: Current simulation period

        Returns:
            NationalAccountsOutput with all measured series
        """
        out = NationalAccountsOutput(period=period)

        # --- GDP (expenditure approach) ---
        # C: household consumption spending
        out.consumption = round_money(
            sum(hh.consumption_spending for hh in state.households)
        )

        # I: inventory investment (change in inventories = production - sales)
        total_production_value = sum(
            f.production * f.price for f in state.firms
        )
        out.investment = round_money(
            total_production_value - out.consumption - state.government.goods_spending
        )
        # Clamp investment: it can be negative (inventory drawdown)
        out.investment = round_money(max(out.investment, -out.consumption))

        # G: government spending
        out.government_spending = round_money(state.government.goods_spending)

        # NX: net exports (closed economy = 0)
        out.net_exports = 0.0

        # GDP = C + I + G + NX
        out.gdp_nominal = round_money(
            out.consumption + out.investment + out.government_spending + out.net_exports
        )

        # Ensure GDP is non-negative (use goods market + govt spending as floor)
        market_gdp = round_money(
            state.goods_market.total_transacted + state.government.goods_spending
        )
        out.gdp_nominal = max(out.gdp_nominal, market_gdp)

        # --- Price Index ---
        prices = [f.price for f in state.firms if f.price > 0]
        if prices:
            # Weighted by units sold (Laspeyres-like)
            weights = [max(f.units_sold, 0.01) for f in state.firms if f.price > 0]
            total_w = sum(weights)
            if total_w > 0:
                avg_price = sum(p * w for p, w in zip(prices, weights)) / total_w
            else:
                avg_price = float(np.mean(prices))
        else:
            avg_price = self._prev_price_index

        if self._base_price is None:
            self._base_price = avg_price

        out.price_index = round(100.0 * avg_price / self._base_price, 4) if self._base_price > 0 else 100.0
        self._price_history.append(avg_price)

        # Inflation
        if self._prev_price_index > 0:
            out.inflation_rate = round(
                (out.price_index - self._prev_price_index) / self._prev_price_index, 6
            )

        # Real GDP
        deflator = out.price_index / 100.0 if out.price_index > 0 else 1.0
        out.gdp_real = round_money(out.gdp_nominal / deflator)

        # GDP growth
        if self._prev_gdp > 0:
            out.gdp_growth = round(
                (out.gdp_nominal - self._prev_gdp) / self._prev_gdp, 6
            )

        # --- Labor Market ---
        out.employment = sum(1 for hh in state.households if hh.employed)
        out.labor_force = sum(1 for hh in state.households if hh.labor_participation)
        out.unemployment_rate = round(
            (out.labor_force - out.employment) / max(out.labor_force, 1), 4
        )
        wages = [f.posted_wage for f in state.firms]
        out.avg_wage = round(float(np.mean(wages)), 2) if wages else 0.0
        out.total_wage_income = round_money(
            sum(hh.wage_income for hh in state.households)
        )

        # --- Fiscal ---
        out.govt_revenue = round_money(state.government.tax_revenue)
        out.govt_expenditure = round_money(
            state.government.transfers_paid + state.government.goods_spending
        )
        out.govt_budget_balance = round_money(out.govt_revenue - out.govt_expenditure)
        out.govt_debt = round_money(state.government.cumulative_money_created)

        if state.debt_manager is not None:
            obs = state.debt_manager.get_observation()
            out.govt_debt = round_money(obs.get("total_outstanding", out.govt_debt))

        out.debt_to_gdp = round(
            out.govt_debt / max(out.gdp_nominal, 1.0), 4
        )

        # --- Financial ---
        out.total_bank_credit = round_money(state.bank.total_loans)
        if self._prev_credit > 0:
            out.credit_growth = round(
                (out.total_bank_credit - self._prev_credit) / self._prev_credit, 6
            )
        out.bank_capital_ratio = round(state.bank.capital_ratio, 4)
        out.default_rate = round(
            state.bank.default_losses / max(out.total_bank_credit, 1.0), 6
        )
        out.lending_rate = round(state.bank.lending_rate, 6)

        # --- Distribution ---
        hh_deposits = np.array([hh.deposits for hh in state.households])
        out.gini_wealth = round(float(_gini(hh_deposits)), 4)

        hh_incomes = np.array([
            hh.wage_income + hh.transfers_received - hh.taxes_paid
            for hh in state.households
        ])
        out.gini_income = round(float(_gini(np.maximum(hh_incomes, 0))), 4)

        # --- Sectoral ---
        out.total_production = round_money(sum(f.production for f in state.firms))
        out.total_inventory = round_money(sum(f.inventory.quantity for f in state.firms))
        out.inventory_to_output_ratio = round(
            out.total_inventory / max(out.total_production, 1.0), 4
        )

        # Capacity utilization proxy: actual workers / potential workers
        total_workers = sum(len(f.workers) for f in state.firms)
        potential = sum(max(f.vacancies + len(f.workers), 1) for f in state.firms)
        out.capacity_utilization = round(total_workers / max(potential, 1), 4)

        # --- Money ---
        out.total_deposits = round_money(
            sum(hh.deposits for hh in state.households)
            + sum(f.deposits for f in state.firms)
        )
        out.money_velocity = round(
            out.gdp_nominal / max(out.total_deposits, 1.0), 4
        )

        # Private leverage: total private debt / total private deposits
        firm_debt = sum(f.total_debt for f in state.firms)
        hh_debt = sum(hh.total_debt for hh in state.households)
        out.private_leverage = round(
            (firm_debt + hh_debt) / max(out.total_deposits, 1.0), 4
        )

        # Update state for next period
        self._prev_gdp = out.gdp_nominal
        self._prev_credit = out.total_bank_credit
        self._prev_price_index = out.price_index

        return out


class LaborMarketMetrics:
    """Detailed labor market measurement."""

    @staticmethod
    def measure(state: Any) -> dict[str, Any]:
        employed = sum(1 for hh in state.households if hh.employed)
        labor_force = sum(1 for hh in state.households if hh.labor_participation)
        total_pop = len(state.households)
        vacancies = sum(f.vacancies for f in state.firms)
        wages = [f.posted_wage for f in state.firms]

        return {
            "employment": employed,
            "labor_force": labor_force,
            "population": total_pop,
            "unemployment_rate": (labor_force - employed) / max(labor_force, 1),
            "participation_rate": labor_force / max(total_pop, 1),
            "vacancy_rate": vacancies / max(employed + vacancies, 1),
            "beveridge_ratio": vacancies / max(labor_force - employed, 1),
            "avg_wage": float(np.mean(wages)) if wages else 0.0,
            "wage_dispersion": float(np.std(wages)) if wages else 0.0,
        }


class FinancialSystemMetrics:
    """Detailed financial system measurement."""

    @staticmethod
    def measure(state: Any) -> dict[str, Any]:
        bank = state.bank
        firms = state.firms

        firm_leverage = [
            f.total_debt / max(f.equity_value, 1.0) for f in firms
        ]

        return {
            "total_loans": bank.total_loans,
            "total_deposits": bank.total_deposits_liability,
            "bank_equity": bank.equity_value,
            "capital_ratio": bank.capital_ratio,
            "lending_rate": bank.lending_rate,
            "interest_income": bank.interest_income,
            "default_losses": bank.default_losses,
            "loans_issued": bank.loans_issued_this_period,
            "active_loans": len(bank.loan_book.active_loans()),
            "avg_firm_leverage": float(np.mean(firm_leverage)) if firm_leverage else 0.0,
            "max_firm_leverage": float(np.max(firm_leverage)) if firm_leverage else 0.0,
            "firms_in_debt": sum(1 for f in firms if f.total_debt > 0),
        }


# --- Series definitions for mapping to FRED ---

MEASUREMENT_SERIES = {
    "gdp_nominal": MeasuredSeries(
        name="gdp_nominal",
        description="Nominal GDP (expenditure approach: C + I + G)",
        units="level",
        frequency="period",
        source_variables=["consumption", "investment", "government_spending"],
    ),
    "gdp_growth": MeasuredSeries(
        name="gdp_growth",
        description="GDP growth rate (period-over-period)",
        units="rate",
        frequency="period",
        source_variables=["gdp_nominal"],
        transformation="pct_change",
    ),
    "inflation_rate": MeasuredSeries(
        name="inflation_rate",
        description="Inflation rate (price index change)",
        units="rate",
        frequency="period",
        source_variables=["price_index"],
        transformation="pct_change",
    ),
    "unemployment_rate": MeasuredSeries(
        name="unemployment_rate",
        description="Unemployment rate (unemployed / labor force)",
        units="rate",
        frequency="period",
        source_variables=["employment", "labor_force"],
        transformation="ratio",
    ),
    "credit_growth": MeasuredSeries(
        name="credit_growth",
        description="Bank credit growth rate",
        units="rate",
        frequency="period",
        source_variables=["total_bank_credit"],
        transformation="pct_change",
    ),
    "govt_budget_balance": MeasuredSeries(
        name="govt_budget_balance",
        description="Government budget balance (revenue - expenditure)",
        units="level",
        frequency="period",
        source_variables=["govt_revenue", "govt_expenditure"],
    ),
    "gini_wealth": MeasuredSeries(
        name="gini_wealth",
        description="Gini coefficient of household wealth (deposits)",
        units="ratio",
        frequency="period",
        source_variables=["household_deposits"],
    ),
}


def _gini(values: np.ndarray) -> float:
    """Compute Gini coefficient."""
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    cumulative = np.cumsum(values)
    return float(
        (2.0 * np.sum((np.arange(1, n + 1) * values)) / (n * np.sum(values)))
        - (n + 1) / n
    )
