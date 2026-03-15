"""
Simulation engine: orchestrates the economic simulation loop.

Each period executes sub-steps in a strict, explicit order:
1. Government policy update
2. Credit market (bank lending terms / loan applications)
3. Labor market (matching + wage payment)
4. Production
5. Goods pricing and goods market
6. Transaction settlement (already done inline)
7. Taxes and transfers
8. Debt service / delinquency / default
9. Metrics computation
10. Logging / persistence
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np

from econosim.core.accounting import Ledger, round_money
from econosim.config.schema import SimulationConfig, ShockSpec
from econosim.agents.household import Household
from econosim.agents.firm import Firm
from econosim.agents.bank import Bank
from econosim.agents.government import Government
from econosim.markets.labor import LaborMarket
from econosim.markets.goods import GoodsMarket
from econosim.markets.credit import CreditMarket

logger = logging.getLogger(__name__)


class SimulationState:
    """Container for all simulation state."""

    def __init__(
        self,
        config: SimulationConfig,
        ledger: Ledger,
        households: list[Household],
        firms: list[Firm],
        bank: Bank,
        government: Government,
        labor_market: LaborMarket,
        goods_market: GoodsMarket,
        credit_market: CreditMarket,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.households = households
        self.firms = firms
        self.bank = bank
        self.government = government
        self.labor_market = labor_market
        self.goods_market = goods_market
        self.credit_market = credit_market
        self.rng = rng
        self.current_period: int = 0
        self.history: list[dict[str, Any]] = []


def build_simulation(config: SimulationConfig) -> SimulationState:
    """Construct all agents and markets from a configuration."""
    rng = np.random.default_rng(config.seed)
    ledger = Ledger()

    # Create bank
    bank = Bank(
        agent_id="bank_0",
        ledger=ledger,
        initial_equity=config.bank.initial_equity,
        initial_reserves=config.bank.initial_reserves,
        base_interest_rate=config.bank.base_interest_rate,
        risk_premium=config.bank.risk_premium,
        capital_adequacy_ratio=config.bank.capital_adequacy_ratio,
        max_loan_to_value=config.bank.max_loan_to_value,
        default_threshold_periods=config.bank.default_threshold_periods,
        loan_term_periods=config.bank.loan_term_periods,
    )

    # Create government
    government = Government(
        agent_id="govt_0",
        ledger=ledger,
        income_tax_rate=config.government.income_tax_rate,
        transfer_per_unemployed=config.government.transfer_per_unemployed,
        spending_per_period=config.government.spending_per_period,
        initial_deposits=config.government.initial_deposits,
    )

    # Create firms
    firms = []
    for i in range(config.firm.count):
        firm = Firm(
            agent_id=f"firm_{i:03d}",
            ledger=ledger,
            initial_deposits=config.firm.initial_deposits,
            initial_inventory=config.firm.initial_inventory,
            initial_price=config.firm.initial_price,
            initial_wage=config.firm.initial_wage,
            labor_productivity=config.firm.labor_productivity,
            target_inventory_ratio=config.firm.target_inventory_ratio,
            price_adjustment_speed=config.firm.price_adjustment_speed,
            wage_adjustment_speed=config.firm.wage_adjustment_speed,
            max_leverage=config.firm.max_leverage,
        )
        firms.append(firm)

    # Set initial demand estimate so firms don't assume zero demand at t=0.
    # Estimate: each firm gets an equal share of household consumption.
    # Must set BOTH units_sold AND revenue so that reset_period_state()
    # correctly propagates them to prev_units_sold / prev_revenue.
    hh_count = config.household.count
    initial_consumption_est = (
        config.household.wealth_propensity * config.household.initial_deposits * hh_count
    ) / max(config.firm.count, 1)
    initial_units_est = initial_consumption_est / max(config.firm.initial_price, 0.01)
    for firm in firms:
        firm.units_sold = initial_units_est
        firm.revenue = initial_consumption_est

    # Create households
    households = []
    for i in range(config.household.count):
        participates = rng.random() < config.household.labor_participation_rate
        hh = Household(
            agent_id=f"hh_{i:04d}",
            ledger=ledger,
            initial_deposits=config.household.initial_deposits,
            consumption_propensity=config.household.consumption_propensity,
            wealth_propensity=config.household.wealth_propensity,
            reservation_wage=config.household.reservation_wage,
            labor_participation=participates,
        )
        households.append(hh)

    # Create markets
    labor_market = LaborMarket(ledger)
    goods_market = GoodsMarket(ledger)
    credit_market = CreditMarket(ledger)

    return SimulationState(
        config=config,
        ledger=ledger,
        households=households,
        firms=firms,
        bank=bank,
        government=government,
        labor_market=labor_market,
        goods_market=goods_market,
        credit_market=credit_market,
        rng=rng,
    )


def apply_shocks(state: SimulationState, period: int) -> None:
    """Apply any shocks scheduled for this period."""
    for shock in state.config.shocks:
        if shock.period != period:
            continue

        logger.info(f"Period {period}: Applying shock {shock.shock_type}/{shock.parameter} = {shock.magnitude}")

        if shock.shock_type == "supply":
            # Modify firm labor productivity
            if shock.parameter == "labor_productivity":
                for firm in state.firms:
                    if shock.additive:
                        firm.labor_productivity += shock.magnitude
                    else:
                        firm.labor_productivity *= shock.magnitude
                    firm.labor_productivity = max(0.1, firm.labor_productivity)

        elif shock.shock_type == "demand":
            # Modify household consumption propensity
            if shock.parameter == "consumption_propensity":
                for hh in state.households:
                    if shock.additive:
                        hh.consumption_propensity += shock.magnitude
                    else:
                        hh.consumption_propensity *= shock.magnitude
                    hh.consumption_propensity = max(0.01, min(1.0, hh.consumption_propensity))

        elif shock.shock_type == "credit":
            # Modify bank lending conditions
            if shock.parameter == "capital_adequacy_ratio":
                if shock.additive:
                    state.bank.capital_adequacy_ratio += shock.magnitude
                else:
                    state.bank.capital_adequacy_ratio *= shock.magnitude

        elif shock.shock_type == "fiscal":
            # Modify government policy
            if shock.parameter == "income_tax_rate":
                if shock.additive:
                    state.government.income_tax_rate += shock.magnitude
                else:
                    state.government.income_tax_rate *= shock.magnitude
                state.government.income_tax_rate = max(0.0, min(1.0, state.government.income_tax_rate))
            elif shock.parameter == "spending_per_period":
                if shock.additive:
                    state.government.spending_per_period += shock.magnitude
                else:
                    state.government.spending_per_period *= shock.magnitude


def step(state: SimulationState) -> dict[str, Any]:
    """Execute one period of the simulation.

    Returns a metrics dict for the period.
    """
    period = state.current_period

    # ── 0. Reset period state ────────────────────────────────────
    for hh in state.households:
        hh.reset_period_state()
    for firm in state.firms:
        firm.reset_period_state()
    state.bank.reset_period_state()
    state.government.reset_period_state()

    # ── 1. Apply shocks ──────────────────────────────────────────
    apply_shocks(state, period)

    # ── 2. Credit market ─────────────────────────────────────────
    state.credit_market.clear(
        firms=state.firms,
        bank=state.bank,
        period=period,
    )

    # ── 3. Labor market ──────────────────────────────────────────
    state.labor_market.clear(
        households=state.households,
        firms=state.firms,
        period=period,
        rng=state.rng,
    )

    # ── 4. Production ────────────────────────────────────────────
    for firm in state.firms:
        firm.produce()

    # ── 5. Goods pricing + goods market ──────────────────────────
    for firm in state.firms:
        firm.adjust_price()

    state.goods_market.clear(
        households=state.households,
        firms=state.firms,
        period=period,
        rng=state.rng,
    )

    # ── 6. Taxes and transfers ───────────────────────────────────
    for hh in state.households:
        if hh.wage_income > 0:
            tax = state.government.compute_tax(hh.wage_income)
            tax = min(tax, hh.deposits)
            if tax > 0:
                state.government.collect_tax(period, hh.agent_id, tax)
                hh.taxes_paid = round_money(hh.taxes_paid + tax)

    # Estimate total fiscal spending needed and ensure solvency
    n_unemployed = sum(
        1 for hh in state.households if not hh.employed and hh.labor_participation
    )
    fiscal_need = round_money(
        n_unemployed * state.government.transfer_per_unemployed
        + state.government.spending_per_period
    )
    state.government.ensure_solvency(fiscal_need, period)

    # Transfers to unemployed
    for hh in state.households:
        if not hh.employed and hh.labor_participation:
            transfer = state.government.transfer_per_unemployed
            if transfer > 0:
                state.government.pay_transfer(period, hh.agent_id, transfer)
                hh.transfers_received = round_money(hh.transfers_received + transfer)

    # Government spending distributed across firms.
    # Buys goods when available; otherwise injects as service contracts.
    # This is the key fiscal stabilizer that prevents deflationary collapse.
    if state.government.spending_per_period > 0 and len(state.firms) > 0:
        spend_per_firm = round_money(state.government.spending_per_period / len(state.firms))
        for firm in state.firms:
            actual_spend = min(spend_per_firm, state.government.deposits)
            if actual_spend > 0.01:
                available_units = firm.inventory.quantity
                if available_units > 0 and firm.price > 0:
                    affordable_units = actual_spend / firm.price
                    purchase_units = min(affordable_units, available_units)
                    purchase_cost = round_money(purchase_units * firm.price)
                    if purchase_cost > 0.01:
                        state.government.purchase_goods(period, firm.agent_id, purchase_cost)
                        firm.inventory.sell(purchase_units)
                        firm.revenue = round_money(firm.revenue + purchase_cost)
                        firm.units_sold = round_money(firm.units_sold + purchase_units)
                        actual_spend = round_money(actual_spend - purchase_cost)
                # Remaining budget goes as direct service contracts (no goods exchanged)
                if actual_spend > 0.01:
                    state.government.purchase_goods(period, firm.agent_id, actual_spend)
                    firm.revenue = round_money(firm.revenue + actual_spend)

    # ── 7. Debt service / delinquency / default ──────────────────
    state.bank.process_loan_payments(period)
    defaulted = state.bank.process_defaults(period)

    # ── 8. Wage adjustment (for next period) ─────────────────────
    for firm in state.firms:
        firm.adjust_wage()

    # ── 9. Compute metrics ───────────────────────────────────────
    metrics = compute_period_metrics(state, period)
    state.history.append(metrics)

    # ── 10. Validate invariants ──────────────────────────────────
    validation = state.ledger.validate_all_balanced()
    unbalanced = [k for k, v in validation.items() if not v]
    if unbalanced:
        logger.warning(f"Period {period}: Unbalanced sheets: {unbalanced}")
    metrics["unbalanced_sheets"] = unbalanced

    state.current_period += 1
    return metrics


def compute_period_metrics(state: SimulationState, period: int) -> dict[str, Any]:
    """Compute aggregate metrics for the current period."""
    # GDP proxy: total goods market transactions + government spending
    gdp = state.goods_market.total_transacted + state.government.goods_spending

    # Unemployment
    employed = sum(1 for hh in state.households if hh.employed)
    labor_force = sum(1 for hh in state.households if hh.labor_participation)
    unemployment_rate = (labor_force - employed) / max(labor_force, 1)

    # Price level (CPI proxy)
    prices = [f.price for f in state.firms]
    avg_price = np.mean(prices) if prices else 0.0

    # Aggregate deposits
    hh_deposits = sum(hh.deposits for hh in state.households)
    firm_deposits = sum(f.deposits for f in state.firms)
    govt_deposits = state.government.deposits
    bank_equity = state.bank.equity_value

    # Income and wealth inequality (simple Gini on deposits)
    deposits_array = np.array([hh.deposits for hh in state.households])
    gini = _gini_coefficient(deposits_array)

    # Total inventory
    total_inventory = sum(f.inventory.quantity for f in state.firms)

    # Credit metrics
    total_loans = state.bank.total_loans
    active_loans = len(state.bank.loan_book.active_loans())
    default_losses = state.bank.default_losses

    # Wages
    wages = [f.posted_wage for f in state.firms]
    avg_wage = float(np.mean(wages)) if wages else 0.0

    # Vacancies
    total_vacancies = sum(f.vacancies for f in state.firms)
    total_employed = employed

    return {
        "period": period,
        "gdp": round_money(gdp),
        "unemployment_rate": round(unemployment_rate, 4),
        "avg_price": round(float(avg_price), 4),
        "avg_wage": round(avg_wage, 2),
        "total_hh_deposits": round_money(hh_deposits),
        "total_firm_deposits": round_money(firm_deposits),
        "govt_deposits": round_money(govt_deposits),
        "govt_tax_revenue": round_money(state.government.tax_revenue),
        "govt_transfers": round_money(state.government.transfers_paid),
        "govt_spending": round_money(state.government.goods_spending),
        "govt_budget_balance": round_money(state.government.budget_balance),
        "govt_money_created": round_money(state.government.money_created),
        "govt_cumulative_money_created": round_money(state.government.cumulative_money_created),
        "bank_equity": round_money(bank_equity),
        "bank_capital_ratio": round(state.bank.capital_ratio, 4),
        "total_loans_outstanding": round_money(total_loans),
        "active_loans_count": active_loans,
        "loans_issued": round_money(state.bank.loans_issued_this_period),
        "default_losses": round_money(default_losses),
        "total_inventory": round_money(total_inventory),
        "total_production": round_money(sum(f.production for f in state.firms)),
        "total_consumption": round_money(sum(hh.consumption_spending for hh in state.households)),
        "total_wage_income": round_money(sum(hh.wage_income for hh in state.households)),
        "total_taxes": round_money(sum(hh.taxes_paid for hh in state.households)),
        "total_transfers": round_money(sum(hh.transfers_received for hh in state.households)),
        "gini_deposits": round(float(gini), 4),
        "total_employed": total_employed,
        "total_vacancies": total_vacancies,
        "employed": employed,
        "labor_force": labor_force,
    }


def _gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient for a distribution of values."""
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    cumulative = np.cumsum(values)
    return float((2.0 * np.sum((np.arange(1, n + 1) * values)) / (n * np.sum(values))) - (n + 1) / n)


def run_simulation(config: SimulationConfig) -> SimulationState:
    """Build and run a complete simulation from config."""
    state = build_simulation(config)
    logger.info(f"Starting simulation '{config.name}' for {config.num_periods} periods")

    for t in range(config.num_periods):
        metrics = step(state)
        if t % config.log_every == 0:
            logger.info(
                f"Period {t}: GDP={metrics['gdp']:.0f}, "
                f"Unemp={metrics['unemployment_rate']:.1%}, "
                f"Price={metrics['avg_price']:.2f}, "
                f"Loans={metrics['total_loans_outstanding']:.0f}"
            )

    logger.info(f"Simulation '{config.name}' completed: {config.num_periods} periods")
    return state
