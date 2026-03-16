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
from econosim.extensions.expectations import AgentExpectations
from econosim.extensions.networks import TradeNetwork, CreditNetwork
from econosim.extensions.bonds import BondMarket, GovernmentDebtManager
from econosim.policies.interfaces import (
    FirmPolicy, HouseholdPolicy, BankPolicy, GovernmentPolicy,
    FirmState, HouseholdState, BankState, GovernmentState,
    MacroState, FirmAction, BankAction, GovernmentAction,
)

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

        # Phase 4 extensions (initialized by build_simulation when enabled)
        self.expectations: dict[str, AgentExpectations] = {}
        self.trade_network: TradeNetwork | None = None
        self.credit_network: CreditNetwork | None = None
        self.bond_market: BondMarket | None = None
        self.debt_manager: GovernmentDebtManager | None = None

        # Policy interfaces (None = use hardcoded agent logic)
        self.firm_policy: FirmPolicy | None = None
        self.household_policy: HouseholdPolicy | None = None
        self.bank_policy: BankPolicy | None = None
        self.government_policy: GovernmentPolicy | None = None


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

    state = SimulationState(
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

    # Initialize Phase 4 extensions based on feature flags
    ext = config.extensions

    if ext.enable_expectations:
        exp_cfg = ext.expectations
        for firm in firms:
            state.expectations[firm.agent_id] = AgentExpectations(
                agent_id=firm.agent_id,
            )
        logger.info("Expectations extension enabled for %d firms", len(firms))

    if ext.enable_networks:
        state.trade_network = TradeNetwork()
        state.credit_network = CreditNetwork()
        logger.info("Network tracking extension enabled")

    if ext.enable_bonds:
        bond_cfg = ext.bonds
        state.bond_market = BondMarket()
        state.debt_manager = GovernmentDebtManager(
            bond_market=state.bond_market,
            default_maturity=bond_cfg.default_maturity,
            default_coupon_rate=bond_cfg.default_coupon_rate,
            max_debt_to_gdp=bond_cfg.max_debt_to_gdp,
        )
        logger.info("Bond market extension enabled")

    return state


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


def build_macro_state(state: SimulationState) -> MacroState:
    """Build a MacroState snapshot from current simulation state."""
    h = state.history
    period = state.current_period

    if len(h) >= 1:
        last = h[-1]
        gdp = last["gdp"]
        gdp_growth = (gdp - h[-2]["gdp"]) / max(h[-2]["gdp"], 1.0) if len(h) >= 2 else 0.0
        prev_price = h[-2]["avg_price"] if len(h) >= 2 else last["avg_price"]
        inflation = (last["avg_price"] - prev_price) / max(prev_price, 0.01)
        prev_loans = h[-2]["total_loans_outstanding"] if len(h) >= 2 else last["total_loans_outstanding"]
        credit_growth = (last["total_loans_outstanding"] - prev_loans) / max(prev_loans, 1.0)
        return MacroState(
            period=period,
            gdp=gdp,
            gdp_growth=gdp_growth,
            inflation_rate=inflation,
            unemployment_rate=last["unemployment_rate"],
            avg_price=last["avg_price"],
            avg_wage=last["avg_wage"],
            total_credit=last["total_loans_outstanding"],
            credit_growth=credit_growth,
            bank_capital_ratio=last["bank_capital_ratio"],
            lending_rate=state.bank.lending_rate,
        )
    else:
        # First period — use initial values
        return MacroState(
            period=period,
            lending_rate=state.bank.lending_rate,
            bank_capital_ratio=state.bank.capital_ratio,
        )


def build_firm_state(firm: Firm) -> FirmState:
    """Build a FirmState observation from a firm agent."""
    return FirmState(
        deposits=firm.deposits,
        inventory=firm.inventory.quantity,
        price=firm.price,
        posted_wage=firm.posted_wage,
        workers_count=len(firm.workers),
        revenue=firm.revenue,
        prev_revenue=firm.prev_revenue,
        wage_bill=firm.wage_bill,
        units_sold=firm.units_sold,
        prev_units_sold=firm.prev_units_sold,
        total_debt=firm.total_debt,
        equity=firm.equity_value,
        labor_productivity=firm.labor_productivity,
        target_inventory_ratio=firm.target_inventory_ratio,
    )


def build_bank_state(bank: Bank) -> BankState:
    """Build a BankState observation from the bank agent."""
    return BankState(
        total_loans=bank.total_loans,
        total_deposits=bank.total_deposits_liability,
        equity=bank.equity_value,
        capital_ratio=bank.capital_ratio,
        lending_rate=bank.lending_rate,
        interest_income=bank.interest_income,
        default_losses=bank.default_losses,
        active_loans_count=len(bank.loan_book.active_loans()),
        base_interest_rate=bank.base_interest_rate,
        risk_premium=bank.risk_premium,
    )


def build_govt_state(govt: Government) -> GovernmentState:
    """Build a GovernmentState observation from the government agent."""
    return GovernmentState(
        deposits=govt.deposits,
        tax_revenue=govt.tax_revenue,
        transfers_paid=govt.transfers_paid,
        goods_spending=govt.goods_spending,
        budget_balance=govt.budget_balance,
        income_tax_rate=govt.income_tax_rate,
        transfer_per_unemployed=govt.transfer_per_unemployed,
        spending_per_period=govt.spending_per_period,
        cumulative_debt=govt.cumulative_money_created,
    )


def _apply_firm_policy(state: SimulationState, macro: MacroState) -> None:
    """Apply firm policy actions: set vacancies and price adjustments."""
    policy = state.firm_policy
    if policy is None:
        return

    for firm in state.firms:
        fs = build_firm_state(firm)
        action = policy.act(fs, macro)

        # Set vacancies directly (overrides decide_vacancies)
        firm.vacancies = max(0, action.vacancies)

        # Apply price adjustment
        firm.price = round_money(firm.price * action.price_adjustment)
        firm.price = max(0.01, firm.price)


def _apply_bank_policy(state: SimulationState, macro: MacroState) -> None:
    """Apply bank policy actions: adjust rates and targets."""
    policy = state.bank_policy
    if policy is None:
        return

    bs = build_bank_state(state.bank)
    action = policy.act(bs, macro)

    state.bank.base_interest_rate = max(0.0, state.bank.base_interest_rate + action.base_rate_adjustment)
    state.bank.capital_adequacy_ratio = max(0.01, state.bank.capital_adequacy_ratio + action.capital_target_adjustment)
    state.bank.risk_premium = max(0.0, state.bank.risk_premium + action.risk_premium_adjustment)


def _apply_govt_policy(state: SimulationState, macro: MacroState) -> None:
    """Apply government policy actions: update fiscal parameters."""
    policy = state.government_policy
    if policy is None:
        return

    gs = build_govt_state(state.government)
    action = policy.act(gs, macro)

    state.government.income_tax_rate = max(0.0, min(1.0, action.tax_rate))
    state.government.transfer_per_unemployed = max(0.0, action.transfer_per_unemployed)
    state.government.spending_per_period = max(0.0, action.spending_per_period)


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

    # ── 1a. Apply policy actions ─────────────────────────────────
    macro = build_macro_state(state)
    _apply_bank_policy(state, macro)
    _apply_govt_policy(state, macro)
    if state.firm_policy is not None:
        _apply_firm_policy(state, macro)

    # ── 1b. Reset extension period state ─────────────────────────
    if state.debt_manager is not None:
        state.debt_manager.reset_period_state()
    if state.trade_network is not None:
        decay_rate = state.config.extensions.networks.edge_decay_rate
        state.trade_network.decay_edges(decay_rate)
    if state.credit_network is not None:
        decay_rate = state.config.extensions.networks.edge_decay_rate
        state.credit_network.decay_edges(decay_rate)

    # ── 2. Credit market ─────────────────────────────────────────
    state.credit_market.clear(
        firms=state.firms,
        bank=state.bank,
        period=period,
    )

    # Record credit flows in network
    if state.credit_network is not None and state.credit_market.total_issued > 0:
        for loan in state.bank.loan_book.active_loans():
            if loan.origination_period == period:
                state.credit_network.record_loan(
                    state.bank.agent_id, loan.borrower_id, loan.principal, period
                )

    # ── 3. Labor market ──────────────────────────────────────────
    state.labor_market.clear(
        households=state.households,
        firms=state.firms,
        period=period,
        rng=state.rng,
        skip_vacancy_decision=state.firm_policy is not None,
    )

    # ── 4. Production ────────────────────────────────────────────
    for firm in state.firms:
        firm.produce()

    # ── 5. Goods pricing + goods market ──────────────────────────
    if state.firm_policy is None:
        for firm in state.firms:
            firm.adjust_price()

    state.goods_market.clear(
        households=state.households,
        firms=state.firms,
        period=period,
        rng=state.rng,
    )

    # Record trade flows in network (aggregate firm sales this period)
    if state.trade_network is not None:
        for firm in state.firms:
            if firm.units_sold > 0:
                # Record aggregate sales volume per firm
                state.trade_network.record_trade(
                    "households", firm.agent_id, firm.revenue, period
                )

    # Sync inventory asset on balance sheet after goods market sales
    for firm in state.firms:
        firm.sync_after_sales()

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

    # Bond-financed spending: track debt issuance alongside sovereign money creation
    if state.debt_manager is not None and fiscal_need > state.government.deposits:
        shortfall = round_money(fiscal_need - state.government.deposits)
        last_gdp = state.history[-1]["gdp"] if state.history else fiscal_need * 10
        if state.debt_manager.can_issue(shortfall, last_gdp):
            # Record bond issuance (the actual funding comes via ensure_solvency)
            state.debt_manager.issue_debt(
                amount=shortfall,
                buyer_id=state.bank.agent_id,
                period=period,
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

    # Sync inventory asset after government purchases
    for firm in state.firms:
        firm.sync_after_sales()

    # ── 7. Debt service / delinquency / default ──────────────────
    state.bank.process_loan_payments(period)
    defaulted = state.bank.process_defaults(period)

    # Bond debt service: process coupons and maturities (tracking only)
    if state.debt_manager is not None:
        state.debt_manager.service_debt(period)

    # ── 8. Wage adjustment (for next period) ─────────────────────
    for firm in state.firms:
        firm.adjust_wage()

    # ── 8b. Update expectations ────────────────────────────────
    if state.expectations:
        avg_price = float(np.mean([f.price for f in state.firms])) if state.firms else 0.0
        prev_price = state.history[-1]["avg_price"] if state.history else avg_price
        inflation = (avg_price - prev_price) / max(prev_price, 0.01) if prev_price > 0 else 0.0
        for firm in state.firms:
            exp = state.expectations.get(firm.agent_id)
            if exp is not None:
                exp.update_all(
                    actual_price=avg_price,
                    actual_wage=firm.posted_wage,
                    actual_demand=firm.units_sold,
                    actual_inflation=inflation,
                )

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

    metrics = {
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

    # Extension metrics
    if state.trade_network is not None:
        obs = state.trade_network.get_observation()
        metrics["trade_network_density"] = obs["density"]
        metrics["trade_network_concentration"] = obs["concentration_hhi"]
        metrics["trade_seller_concentration"] = state.trade_network.seller_concentration()

    if state.credit_network is not None:
        obs = state.credit_network.get_observation()
        metrics["credit_network_density"] = obs["density"]
        metrics["credit_network_concentration"] = obs["concentration_hhi"]
        metrics["credit_systemic_risk"] = state.credit_network.systemic_risk_score()

    if state.debt_manager is not None:
        obs = state.debt_manager.get_observation()
        metrics["bond_outstanding"] = obs["total_outstanding"]
        metrics["bond_interest_expense"] = obs["period_interest_expense"]
        metrics["bond_issued"] = obs["period_bonds_issued"]
        metrics["bond_redeemed"] = obs["period_bonds_redeemed"]
        metrics["bond_debt_to_gdp"] = state.debt_manager.debt_to_gdp(gdp)

    if state.expectations:
        # Average forecast errors across firms
        price_errors = []
        demand_errors = []
        for exp in state.expectations.values():
            price_errors.append(abs(exp.price.forecast_error()))
            demand_errors.append(abs(exp.demand.forecast_error()))
        metrics["avg_price_forecast_error"] = round(float(np.mean(price_errors)), 4) if price_errors else 0.0
        metrics["avg_demand_forecast_error"] = round(float(np.mean(demand_errors)), 4) if demand_errors else 0.0

    return metrics


def _gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient for a distribution of values."""
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    cumulative = np.cumsum(values)
    return float((2.0 * np.sum((np.arange(1, n + 1) * values)) / (n * np.sum(values))) - (n + 1) / n)


def run_simulation(
    config: SimulationConfig,
    firm_policy: FirmPolicy | None = None,
    household_policy: HouseholdPolicy | None = None,
    bank_policy: BankPolicy | None = None,
    government_policy: GovernmentPolicy | None = None,
) -> SimulationState:
    """Build and run a complete simulation from config.

    Optional policy arguments override the default hardcoded agent logic.
    """
    state = build_simulation(config)
    state.firm_policy = firm_policy
    state.household_policy = household_policy
    state.bank_policy = bank_policy
    state.government_policy = government_policy
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
