"""Rule-based baseline policies extracted from current agent logic.

These replicate the existing hardcoded decision rules as formal policy objects,
ensuring behavioral equivalence when swapping from inline logic to policy interfaces.
"""

from __future__ import annotations

from econosim.policies.interfaces import (
    FirmPolicy, FirmState, FirmAction,
    HouseholdPolicy, HouseholdState, HouseholdAction,
    BankPolicy, BankState, BankAction,
    GovernmentPolicy, GovernmentState, GovernmentAction,
    MacroState,
)


class RuleBasedFirmPolicy(FirmPolicy):
    """Replicates the existing firm decision rules as a policy object.

    - Vacancy decision: based on expected demand, inventory target, affordability
    - Price adjustment: lower if inventory high, raise if inventory low + sales
    - Wage adjustment: raise if fill rate low, lower if fill rate high
    - Borrowing: request loan if cash < expected wage bill
    """

    def __init__(
        self,
        price_adjustment_speed: float = 0.03,
        wage_adjustment_speed: float = 0.02,
        max_leverage: float = 3.0,
    ) -> None:
        self.price_adjustment_speed = price_adjustment_speed
        self.wage_adjustment_speed = wage_adjustment_speed
        self.max_leverage = max_leverage

    def act(self, firm_state: FirmState, macro_state: MacroState) -> FirmAction:
        fs = firm_state
        action = FirmAction()

        # --- Vacancy decision ---
        expected_sales = max(fs.prev_units_sold, 1.0)
        revenue_units = fs.prev_revenue / max(fs.price, 0.01)
        demand_estimate = max(expected_sales, revenue_units)
        target_inv = demand_estimate * fs.target_inventory_ratio
        production_needed = max(0.0, demand_estimate + target_inv - fs.inventory)
        workers_needed = int(production_needed / max(fs.labor_productivity, 0.01)) + 1
        affordable = int(fs.deposits / max(fs.posted_wage, 1.0))
        min_hire = 1 if affordable >= 1 else 0
        action.vacancies = max(min_hire, min(workers_needed, affordable))

        # --- Price adjustment ---
        target_inv = max(fs.prev_units_sold, 1.0) * fs.target_inventory_ratio
        inv_ratio = fs.inventory / max(target_inv, 1.0)
        if inv_ratio > 1.2:
            action.price_adjustment = 1.0 - self.price_adjustment_speed
        elif inv_ratio < 0.8 and fs.prev_units_sold > 0.1:
            action.price_adjustment = 1.0 + self.price_adjustment_speed
        else:
            action.price_adjustment = 1.0

        # --- Loan request ---
        expected_wage_bill = fs.posted_wage * max(action.vacancies, 1)
        cash_shortfall = expected_wage_bill - fs.deposits
        if cash_shortfall > 0:
            if fs.equity > 0 and (fs.total_debt + cash_shortfall * 1.2) / fs.equity <= self.max_leverage:
                action.loan_request = cash_shortfall * 1.2
            else:
                action.loan_request = 0.0

        return action


class RuleBasedHouseholdPolicy(HouseholdPolicy):
    """Replicates existing household decision rules."""

    def act(self, hh_state: HouseholdState, macro_state: MacroState) -> HouseholdAction:
        hs = hh_state
        action = HouseholdAction()

        # Buffer-stock consumption: C = α1 * income + α2 * wealth
        disposable = hs.wage_income - 0  # taxes handled externally
        income_part = hs.consumption_propensity * max(0.0, disposable)
        wealth_part = hs.wealth_propensity * max(0.0, hs.deposits)
        desired = income_part + wealth_part
        budget = min(desired, max(0.0, hs.deposits))
        action.consumption_fraction = budget / max(hs.deposits, 0.01)
        action.consumption_fraction = min(1.0, max(0.0, action.consumption_fraction))

        action.labor_participation = True  # always participate
        action.reservation_wage_adjustment = 1.0

        return action


class RuleBasedBankPolicy(BankPolicy):
    """Replicates existing bank decision rules (passive — no policy actions)."""

    def act(self, bank_state: BankState, macro_state: MacroState) -> BankAction:
        return BankAction()  # No adjustments — keep current settings


class RuleBasedGovernmentPolicy(GovernmentPolicy):
    """Replicates existing government decision rules (fixed policy parameters)."""

    def act(self, govt_state: GovernmentState, macro_state: MacroState) -> GovernmentAction:
        return GovernmentAction(
            tax_rate=govt_state.income_tax_rate,
            transfer_per_unemployed=govt_state.transfer_per_unemployed,
            spending_per_period=govt_state.spending_per_period,
        )
