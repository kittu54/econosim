"""Tests for the Household agent."""

import pytest

from econosim.core.accounting import Ledger
from econosim.agents.household import Household


@pytest.fixture
def ledger():
    return Ledger()


@pytest.fixture
def household(ledger):
    return Household(
        agent_id="hh_0001",
        ledger=ledger,
        initial_deposits=1000.0,
        consumption_propensity=0.8,
        wealth_propensity=0.4,
        reservation_wage=50.0,
    )


class TestHouseholdInit:
    def test_initial_deposits(self, household):
        assert household.deposits == 1000.0

    def test_balance_sheet_balanced(self, household):
        assert household.balance_sheet.check_balanced()

    def test_initial_equity(self, household):
        assert household.balance_sheet.total_equity == pytest.approx(1000.0)

    def test_agent_type(self, household):
        assert household.agent_type == "household"


class TestHouseholdDecisions:
    def test_desired_consumption_with_income(self, household):
        household.wage_income = 100.0
        # C = alpha1 * disposable + alpha2 * wealth
        # disposable = 100 - 0 + 0 = 100
        # C = 0.8 * 100 + 0.4 * 1000 = 80 + 400 = 480
        assert household.desired_consumption() == pytest.approx(480.0, abs=1.0)

    def test_desired_consumption_capped_by_deposits(self, ledger):
        hh = Household("hh_poor", ledger, initial_deposits=10.0)
        hh.wage_income = 5.0
        # Can't consume more than deposits
        assert hh.desired_consumption() <= hh.deposits + 0.01

    def test_desired_consumption_zero_income(self, household):
        # With zero income, only wealth term matters
        # C = 0.8 * 0 + 0.4 * 1000 = 400
        assert household.desired_consumption() == pytest.approx(400.0, abs=1.0)

    def test_wants_to_work_when_unemployed(self, household):
        assert household.wants_to_work() is True

    def test_does_not_want_to_work_when_employed(self, household):
        household.employed = True
        assert household.wants_to_work() is False

    def test_does_not_want_to_work_no_participation(self, ledger):
        hh = Household("hh_nolabor", ledger, labor_participation=False)
        assert hh.wants_to_work() is False

    def test_accept_wage_above_reservation(self, household):
        assert household.accept_wage(60.0) is True

    def test_reject_wage_below_reservation(self, household):
        assert household.accept_wage(40.0) is False

    def test_accept_wage_at_reservation(self, household):
        assert household.accept_wage(50.0) is True


class TestHouseholdPeriodState:
    def test_reset_clears_flows(self, household):
        household.wage_income = 100.0
        household.consumption_spending = 80.0
        household.taxes_paid = 20.0
        household.transfers_received = 10.0
        household.reset_period_state()
        assert household.wage_income == 0.0
        assert household.consumption_spending == 0.0
        assert household.taxes_paid == 0.0
        assert household.transfers_received == 0.0

    def test_disposable_income(self, household):
        household.wage_income = 100.0
        household.taxes_paid = 20.0
        household.transfers_received = 10.0
        assert household.disposable_income == pytest.approx(90.0)


class TestHouseholdObservation:
    def test_observation_keys(self, household):
        obs = household.get_observation()
        assert "deposits" in obs
        assert "employed" in obs
        assert "wage_income" in obs
        assert "consumption_propensity" in obs
        assert "disposable_income" in obs
