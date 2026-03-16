"""Tests for bond markets and government debt issuance."""

from __future__ import annotations

import numpy as np
import pytest

from econosim.extensions.bonds import (
    Bond,
    BondStatus,
    BondMarket,
    GovernmentDebtManager,
)


class TestBond:
    def test_creation(self):
        bond = Bond("b_0", face_value=1000, coupon_rate=0.05,
                     maturity_periods=12, issue_period=0, holder_id="bank_0")
        assert bond.face_value == 1000
        assert bond.coupon_rate == 0.05
        assert bond.status == BondStatus.OUTSTANDING

    def test_issue_price_default(self):
        bond = Bond("b_0", face_value=1000, coupon_rate=0.05,
                     maturity_periods=12, issue_period=0)
        assert bond.issue_price == 1000  # Par

    def test_coupon_payment(self):
        bond = Bond("b_0", face_value=1000, coupon_rate=0.05,
                     maturity_periods=12, issue_period=0)
        assert bond.coupon_payment == 50.0

    def test_remaining_periods(self):
        bond = Bond("b_0", face_value=1000, coupon_rate=0.05,
                     maturity_periods=12, issue_period=0)
        assert bond.remaining_periods == 12
        bond.process_coupon()
        assert bond.remaining_periods == 11

    def test_maturity(self):
        bond = Bond("b_0", face_value=1000, coupon_rate=0.05,
                     maturity_periods=3, issue_period=0)
        for _ in range(3):
            bond.process_coupon()
        assert bond.is_mature
        assert bond.status == BondStatus.MATURED

    def test_current_yield(self):
        bond = Bond("b_0", face_value=1000, coupon_rate=0.05,
                     maturity_periods=12, issue_period=0)
        assert bond.current_yield == 0.05

    def test_current_yield_discount(self):
        bond = Bond("b_0", face_value=1000, coupon_rate=0.05,
                     maturity_periods=12, issue_period=0, issue_price=900)
        # Yield = 0.05 * 1000 / 900 ≈ 0.0556
        assert bond.current_yield > 0.05

    def test_fair_value(self):
        bond = Bond("b_0", face_value=1000, coupon_rate=0.05,
                     maturity_periods=12, issue_period=0)
        # At the same discount rate as coupon, fair value ≈ face value
        fv = bond.fair_value(0.05)
        assert abs(fv - 1000) < 5  # Close to par

    def test_fair_value_higher_discount(self):
        bond = Bond("b_0", face_value=1000, coupon_rate=0.05,
                     maturity_periods=12, issue_period=0)
        # Higher discount rate => lower fair value
        fv = bond.fair_value(0.10)
        assert fv < 1000

    def test_to_dict(self):
        bond = Bond("b_0", face_value=1000, coupon_rate=0.05,
                     maturity_periods=12, issue_period=0)
        d = bond.to_dict()
        assert d["bond_id"] == "b_0"
        assert d["face_value"] == 1000
        assert d["status"] == "OUTSTANDING"


class TestBondMarket:
    def test_issue_bond(self):
        market = BondMarket()
        bond = market.issue_bond(1000, 0.05, 12, 0, "bank_0")
        assert bond.bond_id == "bond_0000"
        assert bond.holder_id == "bank_0"

    def test_multiple_issuance(self):
        market = BondMarket()
        market.issue_bond(1000, 0.05, 12, 0, "bank_0")
        market.issue_bond(2000, 0.03, 24, 0, "bank_0")
        assert len(market.outstanding_bonds()) == 2
        assert market.total_outstanding() == 3000

    def test_transfer_bond(self):
        market = BondMarket()
        bond = market.issue_bond(1000, 0.05, 12, 0, "bank_0")
        success = market.transfer_bond(bond.bond_id, "hh_0", 950)
        assert success
        assert bond.holder_id == "hh_0"
        assert market.total_traded_volume == 950

    def test_process_coupons(self):
        market = BondMarket()
        market.issue_bond(1000, 0.05, 12, 0, "bank_0")
        market.issue_bond(2000, 0.03, 6, 0, "hh_0")
        coupons = market.process_coupons(1)
        assert coupons["bank_0"] == 50.0   # 1000 * 0.05
        assert coupons["hh_0"] == 60.0     # 2000 * 0.03

    def test_process_maturities(self):
        market = BondMarket()
        bond = market.issue_bond(1000, 0.05, 2, 0, "bank_0")
        market.process_coupons(1)  # period 1
        market.process_coupons(2)  # period 2 - matures
        redemptions = market.process_maturities(2)
        assert redemptions["bank_0"] == 1000

    def test_yield_curve(self):
        market = BondMarket()
        market.issue_bond(1000, 0.03, 6, 0, "bank_0")
        market.issue_bond(1000, 0.05, 12, 0, "bank_0")
        market.issue_bond(1000, 0.07, 24, 0, "bank_0")
        curve = market.yield_curve()
        assert len(curve) == 3
        # Normal yield curve: short < long
        maturities = [m for m, _ in curve]
        yields = [y for _, y in curve]
        assert maturities == sorted(maturities)

    def test_total_coupon_obligation(self):
        market = BondMarket()
        market.issue_bond(1000, 0.05, 12, 0, "bank_0")
        market.issue_bond(2000, 0.03, 12, 0, "bank_0")
        assert market.total_coupon_obligation() == 110.0  # 50 + 60

    def test_get_observation(self):
        market = BondMarket()
        market.issue_bond(1000, 0.05, 12, 0, "bank_0")
        obs = market.get_observation()
        assert obs["total_outstanding"] == 1000
        assert obs["num_outstanding_bonds"] == 1


class TestGovernmentDebtManager:
    def test_issue_debt(self):
        market = BondMarket()
        dm = GovernmentDebtManager(market)
        bond = dm.issue_debt(1000, "bank_0", period=0)
        assert bond is not None
        assert dm.total_debt_issued == 1000

    def test_service_debt(self):
        market = BondMarket()
        dm = GovernmentDebtManager(market, default_coupon_rate=0.05)
        dm.issue_debt(1000, "bank_0", period=0)
        coupons, maturities = dm.service_debt(1)
        assert coupons["bank_0"] == 50.0
        assert dm.period_interest_expense == 50.0

    def test_net_debt(self):
        market = BondMarket()
        dm = GovernmentDebtManager(market, default_maturity=2, default_coupon_rate=0.05)
        dm.issue_debt(1000, "bank_0", period=0)
        assert dm.net_debt == 1000
        dm.service_debt(1)
        dm.service_debt(2)  # matures
        assert dm.net_debt == 0

    def test_debt_to_gdp(self):
        market = BondMarket()
        dm = GovernmentDebtManager(market)
        dm.issue_debt(5000, "bank_0", period=0)
        ratio = dm.debt_to_gdp(10000)
        assert ratio == 0.5

    def test_can_issue_limit(self):
        market = BondMarket()
        dm = GovernmentDebtManager(market, max_debt_to_gdp=1.0)
        dm.issue_debt(8000, "bank_0", period=0)
        assert not dm.can_issue(5000, 10000)  # Would exceed 1.0 ratio
        assert dm.can_issue(1000, 10000)       # Under limit

    def test_reset_period(self):
        market = BondMarket()
        dm = GovernmentDebtManager(market)
        dm.issue_debt(1000, "bank_0", period=0)
        dm.service_debt(1)
        dm.reset_period_state()
        assert dm.period_interest_expense == 0
        assert dm.period_bonds_issued == 0

    def test_get_observation(self):
        market = BondMarket()
        dm = GovernmentDebtManager(market)
        dm.issue_debt(1000, "bank_0", period=0)
        obs = dm.get_observation()
        assert obs["net_debt"] == 1000
        assert "bond_market" in obs
