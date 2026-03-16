"""
Bond markets and government debt issuance.

Extends the government from pure currency issuance to explicit debt
instruments with yield curves, maturity profiles, and secondary trading.

Key concepts:
- Bond: A fixed-income security with face value, coupon rate, and maturity
- BondMarket: Primary and secondary market for bonds
- GovernmentDebtManager: Manages bond issuance and debt service
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from econosim.core.accounting import round_money


class BondStatus(Enum):
    OUTSTANDING = auto()
    MATURED = auto()
    DEFAULTED = auto()


@dataclass
class Bond:
    """A fixed-income government security."""

    bond_id: str
    face_value: float           # Par value at maturity
    coupon_rate: float          # Annual coupon rate
    maturity_periods: int       # Periods until maturity
    issue_period: int           # When issued
    issue_price: float = 0.0   # Price at issuance (may differ from face value)
    holder_id: str = ""        # Current holder
    status: BondStatus = BondStatus.OUTSTANDING
    coupon_payments_made: int = 0

    def __post_init__(self) -> None:
        if self.issue_price == 0.0:
            self.issue_price = self.face_value

    @property
    def coupon_payment(self) -> float:
        """Per-period coupon payment."""
        return round_money(self.face_value * self.coupon_rate)

    @property
    def remaining_periods(self) -> int:
        """Periods remaining until maturity (relative to issue)."""
        return max(0, self.maturity_periods - self.coupon_payments_made)

    @property
    def is_mature(self) -> bool:
        return self.coupon_payments_made >= self.maturity_periods

    @property
    def current_yield(self) -> float:
        """Current yield = coupon / market price (approximated by issue price)."""
        if self.issue_price <= 0:
            return 0.0
        return self.coupon_rate * self.face_value / self.issue_price

    def fair_value(self, discount_rate: float) -> float:
        """Present value of remaining cash flows at a given discount rate."""
        if discount_rate <= 0:
            return self.face_value + self.coupon_payment * self.remaining_periods
        pv_coupons = 0.0
        for t in range(1, self.remaining_periods + 1):
            pv_coupons += self.coupon_payment / (1 + discount_rate) ** t
        pv_face = self.face_value / (1 + discount_rate) ** self.remaining_periods
        return round_money(pv_coupons + pv_face)

    def process_coupon(self) -> float:
        """Process one coupon payment. Returns the amount paid."""
        if self.status != BondStatus.OUTSTANDING:
            return 0.0
        self.coupon_payments_made += 1
        if self.is_mature:
            self.status = BondStatus.MATURED
        return self.coupon_payment

    def to_dict(self) -> dict[str, Any]:
        return {
            "bond_id": self.bond_id,
            "face_value": self.face_value,
            "coupon_rate": self.coupon_rate,
            "maturity_periods": self.maturity_periods,
            "remaining_periods": self.remaining_periods,
            "holder_id": self.holder_id,
            "status": self.status.name,
            "coupon_payment": self.coupon_payment,
        }


class BondMarket:
    """Primary and secondary market for government bonds.

    Handles bond issuance (primary) and trading between agents (secondary).
    Tracks market prices and constructs yield curves.
    """

    def __init__(self) -> None:
        self._bonds: dict[str, Bond] = {}
        self._next_id: int = 0
        self.last_auction_price: float = 0.0
        self.last_auction_yield: float = 0.0
        self.total_traded_volume: float = 0.0
        self.period_issuance: float = 0.0

    def issue_bond(
        self,
        face_value: float,
        coupon_rate: float,
        maturity_periods: int,
        issue_period: int,
        holder_id: str,
        issue_price: float | None = None,
    ) -> Bond:
        """Issue a new bond in the primary market."""
        bond_id = f"bond_{self._next_id:04d}"
        self._next_id += 1

        bond = Bond(
            bond_id=bond_id,
            face_value=face_value,
            coupon_rate=coupon_rate,
            maturity_periods=maturity_periods,
            issue_period=issue_period,
            issue_price=issue_price or face_value,
            holder_id=holder_id,
        )
        self._bonds[bond_id] = bond
        self.last_auction_price = bond.issue_price
        self.last_auction_yield = bond.current_yield
        self.period_issuance += face_value
        return bond

    def transfer_bond(self, bond_id: str, new_holder: str, price: float) -> bool:
        """Transfer a bond to a new holder at a given price (secondary market)."""
        bond = self._bonds.get(bond_id)
        if bond is None or bond.status != BondStatus.OUTSTANDING:
            return False
        bond.holder_id = new_holder
        self.total_traded_volume += price
        return True

    def process_coupons(self, current_period: int) -> dict[str, float]:
        """Process all outstanding coupon payments.

        Returns dict of holder_id -> total coupon income.
        """
        coupon_income: dict[str, float] = {}
        for bond in self._bonds.values():
            if bond.status != BondStatus.OUTSTANDING:
                continue
            payment = bond.process_coupon()
            if payment > 0:
                coupon_income[bond.holder_id] = round_money(
                    coupon_income.get(bond.holder_id, 0.0) + payment
                )
        return coupon_income

    def process_maturities(self, current_period: int) -> dict[str, float]:
        """Process maturing bonds. Returns dict of holder_id -> total face value returned."""
        redemptions: dict[str, float] = {}
        for bond in self._bonds.values():
            if bond.status == BondStatus.MATURED:
                redemptions[bond.holder_id] = round_money(
                    redemptions.get(bond.holder_id, 0.0) + bond.face_value
                )
        return redemptions

    def outstanding_bonds(self) -> list[Bond]:
        return [b for b in self._bonds.values() if b.status == BondStatus.OUTSTANDING]

    def total_outstanding(self) -> float:
        return round_money(sum(b.face_value for b in self.outstanding_bonds()))

    def total_coupon_obligation(self) -> float:
        return round_money(sum(b.coupon_payment for b in self.outstanding_bonds()))

    def yield_curve(self) -> list[tuple[int, float]]:
        """Construct a simple yield curve from outstanding bonds.

        Returns list of (remaining_periods, yield) sorted by maturity.
        """
        points: dict[int, list[float]] = {}
        for bond in self.outstanding_bonds():
            rem = bond.remaining_periods
            if rem > 0:
                y = bond.current_yield
                points.setdefault(rem, []).append(y)

        curve = []
        for maturity in sorted(points.keys()):
            avg_yield = float(np.mean(points[maturity]))
            curve.append((maturity, round(avg_yield, 6)))
        return curve

    def reset_period_state(self) -> None:
        self.period_issuance = 0.0

    def get_observation(self) -> dict[str, Any]:
        return {
            "total_outstanding": self.total_outstanding(),
            "total_coupon_obligation": self.total_coupon_obligation(),
            "num_outstanding_bonds": len(self.outstanding_bonds()),
            "last_auction_price": self.last_auction_price,
            "last_auction_yield": self.last_auction_yield,
            "yield_curve": self.yield_curve(),
            "period_issuance": self.period_issuance,
        }


class GovernmentDebtManager:
    """Manages government bond issuance and debt service.

    Replaces pure money creation with explicit debt instruments.
    Government can still create money (monetization), but can also
    issue bonds to finance spending.
    """

    def __init__(
        self,
        bond_market: BondMarket,
        default_maturity: int = 12,
        default_coupon_rate: float = 0.005,
        max_debt_to_gdp: float = 2.0,
    ) -> None:
        self.bond_market = bond_market
        self.default_maturity = default_maturity
        self.default_coupon_rate = default_coupon_rate
        self.max_debt_to_gdp = max_debt_to_gdp

        # Tracking
        self.total_debt_issued: float = 0.0
        self.total_interest_paid: float = 0.0
        self.total_principal_redeemed: float = 0.0
        self.period_interest_expense: float = 0.0
        self.period_bonds_issued: float = 0.0
        self.period_bonds_redeemed: float = 0.0

    def issue_debt(
        self,
        amount: float,
        buyer_id: str,
        period: int,
        maturity: int | None = None,
        coupon_rate: float | None = None,
    ) -> Bond | None:
        """Issue government bonds to finance spending."""
        if amount <= 0:
            return None

        bond = self.bond_market.issue_bond(
            face_value=amount,
            coupon_rate=coupon_rate or self.default_coupon_rate,
            maturity_periods=maturity or self.default_maturity,
            issue_period=period,
            holder_id=buyer_id,
        )

        self.total_debt_issued += amount
        self.period_bonds_issued += amount
        return bond

    def service_debt(self, current_period: int) -> tuple[dict[str, float], dict[str, float]]:
        """Process coupon payments and bond maturities.

        Returns (coupon_payments, redemptions) as dicts of holder_id -> amount.
        """
        coupons = self.bond_market.process_coupons(current_period)
        maturities = self.bond_market.process_maturities(current_period)

        total_coupons = sum(coupons.values())
        total_redemptions = sum(maturities.values())

        self.period_interest_expense = total_coupons
        self.total_interest_paid += total_coupons
        self.period_bonds_redeemed = total_redemptions
        self.total_principal_redeemed += total_redemptions

        return coupons, maturities

    @property
    def net_debt(self) -> float:
        return round_money(self.total_debt_issued - self.total_principal_redeemed)

    def debt_to_gdp(self, gdp: float) -> float:
        """Compute debt-to-GDP ratio."""
        if gdp <= 0:
            return 0.0
        return round(self.bond_market.total_outstanding() / gdp, 4)

    def can_issue(self, amount: float, gdp: float) -> bool:
        """Check if new issuance would exceed debt-to-GDP limit."""
        projected = self.bond_market.total_outstanding() + amount
        if gdp <= 0:
            return True
        return projected / gdp <= self.max_debt_to_gdp

    def reset_period_state(self) -> None:
        self.period_interest_expense = 0.0
        self.period_bonds_issued = 0.0
        self.period_bonds_redeemed = 0.0
        self.bond_market.reset_period_state()

    def get_observation(self) -> dict[str, Any]:
        return {
            "net_debt": self.net_debt,
            "total_outstanding": self.bond_market.total_outstanding(),
            "total_interest_paid": self.total_interest_paid,
            "period_interest_expense": self.period_interest_expense,
            "period_bonds_issued": self.period_bonds_issued,
            "period_bonds_redeemed": self.period_bonds_redeemed,
            "bond_market": self.bond_market.get_observation(),
        }
