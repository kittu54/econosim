"""
Multi-sector production with input-output matrices.

Extends the single-good MVP to support multiple goods and sectors.
Each sector produces a distinct good using labor and intermediate inputs
from other sectors, governed by an input-output (Leontief) matrix.

Key concepts:
- Good: A distinct type of commodity (consumption good, intermediate, capital)
- Sector: A group of firms producing the same good
- InputOutputMatrix: Technical coefficients defining inter-sector requirements
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from econosim.core.accounting import round_money


class GoodType(Enum):
    """Classification of goods."""
    CONSUMPTION = auto()    # Final goods for households
    INTERMEDIATE = auto()   # Used as inputs by other sectors
    CAPITAL = auto()        # Durable goods for production


@dataclass
class Good:
    """A distinct type of commodity in the economy."""

    good_id: str
    name: str
    good_type: GoodType = GoodType.CONSUMPTION
    unit: str = "units"
    perishable: bool = False  # If True, unsold inventory depreciates
    depreciation_rate: float = 0.0  # Per-period inventory depreciation

    def __hash__(self) -> int:
        return hash(self.good_id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Good):
            return self.good_id == other.good_id
        return NotImplemented


@dataclass
class SectorInventory:
    """Multi-good inventory for a sector/firm."""

    owner_id: str
    stocks: dict[str, float] = field(default_factory=dict)     # good_id -> quantity
    unit_costs: dict[str, float] = field(default_factory=dict)  # good_id -> avg cost

    def add(self, good_id: str, quantity: float, cost: float) -> None:
        """Add produced or purchased goods to inventory."""
        if quantity <= 0:
            return
        old_qty = self.stocks.get(good_id, 0.0)
        old_cost = self.unit_costs.get(good_id, 0.0)
        old_value = old_qty * old_cost
        new_qty = round_money(old_qty + quantity)
        new_value = old_value + cost
        self.stocks[good_id] = new_qty
        self.unit_costs[good_id] = round_money(new_value / new_qty) if new_qty > 0 else 0.0

    def remove(self, good_id: str, quantity: float) -> float:
        """Remove goods from inventory. Returns cost of goods removed."""
        available = self.stocks.get(good_id, 0.0)
        actual = min(quantity, available)
        if actual <= 0:
            return 0.0
        cost = round_money(actual * self.unit_costs.get(good_id, 0.0))
        self.stocks[good_id] = round_money(available - actual)
        if self.stocks[good_id] < 0.01:
            self.stocks[good_id] = 0.0
            self.unit_costs[good_id] = 0.0
        return cost

    def quantity(self, good_id: str) -> float:
        return self.stocks.get(good_id, 0.0)

    def total_value(self) -> float:
        return round_money(sum(
            qty * self.unit_costs.get(gid, 0.0)
            for gid, qty in self.stocks.items()
        ))

    def depreciate(self, goods: dict[str, Good]) -> float:
        """Apply depreciation to perishable goods. Returns total value lost."""
        total_loss = 0.0
        for good_id, qty in list(self.stocks.items()):
            good = goods.get(good_id)
            if good and good.perishable and good.depreciation_rate > 0 and qty > 0:
                loss_qty = round_money(qty * good.depreciation_rate)
                loss_val = round_money(loss_qty * self.unit_costs.get(good_id, 0.0))
                self.stocks[good_id] = round_money(qty - loss_qty)
                total_loss += loss_val
        return total_loss


class InputOutputMatrix:
    """Leontief input-output matrix defining inter-sector technical coefficients.

    a[i][j] = units of good i required to produce 1 unit of good j.
    Labor coefficients: l[j] = workers needed per unit of good j.

    Example for a 3-sector economy (agriculture, manufacturing, services):
        A = [[0.1, 0.2, 0.05],  # agriculture inputs
             [0.1, 0.15, 0.1],  # manufacturing inputs
             [0.05, 0.1, 0.1]]  # services inputs
        L = [0.3, 0.5, 0.2]     # labor per unit
    """

    def __init__(
        self,
        goods: list[Good],
        coefficients: np.ndarray | None = None,
        labor_coefficients: np.ndarray | None = None,
    ) -> None:
        self.goods = list(goods)
        self.n_sectors = len(goods)
        self._good_index = {g.good_id: i for i, g in enumerate(goods)}

        if coefficients is not None:
            assert coefficients.shape == (self.n_sectors, self.n_sectors), \
                f"Expected {self.n_sectors}x{self.n_sectors} matrix"
            self.A = coefficients.copy()
        else:
            self.A = np.zeros((self.n_sectors, self.n_sectors))

        if labor_coefficients is not None:
            assert len(labor_coefficients) == self.n_sectors
            self.L = np.array(labor_coefficients, dtype=float)
        else:
            self.L = np.ones(self.n_sectors) * 0.5

    def get_coefficient(self, input_good: str, output_good: str) -> float:
        """Get the technical coefficient a[i][j]."""
        i = self._good_index[input_good]
        j = self._good_index[output_good]
        return float(self.A[i, j])

    def set_coefficient(self, input_good: str, output_good: str, value: float) -> None:
        """Set a technical coefficient."""
        i = self._good_index[input_good]
        j = self._good_index[output_good]
        self.A[i, j] = value

    def inputs_required(self, output_good: str, quantity: float) -> dict[str, float]:
        """Calculate intermediate inputs needed to produce a quantity of output_good."""
        j = self._good_index[output_good]
        required = {}
        for i, good in enumerate(self.goods):
            amount = round_money(self.A[i, j] * quantity)
            if amount > 0:
                required[good.good_id] = amount
        return required

    def labor_required(self, output_good: str, quantity: float) -> float:
        """Calculate labor needed to produce a quantity of output_good."""
        j = self._good_index[output_good]
        return round_money(self.L[j] * quantity)

    def leontief_inverse(self) -> np.ndarray:
        """Compute the Leontief inverse (I - A)^(-1) for total requirements."""
        I = np.eye(self.n_sectors)
        return np.linalg.inv(I - self.A)

    def total_requirements(self, final_demand: np.ndarray) -> np.ndarray:
        """Calculate total output needed to satisfy a final demand vector."""
        L_inv = self.leontief_inverse()
        return L_inv @ final_demand

    def is_productive(self) -> bool:
        """Check if the IO matrix is productive (Hawkins-Simon condition).

        All eigenvalues of A must have modulus less than 1.
        """
        eigenvalues = np.linalg.eigvals(self.A)
        return bool(np.all(np.abs(eigenvalues) < 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "goods": [g.good_id for g in self.goods],
            "coefficients": self.A.tolist(),
            "labor_coefficients": self.L.tolist(),
        }


@dataclass
class Sector:
    """A sector groups firms producing the same good.

    Aggregates sector-level statistics and coordinates production
    using the input-output matrix.
    """

    sector_id: str
    good: Good
    firm_ids: list[str] = field(default_factory=list)

    # Sector-level parameters
    base_productivity: float = 1.0
    price_level: float = 10.0

    # Sector-level statistics (updated each period)
    total_output: float = 0.0
    total_revenue: float = 0.0
    total_employment: int = 0
    capacity_utilization: float = 0.0

    def reset_period_stats(self) -> None:
        self.total_output = 0.0
        self.total_revenue = 0.0
        self.total_employment = 0
        self.capacity_utilization = 0.0

    def compute_sector_price(self, firm_prices: list[float]) -> float:
        """Compute sector price as output-weighted average."""
        if not firm_prices:
            return self.price_level
        self.price_level = round_money(float(np.mean(firm_prices)))
        return self.price_level

    def get_observation(self) -> dict[str, Any]:
        return {
            "sector_id": self.sector_id,
            "good_id": self.good.good_id,
            "total_output": self.total_output,
            "total_revenue": self.total_revenue,
            "total_employment": self.total_employment,
            "price_level": self.price_level,
            "capacity_utilization": self.capacity_utilization,
            "num_firms": len(self.firm_ids),
        }


def create_default_sectors() -> tuple[list[Good], list[Sector], InputOutputMatrix]:
    """Create a default 3-sector economy (agriculture, manufacturing, services).

    Returns (goods, sectors, io_matrix).
    """
    goods = [
        Good("agri", "Agriculture", GoodType.CONSUMPTION, perishable=True,
             depreciation_rate=0.1),
        Good("mfg", "Manufacturing", GoodType.INTERMEDIATE),
        Good("svc", "Services", GoodType.CONSUMPTION),
    ]

    sectors = [
        Sector("sector_agri", goods[0], base_productivity=1.2),
        Sector("sector_mfg", goods[1], base_productivity=1.0),
        Sector("sector_svc", goods[2], base_productivity=0.8),
    ]

    # Input-output coefficients
    # Rows = input goods, Columns = output sector
    # a[i][j] = units of good i needed per unit of good j
    coefficients = np.array([
        [0.10, 0.05, 0.02],  # agri inputs to agri, mfg, svc
        [0.05, 0.15, 0.10],  # mfg inputs to agri, mfg, svc
        [0.02, 0.08, 0.05],  # svc inputs to agri, mfg, svc
    ])

    labor_coefficients = np.array([0.3, 0.5, 0.4])

    io_matrix = InputOutputMatrix(goods, coefficients, labor_coefficients)

    return goods, sectors, io_matrix
