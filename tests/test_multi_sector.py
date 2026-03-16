"""Tests for multi-sector production and input-output matrices."""

from __future__ import annotations

import numpy as np
import pytest

from econosim.extensions.multi_sector import (
    Good,
    GoodType,
    SectorInventory,
    InputOutputMatrix,
    Sector,
    create_default_sectors,
)


class TestGood:
    def test_good_creation(self):
        g = Good("food", "Food", GoodType.CONSUMPTION)
        assert g.good_id == "food"
        assert g.good_type == GoodType.CONSUMPTION
        assert not g.perishable

    def test_good_equality(self):
        g1 = Good("food", "Food")
        g2 = Good("food", "Food Item")
        assert g1 == g2

    def test_good_hash(self):
        g1 = Good("food", "Food")
        g2 = Good("steel", "Steel")
        assert hash(g1) != hash(g2)
        assert len({g1, g2}) == 2

    def test_perishable_good(self):
        g = Good("food", "Food", perishable=True, depreciation_rate=0.1)
        assert g.perishable
        assert g.depreciation_rate == 0.1


class TestSectorInventory:
    def test_add_goods(self):
        inv = SectorInventory("firm_0")
        inv.add("food", 100, 500)
        assert inv.quantity("food") == 100
        assert inv.unit_costs["food"] == 5.0

    def test_add_multiple_batches(self):
        inv = SectorInventory("firm_0")
        inv.add("food", 100, 500)   # unit cost 5
        inv.add("food", 100, 1000)  # unit cost 10
        assert inv.quantity("food") == 200
        # Weighted average: (500 + 1000) / 200 = 7.5
        assert inv.unit_costs["food"] == 7.5

    def test_remove_goods(self):
        inv = SectorInventory("firm_0")
        inv.add("food", 100, 500)
        cost = inv.remove("food", 40)
        assert inv.quantity("food") == 60
        assert cost == 200  # 40 * 5.0

    def test_remove_more_than_available(self):
        inv = SectorInventory("firm_0")
        inv.add("food", 50, 250)
        cost = inv.remove("food", 100)
        assert inv.quantity("food") == 0
        assert cost == 250  # 50 * 5.0

    def test_total_value(self):
        inv = SectorInventory("firm_0")
        inv.add("food", 100, 500)
        inv.add("steel", 50, 1000)
        assert inv.total_value() == 1500

    def test_depreciation(self):
        goods = {"food": Good("food", "Food", perishable=True, depreciation_rate=0.1)}
        inv = SectorInventory("firm_0")
        inv.add("food", 100, 500)
        loss = inv.depreciate(goods)
        assert inv.quantity("food") == 90
        assert loss > 0


class TestInputOutputMatrix:
    def test_creation(self):
        goods = [Good("a", "A"), Good("b", "B")]
        io = InputOutputMatrix(goods)
        assert io.n_sectors == 2
        assert io.A.shape == (2, 2)

    def test_coefficients(self):
        goods = [Good("a", "A"), Good("b", "B")]
        A = np.array([[0.1, 0.2], [0.3, 0.1]])
        io = InputOutputMatrix(goods, A)
        assert io.get_coefficient("a", "b") == 0.2
        assert io.get_coefficient("b", "a") == 0.3

    def test_set_coefficient(self):
        goods = [Good("a", "A"), Good("b", "B")]
        io = InputOutputMatrix(goods)
        io.set_coefficient("a", "b", 0.5)
        assert io.get_coefficient("a", "b") == 0.5

    def test_inputs_required(self):
        goods = [Good("agri", "Agriculture"), Good("mfg", "Manufacturing")]
        A = np.array([[0.1, 0.2], [0.05, 0.15]])
        io = InputOutputMatrix(goods, A)

        inputs = io.inputs_required("mfg", 100)
        assert "agri" in inputs
        assert inputs["agri"] == 20.0  # 0.2 * 100
        assert inputs["mfg"] == 15.0   # 0.15 * 100

    def test_labor_required(self):
        goods = [Good("a", "A"), Good("b", "B")]
        io = InputOutputMatrix(goods, labor_coefficients=np.array([0.3, 0.5]))
        assert io.labor_required("a", 100) == 30.0
        assert io.labor_required("b", 100) == 50.0

    def test_leontief_inverse(self):
        goods = [Good("a", "A"), Good("b", "B")]
        A = np.array([[0.1, 0.2], [0.1, 0.1]])
        io = InputOutputMatrix(goods, A)
        L_inv = io.leontief_inverse()
        # (I - A)^(-1) should exist for productive systems
        assert L_inv.shape == (2, 2)
        # Diagonal elements should be > 1 (positive multiplier effect)
        assert L_inv[0, 0] > 1.0
        assert L_inv[1, 1] > 1.0

    def test_total_requirements(self):
        goods = [Good("a", "A"), Good("b", "B")]
        A = np.array([[0.1, 0.2], [0.1, 0.1]])
        io = InputOutputMatrix(goods, A)
        final_demand = np.array([100.0, 50.0])
        total = io.total_requirements(final_demand)
        # Total output should exceed final demand (multiplier)
        assert total[0] > 100.0
        assert total[1] > 50.0

    def test_is_productive(self):
        goods = [Good("a", "A"), Good("b", "B")]
        # Small coefficients => productive
        A = np.array([[0.1, 0.2], [0.1, 0.1]])
        io = InputOutputMatrix(goods, A)
        assert io.is_productive()

    def test_not_productive(self):
        goods = [Good("a", "A"), Good("b", "B")]
        # Large coefficients => not productive
        A = np.array([[0.9, 0.8], [0.8, 0.9]])
        io = InputOutputMatrix(goods, A)
        assert not io.is_productive()

    def test_to_dict(self):
        goods = [Good("a", "A"), Good("b", "B")]
        io = InputOutputMatrix(goods)
        d = io.to_dict()
        assert "goods" in d
        assert "coefficients" in d
        assert len(d["goods"]) == 2


class TestSector:
    def test_creation(self):
        good = Good("food", "Food")
        sector = Sector("s_food", good, firm_ids=["f1", "f2"])
        assert sector.sector_id == "s_food"
        assert len(sector.firm_ids) == 2

    def test_reset_stats(self):
        good = Good("food", "Food")
        sector = Sector("s_food", good)
        sector.total_output = 1000
        sector.reset_period_stats()
        assert sector.total_output == 0

    def test_compute_sector_price(self):
        good = Good("food", "Food")
        sector = Sector("s_food", good)
        price = sector.compute_sector_price([10.0, 12.0, 8.0])
        assert price == 10.0

    def test_get_observation(self):
        good = Good("food", "Food")
        sector = Sector("s_food", good, firm_ids=["f1"])
        obs = sector.get_observation()
        assert obs["sector_id"] == "s_food"
        assert obs["num_firms"] == 1


class TestCreateDefaultSectors:
    def test_creates_3_sectors(self):
        goods, sectors, io_matrix = create_default_sectors()
        assert len(goods) == 3
        assert len(sectors) == 3
        assert io_matrix.n_sectors == 3

    def test_is_productive(self):
        _, _, io_matrix = create_default_sectors()
        assert io_matrix.is_productive()

    def test_goods_types(self):
        goods, _, _ = create_default_sectors()
        types = {g.good_type for g in goods}
        assert GoodType.CONSUMPTION in types
        assert GoodType.INTERMEDIATE in types
