"""Tests for labor skill differentiation and wage dispersion."""

from __future__ import annotations

import numpy as np
import pytest

from econosim.extensions.skilled_labor import (
    SkillLevel,
    SkillDistribution,
    SkilledHousehold,
    SkillRequirement,
    SkilledFirm,
    SkilledLaborMarket,
)


class TestSkillLevel:
    def test_ordering(self):
        assert SkillLevel.UNSKILLED < SkillLevel.SEMI_SKILLED
        assert SkillLevel.SEMI_SKILLED < SkillLevel.SKILLED
        assert SkillLevel.SKILLED < SkillLevel.HIGHLY_SKILLED

    def test_productivity_multiplier(self):
        assert SkillLevel.UNSKILLED.productivity_multiplier == 1.0
        assert SkillLevel.SKILLED.productivity_multiplier > SkillLevel.UNSKILLED.productivity_multiplier
        assert SkillLevel.HIGHLY_SKILLED.productivity_multiplier > SkillLevel.SKILLED.productivity_multiplier

    def test_wage_premium(self):
        assert SkillLevel.UNSKILLED.wage_premium == 1.0
        assert SkillLevel.HIGHLY_SKILLED.wage_premium > 1.0

    def test_training_cost(self):
        assert SkillLevel.UNSKILLED.training_cost == 0.0
        assert SkillLevel.HIGHLY_SKILLED.training_cost > SkillLevel.SKILLED.training_cost


class TestSkillDistribution:
    def test_empty(self):
        dist = SkillDistribution()
        assert dist.total == 0

    def test_add_remove(self):
        dist = SkillDistribution()
        dist.add(SkillLevel.UNSKILLED, 10)
        dist.add(SkillLevel.SKILLED, 5)
        assert dist.total == 15
        dist.remove(SkillLevel.UNSKILLED, 3)
        assert dist.total == 12

    def test_fraction(self):
        dist = SkillDistribution()
        dist.add(SkillLevel.UNSKILLED, 8)
        dist.add(SkillLevel.SKILLED, 2)
        assert dist.fraction(SkillLevel.UNSKILLED) == 0.8
        assert dist.fraction(SkillLevel.SKILLED) == 0.2

    def test_average_skill(self):
        dist = SkillDistribution()
        dist.add(SkillLevel.UNSKILLED, 10)
        assert dist.average_skill() == 0.0
        dist.add(SkillLevel.SKILLED, 10)
        assert dist.average_skill() == 1.0  # (0*10 + 2*10) / 20 = 1.0


class TestSkilledHousehold:
    def test_creation(self):
        hh = SkilledHousehold("hh_0", SkillLevel.UNSKILLED)
        assert hh.skill_level == SkillLevel.UNSKILLED
        assert hh.experience == 0.0

    def test_effective_productivity(self):
        hh_low = SkilledHousehold("hh_0", SkillLevel.UNSKILLED)
        hh_high = SkilledHousehold("hh_1", SkillLevel.SKILLED)
        assert hh_high.effective_productivity(10.0) > hh_low.effective_productivity(10.0)

    def test_reservation_wage(self):
        hh = SkilledHousehold("hh_0", SkillLevel.SKILLED)
        assert hh.reservation_wage(50.0) == 100.0  # 50 * 2.0 premium

    def test_update_experience_employed(self):
        hh = SkilledHousehold("hh_0", SkillLevel.UNSKILLED)
        hh.update_experience(employed=True)
        assert hh.experience == 1.0
        assert hh.training_progress == 1.0

    def test_update_experience_unemployed(self):
        hh = SkilledHousehold("hh_0", SkillLevel.UNSKILLED, experience=5.0)
        hh.update_experience(employed=False)
        assert hh.experience < 5.0

    def test_skill_upgrade(self):
        hh = SkilledHousehold("hh_0", SkillLevel.UNSKILLED,
                              training_progress=10.0, training_threshold=10.0)
        assert hh.try_upgrade_skill()
        assert hh.skill_level == SkillLevel.SEMI_SKILLED
        assert hh.training_progress == 0.0

    def test_no_upgrade_above_max(self):
        hh = SkilledHousehold("hh_0", SkillLevel.HIGHLY_SKILLED,
                              training_progress=100.0)
        assert not hh.try_upgrade_skill()

    def test_get_observation(self):
        hh = SkilledHousehold("hh_0", SkillLevel.SKILLED)
        obs = hh.get_observation()
        assert obs["skill_level"] == 2
        assert obs["skill_name"] == "SKILLED"


class TestSkillRequirement:
    def test_meets_requirement(self):
        req = SkillRequirement(minimum_level=SkillLevel.SEMI_SKILLED)
        assert not req.meets_requirement(SkillLevel.UNSKILLED)
        assert req.meets_requirement(SkillLevel.SEMI_SKILLED)
        assert req.meets_requirement(SkillLevel.SKILLED)

    def test_wage_for_skill(self):
        req = SkillRequirement()
        wage = req.wage_for(SkillLevel.SKILLED, 50.0)
        assert wage == 100.0  # 50 * 2.0

    def test_custom_wages(self):
        req = SkillRequirement(wage_by_skill={SkillLevel.UNSKILLED: 40.0})
        assert req.wage_for(SkillLevel.UNSKILLED, 50.0) == 40.0


class TestSkilledFirm:
    def test_hire(self):
        firm = SkilledFirm("firm_0")
        firm.hire("w1", SkillLevel.UNSKILLED, 50.0)
        firm.hire("w2", SkillLevel.SKILLED, 100.0)
        assert firm.total_workers() == 2
        assert firm.total_wage_bill() == 150.0

    def test_fire(self):
        firm = SkilledFirm("firm_0")
        firm.hire("w1", SkillLevel.UNSKILLED, 50.0)
        firm.fire("w1", SkillLevel.UNSKILLED)
        assert firm.total_workers() == 0

    def test_effective_productivity(self):
        firm = SkilledFirm("firm_0")
        firm.hire("w1", SkillLevel.UNSKILLED, 50.0)  # mult=1.0
        firm.hire("w2", SkillLevel.SKILLED, 100.0)    # mult=2.5
        prod = firm.total_effective_productivity(10.0)
        assert prod == 35.0  # 1*10*1.0 + 1*10*2.5

    def test_skill_composition(self):
        firm = SkilledFirm("firm_0")
        firm.hire("w1", SkillLevel.UNSKILLED, 50.0)
        firm.hire("w2", SkillLevel.UNSKILLED, 50.0)
        firm.hire("w3", SkillLevel.SKILLED, 100.0)
        comp = firm.skill_composition()
        assert comp["UNSKILLED"] == 2
        assert comp["SKILLED"] == 1

    def test_reset_period(self):
        firm = SkilledFirm("firm_0")
        firm.hire("w1", SkillLevel.UNSKILLED, 50.0)
        firm.reset_period_state()
        assert firm.total_wage_bill() == 0


class TestSkilledLaborMarket:
    def test_basic_matching(self):
        market = SkilledLaborMarket()
        rng = np.random.default_rng(42)

        households = [
            ("hh_0", SkilledHousehold("hh_0", SkillLevel.UNSKILLED), True),
            ("hh_1", SkilledHousehold("hh_1", SkillLevel.SKILLED), True),
        ]
        firms = [
            ("firm_0", SkilledFirm("firm_0"),
             {SkillLevel.UNSKILLED: 1, SkillLevel.SKILLED: 1}, 50.0),
        ]

        result = market.clear(households, firms, rng)
        assert result["total_matches"] == 2

    def test_skill_requirement_filtering(self):
        market = SkilledLaborMarket()
        rng = np.random.default_rng(42)

        # Only unskilled workers, but firm wants skilled
        households = [
            ("hh_0", SkilledHousehold("hh_0", SkillLevel.UNSKILLED), True),
        ]
        firms = [
            ("firm_0", SkilledFirm("firm_0"),
             {SkillLevel.SKILLED: 1}, 50.0),
        ]

        result = market.clear(households, firms, rng)
        # Unskilled can't fill skilled vacancy
        assert result["total_matches"] == 0
        assert result["unfilled_vacancies"] == 1

    def test_wage_dispersion(self):
        market = SkilledLaborMarket()
        rng = np.random.default_rng(42)

        households = [
            ("hh_0", SkilledHousehold("hh_0", SkillLevel.UNSKILLED), True),
            ("hh_1", SkilledHousehold("hh_1", SkillLevel.SKILLED), True),
            ("hh_2", SkilledHousehold("hh_2", SkillLevel.HIGHLY_SKILLED), True),
        ]
        firms = [
            ("firm_0", SkilledFirm("firm_0"),
             {SkillLevel.UNSKILLED: 1, SkillLevel.SKILLED: 1,
              SkillLevel.HIGHLY_SKILLED: 1}, 50.0),
        ]

        market.clear(households, firms, rng)
        dispersion = market.wage_dispersion()
        assert dispersion > 0  # Should have wage variation

    def test_get_observation(self):
        market = SkilledLaborMarket()
        obs = market.get_observation()
        assert "matches_by_skill" in obs
        assert "wage_dispersion" in obs
