"""
Labor skill differentiation and wage dispersion.

Extends the homogeneous labor market to support multiple skill levels,
creating wage dispersion and unemployment stratification.

Key concepts:
- SkillLevel: Discrete skill tiers with productivity multipliers
- SkilledHousehold: Household with skill level, training, and skill-based wages
- SkilledFirm: Firm with skill-specific vacancy posting and hiring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

from econosim.core.accounting import round_money


class SkillLevel(IntEnum):
    """Discrete skill levels for labor differentiation."""
    UNSKILLED = 0
    SEMI_SKILLED = 1
    SKILLED = 2
    HIGHLY_SKILLED = 3

    @property
    def productivity_multiplier(self) -> float:
        """Productivity multiplier relative to unskilled baseline."""
        return {
            SkillLevel.UNSKILLED: 1.0,
            SkillLevel.SEMI_SKILLED: 1.5,
            SkillLevel.SKILLED: 2.5,
            SkillLevel.HIGHLY_SKILLED: 4.0,
        }[self]

    @property
    def wage_premium(self) -> float:
        """Wage premium multiplier relative to unskilled baseline."""
        return {
            SkillLevel.UNSKILLED: 1.0,
            SkillLevel.SEMI_SKILLED: 1.3,
            SkillLevel.SKILLED: 2.0,
            SkillLevel.HIGHLY_SKILLED: 3.5,
        }[self]

    @property
    def training_cost(self) -> float:
        """Cost to upgrade to this skill level."""
        return {
            SkillLevel.UNSKILLED: 0.0,
            SkillLevel.SEMI_SKILLED: 500.0,
            SkillLevel.SKILLED: 2000.0,
            SkillLevel.HIGHLY_SKILLED: 5000.0,
        }[self]


@dataclass
class SkillDistribution:
    """Distribution of skill levels in the labor force."""

    counts: dict[SkillLevel, int] = field(default_factory=lambda: {
        SkillLevel.UNSKILLED: 0,
        SkillLevel.SEMI_SKILLED: 0,
        SkillLevel.SKILLED: 0,
        SkillLevel.HIGHLY_SKILLED: 0,
    })

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def fraction(self, level: SkillLevel) -> float:
        total = self.total
        return self.counts.get(level, 0) / max(total, 1)

    def add(self, level: SkillLevel, count: int = 1) -> None:
        self.counts[level] = self.counts.get(level, 0) + count

    def remove(self, level: SkillLevel, count: int = 1) -> None:
        self.counts[level] = max(0, self.counts.get(level, 0) - count)

    def average_skill(self) -> float:
        total = self.total
        if total == 0:
            return 0.0
        return sum(int(level) * count for level, count in self.counts.items()) / total


@dataclass
class SkilledHousehold:
    """Extended household with skill level and training mechanics."""

    household_id: str
    skill_level: SkillLevel = SkillLevel.UNSKILLED
    experience: float = 0.0   # accumulated work experience
    training_progress: float = 0.0  # progress toward next skill level

    # Skill evolution parameters
    experience_per_period: float = 1.0  # experience gained per period employed
    training_threshold: float = 10.0     # experience needed for skill upgrade
    skill_decay_rate: float = 0.01       # experience loss per period unemployed

    def effective_productivity(self, base_productivity: float) -> float:
        """Productivity adjusted for skill level and experience."""
        skill_mult = self.skill_level.productivity_multiplier
        exp_bonus = 1.0 + 0.01 * min(self.experience, 50.0)
        return round_money(base_productivity * skill_mult * exp_bonus)

    def reservation_wage(self, base_wage: float) -> float:
        """Minimum acceptable wage, adjusted for skill level."""
        return round_money(base_wage * self.skill_level.wage_premium)

    def update_experience(self, employed: bool) -> None:
        """Update experience based on employment status."""
        if employed:
            self.experience += self.experience_per_period
            self.training_progress += self.experience_per_period
        else:
            # Skill decay when unemployed
            self.experience = max(0.0, self.experience - self.skill_decay_rate)
            self.training_progress = max(0.0, self.training_progress - self.skill_decay_rate * 0.5)

    def try_upgrade_skill(self) -> bool:
        """Check if household qualifies for skill upgrade."""
        if self.skill_level >= SkillLevel.HIGHLY_SKILLED:
            return False
        if self.training_progress >= self.training_threshold:
            self.skill_level = SkillLevel(int(self.skill_level) + 1)
            self.training_progress = 0.0
            return True
        return False

    def get_observation(self) -> dict[str, Any]:
        return {
            "household_id": self.household_id,
            "skill_level": int(self.skill_level),
            "skill_name": self.skill_level.name,
            "experience": self.experience,
            "training_progress": self.training_progress,
            "productivity_multiplier": self.skill_level.productivity_multiplier,
            "wage_premium": self.skill_level.wage_premium,
        }


@dataclass
class SkillRequirement:
    """Skill requirement for a job posting."""
    minimum_level: SkillLevel = SkillLevel.UNSKILLED
    preferred_level: SkillLevel = SkillLevel.UNSKILLED
    wage_by_skill: dict[SkillLevel, float] = field(default_factory=dict)

    def wage_for(self, skill: SkillLevel, base_wage: float) -> float:
        """Calculate wage offered to a worker of given skill level."""
        if self.wage_by_skill:
            return self.wage_by_skill.get(skill, base_wage * skill.wage_premium)
        return round_money(base_wage * skill.wage_premium)

    def meets_requirement(self, skill: SkillLevel) -> bool:
        return skill >= self.minimum_level


@dataclass
class SkilledFirm:
    """Extended firm with skill-differentiated hiring."""

    firm_id: str
    skill_requirements: dict[str, SkillRequirement] = field(default_factory=dict)
    workers_by_skill: dict[SkillLevel, list[str]] = field(default_factory=lambda: {
        level: [] for level in SkillLevel
    })

    # Wage bill by skill level
    wage_bill_by_skill: dict[SkillLevel, float] = field(default_factory=lambda: {
        level: 0.0 for level in SkillLevel
    })

    def total_effective_productivity(self, base_productivity: float) -> float:
        """Total productivity from all workers, accounting for skill levels."""
        total = 0.0
        for level, worker_ids in self.workers_by_skill.items():
            n_workers = len(worker_ids)
            total += n_workers * base_productivity * level.productivity_multiplier
        return round_money(total)

    def hire(self, worker_id: str, skill: SkillLevel, wage: float) -> None:
        """Hire a worker at the given skill level and wage."""
        self.workers_by_skill[skill].append(worker_id)
        self.wage_bill_by_skill[skill] = round_money(
            self.wage_bill_by_skill.get(skill, 0.0) + wage
        )

    def fire(self, worker_id: str, skill: SkillLevel) -> None:
        """Remove a worker."""
        if worker_id in self.workers_by_skill.get(skill, []):
            self.workers_by_skill[skill].remove(worker_id)

    def total_workers(self) -> int:
        return sum(len(workers) for workers in self.workers_by_skill.values())

    def total_wage_bill(self) -> float:
        return round_money(sum(self.wage_bill_by_skill.values()))

    def reset_period_state(self) -> None:
        for level in SkillLevel:
            self.wage_bill_by_skill[level] = 0.0

    def skill_composition(self) -> dict[str, int]:
        """Return worker count by skill level."""
        return {level.name: len(workers) for level, workers in self.workers_by_skill.items()}

    def get_observation(self) -> dict[str, Any]:
        return {
            "firm_id": self.firm_id,
            "total_workers": self.total_workers(),
            "total_wage_bill": self.total_wage_bill(),
            "skill_composition": self.skill_composition(),
        }


class SkilledLaborMarket:
    """Labor market with skill-based matching.

    Workers are matched to jobs based on skill requirements.
    Higher-skilled workers get priority and higher wages.
    """

    def __init__(self) -> None:
        self.matches_by_skill: dict[SkillLevel, int] = {level: 0 for level in SkillLevel}
        self.vacancies_by_skill: dict[SkillLevel, int] = {level: 0 for level in SkillLevel}
        self.seekers_by_skill: dict[SkillLevel, int] = {level: 0 for level in SkillLevel}
        self.avg_wage_by_skill: dict[SkillLevel, float] = {level: 0.0 for level in SkillLevel}

    def clear(
        self,
        households: list[tuple[str, SkilledHousehold, bool]],  # (id, skilled_hh, wants_work)
        firms: list[tuple[str, SkilledFirm, dict[SkillLevel, int], float]],  # (id, skilled_firm, vacancies, base_wage)
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Match workers to jobs based on skill requirements.

        Returns matching statistics.
        """
        # Reset stats
        for level in SkillLevel:
            self.matches_by_skill[level] = 0
            self.vacancies_by_skill[level] = 0
            self.seekers_by_skill[level] = 0

        # Collect job seekers by skill
        seekers: dict[SkillLevel, list[tuple[str, SkilledHousehold]]] = {
            level: [] for level in SkillLevel
        }
        for hh_id, skilled_hh, wants_work in households:
            if wants_work:
                seekers[skilled_hh.skill_level].append((hh_id, skilled_hh))
                self.seekers_by_skill[skilled_hh.skill_level] += 1

        # Shuffle seekers for fairness
        for level in SkillLevel:
            rng.shuffle(seekers[level])

        # Collect vacancies
        vacancies: list[tuple[str, SkilledFirm, SkillLevel, float]] = []
        for firm_id, skilled_firm, skill_vacancies, base_wage in firms:
            for level, count in skill_vacancies.items():
                for _ in range(count):
                    wage = round_money(base_wage * level.wage_premium)
                    vacancies.append((firm_id, skilled_firm, level, wage))
                    self.vacancies_by_skill[level] += 1

        # Shuffle vacancies
        rng.shuffle(vacancies)

        # Match: iterate vacancies, find best available worker
        total_wages = {level: 0.0 for level in SkillLevel}
        total_matches = {level: 0 for level in SkillLevel}

        for firm_id, skilled_firm, required_level, wage in vacancies:
            # Try to match with required skill level, then look at higher skills
            matched = False
            for try_level in sorted(SkillLevel, key=lambda l: abs(int(l) - int(required_level))):
                if try_level < required_level:
                    continue
                if seekers[try_level]:
                    hh_id, skilled_hh = seekers[try_level].pop(0)
                    actual_wage = round_money(wage * (try_level.wage_premium / required_level.wage_premium))
                    skilled_firm.hire(hh_id, try_level, actual_wage)
                    total_wages[try_level] += actual_wage
                    total_matches[try_level] += 1
                    self.matches_by_skill[try_level] += 1
                    matched = True
                    break

        # Compute average wages
        for level in SkillLevel:
            if total_matches[level] > 0:
                self.avg_wage_by_skill[level] = round_money(
                    total_wages[level] / total_matches[level]
                )

        return {
            "total_matches": sum(total_matches.values()),
            "matches_by_skill": dict(total_matches),
            "avg_wage_by_skill": dict(self.avg_wage_by_skill),
            "unfilled_vacancies": sum(
                self.vacancies_by_skill[l] - self.matches_by_skill[l]
                for l in SkillLevel
            ),
        }

    def wage_dispersion(self) -> float:
        """Compute coefficient of variation of wages across skill levels."""
        wages = [w for w in self.avg_wage_by_skill.values() if w > 0]
        if len(wages) < 2:
            return 0.0
        mean_wage = float(np.mean(wages))
        std_wage = float(np.std(wages))
        return round(std_wage / max(mean_wage, 0.01), 4)

    def get_observation(self) -> dict[str, Any]:
        return {
            "matches_by_skill": {k.name: v for k, v in self.matches_by_skill.items()},
            "vacancies_by_skill": {k.name: v for k, v in self.vacancies_by_skill.items()},
            "seekers_by_skill": {k.name: v for k, v in self.seekers_by_skill.items()},
            "avg_wage_by_skill": {k.name: v for k, v in self.avg_wage_by_skill.items()},
            "wage_dispersion": self.wage_dispersion(),
        }
