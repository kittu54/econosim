"""
Labor market: matches job-seeking households with firms posting vacancies.

MVP matching mechanism:
- Firms post vacancies with wages
- Unemployed households seek jobs
- Random matching: shuffle seekers, iterate over firms with openings
- Household accepts if wage >= reservation_wage
- Wage is paid via ledger transfer from firm to household
"""

from __future__ import annotations

import numpy as np

from econosim.core.accounting import Ledger, round_money
from econosim.agents.household import Household
from econosim.agents.firm import Firm


class LaborMarket:
    """Clears the labor market each period via random matching."""

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        # Period stats
        self.total_matches: int = 0
        self.total_vacancies: int = 0
        self.total_seekers: int = 0

    def clear(
        self,
        households: list[Household],
        firms: list[Firm],
        period: int,
        rng: np.random.Generator,
        skip_vacancy_decision: bool = False,
    ) -> None:
        """Run labor market matching and wage payment for one period.

        Steps:
        1. Collect vacancies from firms
        2. Collect job seekers from households
        3. Shuffle seekers randomly
        4. Match seekers to firms with open vacancies
        5. Pay wages via ledger

        If skip_vacancy_decision is True, assumes firm.vacancies were
        already set externally (e.g. by a policy) and skips decide_vacancies().
        """
        # Reset worker lists
        for firm in firms:
            firm.workers = []

        for hh in households:
            hh.employed = False
            hh.employer_id = None

        # Collect vacancies
        firm_vacancies: list[tuple[Firm, int]] = []
        for firm in firms:
            if skip_vacancy_decision:
                v = firm.vacancies
            else:
                v = firm.decide_vacancies()
            if v > 0:
                firm_vacancies.append((firm, v))

        self.total_vacancies = sum(v for _, v in firm_vacancies)

        # Collect seekers
        seekers = [hh for hh in households if hh.wants_to_work()]
        rng.shuffle(seekers)
        self.total_seekers = len(seekers)

        # Shuffle firm order too for fairness
        firm_order = list(range(len(firm_vacancies)))
        rng.shuffle(firm_order)

        # Match
        self.total_matches = 0
        remaining_vacancies = {i: v for i, (_, v) in enumerate(firm_vacancies)}

        for hh in seekers:
            for fi in firm_order:
                if remaining_vacancies.get(fi, 0) <= 0:
                    continue
                firm, _ = firm_vacancies[fi]
                if hh.accept_wage(firm.posted_wage):
                    # Check firm can afford wage
                    if firm.deposits >= firm.posted_wage:
                        # Match!
                        hh.employed = True
                        hh.employer_id = firm.agent_id
                        firm.workers.append(hh.agent_id)
                        firm.vacancies_filled += 1
                        remaining_vacancies[fi] -= 1

                        # Pay wage
                        wage = firm.posted_wage
                        self.ledger.transfer_deposits(
                            period=period,
                            from_id=firm.agent_id,
                            to_id=hh.agent_id,
                            amount=wage,
                            description=f"wage {firm.agent_id}->{hh.agent_id}",
                        )
                        hh.wage_income = round_money(hh.wage_income + wage)
                        firm.wage_bill = round_money(firm.wage_bill + wage)

                        self.total_matches += 1
                        break  # This household is matched

    @property
    def unemployment_rate(self) -> float:
        if self.total_seekers + self.total_matches == 0:
            return 0.0
        total_labor_force = self.total_seekers + self.total_matches
        unemployed = self.total_seekers - self.total_matches
        # total_seekers was the count BEFORE matching, total_matches is how many got jobs
        # Actually: seekers = those wanting work. matches = those who got jobs.
        # unemployed = seekers - matches (those who wanted work but didn't get it)
        # labor force = all participants = total seekers (since they all wanted to work)
        if self.total_seekers == 0:
            return 0.0
        return max(0.0, (self.total_seekers - self.total_matches) / self.total_seekers)
