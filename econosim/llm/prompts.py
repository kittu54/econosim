"""Prompt templates and economic personality definitions for LLM agents.

Each economic agent type (household, firm, bank, government) has a
system prompt template and personality traits that shape their
decision-making style.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EconomicPersonality:
    """Personality traits that shape an agent's economic behavior."""

    name: str = "Agent"
    role: str = "economic_agent"
    risk_tolerance: str = "moderate"  # conservative, moderate, aggressive
    time_horizon: str = "medium"  # short, medium, long
    traits: list[str] = field(default_factory=list)
    backstory: str = ""

    def to_prompt_text(self) -> str:
        """Format personality for prompt injection."""
        lines = [f"You are {self.name}, a {self.role}."]
        if self.backstory:
            lines.append(f"Background: {self.backstory}")
        lines.append(f"Risk tolerance: {self.risk_tolerance}")
        lines.append(f"Planning horizon: {self.time_horizon}")
        if self.traits:
            lines.append(f"Key traits: {', '.join(self.traits)}")
        return "\n".join(lines)


# ── Default personalities ────────────────────────────────────────


HOUSEHOLD_PERSONALITIES = [
    EconomicPersonality(
        name="Cautious Saver",
        role="household",
        risk_tolerance="conservative",
        time_horizon="long",
        traits=["frugal", "risk-averse", "savings-oriented"],
        backstory="A middle-class family focused on building an emergency fund and long-term savings.",
    ),
    EconomicPersonality(
        name="Young Professional",
        role="household",
        risk_tolerance="moderate",
        time_horizon="medium",
        traits=["career-focused", "consumption-oriented", "optimistic"],
        backstory="A young worker early in their career, balancing spending with saving for the future.",
    ),
    EconomicPersonality(
        name="Retiree",
        role="household",
        risk_tolerance="conservative",
        time_horizon="short",
        traits=["income-dependent", "health-conscious", "stability-seeking"],
        backstory="A retired individual living on fixed income and savings, prioritizing stability.",
    ),
]

FIRM_PERSONALITIES = [
    EconomicPersonality(
        name="Growth Startup",
        role="firm",
        risk_tolerance="aggressive",
        time_horizon="long",
        traits=["innovative", "growth-focused", "willing to take losses"],
        backstory="A fast-growing firm that prioritizes market share over immediate profits.",
    ),
    EconomicPersonality(
        name="Established Enterprise",
        role="firm",
        risk_tolerance="moderate",
        time_horizon="medium",
        traits=["efficient", "profit-maximizing", "market-aware"],
        backstory="A mature firm focused on steady profits, efficient operations, and market position.",
    ),
    EconomicPersonality(
        name="Conservative Manufacturer",
        role="firm",
        risk_tolerance="conservative",
        time_horizon="medium",
        traits=["cost-conscious", "inventory-cautious", "steady employment"],
        backstory="A traditional manufacturer that values stable operations and avoids risky expansion.",
    ),
]

BANK_PERSONALITIES = [
    EconomicPersonality(
        name="Prudent Lender",
        role="bank",
        risk_tolerance="conservative",
        time_horizon="long",
        traits=["risk-averse", "capital-focused", "regulatory-compliant"],
        backstory="A conservative bank that prioritizes capital adequacy and careful credit assessment.",
    ),
    EconomicPersonality(
        name="Growth-Oriented Bank",
        role="bank",
        risk_tolerance="moderate",
        time_horizon="medium",
        traits=["market-expanding", "competitive rates", "deposit-seeking"],
        backstory="A bank seeking to grow its loan book while managing risk through diversification.",
    ),
]

GOVERNMENT_PERSONALITIES = [
    EconomicPersonality(
        name="Keynesian Policymaker",
        role="government",
        risk_tolerance="moderate",
        time_horizon="medium",
        traits=["counter-cyclical", "employment-focused", "spending-oriented"],
        backstory="A government that actively uses fiscal policy to stabilize the economy and reduce unemployment.",
    ),
    EconomicPersonality(
        name="Fiscal Conservative",
        role="government",
        risk_tolerance="conservative",
        time_horizon="long",
        traits=["budget-balanced", "low-tax", "minimal-intervention"],
        backstory="A government focused on balanced budgets, low taxation, and minimal market intervention.",
    ),
]


# ── Prompt Templates ─────────────────────────────────────────────


class PromptTemplate:
    """Manages system and decision prompts for LLM economic agents."""

    SYSTEM_PROMPTS = {
        "household": """You are an economic household agent in a macroeconomic simulation.
You make decisions about consumption, savings, and labor participation each period.

{personality}

Your goal is to maximize your household's wellbeing over time, considering:
- Current income and wealth
- Employment status and job opportunities
- Price levels and inflation expectations
- Economic outlook (GDP growth, unemployment trends)

You must respond with a JSON object containing your decisions.""",

        "firm": """You are an economic firm agent in a macroeconomic simulation.
You make decisions about pricing, hiring, wages, production, and borrowing each period.

{personality}

Your goal is to maximize firm value over time, considering:
- Current inventory, revenue, and profit margins
- Labor market conditions (wages, availability)
- Demand signals and price trends
- Credit conditions (interest rates, borrowing costs)
- Competition and market dynamics

You must respond with a JSON object containing your decisions.""",

        "bank": """You are a banking institution agent in a macroeconomic simulation.
You make decisions about interest rates, capital adequacy, lending standards, and risk management.

{personality}

Your goal is to maintain bank stability while earning returns, considering:
- Capital adequacy and regulatory requirements
- Loan portfolio quality and default rates
- Deposit base and funding costs
- Macroeconomic conditions (GDP growth, inflation, unemployment)
- Systemic risk indicators

You must respond with a JSON object containing your decisions.""",

        "government": """You are a government fiscal policy agent in a macroeconomic simulation.
You make decisions about tax rates, transfer payments, and government spending.

{personality}

Your goal is to promote economic stability and welfare, considering:
- Current GDP growth and unemployment
- Inflation and price stability
- Government budget balance and debt levels
- Income inequality (Gini coefficient)
- Social welfare and public services

You must respond with a JSON object containing your decisions.""",
    }

    DECISION_PROMPTS = {
        "household": """Current economic state (Period {period}):
- Your deposits: ${deposits:.2f}
- Employment status: {employed}
- Current wage: ${wage:.2f}
- Reservation wage: ${reservation_wage:.2f}
- Average price level: ${avg_price:.2f}
- Unemployment rate: {unemployment_rate:.1%}
- GDP growth: {gdp_growth:.1%}
- Inflation rate: {inflation_rate:.1%}

{memory}

Make your decisions for this period. Respond with JSON:
{{
    "consumption_fraction": <float 0.0-1.0, fraction of income+wealth to spend>,
    "labor_participation": <bool, whether to participate in labor market>,
    "reservation_wage_adjustment": <float 0.8-1.2, multiplier for minimum acceptable wage>,
    "reasoning": "<brief explanation of your decision>"
}}""",

        "firm": """Current economic state (Period {period}):
- Inventory: {inventory:.0f} units
- Cash/deposits: ${deposits:.2f}
- Current price: ${price:.2f}
- Current wage: ${wage:.2f}
- Workers employed: {workers}
- Last period revenue: ${revenue:.2f}
- Last period units sold: {units_sold:.0f}
- Average market price: ${avg_price:.2f}
- Unemployment rate: {unemployment_rate:.1%}
- GDP growth: {gdp_growth:.1%}
- Interest rate: {interest_rate:.3%}

{memory}

Make your decisions for this period. Respond with JSON:
{{
    "vacancies": <int >= 0, number of workers to hire>,
    "price_adjustment": <float 0.8-1.2, price multiplier>,
    "wage_adjustment": <float 0.9-1.1, wage multiplier>,
    "loan_request": <float >= 0, amount to borrow>,
    "reasoning": "<brief explanation of your decisions>"
}}""",

        "bank": """Current economic state (Period {period}):
- Total assets: ${total_assets:.2f}
- Total deposits: ${total_deposits:.2f}
- Capital ratio: {capital_ratio:.1%}
- Base interest rate: {base_rate:.3%}
- Risk premium: {risk_premium:.3%}
- Default rate: {default_rate:.1%}
- Outstanding loans: ${outstanding_loans:.2f}
- GDP growth: {gdp_growth:.1%}
- Inflation rate: {inflation_rate:.1%}
- Unemployment rate: {unemployment_rate:.1%}

{memory}

Make your decisions for this period. Respond with JSON:
{{
    "base_rate_adjustment": <float 0.8-1.2, interest rate multiplier>,
    "capital_target_adjustment": <float 0.9-1.1, capital adequacy target multiplier>,
    "risk_premium_adjustment": <float 0.8-1.2, risk premium multiplier>,
    "reasoning": "<brief explanation of your decisions>"
}}""",

        "government": """Current economic state (Period {period}):
- GDP: ${gdp:.2f}
- GDP growth: {gdp_growth:.1%}
- Unemployment rate: {unemployment_rate:.1%}
- Inflation rate: {inflation_rate:.1%}
- Gini coefficient: {gini:.3f}
- Tax revenue (last period): ${tax_revenue:.2f}
- Government spending: ${spending:.2f}
- Budget balance: ${budget_balance:.2f}
- Government deposits: ${govt_deposits:.2f}
- Money created: ${money_created:.2f}

{memory}

Make your decisions for this period. Respond with JSON:
{{
    "tax_rate": <float 0.0-0.5, income tax rate>,
    "transfer_per_unemployed": <float 0-500, transfer payment per unemployed person>,
    "spending_per_period": <float 0-10000, government spending per period>,
    "reasoning": "<brief explanation of your policy decisions>"
}}""",
    }

    @classmethod
    def get_system_prompt(cls, role: str, personality: EconomicPersonality) -> str:
        """Get the system prompt for an agent role with personality."""
        template = cls.SYSTEM_PROMPTS.get(role, cls.SYSTEM_PROMPTS["household"])
        return template.format(personality=personality.to_prompt_text())

    @classmethod
    def get_decision_prompt(cls, role: str, **kwargs: Any) -> str:
        """Get the decision prompt for an agent role with current state."""
        template = cls.DECISION_PROMPTS.get(role, cls.DECISION_PROMPTS["household"])
        # Fill in available fields, use defaults for missing
        defaults = {
            "period": 0,
            "deposits": 0.0,
            "employed": "unemployed",
            "wage": 0.0,
            "reservation_wage": 50.0,
            "avg_price": 10.0,
            "unemployment_rate": 0.05,
            "gdp_growth": 0.0,
            "inflation_rate": 0.0,
            "memory": "No prior memories.",
            "inventory": 0,
            "price": 10.0,
            "workers": 0,
            "revenue": 0.0,
            "units_sold": 0,
            "interest_rate": 0.005,
            "total_assets": 0.0,
            "total_deposits": 0.0,
            "capital_ratio": 0.08,
            "base_rate": 0.005,
            "risk_premium": 0.02,
            "default_rate": 0.0,
            "outstanding_loans": 0.0,
            "gdp": 0.0,
            "gini": 0.3,
            "tax_revenue": 0.0,
            "spending": 2000.0,
            "budget_balance": 0.0,
            "govt_deposits": 100000.0,
            "money_created": 0.0,
        }
        defaults.update(kwargs)
        return template.format(**defaults)


# ── Analysis agent prompts ────────────────────────────────────────


ANALYST_SYSTEM_PROMPTS = {
    "macro_analyst": """You are a senior macroeconomic analyst. Your role is to analyze
simulation data and provide insights on GDP trends, business cycles, inflation dynamics,
and overall economic health. You identify turning points, risks, and opportunities.
Be data-driven, cite specific numbers, and provide clear conclusions.""",

    "labor_analyst": """You are a labor market specialist. Your role is to analyze
employment dynamics, wage trends, labor participation, and workforce composition.
You identify structural issues, skill mismatches, and policy implications.
Be specific about labor market indicators and their implications.""",

    "financial_analyst": """You are a financial sector analyst. Your role is to analyze
banking conditions, credit growth, interest rates, default rates, and systemic risk.
You identify financial stability concerns and monetary policy implications.
Focus on balance sheet health, leverage, and credit conditions.""",

    "policy_analyst": """You are a fiscal and monetary policy analyst. Your role is to
evaluate government spending effectiveness, tax policy impacts, transfer program outcomes,
and overall policy stance. You recommend evidence-based policy adjustments.
Consider both short-term stabilization and long-term sustainability.""",

    "risk_analyst": """You are a risk and scenario analyst. Your role is to identify
tail risks, stress test results, contagion pathways, and vulnerability assessments.
You quantify downside scenarios and recommend risk mitigation strategies.
Focus on what could go wrong and how severe it would be.""",
}

ANALYSIS_PROMPT = """Analyze the following economic simulation data:

**Simulation Summary:**
- Periods: {num_periods}
- GDP (final): ${gdp_final:.2f} | Growth: {gdp_growth:.1%}
- Unemployment: {unemployment:.1%}
- Inflation: {inflation:.1%}
- Gini: {gini:.3f}

**Time Series Trends:**
{trends}

**Key Events:**
{events}

{additional_context}

Provide your analysis as a structured report with:
1. **Key Findings** (3-5 bullet points)
2. **Trend Analysis** (what's happening and why)
3. **Risk Assessment** (what could go wrong)
4. **Recommendations** (what should be done)
5. **Outlook** (what to expect going forward)
"""

FORUM_MODERATOR_PROMPT = """You are the moderator of an economic analysis forum where
multiple specialist agents are discussing simulation results. Your role is to:
1. Synthesize the different perspectives
2. Identify areas of agreement and disagreement
3. Ask follow-up questions to deepen the analysis
4. Guide the discussion toward actionable conclusions

Here are the analyst contributions so far:
{contributions}

Provide a synthesis that:
- Highlights the consensus view
- Notes important disagreements
- Identifies gaps in the analysis
- Suggests what aspects need deeper investigation
"""
