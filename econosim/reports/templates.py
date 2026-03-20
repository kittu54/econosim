"""Report template definitions for different analysis types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReportSection:
    """A section within a report template."""

    id: str
    title: str
    description: str
    required_data: list[str] = field(default_factory=list)
    llm_prompt: str = ""


@dataclass
class ReportTemplate:
    """Template defining the structure of an economic report."""

    name: str
    description: str
    sections: list[ReportSection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Built-in templates ───────────────────────────────────────────


MACRO_FORECAST_TEMPLATE = ReportTemplate(
    name="Macro Forecast Report",
    description="Comprehensive macroeconomic forecast with scenario analysis",
    sections=[
        ReportSection(
            id="executive_summary",
            title="Executive Summary",
            description="High-level overview of economic outlook and key findings",
            llm_prompt="Write an executive summary of the economic simulation results. "
            "Include the key GDP growth rate, unemployment trajectory, inflation outlook, "
            "and overall economic health assessment. Be concise (3-4 paragraphs).",
        ),
        ReportSection(
            id="gdp_analysis",
            title="GDP and Growth Analysis",
            description="Detailed GDP decomposition and growth trajectory",
            required_data=["gdp", "gdp_growth"],
            llm_prompt="Analyze GDP dynamics including growth rates, trend components, "
            "and cyclical patterns. Identify key drivers and risks to growth.",
        ),
        ReportSection(
            id="labor_market",
            title="Labor Market Assessment",
            description="Employment, wages, and labor force dynamics",
            required_data=["unemployment_rate", "avg_wage"],
            llm_prompt="Assess the labor market including unemployment trends, wage growth, "
            "labor participation, and structural vs cyclical factors.",
        ),
        ReportSection(
            id="inflation_prices",
            title="Inflation and Price Dynamics",
            description="Price level trends and inflation outlook",
            required_data=["avg_price", "inflation_rate"],
            llm_prompt="Analyze inflation dynamics including price level trends, "
            "cost-push vs demand-pull factors, and inflation expectations.",
        ),
        ReportSection(
            id="financial_conditions",
            title="Financial Conditions",
            description="Credit markets, banking sector, and financial stability",
            required_data=["total_credit"],
            llm_prompt="Evaluate financial conditions including credit growth, "
            "banking sector health, interest rate environment, and systemic risk indicators.",
        ),
        ReportSection(
            id="inequality",
            title="Income Distribution",
            description="Inequality metrics and distributional impacts",
            required_data=["gini_coefficient"],
            llm_prompt="Assess income distribution trends including Gini coefficient dynamics, "
            "wage inequality, and the impact of fiscal policy on distribution.",
        ),
        ReportSection(
            id="risks_outlook",
            title="Risks and Outlook",
            description="Key risks and forward-looking assessment",
            llm_prompt="Identify the top 3-5 risks to the economic outlook and provide "
            "a forward-looking assessment. Include both upside and downside scenarios.",
        ),
        ReportSection(
            id="policy_recommendations",
            title="Policy Recommendations",
            description="Evidence-based policy suggestions",
            llm_prompt="Based on the analysis, provide 3-5 specific, evidence-based "
            "policy recommendations for fiscal and monetary policy.",
        ),
    ],
)


SCENARIO_COMPARISON_TEMPLATE = ReportTemplate(
    name="Scenario Comparison Report",
    description="Side-by-side comparison of economic scenarios",
    sections=[
        ReportSection(
            id="executive_summary",
            title="Executive Summary",
            description="Overview of scenarios compared and key differences",
            llm_prompt="Summarize the scenario comparison, highlighting the most "
            "significant differences in outcomes across scenarios.",
        ),
        ReportSection(
            id="scenario_overview",
            title="Scenario Descriptions",
            description="What each scenario assumes",
            llm_prompt="Describe each scenario's key assumptions and parameters.",
        ),
        ReportSection(
            id="comparative_analysis",
            title="Comparative Analysis",
            description="Side-by-side outcome comparison",
            llm_prompt="Compare the scenarios across all key metrics (GDP, unemployment, "
            "inflation, credit, inequality). Use tables where appropriate.",
        ),
        ReportSection(
            id="sensitivity",
            title="Sensitivity Analysis",
            description="Which parameters drive the largest differences",
            llm_prompt="Identify which parameter changes drive the largest outcome "
            "differences between scenarios.",
        ),
        ReportSection(
            id="recommendations",
            title="Recommendations",
            description="Which scenario/policy mix is optimal",
            llm_prompt="Based on the comparison, which policy configuration best "
            "achieves the dual mandate of growth and stability?",
        ),
    ],
)


STRESS_TEST_TEMPLATE = ReportTemplate(
    name="Stress Test Report",
    description="Economic stress test and vulnerability assessment",
    sections=[
        ReportSection(
            id="executive_summary",
            title="Executive Summary",
            description="Overview of stress test results",
            llm_prompt="Summarize the stress test results: which scenarios were tested, "
            "which vulnerabilities were found, and overall system resilience.",
        ),
        ReportSection(
            id="methodology",
            title="Methodology",
            description="Stress test design and scenarios",
            llm_prompt="Describe the stress test methodology: scenarios applied, "
            "severity levels, and evaluation criteria.",
        ),
        ReportSection(
            id="results",
            title="Results by Scenario",
            description="Detailed results for each stress scenario",
            llm_prompt="Present detailed results for each stress scenario, "
            "highlighting where the system showed vulnerability.",
        ),
        ReportSection(
            id="vulnerability_assessment",
            title="Vulnerability Assessment",
            description="Key weaknesses and contagion pathways",
            llm_prompt="Identify the key vulnerabilities and potential contagion "
            "pathways discovered during stress testing.",
        ),
        ReportSection(
            id="resilience_score",
            title="Resilience Scorecard",
            description="Overall system resilience rating",
            llm_prompt="Rate the system's resilience on a scale and provide "
            "specific metrics supporting the rating.",
        ),
    ],
)

REPORT_TEMPLATES: dict[str, ReportTemplate] = {
    "macro_forecast": MACRO_FORECAST_TEMPLATE,
    "scenario_comparison": SCENARIO_COMPARISON_TEMPLATE,
    "stress_test": STRESS_TEST_TEMPLATE,
}
