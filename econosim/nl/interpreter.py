"""Natural language query interpreter for EconoSim.

Translates natural language economic questions and scenarios into
simulation configurations, runs the simulation, and returns structured
results with analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from econosim.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class InterpretedQuery:
    """A natural language query translated into simulation parameters."""

    original_query: str
    intent: str  # "simulate", "forecast", "compare", "analyze", "explain"
    simulation_config: dict[str, Any] = field(default_factory=dict)
    scenarios: list[dict[str, Any]] = field(default_factory=list)
    analysis_focus: list[str] = field(default_factory=list)
    report_type: str = "macro_forecast"
    confidence: float = 0.0
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "intent": self.intent,
            "simulation_config": self.simulation_config,
            "scenarios": self.scenarios,
            "analysis_focus": self.analysis_focus,
            "report_type": self.report_type,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


INTERPRETATION_SYSTEM_PROMPT = """You are an economic simulation configuration assistant.
Given a natural language query about economics, you must translate it into a simulation
configuration for the EconoSim platform.

The simulation has these configurable parameters:
- num_periods: int (5-500, default 60)
- seed: int (default 42)
- household.count: int (1-500, default 100)
- household.initial_deposits: float (default 1000)
- household.consumption_propensity: float (0-1, default 0.8)
- household.wealth_propensity: float (0-1, default 0.4)
- household.reservation_wage: float (default 50)
- firm.count: int (1-50, default 5)
- firm.initial_deposits: float (default 15000)
- firm.initial_price: float (default 10)
- firm.initial_wage: float (default 60)
- firm.labor_productivity: float (default 8)
- firm.price_adjustment_speed: float (default 0.03)
- firm.wage_adjustment_speed: float (default 0.02)
- government.income_tax_rate: float (0-0.5, default 0.2)
- government.transfer_per_unemployed: float (default 50)
- government.spending_per_period: float (default 2000)
- government.initial_deposits: float (default 100000)
- bank.base_interest_rate: float (default 0.005)
- bank.capital_adequacy_ratio: float (default 0.08)

Intents:
- "simulate": Run a single simulation with specific parameters
- "forecast": Generate probabilistic forecasts
- "compare": Compare multiple scenarios
- "analyze": Analyze specific economic phenomena
- "explain": Explain economic concepts

You must respond with a JSON object."""

INTERPRETATION_PROMPT = """User query: "{query}"

Translate this into a simulation configuration. Respond with JSON:
{{
    "intent": "<simulate|forecast|compare|analyze|explain>",
    "simulation_config": {{
        "num_periods": <int>,
        "household": {{"consumption_propensity": <float>, ...}},
        "firm": {{"labor_productivity": <float>, ...}},
        "government": {{"income_tax_rate": <float>, ...}},
        "bank": {{"base_interest_rate": <float>, ...}}
    }},
    "scenarios": [
        {{"name": "<scenario_name>", "description": "<what it represents>", "overrides": {{...}}}}
    ],
    "analysis_focus": ["<gdp|unemployment|inflation|credit|inequality|...>"],
    "report_type": "<macro_forecast|scenario_comparison|stress_test>",
    "confidence": <float 0-1, how confident you are in the interpretation>,
    "explanation": "<brief explanation of how you interpreted the query>"
}}

Only include parameters that the user explicitly or implicitly specified.
For unspecified parameters, omit them (defaults will be used).
If comparing scenarios, include multiple entries in "scenarios"."""


class NLInterpreter:
    """Interprets natural language economic queries.

    Uses an LLM to translate natural language questions into simulation
    configurations, then orchestrates simulation and analysis.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def interpret(self, query: str) -> InterpretedQuery:
        """Translate a natural language query into simulation parameters.

        Args:
            query: Natural language economic question or scenario description.

        Returns:
            InterpretedQuery with simulation configuration and intent.
        """
        prompt = INTERPRETATION_PROMPT.format(query=query)

        try:
            response = self.llm_client.complete(
                prompt,
                system=INTERPRETATION_SYSTEM_PROMPT,
                json_mode=True,
            )
            data = response.as_json()
        except Exception as e:
            logger.warning(f"NL interpretation failed: {e}")
            return self._fallback_interpret(query)

        if not data:
            return self._fallback_interpret(query)

        return InterpretedQuery(
            original_query=query,
            intent=data.get("intent", "simulate"),
            simulation_config=data.get("simulation_config", {}),
            scenarios=data.get("scenarios", []),
            analysis_focus=data.get("analysis_focus", []),
            report_type=data.get("report_type", "macro_forecast"),
            confidence=data.get("confidence", 0.5),
            explanation=data.get("explanation", ""),
        )

    def interpret_and_run(self, query: str) -> dict[str, Any]:
        """Interpret a query, run the simulation, and return results.

        Args:
            query: Natural language economic question.

        Returns:
            Dict with interpretation, simulation results, and analysis.
        """
        from econosim.config.schema import SimulationConfig
        from econosim.data.analysis import analyze_simulation_data
        from econosim.experiments.runner import run_experiment

        interpreted = self.interpret(query)

        results: dict[str, Any] = {
            "query": query,
            "interpretation": interpreted.to_dict(),
        }

        if interpreted.intent == "explain":
            # Pure explanation — no simulation needed
            explanation = self._generate_explanation(query)
            results["explanation"] = explanation
            return results

        if interpreted.intent == "compare" and interpreted.scenarios:
            # Run multiple scenarios
            scenario_results = []
            for scenario in interpreted.scenarios:
                cfg_dict = {**interpreted.simulation_config}
                overrides = scenario.get("overrides", {})
                for key, val in overrides.items():
                    parts = key.split(".")
                    target = cfg_dict
                    for part in parts[:-1]:
                        target = target.setdefault(part, {})
                    target[parts[-1]] = val

                config = SimulationConfig(**self._flatten_config(cfg_dict))
                result = run_experiment(config)
                analysis = analyze_simulation_data(result["dataframe"])
                scenario_results.append({
                    "name": scenario.get("name", "unnamed"),
                    "description": scenario.get("description", ""),
                    "summary": result["summary"],
                    "analysis": {
                        "moments": analysis.moments,
                        "trends": analysis.trends,
                        "regime": analysis.regime,
                        "events": analysis.key_events,
                    },
                })
            results["scenarios"] = scenario_results
        else:
            # Single simulation
            config = SimulationConfig(**self._flatten_config(interpreted.simulation_config))
            result = run_experiment(config)
            analysis = analyze_simulation_data(result["dataframe"])
            results["simulation"] = {
                "summary": result["summary"],
                "analysis": {
                    "moments": analysis.moments,
                    "trends": analysis.trends,
                    "regime": analysis.regime,
                    "events": analysis.key_events,
                    "correlations": analysis.correlations,
                },
                "num_periods": len(result["dataframe"]),
            }

        return results

    def _flatten_config(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """Flatten nested config dict for SimulationConfig constructor."""
        flat: dict[str, Any] = {}
        for key, val in cfg.items():
            if isinstance(val, dict):
                flat[key] = val
            else:
                flat[key] = val
        return flat

    def _fallback_interpret(self, query: str) -> InterpretedQuery:
        """Rule-based fallback when LLM interpretation fails."""
        query_lower = query.lower()

        intent = "simulate"
        config: dict[str, Any] = {}
        focus: list[str] = []

        # Detect intent from keywords
        if any(w in query_lower for w in ["compare", "versus", "vs", "difference"]):
            intent = "compare"
        elif any(w in query_lower for w in ["forecast", "predict", "future", "outlook"]):
            intent = "forecast"
        elif any(w in query_lower for w in ["explain", "what is", "how does", "why"]):
            intent = "explain"
        elif any(w in query_lower for w in ["analyze", "assess", "evaluate"]):
            intent = "analyze"

        # Detect parameter hints
        if "recession" in query_lower:
            config["household"] = {"consumption_propensity": 0.5}
            config["firm"] = {"labor_productivity": 5.0}
        elif "boom" in query_lower or "growth" in query_lower:
            config["household"] = {"consumption_propensity": 0.95}
            config["firm"] = {"labor_productivity": 12.0}
        elif "austerity" in query_lower:
            config["government"] = {"spending_per_period": 500, "income_tax_rate": 0.3}

        if "high interest" in query_lower or "tight money" in query_lower:
            config["bank"] = {"base_interest_rate": 0.03}

        # Detect analysis focus
        if "unemployment" in query_lower or "jobs" in query_lower:
            focus.append("unemployment")
        if "inflation" in query_lower or "prices" in query_lower:
            focus.append("inflation")
        if "gdp" in query_lower or "growth" in query_lower:
            focus.append("gdp")
        if "inequality" in query_lower or "gini" in query_lower:
            focus.append("inequality")
        if "credit" in query_lower or "loans" in query_lower:
            focus.append("credit")

        return InterpretedQuery(
            original_query=query,
            intent=intent,
            simulation_config=config,
            analysis_focus=focus or ["gdp", "unemployment"],
            confidence=0.3,
            explanation="Fallback rule-based interpretation",
        )

    def _generate_explanation(self, query: str) -> str:
        """Generate an economic concept explanation via LLM."""
        try:
            response = self.llm_client.complete(
                f"Explain the following economic concept or question in the context "
                f"of a multi-agent macroeconomic simulation: {query}",
                system="You are an economics professor. Explain concepts clearly, "
                "relate them to agent-based simulation, and give practical examples.",
            )
            return response.content
        except Exception:
            return (
                f"Unable to generate explanation for: {query}. "
                "Please ensure an LLM API key is configured."
            )
