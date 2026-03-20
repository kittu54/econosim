"""Tests for LLM integration, LLM-powered policies, analysis,
reports, NL interpreter, and forum engine.

All tests use MockLLMClient to avoid real API calls.
"""

from __future__ import annotations

import json
import pytest
import pandas as pd
import numpy as np

from econosim.llm.client import LLMClient, LLMConfig, LLMResponse, MockLLMClient
from econosim.llm.memory import AgentMemory, MemoryEntry
from econosim.llm.prompts import (
    PromptTemplate,
    EconomicPersonality,
    HOUSEHOLD_PERSONALITIES,
    FIRM_PERSONALITIES,
    BANK_PERSONALITIES,
    GOVERNMENT_PERSONALITIES,
    ANALYST_SYSTEM_PROMPTS,
)


# ── LLM Client Tests ─────────────────────────────────────────────


class TestLLMConfig:
    def test_default_config(self):
        config = LLMConfig()
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048

    def test_from_env_defaults(self):
        config = LLMConfig.from_env()
        assert isinstance(config, LLMConfig)


class TestMockLLMClient:
    def test_default_response(self):
        client = MockLLMClient()
        resp = client.chat([{"role": "user", "content": "test"}])
        assert resp.content == '{"decision": "default", "reasoning": "mock response"}'
        assert resp.model == "mock"
        assert client.call_count == 1

    def test_custom_responses(self):
        client = MockLLMClient(responses=["first", "second"])
        r1 = client.chat([{"role": "user", "content": "a"}])
        r2 = client.chat([{"role": "user", "content": "b"}])
        r3 = client.chat([{"role": "user", "content": "c"}])
        assert r1.content == "first"
        assert r2.content == "second"
        assert r3.content == '{"decision": "default", "reasoning": "mock response"}'
        assert client.call_count == 3

    def test_complete_shortcut(self):
        client = MockLLMClient(responses=["hello"])
        resp = client.complete("test", system="sys")
        assert resp.content == "hello"
        assert client.last_call is not None
        msgs = client.last_call["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"


class TestLLMResponse:
    def test_as_json_valid(self):
        resp = LLMResponse(content='{"key": "value"}')
        assert resp.as_json() == {"key": "value"}

    def test_as_json_markdown_wrapped(self):
        resp = LLMResponse(content='```json\n{"key": "value"}\n```')
        assert resp.as_json() == {"key": "value"}

    def test_as_json_invalid(self):
        resp = LLMResponse(content="not json")
        assert resp.as_json() == {}


# ── Memory Tests ──────────────────────────────────────────────────


class TestAgentMemory:
    def test_add_and_retrieve(self):
        mem = AgentMemory(max_entries=10, agent_id="test")
        mem.add(1, "observation", "GDP is rising", importance=2.0)
        mem.add(1, "decision", "Increase spending")
        assert len(mem) == 2

    def test_get_recent(self):
        mem = AgentMemory()
        for i in range(20):
            mem.add(i, "observation", f"Period {i}")
        recent = mem.get_recent(5)
        assert len(recent) == 5
        assert recent[-1].period == 19

    def test_get_by_category(self):
        mem = AgentMemory()
        mem.add(1, "observation", "obs1")
        mem.add(1, "decision", "dec1")
        mem.add(2, "observation", "obs2")
        obs = mem.get_by_category("observation")
        assert len(obs) == 2

    def test_pruning(self):
        mem = AgentMemory(max_entries=15)
        for i in range(25):
            mem.add(i, "observation", f"Event {i}", importance=float(i))
        assert len(mem) <= 15

    def test_to_prompt_text(self):
        mem = AgentMemory()
        mem.add(1, "observation", "GDP grew 2%")
        text = mem.to_prompt_text()
        assert "Period 1" in text
        assert "GDP grew 2%" in text

    def test_serialization(self):
        mem = AgentMemory(max_entries=10, agent_id="firm_0")
        mem.add(1, "decision", "Hired 5 workers")
        mem.add(2, "outcome", "Revenue increased")

        data = mem.to_dict()
        restored = AgentMemory.from_dict(data)
        assert len(restored) == 2
        assert restored.agent_id == "firm_0"

    def test_clear(self):
        mem = AgentMemory()
        mem.add(1, "observation", "test")
        mem.clear()
        assert len(mem) == 0

    def test_period_range(self):
        mem = AgentMemory()
        for i in range(10):
            mem.add(i, "observation", f"p{i}")
        result = mem.get_by_period_range(3, 6)
        assert len(result) == 4


# ── Prompt Tests ──────────────────────────────────────────────────


class TestPromptTemplate:
    def test_system_prompt_all_roles(self):
        for role in ["household", "firm", "bank", "government"]:
            personality = EconomicPersonality(name="Test", role=role)
            prompt = PromptTemplate.get_system_prompt(role, personality)
            assert "Test" in prompt
            assert role in prompt.lower() or "agent" in prompt.lower()

    def test_decision_prompt_firm(self):
        prompt = PromptTemplate.get_decision_prompt(
            "firm",
            period=5,
            inventory=100,
            deposits=5000.0,
            price=10.0,
            wage=60.0,
        )
        assert "Period 5" in prompt
        assert "100" in prompt
        assert "JSON" in prompt

    def test_decision_prompt_household(self):
        prompt = PromptTemplate.get_decision_prompt(
            "household",
            period=3,
            deposits=1000.0,
            employed="employed",
        )
        assert "Period 3" in prompt
        assert "employed" in prompt

    def test_personality_text(self):
        p = EconomicPersonality(
            name="Test Agent",
            role="firm",
            risk_tolerance="aggressive",
            traits=["innovative", "bold"],
            backstory="A tech startup.",
        )
        text = p.to_prompt_text()
        assert "Test Agent" in text
        assert "aggressive" in text
        assert "innovative" in text
        assert "tech startup" in text


class TestPersonalityLibrary:
    def test_household_personalities_exist(self):
        assert len(HOUSEHOLD_PERSONALITIES) >= 2

    def test_firm_personalities_exist(self):
        assert len(FIRM_PERSONALITIES) >= 2

    def test_bank_personalities_exist(self):
        assert len(BANK_PERSONALITIES) >= 1

    def test_government_personalities_exist(self):
        assert len(GOVERNMENT_PERSONALITIES) >= 1

    def test_analyst_prompts_exist(self):
        assert "macro_analyst" in ANALYST_SYSTEM_PROMPTS
        assert "risk_analyst" in ANALYST_SYSTEM_PROMPTS
        assert len(ANALYST_SYSTEM_PROMPTS) >= 5


# ── LLM Policy Tests ─────────────────────────────────────────────


class TestLLMPolicies:
    def test_llm_firm_policy(self):
        from econosim.policies.llm_policies import LLMFirmPolicy
        from econosim.policies.interfaces import FirmState, MacroState, FirmAction

        mock_response = json.dumps({
            "vacancies": 3,
            "price_adjustment": 1.05,
            "wage_adjustment": 0.98,
            "loan_request": 1000.0,
            "reasoning": "Market is growing, raising prices slightly",
        })
        client = MockLLMClient(responses=[mock_response])
        policy = LLMFirmPolicy(client)

        action = policy.act(FirmState(deposits=5000), MacroState(period=1))
        assert isinstance(action, FirmAction)
        assert action.vacancies == 3
        assert action.price_adjustment == 1.05
        assert action.wage_adjustment == 0.98
        assert action.loan_request == 1000.0

    def test_llm_firm_policy_fallback(self):
        from econosim.policies.llm_policies import LLMFirmPolicy
        from econosim.policies.interfaces import FirmState, MacroState

        # Invalid JSON response — should use defaults
        client = MockLLMClient(responses=["not valid json"])
        policy = LLMFirmPolicy(client)
        action = policy.act(FirmState(), MacroState())
        assert action.vacancies >= 0
        assert 0.8 <= action.price_adjustment <= 1.2

    def test_llm_household_policy(self):
        from econosim.policies.llm_policies import LLMHouseholdPolicy
        from econosim.policies.interfaces import HouseholdState, MacroState

        mock_response = json.dumps({
            "consumption_fraction": 0.6,
            "labor_participation": True,
            "reservation_wage_adjustment": 0.95,
            "reasoning": "Being cautious given rising unemployment",
        })
        client = MockLLMClient(responses=[mock_response])
        policy = LLMHouseholdPolicy(client)

        action = policy.act(HouseholdState(deposits=1000), MacroState(period=2))
        assert action.consumption_fraction == 0.6
        assert action.labor_participation is True
        assert action.reservation_wage_adjustment == 0.95

    def test_llm_bank_policy(self):
        from econosim.policies.llm_policies import LLMBankPolicy
        from econosim.policies.interfaces import BankState, MacroState

        mock_response = json.dumps({
            "base_rate_adjustment": 1.1,
            "capital_target_adjustment": 1.0,
            "risk_premium_adjustment": 1.05,
            "reasoning": "Tightening due to inflation concerns",
        })
        client = MockLLMClient(responses=[mock_response])
        policy = LLMBankPolicy(client)

        action = policy.act(BankState(), MacroState(period=3))
        assert action.base_rate_adjustment == 1.1
        assert action.risk_premium_adjustment == 1.05

    def test_llm_government_policy(self):
        from econosim.policies.llm_policies import LLMGovernmentPolicy
        from econosim.policies.interfaces import GovernmentState, MacroState

        mock_response = json.dumps({
            "tax_rate": 0.25,
            "transfer_per_unemployed": 75.0,
            "spending_per_period": 3000.0,
            "reasoning": "Counter-cyclical expansion",
        })
        client = MockLLMClient(responses=[mock_response])
        policy = LLMGovernmentPolicy(client)

        action = policy.act(GovernmentState(), MacroState(period=4))
        assert action.tax_rate == 0.25
        assert action.transfer_per_unemployed == 75.0
        assert action.spending_per_period == 3000.0

    def test_policy_memory_accumulation(self):
        from econosim.policies.llm_policies import LLMFirmPolicy
        from econosim.policies.interfaces import FirmState, MacroState

        client = MockLLMClient()
        policy = LLMFirmPolicy(client)

        for i in range(5):
            policy.act(FirmState(), MacroState(period=i))

        assert len(policy.memory) == 5
        assert policy.memory.get_by_category("decision") == policy.memory.entries

    def test_value_clamping(self):
        from econosim.policies.llm_policies import LLMFirmPolicy
        from econosim.policies.interfaces import FirmState, MacroState

        # Out-of-range values should be clamped
        mock_response = json.dumps({
            "vacancies": -5,
            "price_adjustment": 2.0,
            "wage_adjustment": 0.1,
            "loan_request": -100,
        })
        client = MockLLMClient(responses=[mock_response])
        policy = LLMFirmPolicy(client)

        action = policy.act(FirmState(), MacroState())
        assert action.vacancies >= 0
        assert action.price_adjustment <= 1.2
        assert action.wage_adjustment >= 0.9
        assert action.loan_request >= 0.0


# ── Data Analysis Tests ───────────────────────────────────────────


class TestDataAnalysis:
    def _make_sim_data(self, n=60):
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "gdp": 7000 + np.cumsum(rng.normal(50, 100, n)),
            "unemployment_rate": np.clip(0.05 + rng.normal(0, 0.01, n), 0, 1),
            "avg_price": 10 + np.cumsum(rng.normal(-0.05, 0.1, n)),
            "gini_coefficient": np.clip(0.3 + rng.normal(0, 0.01, n), 0, 1),
            "total_credit": 5000 + np.cumsum(rng.normal(10, 50, n)),
        })

    def test_analyze_simulation_data(self):
        from econosim.data.analysis import analyze_simulation_data

        df = self._make_sim_data()
        analysis = analyze_simulation_data(df)

        assert analysis.regime in ("normal", "recession", "expansion", "crisis")
        assert "mean_gdp_growth" in analysis.moments
        assert "mean_unemployment" in analysis.moments
        assert len(analysis.trends) > 0

    def test_correlations(self):
        from econosim.data.analysis import analyze_simulation_data

        df = self._make_sim_data()
        analysis = analyze_simulation_data(df)
        assert len(analysis.correlations) > 0

    def test_narrative(self):
        from econosim.data.analysis import analyze_simulation_data

        df = self._make_sim_data()
        analysis = analyze_simulation_data(df)
        narrative = analysis.to_narrative()
        assert "Regime" in narrative

    def test_load_and_analyze_simulation(self):
        from econosim.data.analysis import load_and_analyze

        analysis = load_and_analyze("simulation")
        assert analysis.regime in ("normal", "recession", "expansion", "crisis")
        assert len(analysis.data) > 0


# ── Report Engine Tests ───────────────────────────────────────────


class TestReportEngine:
    def _make_data(self):
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "gdp": 7000 + np.cumsum(rng.normal(50, 100, 60)),
            "unemployment_rate": np.clip(0.05 + rng.normal(0, 0.01, 60), 0, 1),
            "avg_price": 10 + np.cumsum(rng.normal(-0.05, 0.1, 60)),
            "gini_coefficient": np.clip(0.3 + rng.normal(0, 0.01, 60), 0, 1),
            "total_credit": 5000 + np.cumsum(rng.normal(10, 50, 60)),
        })

    def test_generate_report_no_llm(self):
        from econosim.reports.engine import ReportEngine, ReportConfig

        engine = ReportEngine()
        report = engine.generate(self._make_data())

        assert report.title == "Macro Forecast Report"
        assert len(report.sections) > 0
        assert report.elapsed_seconds >= 0

    def test_report_markdown(self):
        from econosim.reports.engine import ReportEngine

        engine = ReportEngine()
        report = engine.generate(self._make_data())
        md = report.to_markdown()
        assert "# Macro Forecast Report" in md

    def test_report_html(self):
        from econosim.reports.engine import ReportEngine

        engine = ReportEngine()
        report = engine.generate(self._make_data())
        html = report.to_html()
        assert "<!DOCTYPE html>" in html
        assert "EconoSim Report Engine" in html

    def test_report_json(self):
        from econosim.reports.engine import ReportEngine

        engine = ReportEngine()
        report = engine.generate(self._make_data())
        j = report.to_json()
        assert "title" in j
        assert "sections" in j
        assert len(j["sections"]) > 0

    def test_report_with_mock_llm(self):
        from econosim.reports.engine import ReportEngine, ReportConfig

        client = MockLLMClient(default_response="LLM generated analysis content here.")
        engine = ReportEngine(llm_client=client)
        report = engine.generate(self._make_data())

        assert any("LLM generated" in s.content for s in report.sections)

    def test_all_templates(self):
        from econosim.reports.engine import ReportEngine, ReportConfig
        from econosim.reports.templates import REPORT_TEMPLATES

        engine = ReportEngine()
        data = self._make_data()

        for template_name in REPORT_TEMPLATES:
            config = ReportConfig(template_name=template_name)
            report = engine.generate(data, config)
            assert len(report.sections) > 0

    def test_invalid_template(self):
        from econosim.reports.engine import ReportEngine, ReportConfig

        engine = ReportEngine()
        config = ReportConfig(template_name="nonexistent")
        with pytest.raises(ValueError, match="Unknown template"):
            engine.generate(self._make_data(), config)


# ── NL Interpreter Tests ─────────────────────────────────────────


class TestNLInterpreter:
    def test_interpret_simulate(self):
        from econosim.nl.interpreter import NLInterpreter

        mock_response = json.dumps({
            "intent": "simulate",
            "simulation_config": {"num_periods": 120},
            "analysis_focus": ["gdp", "unemployment"],
            "report_type": "macro_forecast",
            "confidence": 0.8,
            "explanation": "User wants a basic simulation",
            "scenarios": [],
        })
        client = MockLLMClient(responses=[mock_response])
        interp = NLInterpreter(client)

        result = interp.interpret("Run a 120-period baseline simulation")
        assert result.intent == "simulate"
        assert result.confidence == 0.8

    def test_interpret_compare(self):
        from econosim.nl.interpreter import NLInterpreter

        mock_response = json.dumps({
            "intent": "compare",
            "simulation_config": {"num_periods": 60},
            "scenarios": [
                {"name": "baseline", "description": "Normal", "overrides": {}},
                {"name": "recession", "description": "Downturn", "overrides": {"household.consumption_propensity": 0.5}},
            ],
            "analysis_focus": ["gdp"],
            "report_type": "scenario_comparison",
            "confidence": 0.9,
            "explanation": "Comparing two scenarios",
        })
        client = MockLLMClient(responses=[mock_response])
        interp = NLInterpreter(client)

        result = interp.interpret("Compare baseline vs recession")
        assert result.intent == "compare"
        assert len(result.scenarios) == 2

    def test_fallback_interpretation(self):
        from econosim.nl.interpreter import NLInterpreter

        # LLM returns invalid JSON → fallback
        client = MockLLMClient(responses=["invalid"])
        interp = NLInterpreter(client)

        result = interp.interpret("What happens during a recession?")
        assert result.confidence == 0.3  # fallback confidence

    def test_fallback_recession_keywords(self):
        from econosim.nl.interpreter import NLInterpreter

        client = MockLLMClient(responses=["invalid"])
        interp = NLInterpreter(client)

        result = interp.interpret("Simulate a deep recession scenario")
        assert "household" in result.simulation_config

    def test_interpret_and_run(self):
        from econosim.nl.interpreter import NLInterpreter

        mock_response = json.dumps({
            "intent": "simulate",
            "simulation_config": {"num_periods": 20},
            "analysis_focus": ["gdp"],
            "report_type": "macro_forecast",
            "confidence": 0.8,
            "explanation": "Basic simulation",
            "scenarios": [],
        })
        client = MockLLMClient(responses=[mock_response])
        interp = NLInterpreter(client)

        result = interp.interpret_and_run("Run a short simulation")
        assert "simulation" in result
        assert "analysis" in result["simulation"]

    def test_interpret_explain(self):
        from econosim.nl.interpreter import NLInterpreter

        mock_responses = [
            json.dumps({
                "intent": "explain",
                "simulation_config": {},
                "scenarios": [],
                "analysis_focus": [],
                "report_type": "macro_forecast",
                "confidence": 0.9,
                "explanation": "User wants an explanation",
            }),
            "GDP measures the total value of goods and services produced.",
        ]
        client = MockLLMClient(responses=mock_responses)
        interp = NLInterpreter(client)

        result = interp.interpret_and_run("What is GDP?")
        assert "explanation" in result


# ── Forum Engine Tests ────────────────────────────────────────────


class TestForumEngine:
    def _make_data(self):
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "gdp": 7000 + np.cumsum(rng.normal(50, 100, 60)),
            "unemployment_rate": np.clip(0.05 + rng.normal(0, 0.01, 60), 0, 1),
            "avg_price": 10 + np.cumsum(rng.normal(-0.05, 0.1, 60)),
            "gini_coefficient": np.clip(0.3 + rng.normal(0, 0.01, 60), 0, 1),
            "total_credit": 5000 + np.cumsum(rng.normal(10, 50, 60)),
        })

    def test_forum_basic(self):
        from econosim.forum.engine import ForumEngine, ForumConfig

        # Each round: 5 agents + 1 moderator = 6 messages
        # 2 rounds = 12 messages + final synthesis call
        # Total LLM calls: 5 + 1 + 5 + 1 + 1 (synthesis) = 13
        responses = ["Analysis from agent perspective."] * 12
        responses.append(json.dumps({
            "consensus": "Economy is stable with moderate growth.",
            "key_findings": ["GDP growing steadily", "Low unemployment"],
            "disagreements": ["Inflation outlook differs"],
            "recommendations": ["Maintain current fiscal stance"],
        }))
        client = MockLLMClient(responses=responses)
        engine = ForumEngine(client)

        config = ForumConfig(num_rounds=2)
        session = engine.run(self._make_data(), config)

        assert len(session.messages) > 0
        assert session.consensus != ""
        assert len(session.key_findings) > 0

    def test_forum_transcript(self):
        from econosim.forum.engine import ForumEngine, ForumConfig

        # 1 round: 5 agents + 1 moderator = 6 messages, then 1 synthesis
        synthesis = json.dumps({
            "consensus": "All good.",
            "key_findings": ["Growth"],
            "disagreements": [],
            "recommendations": ["Continue"],
        })
        responses = ["Agent analysis."] * 6 + [synthesis]
        client = MockLLMClient(responses=responses)
        engine = ForumEngine(client)

        config = ForumConfig(num_rounds=1)
        session = engine.run(self._make_data(), config)

        transcript = session.to_transcript()
        assert "TRANSCRIPT" in transcript
        assert "CONSENSUS" in transcript

    def test_forum_serialization(self):
        from econosim.forum.engine import ForumEngine, ForumConfig

        synthesis = json.dumps({
            "consensus": "Stable.",
            "key_findings": ["A"],
            "disagreements": [],
            "recommendations": ["B"],
        })
        responses = ["Analysis."] * 6 + [synthesis]
        client = MockLLMClient(responses=responses)
        engine = ForumEngine(client)

        session = engine.run(self._make_data(), ForumConfig(num_rounds=1))
        data = session.to_dict()
        assert "messages" in data
        assert "consensus" in data

    def test_forum_with_query(self):
        from econosim.forum.engine import ForumEngine, ForumConfig

        synthesis = json.dumps({
            "consensus": "Policy change needed.",
            "key_findings": ["Unemployment rising"],
            "disagreements": [],
            "recommendations": ["Increase transfers"],
        })
        responses = ["Focused analysis."] * 6 + [synthesis]
        client = MockLLMClient(responses=responses)
        engine = ForumEngine(client)

        config = ForumConfig(num_rounds=1)
        session = engine.run(
            self._make_data(), config,
            query="Should we increase unemployment benefits?",
        )
        assert session.consensus != ""

    def test_forum_custom_agents(self):
        from econosim.forum.engine import ForumEngine, ForumConfig

        synthesis = json.dumps({
            "consensus": "OK.",
            "key_findings": [],
            "disagreements": [],
            "recommendations": [],
        })
        # 2 agents + 1 moderator = 3 messages, then 1 synthesis
        responses = ["Analysis."] * 3 + [synthesis]
        client = MockLLMClient(responses=responses)
        engine = ForumEngine(client)

        config = ForumConfig(
            agents=["macro_analyst", "risk_analyst"],
            num_rounds=1,
        )
        session = engine.run(self._make_data(), config)
        analyst_msgs = [m for m in session.messages if m.role == "analyst"]
        assert len(analyst_msgs) == 2
