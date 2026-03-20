"""Forum engine for multi-agent economic analysis discussion.

Orchestrates a moderated discussion between specialist analyst agents,
each analyzing simulation data from their domain perspective. A
moderator agent synthesizes findings and guides the discussion.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from econosim.data.analysis import EmpiricalAnalysis, analyze_simulation_data
from econosim.llm.client import LLMClient
from econosim.llm.prompts import ANALYST_SYSTEM_PROMPTS, ANALYSIS_PROMPT, FORUM_MODERATOR_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class ForumMessage:
    """A single message in the forum discussion."""

    agent: str
    role: str  # "analyst" or "moderator"
    content: str
    round_num: int = 0
    timestamp: float = 0.0


@dataclass
class ForumConfig:
    """Configuration for the forum discussion."""

    agents: list[str] = field(
        default_factory=lambda: [
            "macro_analyst",
            "labor_analyst",
            "financial_analyst",
            "policy_analyst",
            "risk_analyst",
        ]
    )
    num_rounds: int = 2
    moderator_every_n: int = 5  # moderator speaks after every N agent messages
    max_words_per_message: int = 400
    include_moderator: bool = True


@dataclass
class ForumSession:
    """A complete forum discussion session."""

    messages: list[ForumMessage] = field(default_factory=list)
    consensus: str = ""
    key_findings: list[str] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_transcript(self) -> str:
        """Format the discussion as a readable transcript."""
        lines = ["=" * 60, "ECONOMIC ANALYSIS FORUM — TRANSCRIPT", "=" * 60, ""]

        for msg in self.messages:
            role_label = "MODERATOR" if msg.role == "moderator" else msg.agent.upper()
            lines.append(f"[Round {msg.round_num}] {role_label}:")
            lines.append(msg.content)
            lines.append("")
            lines.append("-" * 40)
            lines.append("")

        if self.consensus:
            lines.extend(["", "=" * 60, "CONSENSUS VIEW", "=" * 60, "", self.consensus])

        if self.key_findings:
            lines.extend(["", "KEY FINDINGS:"])
            for i, finding in enumerate(self.key_findings, 1):
                lines.append(f"  {i}. {finding}")

        if self.recommendations:
            lines.extend(["", "RECOMMENDATIONS:"])
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"  {i}. {rec}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize as JSON-compatible dict."""
        return {
            "messages": [
                {
                    "agent": m.agent,
                    "role": m.role,
                    "content": m.content,
                    "round_num": m.round_num,
                }
                for m in self.messages
            ],
            "consensus": self.consensus,
            "key_findings": self.key_findings,
            "disagreements": self.disagreements,
            "recommendations": self.recommendations,
            "elapsed_seconds": self.elapsed_seconds,
        }


class ForumEngine:
    """Orchestrates multi-agent analysis discussions.

    Like BettaFish's ForumEngine, multiple specialist agents analyze
    the same data from different perspectives, with a moderator
    synthesizing their findings.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def run(
        self,
        data: pd.DataFrame,
        config: ForumConfig | None = None,
        analysis: EmpiricalAnalysis | None = None,
        query: str = "",
    ) -> ForumSession:
        """Run a forum discussion on simulation data.

        Args:
            data: Simulation results DataFrame.
            config: Forum configuration.
            analysis: Pre-computed analysis.
            query: Optional user query to focus the discussion.

        Returns:
            ForumSession with full discussion transcript and synthesis.
        """
        config = config or ForumConfig()
        start = time.time()

        if analysis is None:
            analysis = analyze_simulation_data(data)

        session = ForumSession()

        # Build shared data context
        data_context = self._build_data_context(analysis, query)

        # Run discussion rounds
        for round_num in range(1, config.num_rounds + 1):
            # Each analyst contributes
            for agent_name in config.agents:
                message = self._get_agent_contribution(
                    agent_name, data_context, session, round_num, config
                )
                session.messages.append(message)

            # Moderator synthesizes after each round
            if config.include_moderator:
                moderator_msg = self._get_moderator_synthesis(
                    session, round_num, data_context
                )
                session.messages.append(moderator_msg)

        # Final synthesis
        self._generate_final_synthesis(session, data_context)

        session.elapsed_seconds = round(time.time() - start, 2)
        return session

    def _build_data_context(
        self, analysis: EmpiricalAnalysis, query: str = ""
    ) -> str:
        """Build shared data context for all agents."""
        moments = analysis.moments
        lines = ["SIMULATION DATA SUMMARY:", ""]

        # Key metrics
        lines.append(f"Economic Regime: {analysis.regime}")
        lines.append(f"Periods Analyzed: {len(analysis.data)}")
        lines.append("")

        if moments:
            lines.append("Key Metrics:")
            for name, val in moments.items():
                lines.append(f"  {name}: {val:.4f}")
            lines.append("")

        if analysis.trends:
            lines.append("Trends:")
            for var, trend in analysis.trends.items():
                lines.append(f"  {var}: {trend}")
            lines.append("")

        if analysis.key_events:
            lines.append("Notable Events:")
            for event in analysis.key_events:
                lines.append(f"  - {event}")
            lines.append("")

        if analysis.correlations:
            lines.append("Cross-Variable Correlations:")
            for pair, corr in analysis.correlations.items():
                lines.append(f"  {pair}: {corr:.3f}")

        if query:
            lines.extend(["", f"USER QUESTION: {query}"])

        return "\n".join(lines)

    def _get_agent_contribution(
        self,
        agent_name: str,
        data_context: str,
        session: ForumSession,
        round_num: int,
        config: ForumConfig,
    ) -> ForumMessage:
        """Get a single agent's contribution to the discussion."""
        system = ANALYST_SYSTEM_PROMPTS.get(agent_name, ANALYST_SYSTEM_PROMPTS["macro_analyst"])

        # Build prompt with prior discussion context
        prompt_parts = [data_context]

        if round_num > 1 and session.messages:
            # Include previous round's contributions for context
            prev_messages = [
                m for m in session.messages if m.round_num == round_num - 1
            ]
            if prev_messages:
                prompt_parts.append("\nPREVIOUS ROUND DISCUSSION:")
                for m in prev_messages[-5:]:
                    prompt_parts.append(f"\n{m.agent}: {m.content[:300]}...")

        prompt_parts.append(
            f"\nProvide your {agent_name.replace('_', ' ')} perspective. "
            f"Be specific and data-driven. Max {config.max_words_per_message} words."
        )

        prompt = "\n".join(prompt_parts)

        try:
            response = self.llm_client.complete(prompt, system=system)
            content = response.content
        except Exception as e:
            logger.warning(f"Agent {agent_name} failed: {e}")
            content = f"[{agent_name}] Analysis unavailable due to LLM error."

        return ForumMessage(
            agent=agent_name,
            role="analyst",
            content=content,
            round_num=round_num,
            timestamp=time.time(),
        )

    def _get_moderator_synthesis(
        self,
        session: ForumSession,
        round_num: int,
        data_context: str,
    ) -> ForumMessage:
        """Generate moderator synthesis of the current round."""
        # Gather this round's contributions
        round_messages = [m for m in session.messages if m.round_num == round_num]
        contributions = "\n\n".join(
            f"**{m.agent}**: {m.content}" for m in round_messages if m.role == "analyst"
        )

        prompt = FORUM_MODERATOR_PROMPT.format(contributions=contributions)

        try:
            response = self.llm_client.complete(
                prompt,
                system="You are a forum moderator synthesizing economic analysis.",
            )
            content = response.content
        except Exception as e:
            logger.warning(f"Moderator synthesis failed: {e}")
            content = "Moderator synthesis unavailable."

        return ForumMessage(
            agent="moderator",
            role="moderator",
            content=content,
            round_num=round_num,
            timestamp=time.time(),
        )

    def _generate_final_synthesis(
        self, session: ForumSession, data_context: str
    ) -> None:
        """Generate the final consensus, findings, and recommendations."""
        all_contributions = "\n\n".join(
            f"[{m.agent}] {m.content}" for m in session.messages
        )

        prompt = f"""Based on the following multi-agent economic analysis discussion,
provide a final synthesis.

{data_context}

DISCUSSION:
{all_contributions[:4000]}

Respond with JSON:
{{
    "consensus": "<overall consensus view in 2-3 sentences>",
    "key_findings": ["<finding 1>", "<finding 2>", ...],
    "disagreements": ["<disagreement 1>", ...],
    "recommendations": ["<recommendation 1>", "<recommendation 2>", ...]
}}"""

        try:
            response = self.llm_client.complete(
                prompt,
                system="Synthesize the discussion into structured findings.",
                json_mode=True,
            )
            data = response.as_json()

            session.consensus = data.get("consensus", "")
            session.key_findings = data.get("key_findings", [])
            session.disagreements = data.get("disagreements", [])
            session.recommendations = data.get("recommendations", [])
        except Exception as e:
            logger.warning(f"Final synthesis failed: {e}")
            session.consensus = "Final synthesis unavailable due to LLM error."
