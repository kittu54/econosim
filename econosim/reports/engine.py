"""Report generation engine.

Orchestrates the generation of structured economic reports by combining
simulation data analysis with LLM-powered narrative generation.
Outputs HTML, Markdown, or JSON.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from econosim.data.analysis import EmpiricalAnalysis, analyze_simulation_data
from econosim.llm.client import LLMClient, MockLLMClient
from econosim.reports.templates import ReportTemplate, ReportSection, REPORT_TEMPLATES

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    template_name: str = "macro_forecast"
    title: str = ""
    format: str = "html"  # html, markdown, json
    include_charts: bool = True
    include_raw_data: bool = False
    max_section_words: int = 500


@dataclass
class GeneratedSection:
    """A generated report section with content."""

    id: str
    title: str
    content: str
    data_summary: str = ""
    chart_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedReport:
    """A fully generated report."""

    title: str
    template_name: str
    generated_at: str
    sections: list[GeneratedSection]
    metadata: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def to_markdown(self) -> str:
        """Render as Markdown."""
        lines = [f"# {self.title}", ""]
        lines.append(f"*Generated: {self.generated_at}*")
        lines.append("")

        for section in self.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")

        return "\n".join(lines)

    def to_html(self) -> str:
        """Render as standalone HTML document."""
        sections_html = []
        for section in self.sections:
            content = section.content.replace("\n", "<br>\n")
            # Convert markdown bold to HTML
            import re
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
            # Convert markdown lists
            content = re.sub(r'^- (.+)$', r'<li>\1</li>', content, flags=re.MULTILINE)
            content = re.sub(r'(<li>.*</li>\n?)+', r'<ul>\g<0></ul>', content)

            sections_html.append(f"""
        <section id="{section.id}" class="report-section">
            <h2>{section.title}</h2>
            <div class="section-content">{content}</div>
        </section>""")

        nav_items = "\n".join(
            f'            <li><a href="#{s.id}">{s.title}</a></li>'
            for s in self.sections
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        :root {{
            --bg: #0a0e17;
            --surface: #111827;
            --border: #1f2937;
            --text: #e5e7eb;
            --text-muted: #9ca3af;
            --accent: #3b82f6;
            --accent-light: #60a5fa;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.7;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 2rem;
        }}
        header {{
            text-align: center;
            padding: 3rem 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 2rem;
        }}
        header h1 {{
            font-size: 2.2rem;
            background: linear-gradient(135deg, var(--accent), #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        header .meta {{
            color: var(--text-muted);
            font-size: 0.9rem;
        }}
        nav {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}
        nav h3 {{
            color: var(--accent-light);
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        nav ul {{ list-style: none; }}
        nav li {{ margin: 0.3rem 0; }}
        nav a {{
            color: var(--text-muted);
            text-decoration: none;
            transition: color 0.2s;
        }}
        nav a:hover {{ color: var(--accent-light); }}
        .report-section {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 2rem;
            margin-bottom: 1.5rem;
        }}
        .report-section h2 {{
            color: var(--accent-light);
            font-size: 1.4rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }}
        .section-content {{
            color: var(--text);
        }}
        .section-content ul {{
            margin: 0.5rem 0;
            padding-left: 1.5rem;
        }}
        .section-content li {{
            margin: 0.3rem 0;
            list-style: disc;
        }}
        .section-content strong {{
            color: var(--accent-light);
        }}
        footer {{
            text-align: center;
            padding: 2rem 0;
            color: var(--text-muted);
            font-size: 0.8rem;
            border-top: 1px solid var(--border);
            margin-top: 2rem;
        }}
        @media print {{
            body {{ background: white; color: #1a1a1a; }}
            .report-section {{ border: 1px solid #ddd; }}
            header h1 {{ -webkit-text-fill-color: #2563eb; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{self.title}</h1>
            <p class="meta">Generated: {self.generated_at} | Template: {self.template_name}</p>
        </header>

        <nav>
            <h3>Table of Contents</h3>
            <ul>
{nav_items}
            </ul>
        </nav>

        {"".join(sections_html)}

        <footer>
            <p>Generated by EconoSim Report Engine</p>
        </footer>
    </div>
</body>
</html>"""

    def to_json(self) -> dict[str, Any]:
        """Render as JSON-serializable dict."""
        return {
            "title": self.title,
            "template_name": self.template_name,
            "generated_at": self.generated_at,
            "elapsed_seconds": self.elapsed_seconds,
            "sections": [
                {
                    "id": s.id,
                    "title": s.title,
                    "content": s.content,
                    "data_summary": s.data_summary,
                    "chart_data": s.chart_data,
                }
                for s in self.sections
            ],
            "metadata": self.metadata,
        }


class ReportEngine:
    """Generates economic analysis reports from simulation data.

    Combines structured data analysis with LLM-powered narrative
    generation to produce professional economic reports.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def generate(
        self,
        data: pd.DataFrame,
        config: ReportConfig | None = None,
        analysis: EmpiricalAnalysis | None = None,
    ) -> GeneratedReport:
        """Generate a report from simulation data.

        Args:
            data: Simulation results DataFrame.
            config: Report configuration.
            analysis: Pre-computed analysis (computed if not provided).

        Returns:
            GeneratedReport with all sections populated.
        """
        config = config or ReportConfig()
        start = time.time()

        # Get template
        template = REPORT_TEMPLATES.get(config.template_name)
        if template is None:
            raise ValueError(
                f"Unknown template: {config.template_name}. "
                f"Available: {list(REPORT_TEMPLATES.keys())}"
            )

        # Analyze data
        if analysis is None:
            analysis = analyze_simulation_data(data)

        # Generate title
        title = config.title or template.name

        # Generate each section
        sections: list[GeneratedSection] = []
        for section_def in template.sections:
            section = self._generate_section(section_def, analysis, config)
            sections.append(section)

        elapsed = time.time() - start

        return GeneratedReport(
            title=title,
            template_name=config.template_name,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            sections=sections,
            metadata={
                "num_periods": len(data),
                "regime": analysis.regime,
                "format": config.format,
            },
            elapsed_seconds=round(elapsed, 2),
        )

    def _generate_section(
        self,
        section_def: ReportSection,
        analysis: EmpiricalAnalysis,
        config: ReportConfig,
    ) -> GeneratedSection:
        """Generate a single report section."""
        # Build data context for this section
        data_summary = self._build_data_context(section_def, analysis)

        # Build chart data if applicable
        chart_data = {}
        if config.include_charts and section_def.required_data:
            chart_data = self._extract_chart_data(section_def, analysis)

        # Generate content via LLM or fallback to data-driven narrative
        if self.llm_client is not None:
            content = self._llm_generate(section_def, data_summary, analysis, config)
        else:
            content = self._rule_based_generate(section_def, data_summary, analysis)

        return GeneratedSection(
            id=section_def.id,
            title=section_def.title,
            content=content,
            data_summary=data_summary,
            chart_data=chart_data,
        )

    def _build_data_context(
        self, section_def: ReportSection, analysis: EmpiricalAnalysis
    ) -> str:
        """Build a data context string for the section."""
        lines: list[str] = []

        # Add relevant moments
        if analysis.moments:
            lines.append("Key Statistics:")
            for name, val in analysis.moments.items():
                lines.append(f"  {name}: {val:.4f}")

        # Add relevant trends
        if analysis.trends:
            lines.append("Trends:")
            for var, trend in analysis.trends.items():
                if not section_def.required_data or var in section_def.required_data:
                    lines.append(f"  {var}: {trend}")

        # Add regime
        lines.append(f"Economic regime: {analysis.regime}")

        # Add events
        if analysis.key_events:
            lines.append("Notable events:")
            for event in analysis.key_events:
                lines.append(f"  - {event}")

        return "\n".join(lines) if lines else "No data context available."

    def _extract_chart_data(
        self, section_def: ReportSection, analysis: EmpiricalAnalysis
    ) -> dict[str, Any]:
        """Extract chart-ready data for the section."""
        chart_data: dict[str, Any] = {}

        for var in section_def.required_data:
            if var in analysis.data.columns:
                series = analysis.data[var].dropna()
                chart_data[var] = {
                    "values": series.tolist(),
                    "mean": float(series.mean()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }

        return chart_data

    def _llm_generate(
        self,
        section_def: ReportSection,
        data_summary: str,
        analysis: EmpiricalAnalysis,
        config: ReportConfig,
    ) -> str:
        """Generate section content using LLM."""
        prompt = f"""You are writing the "{section_def.title}" section of an economic analysis report.

{section_def.llm_prompt}

Here is the data to base your analysis on:

{data_summary}

Additional context:
- Economic regime: {analysis.regime}
- Number of periods analyzed: {len(analysis.data)}

Write approximately {config.max_section_words} words. Use markdown formatting.
Be specific, data-driven, and actionable. Cite numbers from the data."""

        system = (
            "You are a senior economic analyst writing a professional report. "
            "Be data-driven, concise, and actionable. Use markdown formatting."
        )

        try:
            response = self.llm_client.complete(prompt, system=system)
            return response.content
        except Exception as e:
            logger.warning(f"LLM generation failed for {section_def.id}: {e}")
            return self._rule_based_generate(section_def, data_summary, analysis)

    def _rule_based_generate(
        self,
        section_def: ReportSection,
        data_summary: str,
        analysis: EmpiricalAnalysis,
    ) -> str:
        """Generate section content using rule-based approach (no LLM)."""
        lines: list[str] = []

        if section_def.id == "executive_summary":
            lines.append(f"The economy is currently in a **{analysis.regime}** regime.")
            if "mean_gdp_growth" in analysis.moments:
                growth = analysis.moments["mean_gdp_growth"]
                lines.append(f"Average GDP growth: **{growth:.2%}** per period.")
            if "mean_unemployment" in analysis.moments:
                unemp = analysis.moments["mean_unemployment"]
                lines.append(f"Average unemployment: **{unemp:.1%}**.")
            if "mean_inflation" in analysis.moments:
                infl = analysis.moments["mean_inflation"]
                lines.append(f"Average inflation: **{infl:.2%}** per period.")
            if analysis.key_events:
                lines.append("\n**Notable events:**")
                for event in analysis.key_events:
                    lines.append(f"- {event}")

        elif section_def.id == "risks_outlook":
            lines.append("**Key Risks:**")
            if analysis.regime in ("recession", "crisis"):
                lines.append("- Economy is in downturn — risk of further contraction")
            if "mean_unemployment" in analysis.moments and analysis.moments["mean_unemployment"] > 0.08:
                lines.append("- Elevated unemployment poses social and fiscal risks")
            if "mean_inflation" in analysis.moments and analysis.moments["mean_inflation"] > 0.03:
                lines.append("- Inflationary pressures may require tighter policy")
            if "mean_inflation" in analysis.moments and analysis.moments["mean_inflation"] < -0.01:
                lines.append("- Deflationary pressure risks a demand spiral")
            if not lines[1:]:
                lines.append("- No significant risks identified under current conditions")

        else:
            lines.append(f"**{section_def.title}**")
            lines.append("")
            lines.append(data_summary)

        return "\n".join(lines)
