"""Report generation engine for EconoSim.

Produces professional economic analysis reports from simulation data,
with support for HTML, Markdown, and JSON output formats.
"""

from econosim.reports.engine import ReportEngine, ReportConfig
from econosim.reports.templates import ReportTemplate, REPORT_TEMPLATES

__all__ = [
    "ReportEngine",
    "ReportConfig",
    "ReportTemplate",
    "REPORT_TEMPLATES",
]
