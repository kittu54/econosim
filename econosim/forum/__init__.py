"""Agent collaboration forum for EconoSim.

Multiple specialist analysis agents (macro, labor, financial, policy,
risk) discuss simulation results and synthesize findings through a
moderated forum mechanism.
"""

from econosim.forum.engine import ForumEngine, ForumConfig, ForumSession

__all__ = ["ForumEngine", "ForumConfig", "ForumSession"]
