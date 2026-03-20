"""Natural language interface for EconoSim.

Users describe economic scenarios and questions in plain English;
the system translates them into simulation configurations and
generates analysis.
"""

from econosim.nl.interpreter import NLInterpreter, InterpretedQuery

__all__ = ["NLInterpreter", "InterpretedQuery"]
