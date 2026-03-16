"""
Phase 4 — Advanced Economic Extensions.

Modular extensions that build on the core EconoSim simulation:
- multi_sector: Multiple goods/sectors with input-output matrices
- skilled_labor: Labor skill differentiation and wage dispersion
- bonds: Bond markets and government debt issuance
- expectations: Adaptive expectations and learning dynamics
- networks: Network effects (trade/credit graphs)
"""

from econosim.extensions.multi_sector import (
    Good,
    Sector,
    InputOutputMatrix,
)
from econosim.extensions.skilled_labor import (
    SkillLevel,
    SkilledHousehold,
    SkilledFirm,
)
from econosim.extensions.bonds import (
    Bond,
    BondMarket,
    GovernmentDebtManager,
)
from econosim.extensions.expectations import (
    ExpectationModel,
    AdaptiveExpectations,
    RollingExpectations,
    WeightedExpectations,
)
from econosim.extensions.networks import (
    EconomicNetwork,
    TradeNetwork,
    CreditNetwork,
)

__all__ = [
    "Good", "Sector", "InputOutputMatrix",
    "SkillLevel", "SkilledHousehold", "SkilledFirm",
    "Bond", "BondMarket", "GovernmentDebtManager",
    "ExpectationModel", "AdaptiveExpectations", "RollingExpectations",
    "WeightedExpectations",
    "EconomicNetwork", "TradeNetwork", "CreditNetwork",
]
