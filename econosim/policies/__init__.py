"""Policy interfaces for agent behavior abstraction.

Policies define how agents make decisions. The simulation core calls
policy.act(...) to get actions, enabling clean separation of:
1. Rule-based baseline policies
2. Econometric/statistical policies
3. Learned RL policies
4. Transformer-backed policies
"""

from econosim.policies.interfaces import (
    FirmPolicy,
    HouseholdPolicy,
    BankPolicy,
    GovernmentPolicy,
    FirmAction,
    HouseholdAction,
    BankAction,
    GovernmentAction,
)
from econosim.policies.rule_based import (
    RuleBasedFirmPolicy,
    RuleBasedHouseholdPolicy,
    RuleBasedBankPolicy,
    RuleBasedGovernmentPolicy,
)

__all__ = [
    "FirmPolicy", "HouseholdPolicy", "BankPolicy", "GovernmentPolicy",
    "FirmAction", "HouseholdAction", "BankAction", "GovernmentAction",
    "RuleBasedFirmPolicy", "RuleBasedHouseholdPolicy",
    "RuleBasedBankPolicy", "RuleBasedGovernmentPolicy",
]
