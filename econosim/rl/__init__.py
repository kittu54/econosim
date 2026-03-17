"""RL environment interface and Gymnasium-compatible environments.

Gymnasium-dependent environments are lazily imported to avoid hard
dependency on gymnasium for non-RL users.
"""

from econosim.rl.env import EconEnvInterface

# MacroEnv does not require gymnasium
from econosim.rl.macro_env import MacroEnv


def register_gymnasium_envs() -> None:
    """Register all EconoSim environments with Gymnasium.

    Call this function before using gym.make("EconoSim-Firm-v0") etc.
    Requires gymnasium to be installed.
    """
    import econosim.rl.registration  # noqa: F401


# Auto-register if gymnasium is available (no-op if not installed)
try:
    register_gymnasium_envs()
except ImportError:
    pass


def __getattr__(name: str):
    """Lazy import for gymnasium-dependent environments."""
    _gym_names = {
        "FirmEnv": "econosim.rl.firm_env",
        "HouseholdEnv": "econosim.rl.household_env",
        "GovernmentEnv": "econosim.rl.government_env",
        "BankEnv": "econosim.rl.bank_env",
    }
    _wrapper_names = {
        "NormalizeObservation": "econosim.rl.wrappers",
        "NormalizeReward": "econosim.rl.wrappers",
        "ScaleReward": "econosim.rl.wrappers",
        "ClipAction": "econosim.rl.wrappers",
        "RecordEpisodeMetrics": "econosim.rl.wrappers",
    }

    if name in _gym_names:
        import importlib
        mod = importlib.import_module(_gym_names[name])
        return getattr(mod, name)

    if name in _wrapper_names:
        import importlib
        mod = importlib.import_module(_wrapper_names[name])
        return getattr(mod, name)

    raise AttributeError(f"module 'econosim.rl' has no attribute {name!r}")


__all__ = [
    "EconEnvInterface",
    "MacroEnv",
    "register_gymnasium_envs",
    "FirmEnv",
    "HouseholdEnv",
    "GovernmentEnv",
    "BankEnv",
    "NormalizeObservation",
    "NormalizeReward",
    "ScaleReward",
    "ClipAction",
    "RecordEpisodeMetrics",
]
