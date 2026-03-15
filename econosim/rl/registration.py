"""Register EconoSim environments with Gymnasium."""

from gymnasium.envs.registration import register

register(
    id="EconoSim-Firm-v0",
    entry_point="econosim.rl.firm_env:FirmEnv",
    max_episode_steps=120,
)

register(
    id="EconoSim-Household-v0",
    entry_point="econosim.rl.household_env:HouseholdEnv",
    max_episode_steps=120,
)

register(
    id="EconoSim-Government-v0",
    entry_point="econosim.rl.government_env:GovernmentEnv",
    max_episode_steps=120,
)

register(
    id="EconoSim-Bank-v0",
    entry_point="econosim.rl.bank_env:BankEnv",
    max_episode_steps=120,
)
