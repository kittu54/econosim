"""Register EconoSim environments with Gymnasium."""

from gymnasium.envs.registration import register

register(
    id="EconoSim-Firm-v0",
    entry_point="econosim.rl.firm_env:FirmEnv",
    max_episode_steps=120,
)
