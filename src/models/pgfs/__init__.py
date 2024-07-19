import gymnasium as gym
from gymnasium.envs.registration import register

if "MoleculeDesign-v0" not in gym.envs.registry:
    register(
        id="MoleculeDesign-v0",
        entry_point="src.models.pgfs.envs:MoleculeDesignEnv",
        max_episode_steps=5,  # environment's typical episode length
    )
