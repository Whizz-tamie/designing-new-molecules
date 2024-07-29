import gymnasium as gym
from gymnasium.envs.registration import register

if "MoleculeDesign-v1" not in gym.envs.registry:
    register(
        id="MoleculeDesign-v1",
        entry_point="src.models.ppo.envs:MoleculeDesignEnv",
        max_episode_steps=5,  # environment's typical episode length
    )
