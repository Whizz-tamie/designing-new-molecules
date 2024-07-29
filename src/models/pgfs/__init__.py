import logging

import gymnasium as gym
from gymnasium.envs.registration import register

logger = logging.getLogger(__name__)

if "MoleculeDesign-v0" not in gym.envs.registry:
    register(
        id="MoleculeDesign-v0",
        entry_point="src.models.pgfs.envs:MoleculeDesignEnv",
        max_episode_steps=5,  # environment's typical episode length
    )

    logger.info("Registered environment: MoleculeDesign-v0")
