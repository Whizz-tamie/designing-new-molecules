import glob
import logging
import os

import gymnasium as gym
import numpy as np
import torch
import yaml

import wandb
from src.models.pgfs.train.replay_buffer import ReplayBuffer
from src.models.pgfs.train.td3_agent import TD3Agent
from src.models.pgfs.wrappers.faiss_new import KNNWrapper

# Configure the logger
logger = logging.getLogger(__name__)


def initialize_seed(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    logger.info("Random seed initialized with value: %s", seed_value)


def load_config(config_file):
    """Load a YAML configuration file."""
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        logger.info("Configuration loaded successfully from %s", config_file)
        return config
    except FileNotFoundError:
        logger.error("Configuration file not found: %s", config_file)
        raise
    except yaml.YAMLError as exc:
        logger.error("Error parsing the configuration file: %s", exc)
        raise


def setup_wandb(config, experiment_name):
    """Initialize Weights & Biases for experiment tracking."""
    try:
        wandb.init(
            project=config["project"],
            entity=config["entity"],
            notes="RL molecule search baseline",
            tags=["pgfs", "td3", "QED"],
            config=config["hyperparameters"],
            name=experiment_name,
            job_type="training",
            save_code=True,
            id=config["id"],
            resume="allow",
        )
        logger.info("Weights & Biases initialized for project: %s", config["project"])
    except Exception as e:
        logger.error("Failed to initialize Weights & Biases: %s", e)
        raise


def initialize_env(config, reactant_file, template_file):
    try:
        env = gym.make(
            "MoleculeDesign-v0",
            reactant_file=config["dataset"][reactant_file],
            template_file=config["dataset"][template_file],
        )
        env = KNNWrapper(env)
        logger.info("Environment initialized successfully.")
        return env
    except Exception as e:
        logger.error("Failed to initialize environment: %s", e)
        raise


def initialize_agent(env, config):
    try:
        agent = TD3Agent(
            env,
            float(config["optimizers"]["actor_lr"]),
            float(config["optimizers"]["critic_lr"]),
            config["training"]["gamma"],
            config["training"]["tau"],
            config["noise"]["policy_noise"],
            config["noise"]["noise_std"],
            config["noise"]["noise_clip"],
            config["training"]["policy_freq"],
            config["temperature"]["initial_temperature"],
            config["temperature"]["min_temperature"],
            config["training"]["start_timesteps"],
            config["training"]["max_timesteps"],
        )
        logger.info("Agent initialized successfully.")
        return agent
    except Exception as e:
        logger.error("Failed to initialize agent: %s", e)
        raise


def initialize_components(config):
    """Modular initialization for better error handling"""
    config_params = config["hyperparameters"]
    env = initialize_env(config_params, "training_file", "templates_file")
    eval_env = initialize_env(config_params, "validation_file", "templates_file")
    state_dim = env.unwrapped.observation_space.shape[0]
    template_dim = env.unwrapped.action_space.n
    r2_vec_dim = state_dim
    replay_buffer = ReplayBuffer(
        state_dim, template_dim, r2_vec_dim, int(config_params["training"]["capacity"])
    )
    agent = initialize_agent(env, config_params)
    return env, eval_env, replay_buffer, agent


def find_latest_checkpoint(checkpoint_dir):
    """Finds the latest checkpoint in the specified directory."""
    try:
        list_of_files = glob.glob(
            os.path.join(checkpoint_dir, "*.pth")
        )  # Adjust the pattern as necessary
        if not list_of_files:  # No files found
            logger.info("No checkpoint files found in directory: %s", checkpoint_dir)
            return None
        latest_file = max(list_of_files, key=os.path.getctime)
        logger.info("Latest checkpoint file found: %s", latest_file)
        return latest_file
    except Exception as e:
        logger.error("Failed to find the latest checkpoint: %s", e)
        raise
