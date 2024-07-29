from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from src.models.ppo.train.ppo import CustomWandbCallback


class SimpleMockEnv(gym.Env):
    def __init__(self):
        super(SimpleMockEnv, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.current_step = 0

    def reset(self, seed=None, options=None):
        self.current_step = 0
        if seed is not None:
            np.random.seed(seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        self.current_step += 1
        reward = 1.0
        terminated = self.current_step > 10  # End episode after 10 steps
        truncated = False  # Placeholder for truncation condition
        return self.observation_space.sample(), reward, terminated, truncated, {}

    def render(self, mode="human"):
        pass


@pytest.fixture(scope="module")
def setup_env():
    env = SimpleMockEnv()
    check_env(env)
    yield env
    env.close()


@pytest.fixture(scope="module")
def setup_model(setup_env):
    env = DummyVecEnv([lambda: setup_env])
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
    )
    yield model
    env.close()


@patch("wandb.log")
def test_wandb_logging(mock_wandb_log, setup_model):
    callback = CustomWandbCallback()
    setup_model.learn(total_timesteps=50, callback=callback)

    # Check if wandb.log was called
    assert mock_wandb_log.called

    # Check if specific metrics are logged
    logged_metrics = [call[0][0] for call in mock_wandb_log.call_args_list]
    print(logged_metrics)  # Debug output to understand what metrics are being logged

    # Assertions to check the presence of specific keys in logged metrics
    assert any(
        "step_reward" in metrics for metrics in logged_metrics
    ), "step_reward not logged"
    assert any(
        "step_qed" in metrics for metrics in logged_metrics
    ), "step_qed not logged"
    assert any("episode" in metrics for metrics in logged_metrics), "episode not logged"


if __name__ == "__main__":
    pytest.main()
