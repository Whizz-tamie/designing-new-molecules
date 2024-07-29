import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import (
    ActorCriticPolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomFeaturesExtractor, self).__init__(
            observation_space, features_dim=features_dim
        )
        input_dim = observation_space["obs"].shape[0]
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.extractor(observations["obs"])


class CustomPolicy(MultiInputActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = CustomFeaturesExtractor(self.observation_space)

    def forward(self, obs, deterministic=False):
        # Extract features from the observation
        features = self.mlp_extractor(obs)

        # Compute action logits and value estimates
        action_logits = self.action_net(features)
        value = self.value_net(features)

        # Apply the action mask
        valid_actions = obs["valid_actions"]
        action_logits = th.where(
            valid_actions, action_logits, th.tensor(-1e6).to(action_logits.device)
        )

        # Compute action probabilities using softmax
        action_probs = F.softmax(action_logits, dim=-1)

        # Select action
        if deterministic:
            action = action_probs.argmax(dim=1)
        else:
            action_distribution = th.distributions.Categorical(action_probs)
            action = action_distribution.sample()

        return action, action_logits, value
