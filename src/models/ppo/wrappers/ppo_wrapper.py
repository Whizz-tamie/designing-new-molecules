from typing import Optional, Tuple

import torch as th
from stable_baselines3.common.distributions import (
    CategoricalDistribution,
    Distribution,
    make_proba_distribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        # Initialize the action mask attribute
        self.action_mask = None
        self.mask_dim = action_space.n  # The number of discrete actions
        self.obs_dim = (
            observation_space.shape[0] - self.mask_dim
        )  # the original observation size without the mask

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Split observations and action masks
        actual_obs = obs[:, : self.obs_dim]
        self.action_mask = obs[:, self.obs_dim :]

        # Preprocess the observation if needed
        features = self.extract_features(actual_obs)

        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            if self.action_mask is not None:
                mean_actions += (1 - self.action_mask) * (-1e9)  # Apply mask
            return self.action_dist.proba_distribution(action_logits=mean_actions)

        else:
            raise ValueError("Invalid action distribution")

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Split observations and action masks
        actual_obs = obs[:, : self.obs_dim]
        self.action_mask = obs[:, self.obs_dim :]

        # Preprocess the observation if needed
        features = self.extract_features(actual_obs)

        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        # Split observations and action masks
        actual_obs = obs[:, : self.obs_dim]
        self.action_mask = obs[:, self.obs_dim :]

        features = super().extract_features(actual_obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """

        return self.get_distribution(observation).get_actions(
            deterministic=deterministic
        )

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        # Split observations and action masks
        actual_obs = obs[:, : self.obs_dim]
        self.action_mask = obs[:, self.obs_dim :]

        features = super().extract_features(actual_obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)
