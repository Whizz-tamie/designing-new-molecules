# td3_agent.py

import logging
import math
import pickle
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.models.pgfs.models.model import ActorNetwork, CriticNetwork

# Configure the logger
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3Agent:

    def __init__(
        self,
        env,
        actor_lr=1e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_std=0.1,
        noise_clip=0.2,
        policy_freq=2,
        temperature_start=1.0,
        temperature_end=0.1,
        start_timesteps=3000,
        max_timesteps=1000000,
    ):
        self.env = env
        self.state_dim = env.unwrapped.observation_space.shape[0]
        self.template_dim = env.unwrapped.action_space.n

        # Continuous action space for second reactant
        self.action_dim = env.unwrapped.observation_space.shape[0]

        self._initialize_networks(actor_lr, critic_lr)
        self._initialize_training_parameters(
            gamma,
            tau,
            policy_noise,
            noise_std,
            noise_clip,
            policy_freq,
            temperature_start,
            temperature_end,
            start_timesteps,
            max_timesteps,
        )
        logger.info("TD3Agent initialized with environment and hyperparameters.")

    def _initialize_networks(self, actor_lr, critic_lr):
        self.actor = ActorNetwork(
            self.state_dim, self.template_dim, self.action_dim
        ).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1 = CriticNetwork(
            self.state_dim, self.template_dim, self.action_dim
        ).to(device)
        self.critic2 = CriticNetwork(
            self.state_dim, self.template_dim, self.action_dim
        ).to(device)
        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr,
        )
        logger.info("Networks and optimizers initialized.")

    def _initialize_training_parameters(
        self,
        gamma,
        tau,
        policy_noise,
        noise_std,
        noise_clip,
        policy_freq,
        temperature_start,
        temperature_end,
        start_timesteps,
        max_timesteps,
    ):
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.temperature = temperature_start
        self.temperature_end = temperature_end
        self.temperature_decay = math.pow(
            (temperature_end / temperature_start),
            (1 / (max_timesteps - start_timesteps)),
        )
        self.max_action = self.env.unwrapped.observation_space.high[0]
        self.total_it = 0
        self.actor_loss = None
        # self.fnet_loss = None
        self.critic_loss = None
        logger.info(
            "Training parameters set... - temp_decay: %s", self.temperature_decay
        )

    def get_action(self, state, evaluate=False):
        logger.debug("Policy searching for action...")
        # Convert the state to a torch tensor and ensure it's on the right device
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        # Obtain template mask info from the environment's current state if available
        if hasattr(self.env.unwrapped, "reaction_manager") and hasattr(
            self.env.unwrapped, "current_state"
        ):
            template_mask = self.env.unwrapped.reaction_manager.get_mask(
                self.env.unwrapped.current_state
            )
            template_types = self.env.unwrapped.reaction_manager.template_types
            template_mask_info = (
                template_mask.unsqueeze(0).to(device),
                template_types.to(device),
            )
        if evaluate:
            self.actor.eval()  # Set to evaluation mode for consistent behavior
        else:
            self.actor.train()  # Ensure it is in training mode

        with torch.no_grad():
            template, r2_vector = self.actor(
                state, template_mask_info, self.temperature, evaluate=evaluate
            )
        if torch.any(r2_vector != 0) and not evaluate:
            noise = torch.randn_like(r2_vector) * self.noise_std
            r2_vector += noise
        logger.info(
            "Policy selected - Template: %s, R2 Vectors: %s (Evaluate: %s)",
            template.shape,
            r2_vector.shape,
            evaluate,
        )

        return (template, r2_vector)

    def train(self, replay_buffer, batch_size=32):
        """Train method to update the actor and critic networks."""
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

        self.total_it += 1
        logger.debug("Training iteration %s...", self.total_it)

        # Sample replay buffer
        (
            state_smiles,
            state_obs,
            templates,
            r2_vectors,
            rewards,
            next_state_smiles,
            next_state_obs,
            not_dones,
        ) = replay_buffer.sample(batch_size)

        # Prepare masks and access template types
        logger.debug(
            "Getting template masks for states and next states in sampled batch..."
        )
        batch_masks = [
            self.env.unwrapped.reaction_manager.get_mask(smile)
            for smile in state_smiles
        ]
        next_batch_masks = [
            self.env.unwrapped.reaction_manager.get_mask(smile)
            for smile in next_state_smiles
        ]
        template_types = self.env.unwrapped.reaction_manager.template_types
        template_types.to(device)

        # Convert list of masks to a tensor and transfer to device
        batch_masks = torch.stack(batch_masks).to(device)
        next_batch_masks = torch.stack(next_batch_masks).to(device)

        masks_info = (batch_masks, template_types)
        next_masks_info = (next_batch_masks, template_types)

        logger.debug(
            "Batch training data shapes - states: %s, templates: %s, r2_vectors: %s, rewards: %s, next_states: %s, not_dones: %s, batch_masks: %s, next_batch_masks: %s",
            state_obs.shape,
            templates.shape,
            r2_vectors.shape,
            rewards.shape,
            next_state_obs.shape,
            not_dones.shape,
            batch_masks.shape,
            next_batch_masks.shape,
        )

        # Update Critic
        # Get the next action from the target actor model
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(r2_vectors) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            # Predict next actions using the target actor model
            next_templates, next_r2_vectors = self.actor_target(
                next_state_obs, next_masks_info, self.temperature
            )
            # Apply noise to next r2_vectors for exploration
            next_r2_vectors = (next_r2_vectors + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q-value
            target_Q1 = self.critic1_target(
                next_state_obs, next_templates, next_r2_vectors
            )
            target_Q2 = self.critic2_target(
                next_state_obs, next_templates, next_r2_vectors
            )
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + not_dones * self.gamma * target_Q

        # Get current Q-value estimates
        current_Q1 = self.critic1(state_obs, templates, r2_vectors)
        current_Q2 = self.critic2(state_obs, templates, r2_vectors)

        # Compute critic loss
        self.critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critics
        self.critic_optimizer.zero_grad()
        self.critic_loss.backward()
        self.critic_optimizer.step()

        logger.info(
            "Target Q-value: %s, Curent Q-values: %s, Critic loss: %s",
            target_Q.mean().item(),
            current_Q1.mean().item(),
            self.critic_loss.item(),
        )

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            logger.debug(
                "Delayed actor update and target_actor soft update for iteration %s...",
                self.total_it,
            )

            c_templates, c_r2_vectors = self.actor(
                state_obs, masks_info, self.temperature
            )

            # Compute actor loss
            actor_loss = -self.critic1(state_obs, c_templates, c_r2_vectors).mean()

            # Compute cross-entropy loss for f network output and templates

            # logits = self.actor.logits
            # target_template = torch.argmax(c_templates, dim=-1)
            # logger.debug(
            #   "Logits: %s, target_template: %s", logits.shape, target_template.shape
            # )
            # self.fnet_loss = F.cross_entropy(logits, target_template)
            self.actor_loss = actor_loss  # No side losses

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            self._update_target(self.critic1, self.critic1_target, self.tau)
            self._update_target(self.critic2, self.critic2_target, self.tau)
            self._update_target(self.actor, self.actor_target, self.tau)

            logger.info(
                "Actor loss: %s, FNet loss: %s",
                self.actor_loss.item(),
                self.fnet_loss.item(),
            )

        # Log or return values
        return {
            "total_iterations": self.total_it,
            "critic_loss": self.critic_loss.item(),
            # "fNet_loss": (
            # self.fnet_loss if self.fnet_loss is None else self.fnet_loss.item()
            # ),
            "actor_loss": (
                self.actor_loss if self.actor_loss is None else self.actor_loss.item()
            ),
            "current_q_values": current_Q1.mean().item(),
            "target_q_values": target_Q.mean().item(),
            "temperature": self.temperature,
        }

    def _update_target(self, source, target, tau):
        for target_param, param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_model(self, filename, steps_done, episode_count, replay_buffer):
        state_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic1_target_state_dict": self.critic1_target.state_dict(),
            "critic2_target_state_dict": self.critic2_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "total_it": self.total_it,
            "temperature": self.temperature,
            # "fnet_loss": self.fnet_loss.item() if self.fnet_loss else None,
            "actor_loss": self.actor_loss.item() if self.actor_loss else None,
            "critic_loss": self.critic_loss.item() if self.critic_loss else None,
            "steps_done": steps_done,
            "episode_count": episode_count,
            "replay_buffer": replay_buffer,
        }

        torch.save(state_dict, filename, pickle_protocol=5)
        logger.info("Model saved to: %s", filename)

    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
        self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target_state_dict"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.total_it = checkpoint.get("total_it", 0)
        self.temperature = checkpoint.get("temperature", 1.0)
        # self.fnet_loss = checkpoint.get("fnet_loss", None)
        self.actor_loss = checkpoint.get("actor_loss", None)
        self.critic_loss = checkpoint.get("critic_loss", None)
        steps_done = checkpoint.get("steps_done", 0)
        episode_count = checkpoint.get("episode_count", 0)
        replay_buffer = checkpoint["replay_buffer"]

        logger.info("Model and training state loaded from %s", filename)
        return steps_done, episode_count, replay_buffer
