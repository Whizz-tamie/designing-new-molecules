# models/td3.py

import src.models.pgfs.logging_config as logging_config
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.models.pgfs.models.actor import FNetwork, PiNetwork, ActorProcedure
from src.models.pgfs.models.critic import CriticNetwork
from src.models.pgfs.train.replay_buffer import ReplayBuffer
import wandb
import os

class TD3:
    def __init__(self, state_dim, template_dim, action_dim, max_action, gamma=0.99, tau=0.005, policy_freq=2,
                    temperature=1.0, noise_std=0.2, noise_clip=0.5, min_temp=0.1, temp_decay=0.99,
                    actor_lr=1e-4, critic_lr=3e-4):
        """
        Initialize the TD3 agent with actor and critic networks, target networks, and optimizers.

        Args:
            state_dim (int): Dimension of state input features.
            template_dim (int): Dimension of template features.
            action_dim (int): Dimension of action space.
            max_action (float): Maximum action value.
            gamma (float): Discount factor.
            tau (float): Soft update rate.
            policy_freq (int): Frequency of policy updates.
            temperature (float): Initial temperature parameter for Gumbel Softmax.
            noise_std (float): Standard deviation of the noise for exploration.
            noise_clip (float): Clipping value for the noise.
            min_temp (float): Minimum temperature for Gumbel Softmax.
            temp_decay (float): Decay factor for temperature.
            actor_lr (float): Learning rate for the actor networks.
            critic_lr (float): Learning rate for the critic networks.
        """
        logging_config.setup_logging() # Set up logging here
        self.logger = logging.getLogger(__name__)
        self.logger.info("TD3 instance created")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.template_dim = template_dim
        self.action_dim = action_dim

        self.actor_procedure = ActorProcedure(state_dim, template_dim, action_dim).to(self.device)
        self.actor_procedure_target = ActorProcedure(state_dim, template_dim, action_dim).to(self.device)
        self.actor_procedure_target.load_state_dict(self.actor_procedure.state_dict())

        self.critic1 = CriticNetwork(state_dim, template_dim, action_dim).to(self.device)
        self.critic2 = CriticNetwork(state_dim, template_dim, action_dim).to(self.device)
        self.critic1_target = CriticNetwork(state_dim, template_dim, action_dim).to(self.device)
        self.critic2_target = CriticNetwork(state_dim, template_dim, action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor_procedure.parameters(), actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_freq = policy_freq
        self.temperature = temperature
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.min_temp = min_temp
        self.temp_decay = temp_decay
        self.current_time_step = 0
        self.total_it = 0

        self.critic1_loss = None
        self.critic2_loss = None
        self.actor_loss = None
        self.f_net_loss = None
        self.target_q = None
        self.current_q1 = None
        self.current_q2 = None
        
        self.episode_rewards = []  # Store rewards for each episode

    def select_action(self, state, T_mask):
        """
        Select an action using the actor networks.

        Args:
            state (torch.Tensor): Current state.
        Returns:
            torch.Tensor: Selected template.
            torch.Tensor: Selected action.
        """
        state = state.to(self.device)
        T_mask = T_mask.to(self.device)
        template, action = self.actor_procedure(state, self.temperature, T_mask)
        self.logger.debug(f"Template and action selected for state!")
        return template, action
    
    def backward(self, replay_buffer, batch_size):
        """
        Perform a training step using a minibatch from the replay buffer.

        Args:
            replay_buffer (ReplayBuffer): Replay buffer containing past experiences.
            batch_size (int): Number of transitions to sample for training.
        """
        self.logger.debug("Starting backward pass...")
        self.total_it += 1

        # Sample a batch of transitions from the replay buffer
        state, state_tmask, template, action, next_state, next_tmask, reward, done = replay_buffer.sample(batch_size)
        
        state = state.to(self.device)
        state_tmask = state_tmask.to(self.device)
        template = template.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        next_tmask = next_tmask.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            next_template, next_action = self.actor_procedure_target(next_state, self.temperature, next_tmask)
            noise = (torch.randn_like(next_action) * self.noise_std).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Double Q-learning: min of the two target Q-values
            target_q1 = self.critic1_target(next_state, next_template, next_action)
            target_q2 = self.critic2_target(next_state, next_template, next_action)
            target_q = torch.min(target_q1, target_q2)
            self.target_q = reward + ((1 - done) * self.gamma * target_q).detach()

        # Get current Q-values estimates
        self.current_q1 = self.critic1(state, template, action)
        self.current_q2 = self.critic2(state, template, action)
        
        # Compute critic loss
        self.critic1_loss = nn.MSELoss()(self.current_q1, self.target_q.detach())
        self.critic2_loss = nn.MSELoss()(self.current_q2, self.target_q.detach())

        # Update critic networks
        self.critic1_optimizer.zero_grad()
        self.critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        self.critic2_loss.backward()
        self.critic2_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            # Freeze Critic networks
            for p in self.critic1.parameters():
                p.requires_grad = False
            for p in self.critic2.parameters():
                p.requires_grad = False

            # Compute actor loss
            template, action = self.actor_procedure(state, self.temperature, state_tmask)
            self.actor_loss = -self.critic1(state, template, action).mean()

            # Add cross-entropy loss for f network output and templates
            target_templates = torch.argmax(template, dim=1)
            self.f_net_loss = nn.CrossEntropyLoss()(self.actor_procedure.f_net(state), target_templates)
            self.actor_loss += self.f_net_loss

             # Update actor networks
            self.actor_optimizer.zero_grad()
            self.actor_loss.backward()
            self.actor_optimizer.step()

            # Unfreeze Critic networks
            for p in self.critic1.parameters():
                p.requires_grad = True
            for p in self.critic2.parameters():
                p.requires_grad = True

            # Update target networks
            self.soft_update(self.actor_procedure_target, self.actor_procedure)
            self.soft_update(self.critic1_target, self.critic1)
            self.soft_update(self.critic2_target, self.critic2)

        # Decay the temperature
        self.temperature = max(self.min_temp, self.temperature * self.temp_decay)  # Decay temperature with a factor (e.g., 0.99)       
        self.logger.debug(f"Updated temperature: {self.temperature}")

    def soft_update(self, target, source):
        """
        Soft update the target network parameters.

        Args:
            target (nn.Module): Target network.
            source (nn.Module): Source network.
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename, replay_buffer):
        """
        Save the current state of the model, optimizer, and other relevant information.

        Args:
            filename (str): The file path to save the checkpoint.
            replay_buffer (ReplayBuffer): The replay buffer containing past experiences.
        """
        self.logger.info(f"Saving model to {filename}")
        checkpoint = {
            'actor_state_dict': self.actor_procedure.state_dict(),
            'actor_target_state_dict': self.actor_procedure_target.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'learning_rates': {
            'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
            'critic1_lr': self.critic1_optimizer.param_groups[0]['lr'],
            'critic2_lr': self.critic2_optimizer.param_groups[0]['lr']
            },
            'total_it': self.total_it,
            'gamma': self.gamma,
            'tau': self.tau,
            'policy_freq': self.policy_freq,
            'temperature': self.temperature,
            'noise_std': self.noise_std,
            'noise_clip': self.noise_clip,
            'min_temp': self.min_temp,
            'temp_decay': self.temp_decay,
            'total_it' : self.total_it,
            'current_time_step': self.current_time_step,
            'replay_buffer': {
                'max_size': replay_buffer.max_size,
                'buffer': replay_buffer.buffer
            },
            'f_net_loss': self.f_net_loss.item(),
            'actor_loss': self.actor_loss.item() if self.actor_loss is not None else None,
            'critic1_loss': self.critic1_loss.item() if self.critic1_loss is not None else None,
            'critic2_loss': self.critic2_loss.item() if self.critic2_loss is not None else None,
            'target_q': self.target_q,
            'current_q1': self.current_q1,
            'current_q2': self.current_q2,
            'episode_rewards': self.episode_rewards  # Save the rewards
        }
        torch.save(checkpoint, filename)
        wandb.save(filename, base_path=os.path.dirname(filename))
        self.logger.info("Model saved successfully!!!")


    def load(self, filename, replay_buffer):
        """
        Load the model, optimizer, and other relevant information from a checkpoint.

        Args:
            filename (str): The file path to the checkpoint.
        """
        self.logger.info(f"Loading model from {filename}")
        checkpoint = torch.load(filename)
        self.actor_procedure.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_procedure_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.total_it = checkpoint['total_it']
        self.gamma = checkpoint['gamma']
        self.tau = checkpoint['tau']
        self.policy_freq = checkpoint['policy_freq']
        self.temperature = checkpoint['temperature']
        self.noise_std = checkpoint['noise_std']
        self.noise_clip = checkpoint['noise_clip']
        self.min_temp = checkpoint['min_temp']
        self.temp_decay = checkpoint['temp_decay']
        self.total_it = checkpoint['total_it']
        self.current_time_step = checkpoint['current_time_step']

        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = checkpoint['learning_rates']['actor_lr']
        for param_group in self.critic1_optimizer.param_groups:
            param_group['lr'] = checkpoint['learning_rates']['critic1_lr']
        for param_group in self.critic2_optimizer.param_groups:
            param_group['lr'] = checkpoint['learning_rates']['critic2_lr']
        
        replay_buffer.max_size = checkpoint['replay_buffer']['max_size']
        replay_buffer.buffer = checkpoint['replay_buffer']['buffer']
        self.f_net_loss = checkpoint['f_net_loss']
        self.actor_loss = checkpoint['actor_loss']
        self.critic1_loss = checkpoint['critic1_loss']
        self.critic2_loss = checkpoint['critic2_loss']
        self.target_q = ['target_q']
        self.current_q1 = ['current_q1']
        self.current_q2 = ['current_q2']
        self.episode_rewards = checkpoint['episode_rewards']

        self.logger.info("Model loaded successfully!!!")
        return replay_buffer
        