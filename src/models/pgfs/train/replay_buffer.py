# replay_buffer.py

import logging
import random
from collections import deque

import numpy as np
import torch

# Configure the logger
logger = logging.getLogger(__name__)


class ReplayBuffer:
    def __init__(self, state_dim, template_dim, r2_vec_dim, capacity=int(1e6)):
        self.capacity = capacity
        self.count = 0  # Tracks current number of elements
        self.index = 0  # Tracks the index to add new data

        # Setup the device for tensor operations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pre-allocate memory
        self.state_smiles = [None] * capacity
        self.state_tensors = torch.zeros((capacity, state_dim), device=self.device)
        self.templates = torch.zeros((capacity, template_dim), device=self.device)
        self.r2_vectors = torch.zeros((capacity, r2_vec_dim), device=self.device)
        self.rewards = torch.zeros((capacity, 1), device=self.device)
        self.next_state_smiles = [None] * capacity
        self.next_state_tensors = torch.zeros((capacity, state_dim), device=self.device)
        self.not_dones = torch.zeros((capacity, 1), device=self.device)

        logger.info(
            "Replay buffer created with capacity: %s, state_dim: %s, template_dim: %s, r2_vec_dim: %s",
            capacity,
            state_dim,
            template_dim,
            r2_vec_dim,
        )

    def add(
        self,
        state_smiles,
        state_obs,
        template,
        r2_vector,
        reward,
        next_state_smiles,
        next_state_obs,
        done,
    ):
        idx = self.index

        # Store data in pre-allocated tensors and lists
        self.state_smiles[idx] = state_smiles
        self.state_tensors[idx] = torch.tensor(
            state_obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self.templates[idx] = template.clone().detach()
        self.r2_vectors[idx] = r2_vector.clone().detach()
        self.rewards[idx] = torch.tensor(
            [reward], dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        self.next_state_smiles[idx] = next_state_smiles
        self.next_state_tensors[idx] = torch.tensor(
            next_state_obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self.not_dones[idx] = torch.tensor(
            [1.0 - done], dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        # Update index and count
        self.index = (idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
        logger.debug("Experience added to the buffer.")

    def sample(self, batch_size):
        if self.count < batch_size:
            logger.error("Attempted to sample more elements than are in the buffer.")
            raise ValueError("Not enough elements in the buffer to sample")

        # indices = random.sample(range(self.count), batch_size)
        indices = np.random.randint(0, self.count, size=batch_size)

        # Return a tuple of samples moved to the appropriate device
        return (
            [self.state_smiles[i] for i in indices],
            self.state_tensors[indices],
            self.templates[indices],
            self.r2_vectors[indices],
            self.rewards[indices],
            [self.next_state_smiles[i] for i in indices],
            self.next_state_tensors[indices],
            self.not_dones[indices],
        )

    def size(self):
        return self.count

    def clear(self):
        self.count = 0
        self.index = 0
        logger.info("Replay buffer cleared.")
