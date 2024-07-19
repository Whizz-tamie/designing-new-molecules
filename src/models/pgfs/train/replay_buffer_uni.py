#replay_buffer_uni.py

import torch
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size):
        """
        Initialize the replay buffer.

        Args:
            max_size (int): Maximum number of transitions the buffer can hold.
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=self.max_size)

    def add(self, state, state_tmask, template, next_state, next_tmask, reward, done):
        """
        Add a transition to the buffer.

        Args:
            state (torch.Tensor): The current state.
            template (torch.Tensor): The selected template.
            action (torch.Tensor): The action taken.
            reward (float): The reward received.
            next_state (torch.Tensor): The next state after the action.
            next_tmask (torch.Tensor): The next state template mask
            done (bool): Whether the episode is done.
        """
        experience = (
            state.clone().detach() if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32),
            state_tmask.clone().detach() if isinstance(state_tmask, torch.Tensor) else torch.tensor(state_tmask, dtype=torch.float32),
            template.clone().detach() if isinstance(template, torch.Tensor) else torch.tensor(template, dtype=torch.float32),
            #action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.float32),
            next_state.clone().detach() if isinstance(next_state, torch.Tensor) else torch.tensor(next_state, dtype=torch.float32),
            next_tmask.clone().detach() if isinstance(next_tmask, torch.Tensor) else torch.tensor(next_tmask, dtype=torch.float32),
            reward.clone().detach() if isinstance(reward, torch.Tensor) else torch.tensor(reward, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32)
        )
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.
        Returns:
            tuple: Batches of states, template, actions, rewards, next states, and done flags.
        """
        batch = random.sample(self.buffer, batch_size)
        state, state_tmask, template, next_state, next_tmask, reward, done = map(torch.stack, zip(*batch))
        return state.squeeze(1), state_tmask.squeeze(1), template.squeeze(1), next_state.squeeze(1), next_tmask.squeeze(1), reward.unsqueeze(1), done.unsqueeze(1)

    def size(self):
        """
        Return the current size of the buffer.

        Returns:
            int: Number of transitions currently stored in the buffer.
        """
        return len(self.buffer)
