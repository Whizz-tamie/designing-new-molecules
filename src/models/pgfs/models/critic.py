# models/critic.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    def __init__(self, state_size, template_size, action_size):
        """
        Initialize the CriticNetwork with fully connected layers.

        Args:
            state_size (int): Dimension of state input features.
            template_size (int): Dimension of template input features.
            action_size (int): Dimension of action input features.
        
        Task:
            The critic (Q network) evaluates the state-action pair to output Q(s, a), which is the quality of the action taken in the given state.
        """
        super(CriticNetwork, self).__init__()
        input_size = state_size + template_size + action_size
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, state, template, action):
        """
        Forward pass through the CriticNetwork.

        Args:
            state (torch.Tensor): State input tensor.
            template (torch.Tensor): Template input tensor.
            action (torch.Tensor): Action input tensor.

        Returns:
            torch.Tensor: Q-value.
        
        Task:
            The forward method combines the state, template, and action to produce a quality score (Q-value) that indicates how good the action is in the given state.
        """
        # Concatenate the state, template, and action tensors
        combined_input = torch.cat([state, template, action], dim=-1)
        
        # Pass the combined input through the first fully connected layer followed by a ReLU activation
        combined_input = F.relu(self.fc1(combined_input))
        
        # Pass the output through the second fully connected layer followed by a ReLU activation
        combined_input = F.relu(self.fc2(combined_input))
        
        # Pass the output through the third fully connected layer followed by a ReLU activation
        combined_input = F.relu(self.fc3(combined_input))
        
        # Pass the output through the fourth fully connected layer to produce the Q-value
        q_value = self.fc4(combined_input)
        
        return q_value
