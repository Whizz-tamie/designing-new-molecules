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

        # Weight initialization for ReLU activation
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(
            self.fc4.weight
        )  # Last layer could be initialized differently if preferred

    def forward(self, state, template, action=None):
        """
        Forward pass through the CriticNetwork.

        Args:
            state (torch.Tensor): State input tensor.
            template (torch.Tensor): Template input tensor.
            action (torch.Tensor or None): Action input tensor (optional).
        Returns:
            torch.Tensor: Q-value.

        Task:
            The forward method combines the state, template, and action to produce a quality score (Q-value) that indicates how good the action is in the given state.
        """
        # Dimension checks
        assert state.dim() == 2, "State tensor must be 2-dimensional"
        assert template.dim() == 2, "Template tensor must be 2-dimensional"
        if action is not None:
            assert action.dim() == 2, "Action tensor must be 2-dimensional"
            assert (
                state.size(0) == template.size(0) == action.size(0)
            ), "Batch sizes of state, template, and action must match"
            combined_input = torch.cat([state, template, action], dim=-1)
        else:
            assert state.size(0) == template.size(
                0
            ), "Batch sizes of state and template must match"
            combined_input = torch.cat([state, template], dim=-1)
            combined_input = F.pad(combined_input, (0, self.action_size), "constant", 0)

        # Pass the combined input through the fully connected layers
        combined_input = F.relu(self.fc1(combined_input))
        combined_input = F.relu(self.fc2(combined_input))
        combined_input = F.relu(self.fc3(combined_input))
        q_value = self.fc4(combined_input)

        return q_value
