# models/critic.py

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models.pgfs.logging_config as logging_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CriticNetwork(nn.Module):
    def __init__(
        self, state_size, template_size, action_size, hidden_dims=[256, 64, 16]
    ):
        logger.info(
            f"Initializing CriticNetwork with state_size={state_size}, template_size={template_size}, action_size={action_size}"
        )
        super(CriticNetwork, self).__init__()
        input_size = state_size + template_size + action_size
        layers = []

        # Constructing layers based on hidden dimensions
        prev_dim = input_size
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim

        # Add the output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.network = nn.Sequential(*layers)

        # Weight initialization using He initialization for better performance with ReLU
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

        # Keeping track of action size for action tensor creation
        self.action_size = action_size

    def forward(self, state, template, action=None):
        logger.debug("Forward pass of the CriticNetwork...")
        if action is None:
            action = torch.zeros(state.size(0), self.action_size, device=state.device)
            logger.debug("Action was None, initialized to zeros.")

        # Ensure that dimensions and batch sizes are consistent
        assert (
            state.dim() == 2 and template.dim() == 2 and action.dim() == 2
        ), "Input tensors must be 2-dimensional"
        assert (
            state.size(0) == template.size(0) == action.size(0)
        ), "Mismatch in batch sizes among state, template, and action tensors"

        combined_input = torch.cat(
            [state, template, action], dim=-1
        )  # Concatenate inputs
        q_value = self.network(combined_input)  # Compute Q-value
        logger.debug(f"Computed Q-value: {q_value}")

        return q_value
