# models/model.py

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure the logger
logger = logging.getLogger(__name__)


class FNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128, 128]):
        super(FNetwork, self).__init__()
        layers = []
        for i in range(len(hidden_dims)):
            layers.append(
                nn.Linear(input_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i])
            )
            layers.append(nn.ReLU())
            input_dim = hidden_dims[i]  # Update input dimension for the next layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.network = nn.Sequential(*layers)

        logger.info(
            "Actor's FNetwork initialised with input_dim: %s, output_dim: %s, hidden_dims: %s...",
            input_dim,
            output_dim,
            hidden_dims,
        )
        # Initialize weights using He initialization
        # for layer in self.network:
        #    if isinstance(layer, nn.Linear):
        #        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

    def forward(self, state):
        output = self.network(state)
        logger.debug("Forward pass of the FNetwork - Output: %s", output.shape)
        return output


class PiNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256, 167]):
        super(PiNetwork, self).__init__()
        layers = []
        for i in range(len(hidden_dims)):
            layers.append(
                nn.Linear(input_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i])
            )
            layers.append(nn.ReLU())
            input_dim = hidden_dims[i]  # Update input dimension for the next layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.network = nn.Sequential(*layers)

        logger.info(
            "Actor's PiNetwork initialised with input_dim: %s, output_dim: %s, hidden_dims: %s...",
            input_dim,
            output_dim,
            hidden_dims,
        )
        # Initialize weights using He initialization
        # for layer in self.network:
        #    if isinstance(layer, nn.Linear):
        #        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

    def forward(self, combined_input):
        output = self.network(combined_input)
        logger.debug("Forward pass of the PiNetwork - Output: %s", output.shape)
        return torch.tanh(output)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, template_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.f_net = FNetwork(state_dim, template_dim)
        self.pi_net = PiNetwork(state_dim + template_dim, action_dim)
        self.logits = None

        logger.info(
            "ActorNetwork initialised with state_dim: %s, template_dim: %s, action_dim: %s",
            state_dim,
            template_dim,
            action_dim,
        )

    def forward(self, state, template_mask_info=None, temperature=1.0, evaluate=False):
        logger.debug("ActorNetwork is activated with evaluate mode: %s", evaluate)

        logits = self.f_net(state)
        self.logits = logits  # Save logits as an attribute

        template_one_hot, template_types = self._apply_template_mask(
            logits, template_mask_info, temperature, evaluate
        )
        template_indices = template_one_hot.argmax(dim=-1)
        logger.debug(
            "template indices: %s",
            template_indices,
        )

        # Ensure template_types_tensor and template_index are on the same device
        if template_indices.device != template_types.device:
            template_indices = template_indices.to(template_types.device)

        batch_template_types = template_types[template_indices]
        logger.debug(
            "batch_template_types: %s, batch_template_types_shape: %s",
            batch_template_types,
            batch_template_types.shape,
        )

        # Identify bimolecular samples
        is_bimolecular = batch_template_types == 1
        logger.debug(
            "is_bimolecular: %s, is_bimolecular_shape: %s",
            is_bimolecular,
            is_bimolecular.shape,
        )

        # Initialize r2_vector with zeros for all samples
        r2_vector = torch.zeros_like(state)

        if is_bimolecular.any():
            logger.debug("Bimolecular templates identified, activating PiNetwork...")

            # Select only bimolecular samples for processing
            bimolecular_states = torch.cat(
                (state[is_bimolecular], template_one_hot[is_bimolecular]), dim=-1
            )
            bimolecular_r2 = self.pi_net(bimolecular_states)

            # Place computed vectors back into the corresponding positions in r2_vector
            r2_vector[is_bimolecular] = bimolecular_r2

        return template_one_hot, r2_vector

    def _apply_template_mask(self, logits, template_mask_info, temperature, evaluate):
        """Applies template masking logic to logits based on the template mask info."""
        if template_mask_info is None:
            if evaluate:
                # For evaluation, return the one-hot encoded vector of the argmax
                selected_templates = logits.argmax(dim=-1)
                return F.one_hot(selected_templates, num_classes=logits.size(1)), {}
            else:
                # For training, apply Gumbel Softmax for stochastic sampling
                return F.gumbel_softmax(logits, tau=temperature, hard=True), {}
        else:
            template_mask, template_types = template_mask_info
            masked_logits = logits + (1 - template_mask) * (-1e9)  # Apply mask
            if evaluate:
                # For evaluation, return the one-hot encoded vector of the argmax of masked logits
                selected_templates = masked_logits.argmax(dim=-1)
                return (
                    F.one_hot(selected_templates, num_classes=masked_logits.size(1)),
                    template_types,
                )
            else:
                # For training, apply Gumbel Softmax to the masked logits
                return (
                    F.gumbel_softmax(masked_logits, tau=temperature, hard=True),
                    template_types,
                )


class CriticNetwork(nn.Module):
    def __init__(
        self, state_size, template_size, action_size, hidden_dims=[256, 64, 16]
    ):
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

        logger.info(
            "Initializing CriticNetwork with state_size= %s, template_size= %s, action_size= %s",
            state_size,
            template_size,
            action_size,
        )

        # Weight initialization using He initialization for better performance with ReLU
        # for layer in self.network:
        #    if isinstance(layer, nn.Linear):
        #        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

    def forward(self, state, template, action=None):
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

        logger.debug("Forward pass of the CriticNetwork - Q-values: %s", q_value)
        return q_value
