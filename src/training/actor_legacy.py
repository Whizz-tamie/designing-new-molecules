# models/actor.py
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models.pgfs.logging_config as logging_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize the FNetwork with fully connected layers.

        Args:
            input_dim (int): Dimension of input features (molecular represerntation).
            output_dim (int): Dimension of output features (number of reaction templates).
        Task:
            The f network predicts the best reaction template T given the current state s_t (R_t^{(1)}).
        """
        super(FNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_dim)

        # Initialize weights using He initialization
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(
            self.fc4.weight
        )  # Last layer can use a different strategy if desired

    def forward(self, state):
        """
        Forward pass through the FNetwork.

        Args:
            state (torch.Tensor): Input state tensor (molecular representation).
        Returns:
            torch.Tensor: Output template tensor after passing through the network.
        Task:
            The forward method processes the input state to produce the best reaction template T.
        """
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        state = F.relu(self.fc3(state))
        template = torch.tanh(self.fc4(state))
        return template


class PiNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize the PiNetwork with fully connected layers.

        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Dimension of output features.
        Task:
            The pi network computes the action a_t using the best reaction template T and R_t^{(1)}.
        """
        super(PiNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 167)
        self.fc4 = nn.Linear(167, output_dim)

        # Initialize weights using He initialization
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(
            self.fc4.weight
        )  # Last layer can use a different strategy if desired

    def forward(self, state_template):
        """
        Forward pass through the PiNetwork.

        Args:
            state_template (torch.Tensor): Concatenated state and template tensor.
        Returns:
            torch.Tensor: Output action tensor after passing through the network.
        Task:
            The forward method processes the input state and template to produce the action a_t.
        """
        state_template = F.relu(self.fc1(state_template))
        state_template = F.relu(self.fc2(state_template))
        state_template = F.relu(self.fc3(state_template))
        action = torch.tanh(self.fc4(state_template))
        return action


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, template_dim, action_dim, env):
        super(ActorNetwork, self).__init__()
        self.f_net = FNetwork(state_dim, template_dim)
        self.pi_net = PiNetwork(state_dim + template_dim, action_dim)
        self.env = env

        logging_config.setup_logging()  # Set up logging here
        self.logger = logging.getLogger(__name__)
        self.logger.info("Actor procedure initiated...")

    def forward(self, state, temperature=1.0):
        assert state.dim() == 2, "State tensor must be 2-dimensional"
        assert (
            state.size(1) == self.f_net.fc1.in_features
        ), f"Expected state dimension: {self.f_net.fc1.in_features}, got: {state.size(1)}"

        logits = self.f_net(state)

        reactant_smiles = self.env.current_state
        template_mask = self.env.reaction_manager.get_template_mask(reactant_smiles)

        # Apply the template mask by setting invalid logits to a very large negative value
        masked_logits = logits + (1 - template_mask) * (-1e9)

        template_one_hot = F.gumbel_softmax(masked_logits, tau=temperature, hard=True)
        template_index = template_one_hot.argmax(dim=-1).item()

        self.logger.info(f"Selected template index: {template_index}")
        self.logger.info(f"Template one-hot vector: {template_one_hot}")

        # Retrieve template type from the environment
        template_type = self.env.reaction_manager.templates[template_index]["type"]

        if template_type == "bimolecular":
            combined_input = torch.cat((state, template_one_hot), dim=-1)
            second_reactant_vector = self.pi_net(combined_input)
        else:
            second_reactant_vector = None  # No action needed for unimolecular templates

        return (template_one_hot, second_reactant_vector)

# legacy code
if evaluate:
            if template_mask_info is None:
                template_one_hot = F.one_hot(
                    logits.argmax(dim=-1), num_classes=masked_logits.size(1)
                )
            else:
                template_mask, template_types = template_mask_info

                masked_logits = logits + (1 - template_mask) * (
                    -1e9
                )  # Mask invalid templates
                template_one_hot = F.one_hot(
                    masked_logits.argmax(dim=-1), num_classes=masked_logits.size(1)
                )
        else:
            if template_mask_info is None:
                template_one_hot = F.gumbel_softmax(logits, tau=temperature, hard=True)
            else:
                template_mask, template_types = template_mask_info

                masked_logits = logits + (1 - template_mask) * (
                    -1e9
                )  # Mask invalid templates

                template_one_hot = F.gumbel_softmax(
                    masked_logits, tau=temperature, hard=True
                )

        template_index = template_one_hot.argmax(dim=-1).item()

        # Determine if bimolecular action is necessary
        if template_types[template_index] == "bimolecular":
            combined_input = torch.cat((state, template_one_hot), dim=-1)
            r2_vector = self.pi_net(combined_input)
        else:
            r2_vector = None

        return template_one_hot, r2_vector
