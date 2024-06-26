# models/actor.py

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ActorProcedure(nn.Module):
    def __init__(self, state_dim, template_dim, action_dim):
        """
        Initialize the FNetwork and PiNetwork

        Args:
            input_dim (int): Dimension of the input features (e.g., ECFP4 fingerprints).
            output_dim (int): Dimension of the output (number of reaction templates).
            action_dim (int): Dimension of the action space (feature representation of reactants).
        Task:
            Actor procedure to select the best reaction template and action.
        """
        super(ActorProcedure, self).__init__()
        self.f_net = FNetwork(state_dim, template_dim)
        self.pi_net = PiNetwork(state_dim + template_dim, action_dim)

    def forward(self, state, temperature, T_mask):
        """
        Forward pass through the PiNetwork.

        Args:
            state (torch.Tensor): Input state tensor (molecular representation).
            T_mask (Tensor): Mask to ensure valid templates.
            temperature (float): Temperature parameter for Gumbel Softmax.
        Returns:
            T (Tensor): Selected reaction template.
            action (Tensor): Selected action feature representation.
        """
        T = self.f_net(state)
       
        # Create a mask with -inf for invalid templates
        #inf_mask = torch.where(T_mask== 1, torch.tensor(0.0), torch.tensor(float('-inf')))
        #T = T + inf_mask
        #T = T * T_mask

        # Apply a very small value to invalid logits before softmax
        very_small_value = -1e9
        T = T + (1 - T_mask) * very_small_value

        T = F.gumbel_softmax(T, tau=temperature, hard=True)
        action = self.pi_net(torch.cat((state, T), dim=-1))
        return T, action
