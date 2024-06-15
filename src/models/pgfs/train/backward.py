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

def actor_procedure(state, f_net, pi_net, T_mask, temperature, input_dim, output_dim, action_dim):
    """
    Actor procedure to select the best reaction template and action.

    Args:
        state (Tensor): Current state (current molecule).
        T_mask (Tensor): Mask to ensure valid templates.
        temperature (float): Temperature parameter for Gumbel Softmax.
        input_dim (int): Dimension of the input features (e.g., ECFP4 fingerprints).
        output_dim (int): Dimension of the output (number of reaction templates).
        action_dim (int): Dimension of the action space (feature representation of reactants).

    Returns:
        T (Tensor): Selected reaction template.
        a (Tensor): Selected action feature representation.
    """
     # Expand T_mask to match batch size
    T_mask = T_mask.repeat(state.size(0), 1)
    pi_input = input_dim + output_dim

    # Initialize networks
    f_net = FNetwork(input_dim, output_dim)
    pi_net = PiNetwork(pi_input, action_dim)


    # Predict the reaction template using f network
    T = f_net(state)

    # Apply mask to ensure only valid templates
    if T_mask.shape != T.shape:
        raise ValueError(f"Dimension mismatch: T_mask shape {T_mask.shape} does not match T shape {T.shape}")

    T  = T * T_mask

    # Use Gumbel Softmax for differentiable sampling
    T = F.gumbel_softmax(T, temperature, hard=True)

    # Compute the action using policy network π
    a = pi_net(torch.cat((state, T), dim=-1))

    return T, a

self.actor_f = FNetwork(state_dim, template_dim).to(device)
        self.actor_pi = PiNetwork(state_dim + template_dim, action_dim).to(device)
        self.actor_f_target = FNetwork(state_dim, template_dim).to(device)
        self.actor_pi_target = PiNetwork(state_dim + template_dim, action_dim).to(device)
        self.actor_f_target.load_state_dict(self.actor_f.state_dict())
        self.actor_pi_target.load_state_dict(self.actor_pi.state_dict())

template, action = actor_procedure(state, self.actor_f, self.actor_pi, self.T_mask, self.temperature, self.state_dim, self.template_dim, self.action_dim)

checkpoint = {
            'actor_f_state_dict': self.actor_f.state_dict(),
            'actor_pi_state_dict': self.actor_pi.state_dict(),
            'actor_f_target_state_dict': self.actor_f_target.state_dict(),
            'actor_pi_target_state_dict': self.actor_pi_target.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_f_optimizer_state_dict': self.actor_f_optimizer.state_dict(),
            'actor_pi_optimizer_state_dict': self.actor_pi_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'total_it': self.total_it,
            'gamma': self.gamma,
            'tau': self.tau,
            'policy_freq': self.policy_freq,
            'temperature': self.temperature,
            'T_mask': self.T_mask,
        }
self.actor_f_optimizer.zero_grad()
            self.actor_pi_optimizer.zero_grad()
            actor_loss.backward()

# Debugging lines to check gradients
            print("actor_pi gradients:")
            for name, param in self.actor_pi.named_parameters():
                if param.grad is not None:
                    print(name, param.grad.abs().mean())
                else:
                    print(name, "No gradient")

            self.actor_f_optimizer.step()
            self.actor_pi_optimizer.step()