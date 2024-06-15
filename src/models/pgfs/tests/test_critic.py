# tests/test_critic.py

import unittest
import torch
from src.models.pgfs.models.critic import CriticNetwork

class TestCriticNetwork(unittest.TestCase):
    
    def setUp(self):
        # Define the dimensions for state, template, and action
        self.state_size = 1024
        self.template_size = 128
        self.action_size = 1024
        
        # Instantiate the CriticNetwork
        self.critic = CriticNetwork(self.state_size, self.template_size, self.action_size)
    
    def test_initialization(self):
        """Test if the network is initialized correctly."""
        self.assertEqual(self.critic.fc1.in_features, self.state_size + self.template_size + self.action_size)
        self.assertEqual(self.critic.fc1.out_features, 256)
        self.assertEqual(self.critic.fc2.in_features, 256)
        self.assertEqual(self.critic.fc2.out_features, 64)
        self.assertEqual(self.critic.fc3.in_features, 64)
        self.assertEqual(self.critic.fc3.out_features, 16)
        self.assertEqual(self.critic.fc4.in_features, 16)
        self.assertEqual(self.critic.fc4.out_features, 1)
    
    def test_forward_pass(self):
        """Test the forward pass of the network."""
        state = torch.randn(1, self.state_size)
        template = torch.randn(1, self.template_size)
        action = torch.randn(1, self.action_size)
        
        # Perform a forward pass
        q_value = self.critic(state, template, action)
        
        # Check the output shape
        self.assertEqual(q_value.shape, (1, 1))
        
        # Check if the output is a tensor
        self.assertIsInstance(q_value, torch.Tensor)

if __name__ == '__main__':
    unittest.main()