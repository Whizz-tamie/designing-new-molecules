# tests/test_actor.py

import unittest
from unittest.mock import patch, MagicMock
import torch
from src.models.pgfs.models.actor import FNetwork, PiNetwork, ActorProcedure

class TestFNetwork(unittest.TestCase):

    def setUp(self):
        self.input_dim = 1024
        self.output_dim = 64
        self.f_network = FNetwork(self.input_dim, self.output_dim)

    def test_initialization(self):
        """Test if the FNetwork is initialized correctly."""
        self.assertEqual(self.f_network.fc1.in_features, self.input_dim)
        self.assertEqual(self.f_network.fc1.out_features, 256)
        self.assertEqual(self.f_network.fc2.in_features, 256)
        self.assertEqual(self.f_network.fc2.out_features, 128)
        self.assertEqual(self.f_network.fc3.in_features, 128)
        self.assertEqual(self.f_network.fc3.out_features, 128)
        self.assertEqual(self.f_network.fc4.in_features, 128)
        self.assertEqual(self.f_network.fc4.out_features, self.output_dim)

    def test_forward_pass(self):
        """Test the forward pass of the FNetwork."""
        state = torch.randn(1, self.input_dim)
        template = self.f_network(state)
        self.assertEqual(template.shape, (1, self.output_dim))
        self.assertIsInstance(template, torch.Tensor)

class TestPiNetwork(unittest.TestCase):

    def setUp(self):
        self.input_dim = 1088
        self.output_dim = 1024
        self.pi_network = PiNetwork(self.input_dim, self.output_dim)

    def test_initialization(self):
        """Test if the PiNetwork is initialized correctly."""
        self.assertEqual(self.pi_network.fc1.in_features, self.input_dim)
        self.assertEqual(self.pi_network.fc1.out_features, 256)
        self.assertEqual(self.pi_network.fc2.in_features, 256)
        self.assertEqual(self.pi_network.fc2.out_features, 256)
        self.assertEqual(self.pi_network.fc3.in_features, 256)
        self.assertEqual(self.pi_network.fc3.out_features, 167)
        self.assertEqual(self.pi_network.fc4.in_features, 167)
        self.assertEqual(self.pi_network.fc4.out_features, self.output_dim)

    def test_forward_pass(self):
        """Test the forward pass of the PiNetwork."""
        state_template = torch.randn(1, self.input_dim)
        action = self.pi_network(state_template)
        self.assertEqual(action.shape, (1, self.output_dim))
        self.assertIsInstance(action, torch.Tensor)

class TestActorProcedure(unittest.TestCase):

    @patch('src.models.pgfs.models.actor.PiNetwork')
    @patch('src.models.pgfs.models.actor.FNetwork')
    def test_actor_procedure(self, MockFNetwork, MockPiNetwork):
        """Test the actor procedure function."""
        input_dim = 1024
        output_dim = 64
        action_dim = 128
        temperature = 1.0

        # Mock input state and mask
        state = torch.randn(1, input_dim)
        T_mask = torch.ones(1, output_dim)

        # Create instances of the mock networks
        mock_f_net_instance = MockFNetwork.return_value
        mock_pi_net_instance = MockPiNetwork.return_value

        # Explicitly mock the return values of the networks
        mock_f_net_instance.return_value = torch.randn(1, output_dim)
        mock_pi_net_instance.return_value = torch.randn(1, action_dim)

        # Call the actor_procedure function
        T, a = actor_procedure(state, mock_f_net_instance, mock_pi_net_instance, T_mask, temperature, input_dim, output_dim, action_dim)

        # Verify the shapes of the outputs
        self.assertEqual(T.shape, (1, output_dim))
        self.assertIsInstance(T, torch.Tensor)
        self.assertEqual(a.shape, (1, action_dim))
        self.assertIsInstance(a, torch.Tensor)

if __name__ == '__main__':
    unittest.main()