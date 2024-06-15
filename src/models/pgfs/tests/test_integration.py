import unittest
import torch
from src.models.pgfs.models.actor import FNetwork, PiNetwork, ActorProcedure
from src.models.pgfs.models.critic import CriticNetwork
from src.models.pgfs.environments.environment import Environment
from src.models.pgfs.train.replay_buffer import ReplayBuffer
from src.models.pgfs.train.td3 import TD3

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = 1024
        template_dim = 64
        action_dim = 1024
        max_action = 1.0
        gamma = 0.99
        tau = 0.005
        policy_freq = 2
        temperature = 1.0
        self.batch_size = 2
        self.episodes = 10
        self.max_steps = 2
        noise_std = 0.2
        noise_clip = 0.5
        min_temp = 0.1
        temp_decay = 0.99
        actor_lr = 1e-4
        critic_lr = 3e-4

        self.env = Environment(precomputed_vectors_file='./data/preprocessed_data/enamine_fpt_uuid.pkl', templates_file='./data/preprocessed_data/rxn_set_processed.txt', max_steps=self.max_steps)
        self.replay_buffer = ReplayBuffer(max_size=100)
        self.td3 = TD3(
            state_dim, template_dim, action_dim, max_action, gamma, tau, policy_freq, temperature,
            torch.ones(1, template_dim), noise_std, noise_clip, min_temp, temp_decay, actor_lr, critic_lr
        )

        # Print the device being used
        print(f"Using device: {self.device}")

    def test_networks_initialization(self):
        self.assertIsInstance(self.td3.actor_procedure.f_net, FNetwork)
        self.assertIsInstance(self.td3.actor_procedure.pi_net, PiNetwork)
        self.assertIsInstance(self.td3.actor_procedure_target.f_net, FNetwork)
        self.assertIsInstance(self.td3.actor_procedure_target.pi_net, PiNetwork)
        self.assertIsInstance(self.td3.critic1, CriticNetwork)
        self.assertIsInstance(self.td3.critic2, CriticNetwork)
        self.assertIsInstance(self.td3.critic1_target, CriticNetwork)
        self.assertIsInstance(self.td3.critic2_target, CriticNetwork)

    def test_training_loop(self):
        def get_params(network):
            return [param.clone() for param in network.parameters()]

        initial_actor_f_params = get_params(self.td3.actor_procedure.f_net)
        initial_actor_pi_params = get_params(self.td3.actor_procedure.pi_net)
        initial_critic1_params = get_params(self.td3.critic1)
        initial_critic2_params = get_params(self.td3.critic2)
        for episode in range(self.episodes):
            state, state_uid = self.env.reset()  # Reset the environment and get initial state
            episode_reward = 0

            for step in range(self.max_steps):
                # Select action according to policy
                template, action = self.td3.select_action(state)

                # Perform action in environment
                next_state, next_state_uid, reward, done = self.env.step(state_uid, template, action)

                # Store transition in replay buffer
                self.replay_buffer.add(state, template, action, next_state, reward, done)

                # Update state
                state = next_state
                state_uid = next_state_uid
                episode_reward += reward  # Accumulate reward

                # If episode is done, break loop
                if done:
                    break

            # Sample a batch from replay buffer and update the TD3 agent
            if self.replay_buffer.size() > self.batch_size:
                self.td3.backward(self.replay_buffer, self.batch_size)

                # Check gradients for f_net
                print("f_net gradients:")
                for name, param in self.td3.actor_procedure.f_net.named_parameters():
                    if param.grad is not None:
                        print(name, param.grad.abs().mean().item())
                    else:
                        print(name, "No gradient")

                # Check gradients for pi_net
                print("pi_net gradients:")
                for name, param in self.td3.actor_procedure.pi_net.named_parameters():
                    if param.grad is not None:
                        print(name, param.grad.abs().mean().item())
                    else:
                        print(name, "No gradient")

        # Check if the models are being updated (not necessary, but good to verify)
        self.assertGreater(len(self.replay_buffer.buffer), 0, "Replay buffer should have stored some experiences.")

        updated_actor_f_params = get_params(self.td3.actor_procedure.f_net)
        updated_actor_pi_params = get_params(self.td3.actor_procedure.pi_net)
        updated_critic1_params = get_params(self.td3.critic1)
        updated_critic2_params = get_params(self.td3.critic2)

        def params_changed(initial_params, updated_params):
            return any(not torch.equal(initial, updated) for initial, updated in zip(initial_params, updated_params))

        self.assertTrue(params_changed(initial_actor_f_params, updated_actor_f_params), "Actor_f parameters should have been updated.")
        self.assertTrue(params_changed(initial_actor_pi_params, updated_actor_pi_params), "Actor_pi parameters should have been updated.")
        self.assertTrue(params_changed(initial_critic1_params, updated_critic1_params), "Critic1 parameters should have been updated.")
        self.assertTrue(params_changed(initial_critic2_params, updated_critic2_params), "Critic2 parameters should have been updated.")

if __name__ == '__main__':
    unittest.main()
