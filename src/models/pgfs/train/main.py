# train/main.py

import argparse
import src.models.pgfs.logging_config as logging_config
import torch
from src.models.pgfs.environments.environment import Environment
from src.models.pgfs.train.replay_buffer import ReplayBuffer
from src.models.pgfs.train.td3 import TD3
import wandb
import yaml
import logging
import os
import numpy as np

def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

def main(experiment_name):
    """
    Main function to run the training loop.
    """
    logging_config.setup_logging()  # Set up logging here
    logger = logging.getLogger(__name__)
    logger.info("Starting main function!")
   
    # Load configuration
    config = load_config("/rds/user/gtj21/hpc-work/designing-new-molecules/src/models/pgfs/config.yaml")
    logger.info("Loading config...")

    # Initialize Weights & Biases
    wandb.init(project=config['project'],
               entity=config['entity'],
               notes="RL molecule search baseline",
               tags=["pgfs", "td3", "QED"], 
               config=config['hyperparameters'],
               name=experiment_name,
               id="td3_v1_cpu",
               resume="allow"
               )
    
    # Extract hyperparameters
    model_config = wandb.config.model
    training_config = wandb.config.training
    optimizer_config = wandb.config.optimizers
    temperature_config = wandb.config.temperature
    noise_config = wandb.config.noise
    datasets_config = wandb.config.dataset

    state_dim = model_config.get("state_dim")
    template_dim = model_config.get("template_dim")
    action_dim = model_config.get("action_dim")
    max_action = model_config.get("max_action")
    gamma = training_config.get("gamma")
    tau = training_config.get("tau")
    batch_size = training_config.get("batch_size")
    max_size = training_config.get("max_size")
    total_time_steps = training_config.get("time_steps")
    max_steps = training_config.get("max_steps")
    policy_freq = training_config.get("policy_freq")

    initial_temperature = temperature_config.get("initial_temperature")
    min_temperature = temperature_config.get("min_temperature")
    temp_decay = temperature_config.get("temp_decay")

    noise_std = noise_config.get("noise_std")
    noise_clip = noise_config.get("noise_clip")

    actor_lr = float(optimizer_config.get("actor_lr"))
    critic_lr = float(optimizer_config.get("critic_lr"))

    molecule_file = datasets_config.get("precomputed_vectors_file")
    template_file = datasets_config.get("templates_file")

    # Initialize environment, replay buffer, and TD3 agent
    env = Environment(precomputed_vectors_file=molecule_file, templates_file=template_file, max_steps=max_steps)
    replay_buffer = ReplayBuffer(max_size)
    td3 = TD3(
        state_dim, template_dim, action_dim, max_action, gamma, tau, policy_freq,
        initial_temperature, noise_std, noise_clip, min_temperature, temp_decay,
        actor_lr, critic_lr
    )

    # Create a directory for checkpoints
    checkpoint_dir ="/rds/user/gtj21/hpc-work/designing-new-molecules/src/models/pgfs/checkpoints_cpu/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load the latest checkpoint if available
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        logger.info(f"Loading checkpoint {checkpoint_path}...")
        td3.load(checkpoint_path, replay_buffer)
        episode = int(latest_checkpoint.split('_')[1].split('.')[0]) + 1
        current_time_step = td3.current_time_step + 1
        logger.info(f"Checkpoints found. Resuming training using checkpoint: {latest_checkpoint} with time step: {current_time_step}...")
    else:
        logger.info("No checkpoints found. Starting training from scratch...")
        current_time_step = 0
        episode = 0
    
    while current_time_step < total_time_steps:
        state, state_uid, state_tmask = env.reset()  # Reset the environment and get initial state and its UUID
        episode_reward = 0  # Initialize episode reward
        episode_steps = 0

        for step in range(max_steps):
            # Select action according to policy
            template, action = td3.select_action(state, state_tmask)

            # Perform action in environment
            next_state, next_state_uid, reward, done, next_tmask = env.step(state_uid, template, action)

            # Store transition in replay buffer
            replay_buffer.add(state, state_tmask, template, action, next_state, next_tmask, reward, done)

            # Update state and state_uid
            state = next_state
            state_uid = next_state_uid
            state_tmask = next_tmask
            episode_reward += reward  # Accumulate reward
            episode_steps += 1
            current_time_step += 1
            td3.current_time_step = current_time_step

            # If episode is done, break loop
            if done:
                break
        
        # Check if we reached the total time steps
        if current_time_step >= total_time_steps:
            break
        
        # Calculate QED metrics
        average_qed_score = np.mean(env.qed_scores) if len(env.qed_scores) > 0 else 0
        highest_qed_score = np.max(env.qed_scores)  if len(env.qed_scores) > 0 else 0
        env.qed_scores = []  # Reset for the next episode
        
        wandb.log({
            "episode": episode,
            "time_step": current_time_step,
            "Reward (QED)": reward,
            "episode_steps": episode_steps,
            "average_qed_score": average_qed_score,
            "highest_qed_score": highest_qed_score,
            "buffer_size": replay_buffer.size(),
        })
        td3.episode_rewards.append(episode_reward)  # Add episode reward to the list

        # Sample a batch from replay buffer and update the TD3 agent
        if replay_buffer.size() > batch_size:
            td3.backward(replay_buffer, batch_size)

            # Log losses and any other metrics here
            wandb.log({
                "critic1_loss": td3.critic1_loss.item(),
                "critic2_loss": td3.critic2_loss.item(),
                "actor_loss": td3.actor_loss,
                "f_net_loss": td3.f_net_loss,
                "current_qvalue": td3.current_q1.mean().item(),
                "target_qvalue": td3.target_q.mean().item(),
                "total_training_iterations": td3.total_it 
            })
            logger.info(f"Episode {episode} - Current time steps: {current_time_step} Critic1 Loss: {td3.critic1_loss}, Critic2 Loss: {td3.critic2_loss}, Actor Loss: {td3.actor_loss}, Fnet Loss {td3.f_net_loss}")
        
        # Save model checkpoints periodically
        if episode % 10 == 0 and episode > 0:  # Adjust the frequency as needed
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{episode}.pth")
            td3.save(checkpoint_path, replay_buffer)
            logger.info(f"Saved checkpoint at episode {episode}")

        episode += 1
    
    # Save the final model
    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
    td3.save(final_model_path, replay_buffer)
    logger.info("Saved final model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Weights & Biases experiment")
    parser.add_argument("--experiment_name", type=str, required=True, help="Custom experiment name")
    args = parser.parse_args()
    
    main(args.experiment_name)