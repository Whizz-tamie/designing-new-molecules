import gymnasium as gym


def eval_policy(policy, eval_env, seed, eval_episodes=10):
    total_reward = 0.0

    for _ in range(eval_episodes):
        state, info, done = eval_env.reset(seed=seed + 100), False
        episode_timesteps = 0
        episode_reward = 0

        while not done:
            action = policy.get_action(
                state, evaluate=True
            )  # Ensure deterministic actions
            next_state, reward, terminated, truncated, next_info = eval_env.step(action)
            state = next_state
            info = next_info
            done = terminated or truncated
            episode_reward += reward
        total_reward += episode_reward

    average_reward = total_reward / eval_episodes
    logger.info("Evaluation over %s episodes: %s", eval_episodes, average_reward)
    return average_reward


import logging
import os
import sys
from datetime import datetime

log_file_name = None


def setup_logging():
    log_dir = "/rds/user/gtj21/hpc-work/designing-new-molecules/logs"
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if "LOG_FILE_NAME" not in os.environ:
        os.makedirs(log_dir, exist_ok=True)
        log_file_name = f"RL_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        os.environ["LOG_FILE_NAME"] = log_file_name
    else:
        log_file_name = os.environ["LOG_FILE_NAME"]

    # Clear all existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(os.path.join(log_dir, log_file_name))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stderr_handler = logging.StreamHandler(sys.stderr)

    file_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(logging.INFO)
    stderr_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)

    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)

    # Ensure the log file is created by writing an initial message
    root_logger.info(f"Logging setup complete. Log file: {log_file_name}")


# main_script.py

import argparse
import cProfile
import logging
import os
import pstats
import sys

import torch

import wandb
from src.models.pgfs.logging_config import *
from src.models.pgfs.utility.random_selector import select_random_action
from src.models.pgfs.utility.setup_module import (
    find_latest_checkpoint,
    initialize_components,
    initialize_seed,
    load_config,
    setup_wandb,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure the logger
setup_logging()
logger = logging.getLogger(__name__)


def eval_policy(policy, eval_env, seed, eval_episodes=10):
    total_steps = 0
    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        done = False
        episode_timesteps = 0
        episode_reward = 0

        while not done:
            total_steps += 1
            episode_timesteps += 1

            action = policy.get_action(
                state, evaluate=True
            )  # Ensure deterministic actions
            eval_env.enable()
            state, reward, done, _ = eval_env.step(action)
            episode_reward += reward
            avg_reward += reward

            wandb.log({"eval_reward_per_step": reward})

        # Log individual episode metrics
        wandb.log(
            {
                "eval_avg_episode_reward": episode_reward / episode_timesteps,
                "eval_episode_length": episode_timesteps,
            }
        )
        # Render the just completed episode in the console
        eval_env.unwrapped.render(mode="console")

    avg_reward /= eval_episodes

    logger.info(
        "Evaluation over %s episodes. Average_reward: %.3f",
        eval_episodes,
        avg_reward,
    )

    # Log average metrics over all evaluation episodes
    wandb.log({"eval_average_reward": avg_reward, "eval_total_steps": total_steps})


def main(experiment_name, profile=False, output_file=None):
    logger.info("Starting training process...")

    # Load the configuration
    config_path = (
        "/rds/user/gtj21/hpc-work/designing-new-molecules/src/models/pgfs/config.yaml"
    )

    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        return

    seed = config["hyperparameters"]["training"]["seed"]

    initialize_seed(seed)

    # Setup wandb for experiment tracking
    try:
        setup_wandb(config, experiment_name)
    except Exception as e:
        logger.error(f"Failed to initialize wandb: {e}")
        return

    # Initialize the environment, replay buffer, and agent
    try:
        env, eval_env, replay_buffer, agent = initialize_components(config)
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return

    # Profiling setup
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    max_timesteps = config["hyperparameters"]["training"]["max_timesteps"]
    eval_freq = config["hyperparameters"]["training"]["eval_freq"]
    save_freq = config["hyperparameters"]["training"]["save_freq"]
    bootstrapping_timesteps = config["hyperparameters"]["training"]["start_timesteps"]
    batch_size = config["hyperparameters"]["training"]["batch_size"]
    checkpoint_dir = "src/models/pgfs/checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Automatically find and load the latest checkpoint if available
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        try:
            steps_done, episode_count, replay_buffer = agent.load_model(
                latest_checkpoint
            )
            logger.info(f"Resumed training from checkpoint: {latest_checkpoint}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {latest_checkpoint}: {e}")
            return
    else:
        steps_done = 0
        episode_count = 0
        logger.info("No checkpoint found, starting training from scratch...")

    # Training loop setup
    try:
        while steps_done < int(max_timesteps):
            state, info = env.reset(seed=seed)
            logger.info("Randomly selected %s", info["SMILES"])

            done = False
            episode_reward = 0
            episode_timesteps = 0
            highest_reward = float("-inf")

            episode_count += 1
            while not done:
                steps_done += 1
                episode_timesteps += 1
                if steps_done < bootstrapping_timesteps:
                    if steps_done == 1:
                        logger.info("Starting bootstrapping phase...")
                    env.disable()  # Disable KNN during bootstrapping
                    action = select_random_action(env, info["SMILES"])
                else:
                    if steps_done == bootstrapping_timesteps:
                        logger.info(
                            "Ending bootstrapping phase. Using agent to predict subsequent actions and enabling the KNN Wrapper..."
                        )
                    env.enable()  # Enable KNN after bootstrapping
                    action = agent.get_action(state)

                # Perform the action
                next_state, reward, terminated, truncated, next_info = env.step(action)

                # Store data in replay buffer
                replay_buffer.add(
                    info["SMILES"],
                    state,
                    action[0],
                    (
                        action[1]
                        if isinstance(action[1], torch.Tensor)
                        else torch.tensor(
                            env.unwrapped.reactants[action[1]], dtype=torch.float32
                        )
                        .unsqueeze(0)
                        .to(device)
                    ),
                    reward,
                    next_info["SMILES"],
                    next_state,
                    done,
                )

                # Update the state
                state = next_state
                info = next_info
                episode_reward += reward
                episode_max_reward = max(highest_reward, reward)
                done = terminated or truncated

                wandb.log({"reward_per_step": reward})

                # Train agent after collecting sufficient data
                if steps_done >= bootstrapping_timesteps:
                    training_metrics = agent.train(replay_buffer, batch_size)
                    wandb.log(training_metrics)

                    # Evaluate policy at specified frequency
                    if (steps_done + 1) % eval_freq == 0:
                        logger.info("Evaluating policy...")
                        eval_policy(agent, eval_env, seed, eval_episodes=10)

                    # Periodically save the model
                    if episode_count % save_freq == 0:
                        checkpoint_path = os.path.join(
                            checkpoint_dir, f"checkpoint_{episode_count}.pth"
                        )
                        agent.save_model(
                            checkpoint_path, steps_done, episode_count, replay_buffer
                        )
                        logger.info(f"Saved checkpoint at episode {episode_count}")

            avg_episode_reward = (
                episode_reward / episode_timesteps if episode_timesteps > 0 else 0
            )

            logger.info(
                "Total Timesteps: %s, Total Episodes: %s, Episode Timesteps: %s, Episode Reward: %.3f, Episode Avg reward: %.3f, Episode Highest reward: %.3f",
                steps_done,
                episode_count,
                episode_timesteps,
                episode_reward,
                avg_episode_reward,
                episode_max_reward,
            )

            wandb.log(
                {
                    "steps_done": steps_done,
                    "total_episode_reward": episode_reward,
                    "average_episode_reward": avg_episode_reward,
                    "max_episode_reward": episode_max_reward,
                    "episode_length": episode_timesteps,
                    "episode": episode_count,
                    "buffer_size": replay_buffer.size(),
                }
            )

        # Save the final model
        final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
        agent.save_model(final_model_path, steps_done, episode_count, replay_buffer)
        logger.info("Saved final model.")
        wandb.finish()

    finally:
        # Ensure resources are released
        env.close()

    # End profiling and print results if profiling was enabled
    if profile:
        profiler.disable()
        if output_file:
            # Save to file
            with open(output_file, "w") as f:
                sys.stdout = f  # Redirect stdout to file
                stats = pstats.Stats(profiler, stream=f).sort_stats("cumtime")
                stats.print_stats()
                sys.stdout = sys.__stdout__  # Reset stdout to original
        else:
            # Print to stdout
            stats = pstats.Stats(profiler).sort_stats("cumtime")
            stats.print_stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the TD3 training")
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="Name of the experiment"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable profiling for this run"
    )
    parser.add_argument(
        "--output_file", type=str, help="File to write profiling stats to"
    )
    args = parser.parse_args()
    main(args.experiment_name, args.profile, args.output_file)
