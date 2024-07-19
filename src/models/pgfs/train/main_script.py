# main_script

import argparse
import cProfile
import logging
import os
import pstats
from datetime import datetime

import torch

import wandb
from src.models.pgfs.logging_config import setup_logging
from src.models.pgfs.utility.random_selector import select_random_action
from src.models.pgfs.utility.setup_module import (
    find_latest_checkpoint,
    initialize_components,
    initialize_seed,
    load_config,
    setup_wandb,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_policy(policy, eval_env, seed, eval_count, eval_episodes=10):
    logger.debug("Starting current policy evaluation...")
    policy.env = eval_env

    steps_done = 0
    avg_reward = 0.0
    for eval_episode in range(eval_episodes):
        logger.info("Evaluation episode %s", eval_episode + 1)

        state, _ = eval_env.reset()
        done = False
        episode_timesteps = 0
        episode_reward = 0
        highest_reward = 0

        while not done:
            steps_done += 1
            episode_timesteps += 1

            action = policy.get_action(state, evaluate=True)
            eval_env.enable()
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            avg_reward += reward
            highest_reward = max(highest_reward, reward)

        eval_env.unwrapped.render(mode="console")

        if (eval_episode + 1) == eval_episodes:
            eval_env.unwrapped.render(
                mode="human",
                save_path=f"eval_results/Eval{eval_count}_episode{eval_episode + 1}.png",
            )
            eval_env.unwrapped.close()

    avg_reward /= eval_episodes
    logger.info(
        "Ending evaluation  %s over %s episodes - Average_reward: %.3f",
        eval_count,
        eval_episodes,
        avg_reward,
    )
    wandb.log(
        {
            "eval_count": eval_count,
            "eval_avg_reward": avg_reward,
            "steps_done (eval)": steps_done,
        }
    )


def setup_training(config_path, experiment_name):
    config = load_config(config_path)
    seed = config["hyperparameters"]["training"]["seed"]
    initialize_seed(seed)
    setup_wandb(config, experiment_name)
    return config, seed


def initialize_training(config):
    env, eval_env, replay_buffer, agent = initialize_components(config)
    checkpoint_dir = "src/models/pgfs/checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)

    if latest_checkpoint:
        steps_done, episode_count, replay_buffer = agent.load_model(latest_checkpoint)
        logger.info(f"Resumed training from checkpoint: {latest_checkpoint}")
    else:
        steps_done, episode_count = 0, 0
        logger.info("No checkpoint found, starting training from scratch...")

    return (
        env,
        eval_env,
        replay_buffer,
        agent,
        steps_done,
        episode_count,
        checkpoint_dir,
    )


def train(
    seed,
    env,
    eval_env,
    replay_buffer,
    agent,
    config,
    steps_done,
    episode_count,
    checkpoint_dir,
):
    max_timesteps = config["hyperparameters"]["training"]["max_timesteps"]
    eval_freq = config["hyperparameters"]["training"]["eval_freq"]
    save_freq = config["hyperparameters"]["training"]["save_freq"]
    bootstrapping_timesteps = config["hyperparameters"]["training"]["start_timesteps"]
    batch_size = config["hyperparameters"]["training"]["batch_size"]

    eval_count = 1
    while steps_done < int(max_timesteps):
        state, info = env.reset()

        done = False
        episode_reward = 0
        episode_timesteps = 0
        highest_reward = 0
        episode_count += 1

        while not done:
            steps_done += 1
            episode_timesteps += 1

            if steps_done < bootstrapping_timesteps:
                if steps_done == 1:
                    logger.info("Starting bootstrapping phase...")
                env.disable()
                action = select_random_action(env, info["SMILES"])
            else:
                if steps_done == bootstrapping_timesteps:
                    logger.info(
                        "Ending bootstrapping phase... Using the agent policy to predict subsequent actions and enabling the KNN Wrapper..."
                    )
                env.enable()
                action = agent.get_action(state)

            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

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

            state = next_state
            info = next_info
            episode_reward += reward
            highest_reward = max(highest_reward, reward)

            wandb.log(
                {
                    "steps_done": steps_done,
                    "reward_per_step": reward,
                    "episode": episode_count,
                }
            )

            if steps_done >= bootstrapping_timesteps:
                training_metrics = agent.train(replay_buffer, batch_size)
                training_metrics["steps_done"] = steps_done
                training_metrics["episode"] = episode_count
                wandb.log(training_metrics)

                if steps_done % int(float(eval_freq)) == 0:
                    logger.info("Evaluating policy on validation set...")
                    eval_policy(agent, eval_env, seed, eval_count=eval_count)
                    eval_count += 1

                if episode_count % int(float(save_freq)) == 0:
                    checkpoint_path = os.path.join(
                        checkpoint_dir, f"checkpoint_{episode_count}.pth"
                    )
                    agent.save_model(
                        checkpoint_path, steps_done, episode_count, replay_buffer
                    )
                    logger.info(f"Saved checkpoint at episode {episode_count}")

                # Anneal the temperature parameter
                if agent.temperature > agent.temperature_end:
                    agent.temperature *= agent.temperature_decay
                    agent.temperature = max(agent.temperature, agent.temperature_end)
                    logger.info("Temperature annealed to: %s", agent.temperature)

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
            highest_reward,
        )

        wandb.log(
            {
                "steps_done": steps_done,
                "episode": episode_count,
                "total_episode_reward": episode_reward,
                "average_episode_reward": avg_episode_reward,
                "max_episode_reward": highest_reward,
                "episode_length": episode_timesteps,
                "buffer_size": replay_buffer.size(),
            }
        )

    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
    agent.save_model(final_model_path, steps_done, episode_count, replay_buffer)
    logger.info("Saved final model.")
    wandb.finish()


def main(experiment_name, profile=False, output_file=None):
    logger.info("Starting training process...")

    config_path = "src/models/pgfs/config.yaml"
    config, seed = setup_training(config_path, experiment_name)

    try:
        (
            env,
            eval_env,
            replay_buffer,
            agent,
            steps_done,
            episode_count,
            checkpoint_dir,
        ) = initialize_training(config)
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        return

    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    try:
        train(
            seed,
            env,
            eval_env,
            replay_buffer,
            agent,
            config,
            steps_done,
            episode_count,
            checkpoint_dir,
        )
    finally:
        env.close()

    if profile:
        profiler.disable()
        if output_file:
            profiler.dump_stats(output_file)
        else:
            stats = pstats.Stats(profiler)
            stats.sort_stats("cumtime")
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
    # Configure logger
    setup_logging()
    logger = logging.getLogger(__name__)

    main(args.experiment_name, args.profile, args.output_file)
