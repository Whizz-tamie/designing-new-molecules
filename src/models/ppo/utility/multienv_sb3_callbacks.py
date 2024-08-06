# sb3_callbacks.py
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

import wandb

logger = logging.getLogger(__name__)


def prune_checkpoints(directory, max_checkpoints=5):
    """Prune the checkpoint directory to keep only the last 'max_checkpoints' files"""
    checkpoints = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".zip")
    ]
    checkpoints.sort(key=os.path.getmtime, reverse=True)

    # Keep only the 'max_checkpoints' most recent files
    for old_checkpoint in checkpoints[max_checkpoints:]:
        os.remove(old_checkpoint)


class PruningCallback(BaseCallback):
    def __init__(self, check_freq, save_path, max_checkpoints=5, verbose=1):
        super(PruningCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.max_checkpoints = max_checkpoints

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            prune_checkpoints(self.save_path, max_checkpoints=self.max_checkpoints)
            if self.verbose > 0:
                logger.info(f"Old checkpoints pruned at {self.num_timesteps} steps.")
        return True


class CustomWandbCallback(BaseCallback):
    def __init__(self, gamma: float, verbose=0):
        super().__init__(verbose)
        self.gamma = gamma
        self.episode_rewards = None
        self.episode_qeds = None
        self.max_reward = float("-inf")
        self.max_qed = float("-inf")
        self.total_rewards = []
        self.total_qeds = []
        self.total_steps = 0
        self.total_episodes = 0

    def _on_training_start(self):
        num_envs = self.training_env.num_envs
        self.num_envs = num_envs  # Store the number of environments
        self.episode_rewards = [[] for _ in range(num_envs)]
        self.episode_qeds = [[] for _ in range(num_envs)]

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        infos = self.locals["infos"]

        for i in range(len(rewards)):
            self.total_steps += 1
            reward = rewards[i]
            info = infos[i]

            self.episode_rewards[i].append(reward)
            qed = info.get("QED", 0)
            self.episode_qeds[i].append(qed)

            # Log step reward and QED for this step
            wandb.log(
                {
                    "step_reward": reward,
                    "step_qed": qed,
                    "total_steps": self.total_steps,
                }
            )

            # Check if the episode is done for this environment
            if "episode" in info.keys():
                self.total_episodes += 1

                episode_max_reward = max(self.episode_rewards[i])
                episode_max_qed = max(self.episode_qeds[i])

                # Update global maxima if this episode's maxima are higher
                if episode_max_reward > self.max_reward:
                    self.max_reward = episode_max_reward
                if episode_max_qed > self.max_qed:
                    self.max_qed = episode_max_qed

                total_episode_reward = sum(self.episode_rewards[i])
                total_episode_qed = sum(self.episode_qeds[i])

                self.total_rewards.append(total_episode_reward)
                self.total_qeds.append(total_episode_qed)

                # Calculate the average reward and QED for this episode only
                avg_episode_reward = total_episode_reward / len(self.episode_rewards[i])
                avg_episode_qed = total_episode_qed / len(self.episode_qeds[i])

                # Log the total and average rewards for this episode
                wandb.log(
                    {
                        "total_reward": total_episode_reward,
                        "avg_reward": avg_episode_reward,
                        "total_qed": total_episode_qed,
                        "avg_qed": avg_episode_qed,
                        "episode": self.total_episodes,
                        "total_steps": self.total_steps,
                    }
                )

                # Reset the episode metrics
                self._reset_episode_metrics(i)

        return True

    def _reset_episode_metrics(self, env_index):
        self.episode_rewards[env_index] = []
        self.episode_qeds[env_index] = []

    def _on_training_end(self) -> None:
        overall_avg_reward = np.mean(self.total_rewards) if self.total_rewards else 0
        overall_avg_qed = np.mean(self.total_qeds) if self.total_qeds else 0

        # Log overall training metrics
        wandb.log(
            {
                "overall_max_reward": self.max_reward,
                "overall_max_qed": self.max_qed,
                "overall_avg_reward": overall_avg_reward,
                "overall_avg_qed": overall_avg_qed,
                "total_episodes": self.total_episodes,
                "total_steps": self.total_steps,
            }
        )


class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_count = 0  # Initialize episode counter
        self.all_episode_QEDs = []
        self.episode_images = {}  # Store images for each episode

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Custom callback to handle rendering and QED logging
            def custom_callback(locals_dict, globals_dict):
                step_info = locals_dict["info"]
                env = locals_dict["env"].envs[0].unwrapped

                # Extract QED from the step info if available
                qed = step_info.get("QED", 0)
                self.all_episode_QEDs.append(qed)

                # Check if the episode is done
                if "episode" in step_info.keys():
                    self.episode_count += 1  # track episode count

                    fig = env.render()
                    if fig:
                        # Store figure in dictionary with episode count
                        self.episode_images[
                            f"eval_trainstep_{self.num_timesteps}_episode_{self.episode_count}"
                        ] = fig
                        plt.close(fig)  # Close the figure to free memory

            # Evaluate policy with the custom callback
            episode_rewards, episode_lengths = self._evaluate_policy(
                callback=custom_callback
            )

            # Log results and QED statistics at the end of evaluation
            self._log_results(episode_rewards, episode_lengths)

            # Log all episode images and QED statistics at the end of evaluation
            if self.episode_images:
                wandb.log(self.episode_images)
                self.episode_images.clear()

            # Log statistics at the end of evaluation
            if self.all_episode_QEDs:
                wandb.log(
                    {
                        "train_steps": self.num_timesteps,
                        "eval_avg_episode_QED": np.mean(self.all_episode_QEDs),
                        "eval_total_episode_QED": np.sum(self.all_episode_QEDs),
                        "eval_max_episode_QED": np.max(self.all_episode_QEDs),
                    }
                )

            # Reset QEDs for next evaluation
            self.all_episode_QEDs = []
            self.episode_count = 0

        return True

    def _evaluate_policy(self, callback=None):
        """
        Custom evaluate function to include a callback that handles rendering.
        """
        return evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render,
            deterministic=self.deterministic,
            callback=callback,
            return_episode_rewards=True,
        )

    def _log_results(self, episode_rewards, episode_lengths):
        """
        Handles logging the results of evaluation, similar to the base class implementation.
        """
        if self.log_path is not None:
            assert isinstance(episode_rewards, list)
            assert isinstance(episode_lengths, list)
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(episode_rewards)
            self.evaluations_length.append(episode_lengths)

            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
            )

        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
            episode_lengths
        )
        self.last_mean_reward = float(mean_reward)

        if self.verbose >= 1:
            print(
                f"Eval num_timesteps={self.num_timesteps}, "
                f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
            )
            print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

        # Add to current Logger
        self.logger.record("eval/mean_reward", float(mean_reward))
        self.logger.record("eval/mean_ep_length", mean_ep_length)

        # Dump log so the evaluation results are printed with the correct timestep
        self.logger.record(
            "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
        )
        self.logger.dump(self.num_timesteps)

        # Check for new best model and perform any additional callbacks
        self._handle_new_best(mean_reward)
        self._perform_additional_callbacks()

    def _handle_new_best(self, mean_reward):
        if mean_reward > self.best_mean_reward:
            if self.verbose >= 1:
                print("New best mean reward!")
            if self.best_model_save_path:
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            self.best_mean_reward = mean_reward

    def _perform_additional_callbacks(self):
        if self.callback_on_new_best:
            self.callback_on_new_best.on_step()
