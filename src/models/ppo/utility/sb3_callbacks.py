# sb3_callbacks.py

import os

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

import wandb


class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_qeds = []
        self.episode_steps = 0
        self.total_rewards = []
        self.total_qeds = []
        self.total_steps = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self.episode_rewards.append(reward)

        if len(self.locals["infos"]) > 0 and "QED" in self.locals["infos"][0]:
            qed = self.locals["infos"][0].get("QED", 0)
            self.episode_qeds.append(qed)

        self.episode_steps += 1
        self.total_steps += 1

        # Check if the episode is done
        if self.locals["dones"][0]:
            self.episode_count += 1

            total_reward = sum(self.episode_rewards)
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            max_reward = max(self.episode_rewards) if self.episode_rewards else 0

            total_qed = sum(self.episode_qeds)
            avg_qed = np.mean(self.episode_qeds) if self.episode_qeds else 0

            # Log episode-wise metrics
            wandb.log(
                {
                    "episode": self.episode_count,
                    "total_reward": total_reward,
                    "avg_reward": avg_reward,
                    "max_reward": max_reward,
                    "total_qed": total_qed,
                    "avg_qed": avg_qed,
                    "episode_steps": self.episode_steps,
                    "total_steps": self.total_steps,
                }
            )

            self.total_rewards.append(total_reward)
            self.total_qeds.append(total_qed)

            self.episode_rewards = []
            self.episode_qeds = []
            self.episode_steps = 0

        # Log step-wise reward and QED
        wandb.log(
            {
                "step_reward": reward,
                "step_qed": qed if "QED" in self.locals["infos"][0] else None,
                "total_steps": self.total_steps,
            }
        )

        return True

    def _on_training_end(self) -> None:
        overall_avg_reward = np.mean(self.total_rewards) if self.total_rewards else 0
        overall_avg_qed = np.mean(self.total_qeds) if self.total_qeds else 0

        # Log overall training metrics
        wandb.log(
            {
                "overall_avg_reward": overall_avg_reward,
                "overall_avg_qed": overall_avg_qed,
                "total_steps": self.total_steps,
            }
        )


class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_count = 0  # Initialize episode counter
        self.all_episode_QEDs = []

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
                        wandb.log(
                            {
                                f"eval_step_{self.num_timesteps}_episode_{self.episode_count}": wandb.Image(
                                    fig
                                ),
                                "train_steps": self.num_timesteps,
                            }
                        )
                        plt.close(fig)  # Close the figure to free memory

            # Evaluate policy with the custom callback
            episode_rewards, episode_lengths = self._evaluate_policy(
                callback=custom_callback
            )

            # Log results and QED statistics at the end of evaluation
            self._log_results(episode_rewards, episode_lengths)
            if self.all_episode_QEDs:
                avg_episode_QED = np.mean(self.all_episode_QEDs)
                total_episode_QED = np.sum(self.all_episode_QEDs)
                max_episode_QED = np.max(self.all_episode_QEDs)
                wandb.log(
                    {
                        "eval_avg_episode_QED": avg_episode_QED,
                        "eval_total_episode_QED": total_episode_QED,
                        "eval_max_episode_QED": max_episode_QED,
                        "timestep": self.num_timesteps,
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
        from stable_baselines3.common.evaluation import evaluate_policy

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
