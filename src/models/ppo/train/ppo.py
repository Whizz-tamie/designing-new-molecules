import argparse
import os
import sys

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

import src.models.ppo.config.config as config
import wandb
from src.models.ppo.utility.sb3_callbacks import CustomEvalCallback, CustomWandbCallback


def main(experiment_name, run_id):
    wandb.require("core")

    wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        name=experiment_name,
        id=run_id,
        job_type="training",
        notes="Running SB3 PPO on the MoleculeDesign-v1 environment",
        sync_tensorboard=True,
        save_code=True,
        resume="allow",
        config={
            "policy_type": config.POLICY_TYPE,
            "total_timesteps": config.TOTAL_TIMESTEPS,
            "env_name": config.ENV_NAME,
        },
    )

    # Initialize paths that depend on wandb.run.id
    run_id = wandb.run.id
    paths = config.initialize_paths(run_id)

    os.makedirs(os.path.dirname(paths["log_file"]), exist_ok=True)

    sys.stdout = open(paths["log_file"], "w")
    sys.stderr = open(paths["log_file"], "w")

    env = gym.make(
        config.ENV_NAME,
        reactant_file=config.REACTANT_FILE,
        template_file=config.TEMPLATE_FILE,
        max_steps=config.MAX_STEPS,
        use_multidiscrete=config.USE_MULTIDISCRETE,
    )

    env = Monitor(
        env,
        filename=os.path.join(config.LOG_DIR, f"run_{run_id}/monitor.csv"),
    )
    env = DummyVecEnv([lambda: env])

    eval_env = gym.make(
        config.ENV_NAME,
        reactant_file=config.EVAL_REACTANT_FILE,
        template_file=config.TEMPLATE_FILE,
        max_steps=config.MAX_STEPS,
        use_multidiscrete=config.USE_MULTIDISCRETE,
        render_mode="human",
    )
    eval_env = Monitor(
        eval_env,
        filename=os.path.join(config.LOG_DIR, f"run_{run_id}/eval_monitor.csv"),
    )
    eval_env = DummyVecEnv([lambda: eval_env])

    model = PPO(
        policy=config.POLICY_TYPE,
        env=env,
        verbose=1,
        tensorboard_log=paths["tensorboard_log_dir"],
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=config.GRADIENT_SAVE_FREQ,
        model_save_freq=config.MODEL_SAVE_FREQ,
        model_save_path=paths["model_save_path"],
        verbose=2,
    )

    eval_callback = CustomEvalCallback(
        eval_env,
        best_model_save_path=paths["eval_model_save_path"],
        log_path=paths["eval_log_path"],
        eval_freq=config.EVAL_FREQ,
        deterministic=config.DETERMINISTIC,
        render=config.RENDER,
    )

    custom_wandb_callback = CustomWandbCallback()

    callback = CallbackList([wandb_callback, custom_wandb_callback, eval_callback])

    model.learn(
        total_timesteps=wandb.config.total_timesteps,
        callback=callback,
        tb_log_name="Mol_PPO",  # Subdirectory for TensorBoard logs
    )

    model.save(paths["final_model_save_path"])

    wandb.finish()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PPO training")
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="Name of the experiment"
    )
    parser.add_argument("--run_id", type=str, help="ID of the run")

    args = parser.parse_args()
    main(experiment_name=args.experiment_name, run_id=args.run_id)
