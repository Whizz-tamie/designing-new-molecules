import argparse
import logging
import os

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

import src.models.ppo.config.config as config
import wandb
from src.models.pgfs.logging_config import setup_logging
from src.models.ppo.utility.multienv_sb3_callbacks import (
    CustomEvalCallback,
    CustomWandbCallback,
    PruningCallback,
)


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the given directory."""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".zip")
    ]
    if not checkpoint_files:
        return None
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    return latest_checkpoint


def main(experiment_name, run_id):
    logger = logging.getLogger(__name__)

    wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        name=experiment_name,
        id=run_id,
        job_type="training",
        notes="Running SB3 A2C on the MoleculeDesign-v1 environment with 4 multiworkers",
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

    logger.info(f"Experiment '{experiment_name}' with run ID '{run_id}' started.")

    env_kwargs = {
        "reactant_file": config.REACTANT_FILE,
        "template_file": config.TEMPLATE_FILE,
        "max_steps": config.MAX_STEPS,
        "use_multidiscrete": config.USE_MULTIDISCRETE,
        "render_mode": "human",
    }

    # Create parallel environments for training
    env = make_vec_env(
        config.ENV_NAME,
        n_envs=10,
        env_kwargs=env_kwargs,
        monitor_dir=os.path.join(config.LOG_DIR, f"run_{run_id}"),
    )

    eval_env_kwargs = {
        "reactant_file": config.EVAL_REACTANT_FILE,
        "template_file": config.TEMPLATE_FILE,
        "max_steps": config.MAX_STEPS,
        "use_multidiscrete": config.USE_MULTIDISCRETE,
        "render_mode": "human",
    }

    # Create a single environment for evaluation
    eval_env = make_vec_env(
        config.ENV_NAME,
        n_envs=1,
        env_kwargs=eval_env_kwargs,
        monitor_dir=os.path.join(config.LOG_DIR, f"run_{run_id}/eval"),
    )

    # Check for a previous checkpoint to resume training
    latest_checkpoint = find_latest_checkpoint(paths["model_save_path"])
    if latest_checkpoint:
        model = A2C.load(latest_checkpoint, env=env, device="cpu")
        logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
    else:
        model = A2C(
            policy=config.POLICY_TYPE,
            env=env,
            verbose=1,
            device="cpu",
            tensorboard_log=paths["tensorboard_log_dir"],
            policy_kwargs=dict(
                optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)
            ),
        )
        logger.info(f"No checkpoint found, starting training from scratch...")

    # Setup callback for periodic checkpoint saving
    checkpoint_callback = CheckpointCallback(
        save_freq=config.MODEL_SAVE_FREQ,
        save_path=paths["model_save_path"],
        name_prefix="a2c_model",
    )

    pruning_callback = PruningCallback(
        check_freq=config.CHECK_FREQ,
        save_path=paths["model_save_path"],
        max_checkpoints=5,
    )

    wandb_callback = WandbCallback(
        model_save_path=paths["model_save_path"],
        model_save_freq=config.MODEL_SAVE_FREQ,
        gradient_save_freq=config.GRADIENT_SAVE_FREQ,
        log="all",
        verbose=2,
    )

    eval_callback = CustomEvalCallback(
        eval_env,
        best_model_save_path=paths["eval_model_save_path"],
        log_path=paths["eval_log_path"],
        eval_freq=config.EVAL_FREQ,
        n_eval_episodes=config.N_EVAL_EPISODES,
        deterministic=config.DETERMINISTIC,
        render=config.RENDER,
    )

    custom_wandb_callback = CustomWandbCallback(gamma=model.gamma)

    callback = CallbackList(
        [
            wandb_callback,
            custom_wandb_callback,
            eval_callback,
            checkpoint_callback,
            pruning_callback,
        ]
    )

    model.learn(
        total_timesteps=wandb.config.total_timesteps,
        callback=callback,
        tb_log_name="Mol_A2C",  # Subdirectory for TensorBoard logs
        progress_bar=True,
    )

    model.save(paths["final_model_save_path"])

    wandb.finish()

    env.close()
    eval_env.close()
    logger.info(f"Experiment '{experiment_name}' with run ID '{run_id}' finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the A2C training")
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="Name of the experiment"
    )
    parser.add_argument("--run_id", type=str, help="ID of the run")

    args = parser.parse_args()
    log_dir = f"src/models/a2c/logs/{args.run_id}/"
    os.makedirs(log_dir, exist_ok=True)

    setup_logging(args.experiment_name, log_dir=log_dir)
    logger = logging.getLogger(__name__)
    main(experiment_name=args.experiment_name, run_id=args.run_id)
