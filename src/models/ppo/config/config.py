# config.py
import os

# WandB configuration
WANDB_PROJECT = "MOLECULE-DESIGN-RL_PPO"
WANDB_ENTITY = "whizz"

# Environment configuration
REACTANT_FILE = "data/new_data/unimolecular_reactants_train.pkl"
EVAL_REACTANT_FILE = "data/new_data/unimolecular_reactants_val.pkl"
TEMPLATE_FILE = "data/new_data/templates.pkl"
ENV_NAME = "MoleculeDesign-v1"
MAX_STEPS = 5
USE_MULTIDISCRETE = False

# Directories for logging and saving models
LOG_DIR = "src/models/ppo/logs/"

# PPO configuration
TOTAL_TIMESTEPS = 1000000
POLICY_TYPE = "MlpPolicy"

# Callback configuration
GRADIENT_SAVE_FREQ = 2500
MODEL_SAVE_FREQ = 50000
EVAL_FREQ = 50000
DETERMINISTIC = True
RENDER = False


def initialize_paths(run_id):
    log_dir = os.path.join(LOG_DIR, f"run_{run_id}")
    model_save_path = f"src/models/ppo/models/run_{run_id}/checkpoints/"
    eval_save_path = f"src/models/ppo/evaluations/run_{run_id}"
    tensorboard_log_dir = "src/models/ppo/tb_logs/"
    final_model_save_path = os.path.join(model_save_path, "final_model")
    eval_model_save_path = f"src/models/ppo/models/run_{run_id}/eval_models/"
    eval_log_path = os.path.join(log_dir, "eval_logs")

    return {
        "model_save_path": model_save_path,
        "eval_save_path": eval_save_path,
        "tensorboard_log_dir": tensorboard_log_dir,
        "final_model_save_path": final_model_save_path,
        "eval_model_save_path": eval_model_save_path,
        "eval_log_path": eval_log_path,
        "log_file": os.path.join(LOG_DIR, f"run_{run_id}/training.log"),
    }
