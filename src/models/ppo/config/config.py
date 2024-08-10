# config.py
import os

# WandB configuration
WANDB_PROJECT = "MolSynthRL"
WANDB_ENTITY = "whizz"

# Environment configuration
REACTANT_FILE = "data/new_data/unimolecular_reactants_train.pkl"
EVAL_REACTANT_FILE = "data/new_data/unimolecular_reactants_val.pkl"
TEMPLATE_FILE = "data/new_data/templates.pkl"
ENV_NAME = "MoleculeDesign-v1"
MAX_STEPS = 5
USE_MULTIDISCRETE = False
N_ENVS = 10

# Directories for logging and saving models
BASE_DIR = "src/models/ppo/"
LOG_DIR = BASE_DIR + "logs/"

# PPO configuration
TOTAL_TIMESTEPS = 1000000
POLICY_TYPE = "MlpPolicy"

# Callback configuration
GRADIENT_SAVE_FREQ = 100
MODEL_SAVE_FREQ = 5000
EVAL_FREQ = 100000
DETERMINISTIC = True
RENDER = False
N_EVAL_EPISODES = 7
CHECK_FREQ = 5000


def initialize_paths(run_id):
    log_dir = LOG_DIR + f"run_{run_id}/"
    model_save_path = BASE_DIR + f"/models/run_{run_id}/checkpoints/"
    tensorboard_log_dir = BASE_DIR + "/tb_logs/"
    final_model_save_path = model_save_path + "ppo_moldesign"
    eval_model_save_path = BASE_DIR + f"/models/run_{run_id}/eval_models/"
    eval_log_path = log_dir + "eval_logs"

    return {
        "model_save_path": model_save_path,
        "tensorboard_log_dir": tensorboard_log_dir,
        "final_model_save_path": final_model_save_path,
        "eval_model_save_path": eval_model_save_path,
        "eval_log_path": eval_log_path,
    }
