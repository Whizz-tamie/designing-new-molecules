# inference.py

import torch
import wandb
from rdkit import Chem
from rdkit.Chem import QED
from src.models.pgfs.environments.environment import Environment
from src.models.pgfs.train.replay_buffer import ReplayBuffer
from src.models.pgfs.train.td3 import TD3
import yaml

def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

def calculate_penalized_clogp(mol):
    try:
        logp = Chem.Crippen.MolLogP(mol)
        qed = QED.qed(mol)
        return logp - qed
    except Exception as e:
        return None

def run_inference(config, checkpoint_path, validation_set, num_molecules=2000):
    # Load configuration
    model_config = config['model']
    training_config = config['training']
    datasets_config = config['dataset']

    state_dim = model_config["state_dim"]
    template_dim = model_config["template_dim"]
    action_dim = model_config["action_dim"]
    max_action = model_config["max_action"]
    gamma = training_config["gamma"]
    tau = training_config["tau"]
    policy_freq = training_config["policy_freq"]

    initial_temperature = config['temperature']["initial_temperature"]
    min_temperature = config['temperature']["min_temperature"]
    temp_decay = config['temperature']["temp_decay"]

    noise_std = config['noise']["noise_std"]
    noise_clip = config['noise']["noise_clip"]

    actor_lr = float(config['optimizers']["actor_lr"])
    critic_lr = float(config['optimizers']["critic_lr"])

    molecule_file = datasets_config["precomputed_vectors_file"]
    template_file = datasets_config["templates_file"]

    # Initialize environment
    env = Environment(precomputed_vectors_file=molecule_file, templates_file=template_file)
    
    # Initialize the TD3 model
    td3 = TD3(
        state_dim, template_dim, action_dim, max_action, gamma, tau, policy_freq,
        initial_temperature, torch.ones(1, template_dim), noise_std, noise_clip, min_temperature, temp_decay,
        actor_lr, critic_lr
    )

    # Load the checkpoint
    td3.load(checkpoint_path)

    # Set model to evaluation mode
    td3.actor_f.eval()
    td3.actor_pi.eval()

    generated_molecules = []

    for state, state_uid in validation_set[:num_molecules]:
        done = False
        reaction_chain = []
        while not done:
            with torch.no_grad():
                # Select action according to policy
                template, action = td3.select_action(state)
                # Perform action in environment
                next_state, next_state_uid, reward, done = env.step(state_uid, template, action)
                
                # Store the step details
                step_details = {
                    'state': state,
                    'state_uid': state_uid,
                    'next_state': next_state,
                    'next_state_uid': next_state_uid,
                    'template': template,
                    'action': action,
                    'reward': reward
                }
                reaction_chain.append(step_details)

                # Update state and state_uid
                state = next_state
                state_uid = next_state_uid

                if done:
                    break

        generated_molecules.append(reaction_chain)
    
    return generated_molecules

def main():
    # Load configuration
    config = load_config("config.yaml")

    # Initialize Weights & Biases
    wandb.init(project=config['project'], entity=config['entity'])

    # Load validation set
    validation_set = [  # Example validation set
        (torch.randn(1, 1024), str(i)) for i in range(2000)
    ]

    # Run inference
    checkpoint_path = "/rds/user/gtj21/hpc-work/designing-new-molecules/src/models/pgfs/checkpoints/final_model.pth"
    generated_molecules = run_inference(config, checkpoint_path, validation_set)

    # Log generated molecules and their metrics to W&B
    for idx, reaction_chain in enumerate(generated_molecules):
        for step_idx, step in enumerate(reaction_chain):
            next_state_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(step['next_state']))
            penalized_clogp = calculate_penalized_clogp(Chem.MolFromSmiles(next_state_smiles))
            qed_score = QED.qed(Chem.MolFromSmiles(next_state_smiles))

            wandb.log({
                f"molecule_{idx}_step_{step_idx}/state": step['state'],
                f"molecule_{idx}_step_{step_idx}/state_uid": step['state_uid'],
                f"molecule_{idx}_step_{step_idx}/next_state": step['next_state'],
                f"molecule_{idx}_step_{step_idx}/next_state_uid": step['next_state_uid'],
                f"molecule_{idx}_step_{step_idx}/template": step['template'],
                f"molecule_{idx}_step_{step_idx}/action": step['action'],
                f"molecule_{idx}_step_{step_idx}/reward": step['reward'],
                f"molecule_{idx}_step_{step_idx}/qed_score": qed_score,
                f"molecule_{idx}_step_{step_idx}/penalized_clogp": penalized_clogp
            })

if __name__ == "__main__":
    main()
