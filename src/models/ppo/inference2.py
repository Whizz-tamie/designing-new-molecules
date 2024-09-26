import argparse

import gymnasium as gym
import pandas as pd
from rdkit.Chem import QED, AllChem
from sb3_contrib import MaskablePPO

import src.models.ppo.config.config as config


def load_model(model_path):
    """Load the trained MaskablePPO model."""
    model = MaskablePPO.load(model_path)
    return model


def run_inference(model, env):
    """Perform inference using the trained model and collect results."""
    results = []
    for episode, reactant in enumerate(env.reactants.keys()):
        # Set the current state manually to the reactant
        env.current_state = reactant
        env.current_step = 0  # Reset steps counter
        env.steps_log = {}  # Clear previous steps
        env.done = False
        obs = env._get_obs()  # Get initial observation based on the chosen reactant

        # Log step 0 with initial molecule state
        initial_info = env._get_info()
        results.append(
            {
                "path_id": episode,
                "step": 0,
                "reactant": reactant,
                "template": "",
                "product": reactant,  # No reaction yet, so product is same as reactant
                "qed": initial_info["QED"],
                "second_reactant": "",
            }
        )

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if not done:
                # Fetch current step information
                step_info = env.steps_log.get(env.current_step, {})
                results.append(
                    {
                        "path_id": episode,
                        "step": env.current_step,
                        "reactant": step_info.get("r1", ""),
                        "template": step_info.get("template", ""),
                        "product": step_info.get("product", ""),
                        "qed": info["QED"],
                        "second_reactant": step_info.get("r2", ""),
                    }
                )
    return pd.DataFrame(results)


def save_results(results, output_file):
    """Save the collected inference results to a CSV file."""
    results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def main(model_path, output_file):
    env = gym.make(
        config.ENV_NAME,
        reactant_file=config.EVAL_REACTANT_FILE,
        template_file=config.TEMPLATE_FILE,
        max_steps=config.MAX_STEPS,
        use_multidiscrete=config.USE_MULTIDISCRETE,
    )
    model = load_model(model_path)

    # Run inference for each reactant
    results_df = run_inference(model, env)
    save_results(results_df, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with a trained MaskablePPO model."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="inference_results.csv",
        help="Path to save the inference results.",
    )
    args = parser.parse_args()

    main(args.model_path, args.output_file)
