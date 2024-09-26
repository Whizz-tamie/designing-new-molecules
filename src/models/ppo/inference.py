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


def run_inference(model, env, num_episodes=100):
    """Perform inference using the trained model and collect results."""
    results = []
    episode = 0
    while episode < num_episodes:
        obs, info = env.reset()
        done = False

        # Log step 0 with initial molecule state
        results.append(
            {
                "path_id": episode,
                "step": env.current_step,
                "reactant": env.current_state,
                "template": "",
                "product": env.current_state,  # No reaction yet, so product is same as reactant
                "qed": info["QED"],
                "second_reactant": "",
            }
        )

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
        episode += 1
    return pd.DataFrame(results)


def save_results(results, output_file):
    """Save the collected inference results to a CSV file."""
    results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def main(model_path, num_episodes, output_file):
    env = gym.make(
        config.ENV_NAME,
        reactant_file=config.EVAL_REACTANT_FILE,
        template_file=config.TEMPLATE_FILE,
        max_steps=config.MAX_STEPS,
        use_multidiscrete=config.USE_MULTIDISCRETE,
    )
    model = load_model(model_path)
    results_df = run_inference(model, env, num_episodes=num_episodes)
    save_results(results_df, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with a trained MaskablePPO model."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model file."
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to run for inference.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="inference_results.csv",
        help="Path to save the inference results.",
    )
    args = parser.parse_args()

    main(args.model_path, args.num_episodes, args.output_file)
# python inference.py --model_path /path/to/your/model.zip --num_episodes 100 --output_file /path/to/save/inference_results.csv
