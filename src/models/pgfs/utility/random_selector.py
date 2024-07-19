import logging
import random

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure the logger
logger = logging.getLogger(__name__)


def select_random_action(env, smiles_string):
    logger.info("Using random action selector...")

    mask = env.unwrapped.reaction_manager.get_mask(smiles_string)
    valid_template_indices = torch.where(mask == 1)[0]

    if len(valid_template_indices) == 0:
        logger.error("No valid templates found for the given SMILES string.")
        raise ValueError("No valid templates found for the given SMILES string.")

    # Randomly select one of the valid indices
    selected_index = valid_template_indices[
        torch.randint(len(valid_template_indices), (1,)).item()
    ]
    logger.debug("Randomly selected template index: %s", selected_index.item())

    # Generate a one-hot encoding for the selected index
    template_one_hot = torch.zeros(len(env.unwrapped.templates), device=device)
    template_one_hot[selected_index] = 1
    template_one_hot = template_one_hot.unsqueeze(0)
    logger.debug(
        "Generated a one-hot encoding for template %s: %s",
        selected_index.item(),
        template_one_hot.shape,
    )

    # Check if the selected template requires a bimolecular reaction
    template_type = env.unwrapped.templates[selected_index.item()]["type"]
    if template_type == "bimolecular":
        logger.debug(
            "template %s is %s. Randomly selecting second reactant...",
            selected_index.item(),
            template_type,
        )
        r2 = random.choice(
            env.unwrapped.reaction_manager.get_valid_reactants(selected_index.item())
        )
        logger.debug(
            "Selected second reactant: %s",
            r2,
        )
    else:
        # Zero tensor for unimolecular reactions
        r2 = torch.zeros(
            env.unwrapped.observation_space.shape[0], device=device
        ).unsqueeze(0)
        logger.debug(
            "Template  %s is %s. Using zero tensor for reactant component: %s",
            selected_index.item(),
            template_type,
            r2.shape,
        )
    return (template_one_hot, r2)
