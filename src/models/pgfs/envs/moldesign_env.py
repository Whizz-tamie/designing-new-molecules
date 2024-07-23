import logging
import os
import pickle

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from rdkit import Chem
from rdkit.Chem import QED, AllChem, DataStructs, Draw
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from src.models.pgfs.utility.reaction_manager import ReactionManager
from src.models.pgfs.utility.render_helper import render_steps

# Configure the logger
logger = logging.getLogger(__name__)


class MoleculeDesignEnv(gym.Env):
    """
    A Gym environment for molecular design using RDKit for handling chemical data.

    Parameters:
    - reactants (dict): A dictionary of SMILES strings representing possible reactants and their molecule fingerprints.
    - templates (dict): A dictionary of reaction templates including their names, smarts, and types.
    - render_mode (str): Specifies the mode used for rendering ('human', 'rgb_array').
    """

    metadata = {"render_modes": ["human", "console"]}

    def __init__(self, reactant_file, template_file, max_steps=5, render_mode=None):
        super(MoleculeDesignEnv, self).__init__()

        self.max_steps = max_steps
        self.render_mode = render_mode
        self.current_state = None
        self.current_step = 0
        self.steps_log = {}

        self.reactants, self.templates = self._load_data(reactant_file, template_file)
        self.reaction_manager = ReactionManager(self.templates, self.reactants)
        self.action_space = spaces.Discrete(len(self.templates))
        self.reactant_action_space = None
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1024,), dtype=np.float32
        )

        # Variables for reward function
        self.previous_qed = 0.0
        self.epsilon = 0.1  # reward for longer synthesis route
        self.penalty = -1  # Penalty factor for invalid molecules

        logger.info("MoleculeDesignEnv instance created...")

    def _load_data(self, reactant_file, template_file):
        with open(reactant_file, "rb") as f:
            reactants = pickle.load(f)
        with open(template_file, "rb") as f:
            templates = pickle.load(f)

        self._validate_data(reactants, ExplicitBitVect, "Reactants")
        self._validate_data(templates, dict, "Templates")
        return reactants, templates

    def _validate_data(self, data, expected_type, message_prefix):
        if not isinstance(data, dict):
            raise ValueError(f"{message_prefix} should be a dictionary.")
        for key, item in data.items():
            if not isinstance(item, expected_type):
                raise ValueError(
                    f"{message_prefix} with key {key} is not a {expected_type.__name__}."
                )

    def _validate_smiles(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except Exception as e:
            logger.error("Invalid SMILES string: %s. Error: %s", smiles, str(e))
            return None

    def reset(self, seed=None, options=None):
        logger.debug("Resetting the environment...")
        super().reset()
        valid_molecule = False
        retry_count = 0
        while not valid_molecule and retry_count < 10:
            self.current_state = self.np_random.choice(list(self.reactants))
            if self._validate_smiles(self.current_state):
                valid_molecule = True
            retry_count += 1
        if not valid_molecule:
            raise ValueError(
                "Failed to select a valid initial molecule after multiple attempts..."
            )

        self.current_step = 0
        self.steps_log = {}
        self.previous_qed = 0.0  # Reset previous QED

        logger.info(
            "Randomly selected a new starting reactant: %s in %s tries",
            self.current_state,
            retry_count,
        )
        return self._get_obs(), self._get_info()

    def _get_obs(self, smiles=None):
        smiles = smiles if smiles is not None else self.current_state

        if smiles is None:
            return np.zeros(1024, dtype=np.float32)

        mol = self._validate_smiles(smiles)
        if mol is None:
            return np.zeros(1024, dtype=np.float32)
        fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
        fingerprint = fpgen.GetFingerprint(mol)

        logger.debug("Getting the observation for reactant: %s", smiles)

        return np.array(fingerprint, dtype=np.float32)

    def _get_info(self):
        """Provides auxiliary information about the current state."""
        logger.debug(
            "Getting auxiliary information for reactant: %s", self.current_state
        )
        if self.current_state is None:
            return {"SMILES": None, "QED": 0}

        try:
            mol = self._validate_smiles(self.current_state)
            qed = QED.qed(mol) if mol else 0
            return {"SMILES": self.current_state, "QED": qed}
        except Exception as e:
            logger.error(
                "Failed to calculate QED for SMILES: %s, Error: %s",
                self.current_state,
                str(e),
            )
            return {"SMILES": self.current_state, "QED": 0}

    def step(self, action):
        logger.debug("Starting the synthesis route...")
        self.current_step += 1
        logger.info("Step %s ...", self.current_step)

        try:
            template_index, reactant = action
            logger.debug(
                "Action received from KNN - template_index: %s, Reactant: %s",
                template_index,
                reactant,
            )
            template = self.templates.get(template_index)
            if not template:
                raise ValueError("Template index %s out of range." % template_index)

            self.previous_qed = self._get_info()["QED"]

            new_state = self.reaction_manager.apply_reaction(
                self.current_state, template["smarts"], reactant
            )

            if new_state is None:
                logger.warning(
                    "Reaction did not produce a valid new state, Ending episode..."
                )

            if new_state is not None:
                self.steps_log[self.current_step] = {
                    "r1": self.current_state,
                    "template": template["name"],
                    "r2": reactant,
                    "product": new_state,
                }
                logger.debug(
                    "Saving log for Step %s: %s",
                    self.current_step,
                    self.steps_log[self.current_step],
                )

            self.current_state = new_state

            observation = self._get_obs()
            reward = self._get_reward()
            terminated = self.current_step >= self.max_steps
            truncated = not self.reaction_manager.get_mask(new_state).any()
            info = self._get_info()

            return observation, reward, terminated, truncated, info

        except Exception as e:
            logger.error("Error during step execution: %s", str(e))
            return None, 0, False, True, {"error": str(e)}

    def _get_reward(self):
        mol = self._validate_smiles(self.current_state)
        if mol:
            current_qed = self._get_info()["QED"]
            delta_qed = current_qed - self.previous_qed

            reward = delta_qed + self.epsilon
        else:
            reward = self.penalty  # Penalty for invalid molecule

        return reward

    def _render_human(self, save_path=None):
        if len(self.steps_log) < 1:
            return

        if save_path:
            directory = os.path.dirname(save_path)
            os.makedirs(directory, exist_ok=True)

        logger.debug(
            "Rendering environment in human mode and saving to %s...", save_path
        )
        render_steps(self.steps_log, save_path)

    def render(self, mode="human", save_path=None):
        if mode == "human":
            self._render_human(save_path)
        elif mode == "console":
            self._render_console()

    def _render_console(self):
        if len(self.steps_log) < 1:
            return

        logger.debug("Rendering environment in console mode...")

        for i, step in enumerate(self.steps_log.values()):
            initial_reactant = step.get("r1", "None")
            template_name = step.get("template", "None")
            second_reactant = step.get("r2", "None")
            product = step.get("product", "None")

            # Calculate QED values where possible, handle missing or invalid SMILES gracefully
            try:
                initial_qed = (
                    QED.qed(self._validate_smiles(initial_reactant))
                    if initial_reactant
                    else "No initial reactant"
                )
                second_qed = (
                    QED.qed(self._validate_smiles(second_reactant))
                    if second_reactant and second_reactant != "None"
                    else "N/A"
                )
                product_qed = (
                    QED.qed(self._validate_smiles(product))
                    if product
                    else "No product formed"
                )

                # Format output for clarity
                print(
                    f"Step {i+1}:",
                    f"Initial Reactant: {initial_reactant} - QED: {initial_qed},",
                    f"Template: {template_name},",
                    f"Second Reactant: {second_reactant} - QED: {second_qed},",
                    f"Product: {product} - QED: {product_qed}",
                )
            except Exception as e:
                print(f"Error processing step {i+1}: {str(e)}")

    def close(self):
        """Performs any necessary cleanup."""
        plt.close("all")
