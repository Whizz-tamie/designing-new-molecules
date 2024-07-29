# molecule_design_env.py

import logging
import os
import pickle

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from src.models.ppo.utility import render_helper
from src.models.ppo.utility.reaction_manager import ReactionManager

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

    def __init__(
        self,
        reactant_file,
        template_file,
        max_steps=5,
        render_mode=None,
        use_multidiscrete=True,
    ):
        super(MoleculeDesignEnv, self).__init__()

        self.max_steps = max_steps
        self.render_mode = render_mode
        self.current_state = None
        self.current_step = 0
        self.steps_log = {}
        self.permanent_log = {}  # New log to hold data across resets

        self.reactants, self.templates = self._load_data(reactant_file, template_file)
        self.num_templates = len(self.templates)
        self.num_reactants = len(self.reactants)

        self.use_multidiscrete = use_multidiscrete
        if self.use_multidiscrete:
            self.action_space = spaces.MultiDiscrete(
                [self.num_templates, self.num_reactants]
            )
        else:
            self.templates = {
                k: v for k, v in self.templates.items() if v["type"] == "unimolecular"
            }
            self.templates = {i: v for i, (k, v) in enumerate(self.templates.items())}
            self.action_space = spaces.Discrete(len(self.templates))

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1024,), dtype=np.float32
        )

        self.reaction_manager = ReactionManager(self.templates, self.reactants)

        # Variables for reward function
        self.previous_qed = 0.0
        self.done = False

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
        super().reset(seed=seed)
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
        self.done = False

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

        # Convert ExplicitBitVect to numpy array
        arr = np.zeros((1024,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fingerprint, arr)

        logger.debug("Getting the observation for reactant: %s", smiles)

        return arr.astype(np.float32)

    def _get_info(self):
        """Provides auxiliary information about the current state."""
        logger.debug(
            "Getting auxillary information for reactant: %s", self.current_state
        )
        if self.current_state is None:
            return {"SMILES": None, "QED": 0.0}

        try:
            mol = self._validate_smiles(self.current_state)
            qed = QED.qed(mol) if mol else 0.0
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
            if self.use_multidiscrete:
                template_index = int(action[0])
                reactant_index = int(action[1])
            else:
                template_index = int(action)
                reactant_index = None

            logger.debug(
                "Template selected: %s, Reactant selected: %s",
                template_index,
                reactant_index,
            )

            template = self.templates.get(template_index)
            if not template:
                raise ValueError("Template index %s out of range." % template_index)

            reactant_smiles = None
            if template["type"] == "bimolecular" and reactant_index is not None:
                valid_reactants = self.reaction_manager.get_valid_reactants(
                    template_index
                )
                if reactant_index < len(valid_reactants):
                    reactant_smiles = valid_reactants[reactant_index]
                else:
                    logger.warning(
                        "Reactant index %s out of range for selected template, using None",
                        reactant_index,
                    )
                    reactant_smiles = None

            new_state = self.reaction_manager.apply_reaction(
                self.current_state, template["smarts"], reactant_smiles
            )

            if new_state is None:
                logger.warning(
                    "Reaction did not produce a valid new state, Ending episode..."
                )

            if new_state is not None:
                self.steps_log[self.current_step] = {
                    "r1": self.current_state,
                    "template": template["name"],
                    "r2": reactant_smiles,
                    "product": new_state,
                }
                self.permanent_log[self.current_step] = {
                    "r1": self.current_state,
                    "template": template["name"],
                    "r2": reactant_smiles,
                    "product": new_state,
                }
                logger.debug(
                    "Saving log for Step %s: %s",
                    self.current_step,
                    self.steps_log[self.current_step],
                )

            self.previous_qed = self._get_info()["QED"]
            self.current_state = new_state

            observation = self._get_obs()
            reward = self._get_reward()
            logger.debug(
                "Current QED: %s, Previous QED: %s, reward: %s",
                self._get_info()["QED"],
                self.previous_qed,
                reward,
            )
            terminated = self.current_step >= self.max_steps
            truncated = not self.reaction_manager.get_mask(new_state).any()
            info = self._get_info()

            self.done = terminated or truncated

            return observation, reward, terminated, truncated, info

        except Exception as e:
            logger.error("Error during step execution: %s", str(e))
            self.done = True
            return None, 0, False, True, {"error": str(e)}

    def _get_reward(self):
        mol = self._validate_smiles(self.current_state)
        if mol:
            current_qed = self._get_info()["QED"]
            delta_qed = current_qed - self.previous_qed
            reward = delta_qed
        else:
            reward = 0

        return round(reward, 3)

    def _render_human(self):
        if len(self.permanent_log) < 1:
            return

        logger.debug("Rendering environment in human mode...")
        fig = render_helper.render_steps(self.permanent_log)
        self.permanent_log.clear()  # Clear after rendering
        return fig

    def render(self):
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "console":
            return self._render_console()

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
