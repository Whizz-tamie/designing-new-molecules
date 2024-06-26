# environments/environment.py

import src.models.pgfs.logging_config as logging_config
import uuid
import random
import torch
import logging
from rdkit import Chem
from rdkit.Chem import AllChem, QED
from src.models.pgfs.models.knn import KNN
from src.chem.precompute_vector import load_precomputed_vectors, load_templates, reactant_to_vector, convert_to_tensor


class Environment:
    def __init__(self, precomputed_vectors_file, templates_file, k=1, max_steps=5):
        """
        Initialize the Environment with initial reactants and parameters.

        Args:
            initial_reactants (list): List of initial reactants.
            precomputed_vectors_file (str): Path to the file with precomputed vectors.
            k (int): Number of nearest neighbors to consider.
            max_steps (int): Maximum number of reaction steps per episode.
        """
        logging_config.setup_logging() # Set up logging here
        self.logger = logging.getLogger(__name__)
        self.logger.info("Environment instance created")
        self.precomputed_vectors = load_precomputed_vectors(precomputed_vectors_file)
        self.templates = load_templates(templates_file)
        self.k = k
        self.max_steps = max_steps
        self.current_step = 0
        self.knn = KNN(self.k)
        self.initial_reactants_uuid = list(self.precomputed_vectors.keys())
        self.current_state_info = {}
        self.qed_scores = []  # List to store QED scores
    
    def reset(self):
        """
        Reset the environment to the initial state.

        Returns:
            Tensor: Initial state molecule in tensor format.
        """
        self.current_step = 0
        self.current_state_info = {}
        uid = random.choice(self.initial_reactants_uuid)
        initial_state = self.precomputed_vectors[uid]["vector"]
        template_mask = self.get_template_mask(self.precomputed_vectors[uid]["smiles"])
        return initial_state.unsqueeze(0), uid, template_mask
    
    def one_hot_to_smarts(self, one_hot_template: torch.Tensor) -> str:
        """
        Convert one-hot encoded template to SMARTS template.

        Args:
            one_hot_template (torch.Tensor): One-hot encoded template.

        Returns:
            str: Corresponding SMARTS template.
        """
        template_index = torch.argmax(one_hot_template).item()
        return self.templates[template_index]
    
    def get_template_mask(self, reactant: str) -> torch.Tensor:
        """
        Identify valid reaction templates for a given reactant.

        Args:
            reactant (str): Reactant in SMILES format.

        Returns:
            list: Valid reaction templates that match the reactant.
        """
        try:
            template_mask = [int(self.match_template(reactant, t)["first"]) for t in self.templates]
            self.logger.info(f"Found {sum(template_mask)} valid templates for reactant: {reactant}")
            
            return torch.tensor(template_mask, dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            self.logger.error(f"Error in getting valid templates: {e}")
            return torch.ones(1,  102)
    
    def get_valid_reactants(self, template: str) -> list:
        """
        Identify valid reactants for a given reaction template.

        Args:
            template (str): Reaction template in SMARTS format.

        Returns:
            list: Valid reactants that match the template.
        """
        try:
            valid_reactants_uuid = [r for r in self.precomputed_vectors.keys() if self.match_template(self.precomputed_vectors[r]['smiles'], template)["second"]]
            self.logger.info(f"Found {len(valid_reactants_uuid)} valid second reactants for template {template}")
            return [(r,self.precomputed_vectors[r]['vector']) for r in valid_reactants_uuid]
        except Exception as e:
            self.logger.error(f"Error in getting valid reactants: {e}")
            return []

    def match_template(self, reactant: str, template: str) -> bool:
        """
        Check if a reactant matches the reaction template.

        Args:
            reactant (str): Reactant molecule in SMILES format.
            template (str): Reaction template in SMARTS format.

        Returns:
            bool: True if reactant matches the template, else False.
        """
        try:
            reaction = AllChem.ReactionFromSmarts(template)
            reactant_mol = Chem.MolFromSmiles(reactant)
            
            if reactant_mol is None:
                return {"first": False, "second": False}

            num_reactants = reaction.GetNumReactantTemplates()
            match_first = False
            match_second = False

            if num_reactants == 1:
                reactant1_template = reaction.GetReactantTemplate(0)
                match_first = reactant_mol.HasSubstructMatch(reactant1_template, useChirality=True)
            elif num_reactants == 2:
                reactant1_template = reaction.GetReactantTemplate(0)
                reactant2_template = reaction.GetReactantTemplate(1)
                match_first = reactant_mol.HasSubstructMatch(reactant1_template, useChirality=True)
                match_second = reactant_mol.HasSubstructMatch(reactant2_template, useChirality=True)
            
            return {"first": match_first, "second": match_second}
        except Exception as e:
            self.logger.error(f"Error in matching template: {e}")
            return {"first": False, "second": False}

    def forward_reaction(self, state: str, template: str, reactant: str) -> list:
        """
        Compute the next state based on the current state and selected reactants.

        Args:
            state (str): Current state molecule in SMILES format.
            template (str): Reaction template in SMARTS format.
            reactants (list): List of selected reactants.

        Returns:
            str: Next state molecule in SMILES format.
        """          
        try:
            reaction = AllChem.ReactionFromSmarts(template)
            state_mol = Chem.MolFromSmiles(state)
            
            if state_mol is None:
                self.logger.error(f"Invalid state molecule: {state}")
                return None

            num_reactants = reaction.GetNumReactantTemplates()

            if num_reactants == 1:
                product_sets = reaction.RunReactants((state_mol,))
                self.logger.info(f"Unimolecular reaction:: Generated {len(product_sets)} using Reactant:{state} Template: {template}")
            elif num_reactants == 2:
                if reactant is None:
                    self.logger.error("No reactants provided for bimolecular reaction")
                    return None
                reactant_mol = Chem.MolFromSmiles(reactant)
                if reactant_mol is None:
                    self.logger.error(f"Invalid reactant molecule: {reactant}")
                    return None
                product_sets = reaction.RunReactants((state_mol, reactant_mol))
                self.logger.info(f"Bimolecular reaction:: Generated {len(product_sets)} using Reactant:{state} Template:{template}, Second Reactant:{reactant}")
            else:
                self.logger.error(f"Unexpected number of reactants: {num_reactants}")
                return None
            
            if not product_sets:
                self.logger.info("No products generated from reaction")
                return None
        
            # Check and correct mapping
            corrected_products = []
            for product_set in product_sets:
                for product in product_set:
                    try:
                        Chem.SanitizeMol(product)
                        corrected_products.append(product)
                    except Exception as e:
                        self.logger.error(f"Error in sanitization: {e}")
                        continue

            self.logger.info(f"Found {len(corrected_products)} after sanitization")
            return corrected_products

        except Exception as e:
            self.logger.error(f"Error in forward reaction: {e}")
            return None
        
    def scoring_function(self, state: Chem.Mol) -> float:
        """
        Compute the reward for a given state using QED.

        Args:
            state (rdkit.Chem.Mol): State molecule in RDKit molecule format.

        Returns:
            float: Reward value.
        """
        if state is None:
            return 0.0
        try:
            return QED.qed(state)
        except Exception as e:
            self.logger.error(f"Error in scoring function: {e}")
            return 0.0

    def step(self, state_uid: str, one_hot_template: torch.Tensor, action: torch.Tensor) -> tuple:
        """
        Execute a step in the environment given the current state, action, and template.

        Args:
            state (str): Current state UUID.
            one_hot_template (torch.Tensor): Reaction template in one hot encoding.
            action (torch.Tensor): Action vector.

        Returns:
            tuple: Next state tensor, next state UUID, reward, done flag, and next_tmask.
        """
        if self.current_step == 0:
            state_smiles = self.precomputed_vectors[state_uid]["smiles"]
        else:
            state_smiles = self.current_state_info[state_uid]["smiles"] 
            self.current_state_info = {}
        
        self.current_step += 1
        self.logger.info(f"Step {self.current_step}: Starting Reactant:{state_smiles}")

        # Convert one-hot encoded template to SMARTS template
        template = self.one_hot_to_smarts(one_hot_template)
        self.logger.info(f"Predicted template:{template}")

        # Check the number of reactants required by the template
        reaction = AllChem.ReactionFromSmarts(template)
        num_reactants = reaction.GetNumReactantTemplates()

        if num_reactants == 1:
            # Unimolecular reaction
            next_state_mols = self.forward_reaction(state_smiles, template, None)
        else:
            # Bimolecular reaction
            # Get valid reactants for the given template
            valid_reactants = self.get_valid_reactants(template)
            if not valid_reactants:
                self.logger.info(f"No valid second reactants for template:{template}")
                return convert_to_tensor(reactant_to_vector(state_smiles)).unsqueeze(0), state_uid, 0.0, True, self.get_template_mask(state_smiles)

            # Find the k-nearest neighbor for the action among valid reactants
            selected_reactant = self.knn.find_neighbors(action, valid_reactants)[0]
            selected_reactant = self.precomputed_vectors[selected_reactant]['smiles']
            self.logger.info(f"KNN action:{selected_reactant}")
            next_state_mols = self.forward_reaction(state_smiles, template, selected_reactant)

        if next_state_mols is None or next_state_mols == []:
            return convert_to_tensor(reactant_to_vector(state_smiles)).unsqueeze(0), state_uid, 0.0, True, self.get_template_mask(state_smiles)

        # Compute the rewards for the generated states (molecules)
        rewards = [self.scoring_function(p) for p in next_state_mols]
        self.logger.info(f"Rewards:{rewards}")

        # Select the product and reward corresponding to the maximum reward
        max_reward_idx = torch.argmax(torch.tensor(rewards))
        next_state_mols = next_state_mols[max_reward_idx]
        next_state_smiles = Chem.MolToSmiles(next_state_mols)
        self.logger.info(f"Generated molecule with highest QED:{max_reward_idx}{next_state_smiles}")
        next_state_fpt = convert_to_tensor(reactant_to_vector(next_state_smiles))
        reward = rewards[max_reward_idx]
        self.qed_scores.append(reward)  # Log QED score

        # Generate new UUID for the next state
        next_state_uuid = str(uuid.uuid4())

        # Update the current state info
        self.current_state_info[next_state_uuid] = {
            'smiles': next_state_smiles,
            'tensor': next_state_fpt
        }

        next_tmask = self.get_template_mask(next_state_smiles)

        # Check if the episode is done
        done = self.is_done()
        return next_state_fpt.unsqueeze(0), next_state_uuid, reward, done, next_tmask

    def is_done(self) -> bool:
        """
        Check if the episode has ended based on the state and current step.

        Returns:
            bool: True if the episode has ended, else False.
        """
        if self.current_step >= self.max_steps:
            return True

        return False
