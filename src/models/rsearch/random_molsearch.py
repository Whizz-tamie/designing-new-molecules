# random_molsearch.py

import logging
import os
import pickle
import random
from datetime import datetime

import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, AllChem, Crippen

import wandb

log_file_name = f"Rmolsearch_uni.log"
log_dir = "/rds/user/gtj21/hpc-work/designing-new-molecules/src/models/rsearch/logs"

# Configure logging
logging.basicConfig(
    filename=os.path.join(log_dir, log_file_name),
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


class SynthesisStep:
    def __init__(self, step, reactant, template, product, clogp, second_reactant=None):
        self.step = step
        self.reactant = reactant
        self.template = template
        self.product = product
        self.clogp = clogp
        self.second_reactant = second_reactant

    def to_dict(self, path_id):
        return {
            "path_id": path_id,
            "step": self.step,
            "reactant": self.reactant,
            "template": self.template,
            "product": self.product,
            "clogp": self.clogp,
            "second_reactant": self.second_reactant,
        }


class SynthesisPath:
    def __init__(self, path_id):
        self.path_id = path_id
        self.steps = []

    def add_step(self, step):
        self.steps.append(step)

    def to_dict(self):
        return [step.to_dict(self.path_id) for step in self.steps]


class RandomMolSearch:
    def __init__(
        self,
        reactants,
        templates,
        experiment_name,
        checkpoint_file,
        result_file,
        max_steps=5,
        max_reactions=100,
        max_attempts=100,
    ):
        self.reactants = reactants
        self.templates = templates
        self.max_steps = max_steps
        self.max_reactions = max_reactions
        self.max_attempts = max_attempts
        self.paths = []
        self.saved_path_ids = set()
        self.total_reactions = 0
        self.attempts = 0
        self.no_compatible_templates = set()
        self.path_id_counter = 0
        self.checkpoint_file = checkpoint_file
        self.result_file = result_file

        # Initialize Weights & Biases
        wandb.init(
            project="random_molsearch",
            entity="whizz",
            notes="random search molecule design baseline",
            name=experiment_name,
            id="rsearch_uni",
            resume="allow",
        )

        # Load checkpoint if it exists
        self.load_checkpoint()

    def compute_qed(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return round(QED.qed(mol), 3)
            return None
        except Exception as e:
            logging.error(f"Error computing QED for SMILES {smiles}: {e}")
            return None

    def compute_clogp(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return round(Crippen.MolLogP(mol), 3)
            return None
        except Exception as e:
            logging.error(f"Error computing clogP for SMILES {smiles}: {e}")
            return None

    def sanitize_molecule(self, mol):
        try:
            Chem.SanitizeMol(mol)
            return True
        except Chem.MolSanitizeException as e:
            logging.error(
                f"Sanitization error for molecule {Chem.MolToSmiles(mol)}: {e}"
            )
            return False

    def forward_reaction(self, reactant1, smarts, reactant2=None):
        reactant1_mol = Chem.MolFromSmiles(reactant1)
        if not reactant1_mol:
            logging.error(f"Invalid reactant1 SMILES: '{reactant1}'")
            return None

        rxn = AllChem.ReactionFromSmarts(smarts)
        if reactant2 is not None:
            reactant2_mol = Chem.MolFromSmiles(reactant2)
            if not reactant2_mol:
                logging.error(f"Invalid reactant2 SMILES: '{reactant2}'")
                return None
            products = rxn.RunReactants((reactant1_mol, reactant2_mol))
        else:
            products = rxn.RunReactants((reactant1_mol,))

        if products:
            sanitized_smiles = [
                Chem.MolToSmiles(product)
                for product_set in products
                for product in product_set
                if self.sanitize_molecule(product)
            ]
            return sanitized_smiles if sanitized_smiles else None

        return None

    def match_template(self, reactant: str, template: str) -> bool:
        try:
            rxn = AllChem.ReactionFromSmarts(template)
            reactant_mol = Chem.MolFromSmiles(reactant)

            if reactant_mol is None:
                return {"first": False, "second": False}

            num_reactants = rxn.GetNumReactantTemplates()
            match_first = False
            match_second = False

            if num_reactants == 1:
                reactant1_template = rxn.GetReactantTemplate(0)
                match_first = reactant_mol.HasSubstructMatch(
                    reactant1_template, useChirality=True
                )
            elif num_reactants == 2:
                reactant1_template = rxn.GetReactantTemplate(0)
                reactant2_template = rxn.GetReactantTemplate(1)
                match_first = reactant_mol.HasSubstructMatch(
                    reactant1_template, useChirality=True
                )
                match_second = reactant_mol.HasSubstructMatch(
                    reactant2_template, useChirality=True
                )

            return {"first": match_first, "second": match_second}
        except Exception as e:
            logging.error(f"Error in matching template: {e}")
            return {"first": False, "second": False}

    def generate_new_molecule(self, current_reactant, template, step, path):
        second_reactant = None
        products = None
        rxn = AllChem.ReactionFromSmarts(template.Smarts)
        if rxn.GetNumReactantTemplates() == 1:
            products = self.forward_reaction(current_reactant, template.Smarts)
        elif rxn.GetNumReactantTemplates() == 2:
            compatible_reactants = [
                r
                for r in self.reactants
                if Chem.MolFromSmiles(r).HasSubstructMatch(rxn.GetReactantTemplate(1))
            ]
            if not compatible_reactants:
                logging.info(
                    f"No compatible second reactant for reaction {template.Reaction}"
                )
                return current_reactant, False
            second_reactant = random.choice(compatible_reactants)
            logging.info(
                f"Found {len(compatible_reactants)} valid secod reactants for {current_reactant} and randomly select {second_reactant}"
            )
            products = self.forward_reaction(
                current_reactant, template.Smarts, reactant2=second_reactant
            )

        if products:
            clogp_scores = [self.compute_clogp(product) for product in products]
            best_clogp = max(clogp_scores)
            best_products = [
                products[i]
                for i in range(len(products))
                if clogp_scores[i] == best_clogp
            ]
            best_product = random.choice(best_products)
            clogp_score = best_clogp
            step_info = SynthesisStep(
                step=step,
                reactant=current_reactant,
                template=template.Reaction,
                product=best_product,
                clogp=clogp_score,
                second_reactant=second_reactant,
            )
            path.add_step(step_info)
            current_reactant = best_product
            self.total_reactions += 1
            return current_reactant, True
        else:
            logging.info(
                f"No valid products generated for {current_reactant} using reaction template {template.Reaction}"
            )
            return current_reactant, False

    def save_results_to_csv(self, filename):
        new_results = []
        for path in self.paths:
            if path.path_id not in self.saved_path_ids:
                new_results.extend(path.to_dict())
                self.saved_path_ids.add(path.path_id)

        if new_results:
            df = pd.DataFrame(new_results)
            if not os.path.isfile(filename):
                df.to_csv(filename, index=False)
            else:
                df.to_csv(filename, mode="a", header=False, index=False)

    def load_existing_results(self):
        if os.path.isfile(self.result_file):
            df = pd.read_csv(self.result_file)
            if not df.empty:
                self.saved_path_ids = set(df["path_id"].unique())
                self.path_id_counter = max(self.saved_path_ids) + 1

    def save_checkpoint(self):
        checkpoint = {
            "paths": self.paths,
            "saved_path_ids": self.saved_path_ids,
            "total_reactions": self.total_reactions,
            "attempts": self.attempts,
            "no_compatible_templates": self.no_compatible_templates,
            "path_id_counter": self.path_id_counter,
        }
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(checkpoint, f)
        logging.info(f"Checkpoint saved to {self.checkpoint_file}...")
        self.save_results_to_csv(self.result_file)
        logging.info(f"Results saved to {self.result_file}...")

    def load_checkpoint(self):
        if os.path.isfile(self.checkpoint_file):
            with open(self.checkpoint_file, "rb") as f:
                checkpoint = pickle.load(f)
                self.paths = checkpoint["paths"]
                self.saved_path_ids = checkpoint["saved_path_ids"]
                self.total_reactions = checkpoint["total_reactions"]
                self.attempts = checkpoint["attempts"]
                self.no_compatible_templates = checkpoint["no_compatible_templates"]
                self.path_id_counter = checkpoint["path_id_counter"]
            logging.info(f"Resumed from checkpoint {self.checkpoint_file}...")
        else:
            logging.info("No checkpoint found. Starting from scratch....")
            self.load_existing_results()

    def run_molsearch(self):
        try:
            logging.info(
                f"Starting Random molecule search... Date:{datetime.now().strftime('%Y%m%d_%H%M%S')}..."
            )
            while (
                self.total_reactions < self.max_reactions
                and self.attempts < self.max_attempts
            ):
                initial_reactant = random.choice(self.reactants)
                logging.info(
                    f"Path:{self.path_id_counter}: Randomly selected starting reactant:{initial_reactant}"
                )

                if initial_reactant in self.no_compatible_templates:
                    self.attempts += 1
                    continue

                current_reactant = initial_reactant
                path = SynthesisPath(path_id=self.path_id_counter)
                initial_step = SynthesisStep(
                    step=0,
                    reactant=initial_reactant,
                    template=None,
                    product=initial_reactant,
                    clogp=self.compute_clogp(initial_reactant),
                )
                path.add_step(initial_step)
                valid_sequence = False

                for step in range(1, self.max_steps + 1):
                    logging.info(
                        f"Path:{self.path_id_counter} - Step:{step} Reactant:{current_reactant}"
                    )
                    if self.total_reactions >= self.max_reactions:
                        logging.info(
                            f"Max number of reactions reached...: {self.max_reactions}"
                        )
                        break

                    compatible_templates = [
                        template
                        for _, template in self.templates.iterrows()
                        if self.match_template(current_reactant, template.Smarts)[
                            "first"
                        ]
                    ]
                    if not compatible_templates:
                        logging.info(
                            f"Path:{self.path_id_counter} ends in Step:{step}: No compatible template for reactant:{current_reactant}"
                        )
                        self.no_compatible_templates.add(current_reactant)
                        break

                    template = random.choice(compatible_templates)
                    logging.info(
                        f"Found {len(compatible_templates)} templates for {current_reactant} and randomly selected {template.Reaction}..."
                    )
                    current_reactant, valid_step = self.generate_new_molecule(
                        current_reactant, template, step, path
                    )
                    if not valid_step:
                        logging.info(
                            f"Path:{self.path_id_counter} ends in Step:{step}: No products generated by reactions..."
                        )
                        break

                    valid_sequence = True

                if valid_sequence:
                    self.paths.append(path)
                    self.path_id_counter += 1
                else:
                    self.attempts += 1

                wandb.log(
                    {
                        "total_reactions": self.total_reactions,
                        "total_attempts": self.attempts,
                        "num_steps": max(1, len(path.steps) - 1),
                        "num_paths": self.path_id_counter,
                    }
                )

                if self.total_reactions > 0 and self.total_reactions % 1000 == 0:
                    self.save_checkpoint()

            if self.total_reactions >= self.max_reactions:
                logging.info(
                    f"Terminated because the total number of reactions {self.total_reactions} reached the maximum limit of {self.max_reactions}. There were {self.attempts} attempts..."
                )
            elif self.attempts >= self.max_attempts:
                logging.info(
                    f"Terminated because the number of attempts {self.attempts} reached the maximum limit of {self.max_attempts}. There were {self.total_reactions} reactions..."
                )

            logging.info("Molecule search completed...")
            wandb.finish()
        except Exception as e:
            logging.error(f"Error during molecule search: {e}")
            self.save_checkpoint()
            raise
