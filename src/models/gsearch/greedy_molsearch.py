import logging
from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit.Chem import Crippen
import pandas as pd
import os
import wandb
import pickle
from datetime import datetime

log_file_name = f"Gmolsearch_uni.log"
log_dir = "/rds/user/gtj21/hpc-work/designing-new-molecules/src/models/gsearch/logs"

# Configure logging
logging.basicConfig(filename=os.path.join(log_dir, log_file_name), level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

class SynthesisStep:
    def __init__(self, step, reactant, template, product, qed, second_reactant=None):
        self.step = step
        self.reactant = reactant
        self.template = template
        self.product = product
        self.qed = qed
        self.second_reactant = second_reactant

    def to_dict(self, path_id):
        return {
            "path_id": path_id,
            "step": self.step,
            "reactant": self.reactant,
            "template": self.template,
            "product": self.product,
            "qed": self.qed,
            "second_reactant": self.second_reactant
        }

class SynthesisPath:
    def __init__(self, path_id):
        self.path_id = path_id
        self.steps = []

    def add_step(self, step):
        self.steps.append(step)

    def to_dict(self):
        return [step.to_dict(self.path_id) for step in self.steps]
    

class GreedyMolSearch:
    def __init__(self, reactants, templates, experiment_name, checkpoint_file, result_file, max_steps=5, max_reactions=100):
        self.reactants = reactants
        self.templates = templates
        self.max_steps = max_steps
        self.max_reactions = max_reactions
        self.paths = []
        self.saved_path_ids = set()
        self.total_reactions = 0
        self.path_id_counter = 0
        self.checkpoint_file = checkpoint_file
        self.result_file = result_file

        # Initialize Weights & Biases
        wandb.init(project="greedy_molsearch",
                entity="whizz",
                notes="greedy search molecule design baseline", 
                name=experiment_name,
                id="gsearch_uni",
                resume="allow"
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
            logging.error(f"Sanitization error for molecule {Chem.MolToSmiles(mol)}: {e}")
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
            sanitized_smiles = [Chem.MolToSmiles(product) for product_set in products for product in product_set if self.sanitize_molecule(product)]
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
                match_first = reactant_mol.HasSubstructMatch(reactant1_template, useChirality=True)
            elif num_reactants == 2:
                reactant1_template = rxn.GetReactantTemplate(0)
                reactant2_template = rxn.GetReactantTemplate(1)
                match_first = reactant_mol.HasSubstructMatch(reactant1_template, useChirality=True)
                match_second = reactant_mol.HasSubstructMatch(reactant2_template, useChirality=True)
            
            return {"first": match_first, "second": match_second}
        except Exception as e:
            logging.error(f"Error in matching template: {e}")
            return {"first": False, "second": False}
    
    def generate_new_molecule(self, current_reactant, templates, step, path):
        best_template = None
        best_qed = -1
        best_product = None
        best_second_reactant = None

        for template in templates:
            rxn = AllChem.ReactionFromSmarts(template.Smarts)
            if rxn.GetNumReactantTemplates() == 1:
                self.total_reactions += 1  # Increment for each attempt
                possible_products = self.forward_reaction(current_reactant, template.Smarts)
            elif rxn.GetNumReactantTemplates() == 2:
                compatible_reactants = [
                    r for r in self.reactants 
                    if Chem.MolFromSmiles(r).HasSubstructMatch(rxn.GetReactantTemplate(1))
                ]
                logging.info(f"Found {len(compatible_reactants)} compatible second reactants for bimolecular template {template.Reaction}...")
                if not compatible_reactants:
                    continue

                # Select the second reactant that maximizes the QED score of the resulting product
                for reactant2 in compatible_reactants:
                    self.total_reactions += 1  # Increment for each attempt
                    possible_products = self.forward_reaction(current_reactant, template.Smarts, reactant2=reactant2)
                    if possible_products:
                        qed_scores = [self.compute_qed(product) for product in possible_products]
                        max_qed = max(qed_scores)
                        if max_qed > best_qed:
                            best_qed = max_qed
                            best_template = template
                            best_second_reactant = reactant2
                            best_product = possible_products[qed_scores.index(max_qed)]
            else:
                continue

            if possible_products is not None and best_template is None:
                qed_scores = [self.compute_qed(product) for product in possible_products]
                max_qed = max(qed_scores)
                if max_qed > best_qed:
                    best_qed = max_qed
                    best_template = template
                    best_product = possible_products[qed_scores.index(max_qed)]

        if best_product:
            logging.info(f"Selected best template:{best_template.Reaction} and best second reactant:{best_second_reactant} producing product:{best_product} with highest QED:{best_qed}...")
            step_info = SynthesisStep(step=step, reactant=current_reactant, template=best_template.Reaction, product=best_product, qed=best_qed, second_reactant=best_second_reactant)
            path.add_step(step_info)
            current_reactant = best_product
            return current_reactant, True
        else:
            logging.info(f"No valid products generated for {current_reactant} using any template.")
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
                df.to_csv(filename, mode='a', header=False, index=False)

    def save_checkpoint(self):
        checkpoint = {
            'paths': self.paths,
            'saved_path_ids': self.saved_path_ids,
            'total_reactions': self.total_reactions,
            'path_id_counter': self.path_id_counter,
        }
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        logging.info(f"Checkpoint saved to {self.checkpoint_file}...")
        self.save_results_to_csv(self.result_file)
        logging.info(f"Results saved to {self.result_file}...")

    def load_checkpoint(self):
        if os.path.isfile(self.checkpoint_file):
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
                self.paths = checkpoint['paths']
                self.saved_path_ids = checkpoint['saved_path_ids']
                self.total_reactions = checkpoint['total_reactions']
                self.path_id_counter = checkpoint['path_id_counter']
            logging.info(f"Resumed from checkpoint {self.checkpoint_file}...")
        else:
            logging.info("No checkpoint found. Starting from scratch...")

    def run_molsearch(self):
        try:
            logging.info(f"Starting Greedy molecule search... Date:{datetime.now().strftime('%Y%m%d_%H%M%S')}...")
            for idx, initial_reactant in enumerate(self.reactants):
                if idx < self.path_id_counter:
                    continue  # Skip already processed reactants

                if self.total_reactions >= self.max_reactions:
                    logging.info(f"Max number of reactions reached...: {self.max_reactions}")
                    self.save_checkpoint()
                    break
                
                logging.info(f"Path:{self.path_id_counter}: Selected starting reactant:{initial_reactant}")

                current_reactant = initial_reactant
                path = SynthesisPath(path_id=self.path_id_counter)
                initial_step = SynthesisStep(step=0, reactant=initial_reactant, template=None, product=initial_reactant, qed=self.compute_qed(initial_reactant))
                path.add_step(initial_step)
                valid_sequence = False
            
                for step in range(1, self.max_steps + 1):
                    logging.info(f"Path:{self.path_id_counter} - Step:{step} Reactant:{current_reactant}")
                    
                    compatible_templates = [template for _, template in self.templates.iterrows() if self.match_template(current_reactant, template.Smarts)["first"]]
                    logging.info(f"Found {len(compatible_templates)} compatible templates for reactant {current_reactant}...")
                    if not compatible_templates:
                        logging.info(f"Path:{self.path_id_counter} ends in Step:{step}: No compatible template for reactant:{current_reactant}")
                        break

                    current_reactant, valid_step = self.generate_new_molecule(current_reactant, compatible_templates, step, path)
                    if not valid_step:
                        logging.info(f"Path:{self.path_id_counter} ends in Step:{step}: No valid products generated.")
                        break

                    valid_sequence = True

                if valid_sequence:
                    self.paths.append(path)
                    self.path_id_counter += 1

                wandb.log({
                    "total_reactions": self.total_reactions,
                    "num_steps": max(1, len(path.steps)-1),
                    "num_paths": self.path_id_counter
                })

                if self.path_id_counter > 0 and self.path_id_counter % 10 == 0:
                    logging.info(f"Saving checkpoint at path:{self.path_id_counter}...")
                    self.save_checkpoint()
            
            logging.info("Molecule search completed...")
            wandb.finish()
        except Exception as e:
            logging.error(f"Error during molecule search: {e}")
            self.save_checkpoint()
            raise
