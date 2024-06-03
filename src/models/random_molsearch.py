import random
import logging
from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit.Chem import Crippen
import pandas as pd
import os

# Configure logging
logging.basicConfig(filename='/rds/user/gtj21/hpc-work/designing-new-molecules/logs/Rmolsearch.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

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
    

class RandomMolSearch:
    def __init__(self, reactants, templates, max_steps=5, max_reactions=100, max_attempts=100):
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

    def parse_template(self, smarts):
        try:
            if ">>" in smarts:
                reactant_smarts, _ = smarts.split(">>")
            elif ">" in smarts:
                parts = smarts.split('>')
                if len(parts) == 3:
                    reactant_smarts, _, _ = parts
                else:
                    raise ValueError(f"Invalid SMARTS format: {smarts}")
            else:
                raise ValueError(f"Invalid SMARTS format: {smarts}")
            return reactant_smarts
        except Exception as e:
            logging.error(f"Error parsing SMARTS {smarts}: {e}")
            return None

    def substructure_match(self, reactant_smiles, template_smarts):
        try:
            reactant = Chem.MolFromSmiles(reactant_smiles)
            pattern = Chem.MolFromSmarts(template_smarts)
            if pattern is None:
                return False
            return reactant.HasSubstructMatch(pattern)
        except Exception as e:
            logging.error(f"Error in substructure matching for reactant {reactant_smiles} and template {template_smarts}: {e}")
            return False

    def sanitize_molecule(self, mol):
        try:
            Chem.SanitizeMol(mol)
            return True
        except Chem.MolSanitizeException as e:
            logging.error(f"Sanitization error for molecule {Chem.MolToSmiles(mol)}: {e}")
            return False

    def apply_template(self, reactant1, smarts, reactant2=None, reaction_type="unimolecular"):
        reactant1_mol = Chem.MolFromSmiles(reactant1)
        if not reactant1_mol:
            logging.error(f"Invalid reactant1 SMILES: '{reactant1}'")
            return None
        
        if reaction_type == 'unimolecular':
            rxn = AllChem.ReactionFromSmarts(smarts)
            products = rxn.RunReactants((reactant1_mol,))
        elif reaction_type == 'bimolecular' and reactant2:
            reactant2_mol = Chem.MolFromSmiles(reactant2)
            if not reactant2_mol:
                logging.error(f"Invalid reactant2 SMILES: '{reactant2}'")
                return None
            rxn = AllChem.ReactionFromSmarts(smarts)
            products = rxn.RunReactants((reactant1_mol, reactant2_mol))
        else:
            logging.error(f"Invalid reaction type or missing reactant2 for bimolecular reaction")
            return None
        
        if products:
            sanitized_smiles = []
            for product_set in products:
                for product in product_set:
                    if self.sanitize_molecule(product):
                        sanitized_smiles.append(Chem.MolToSmiles(product))
            return sanitized_smiles if sanitized_smiles else None
        
        return None
    
    def filter_templates(self, current_reactant):
        compatible_templates = []
        for _, template in self.templates.iterrows():
            template_smarts = self.parse_template(template.Smarts)
            if template.Type == 'unimolecular':
                if self.substructure_match(current_reactant, template_smarts):
                    compatible_templates.append(template)
            elif template.Type == 'bimolecular':
                r1_smarts, r2_smarts = template_smarts.split('.')
                if self.substructure_match(current_reactant, r1_smarts):
                    compatible_templates.append(template)
        return compatible_templates
    
    def generate_new_molecule(self, current_reactant, template, step, path):
        template_smarts = self.parse_template(template.Smarts)
        second_reactant = None
        if template.Type == 'unimolecular':
            products = self.apply_template(current_reactant, template.Smarts, reaction_type='unimolecular')
        elif template.Type == 'bimolecular':
            r1_smarts, r2_smarts = template_smarts.split('.')
            compatible_reactants = [r for r in self.reactants if self.substructure_match(r, r2_smarts)]
            if not compatible_reactants:
                logging.info(f"No compatible second reactant for reaction {template.Reaction}")
                return current_reactant, False
            second_reactant = random.choice(compatible_reactants)
            products = self.apply_template(current_reactant, template.Smarts, reactant2=second_reactant, reaction_type='bimolecular')

        if products:
            qed_scores = [self.compute_qed(product) for product in products]
            best_qed = max(qed_scores)
            best_products = [products[i] for i in range(len(products)) if qed_scores[i] == best_qed]
            best_product = random.choice(best_products)
            qed_score = best_qed
            step_info = SynthesisStep(step=step, reactant=current_reactant, template=template.Reaction, product=best_product, qed=qed_score, second_reactant=second_reactant)
            path.add_step(step_info)
            current_reactant = best_product
            self.total_reactions += 1
            return current_reactant, True
        else:
            logging.info("No products!")
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
    
    def run_molsearch(self):
        logging.info("Starting molecule search.")
        while self.total_reactions < self.max_reactions and self.attempts < self.max_attempts:
            initial_reactant = random.choice(self.reactants)
         
            if initial_reactant in self.no_compatible_templates:
                self.attempts += 1
                continue

            current_reactant = initial_reactant
            path = SynthesisPath(path_id=self.path_id_counter)
            initial_step = SynthesisStep(step=0, reactant=initial_reactant, template=None, product=initial_reactant, qed=self.compute_qed(initial_reactant))
            path.add_step(initial_step)
            valid_sequence = False
         
            for step in range(1, self.max_steps + 1):
                if self.total_reactions >= self.max_reactions:
                    break

                compatible_templates = self.filter_templates(current_reactant)
                if not compatible_templates:
                    logging.info(f"No compatible template for reactant {current_reactant}")
                    self.no_compatible_templates.add(current_reactant)
                    break

                template = random.choice(compatible_templates)
                current_reactant, valid_step = self.generate_new_molecule(current_reactant, template, step, path)
                if not valid_step:
                    break

                valid_sequence = True

            if valid_sequence:
                self.paths.append(path)
                self.path_id_counter += 1
            else:
                self.attempts += 1

            if self.total_reactions > 0 and self.total_reactions % 1000 == 0:
                file_path = f'/rds/user/gtj21/hpc-work/designing-new-molecules/data/results/Rmolsearch_results.csv'
                self.save_results_to_csv(file_path)
                logging.info(f"Intermediate save at {self.total_reactions} reactions.")
        
        if self.total_reactions >= self.max_reactions:
            logging.info(f"Terminated because the total number of reactions {self.total_reactions} reached the maximum limit of {self.max_reactions}. There were {self.attempts} attempts.")
        elif self.attempts >= self.max_attempts:
            logging.info(f"Terminated because the number of attempts {self.attempts} reached the maximum limit of {self.max_attempts}. There were {self.total_reactions} reactions.")
        
        logging.info("Molecule search completed.")
