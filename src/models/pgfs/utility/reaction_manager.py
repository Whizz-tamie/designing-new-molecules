# reaction_manager.py

import logging

import torch
from rdkit import Chem
from rdkit.Chem import QED, AllChem

# Configure the logger
logger = logging.getLogger(__name__)


class ReactionManager:
    def __init__(self, templates, reactants):
        self.templates = templates
        self.reactants = reactants
        self.template_mask_cache = {}
        self.valid_reactants_cache = {}
        self._initialize_template_types()

        logger.info("ReactionManager Initialised...")

    def _initialize_template_types(self):
        self.template_types = {
            i: t["type"] for i, t in enumerate(self.templates.values())
        }
        # Define a mapping from string labels to integers
        type_mapping = {"unimolecular": 0, "bimolecular": 1}
        # Convert the original dictionary to use integer labels
        self.template_types = {
            k: type_mapping[v] for k, v in self.template_types.items()
        }
        # Convert dictionary to tensor
        self.template_types = torch.tensor(
            [self.template_types[i] for i in range(max(self.template_types.keys()) + 1)]
        )
        logger.debug("Template types initialized: %s", self.template_types.shape)

    def apply_reaction(self, state, template, reactant=None):
        try:
            state_mol = Chem.MolFromSmiles(state)
            if not state_mol:
                logger.error("Invalid state molecule SMILES: %s", state)
                return None

            reaction = AllChem.ReactionFromSmarts(template)  # type: ignore
            return self._process_reaction(state_mol, reaction, reactant)
        except Exception as e:
            logger.error("Exception in apply_reaction: %s", e)
            return None

    def _process_reaction(self, state_mol, reaction, reactant):
        num_reactants = reaction.GetNumReactantTemplates()
        product_sets = self._run_reaction(state_mol, reaction, num_reactants, reactant)

        if not product_sets:
            logger.warning("No products generated from reaction...")
            return None

        # Filter and sanitize products separately
        valid_products = []
        for product in (p for subset in product_sets for p in subset):
            try:
                sanitization_result = Chem.SanitizeMol(product, catchErrors=True)
                if sanitization_result == Chem.SanitizeFlags.SANITIZE_NONE:
                    valid_products.append(product)
                else:
                    error_flags = []
                    for flag in Chem.SanitizeFlags:
                        if sanitization_result & flag:
                            error_flags.append(flag.name)
                    logger.error(
                        "Sanitization failed for product with SMILES '%s' due to: %s",
                        Chem.MolToSmiles(product, True),
                        ", ".join(error_flags),
                    )
            except Exception as e:
                logger.error(
                    "Sanitization failed for product: %s. Error: %s",
                    Chem.MolToSmiles(product, True),
                    str(e),
                )
                continue

        # Select the best product based on QED
        if valid_products:
            try:
                best_product = max(valid_products, key=lambda mol: QED.qed(mol))
                best_product_smiles = Chem.MolToSmiles(best_product)

                logger.info(
                    "Generated new product: %s with QED: %s",
                    best_product_smiles,
                    QED.qed(best_product),
                )
                return best_product_smiles
            except Exception as e:
                logger.error("Error in QED calculation: %s", e)
                return None
        return None

    def _run_reaction(self, state_mol, reaction, num_reactants, reactant):
        if num_reactants == 1:
            return reaction.RunReactants((state_mol,))
        elif num_reactants == 2 and reactant:
            reactant_mol = Chem.MolFromSmiles(reactant)
            if reactant_mol:
                return reaction.RunReactants((state_mol, reactant_mol))
            else:
                logger.error("Invalid second reactant molecule SMILES: %s", reactant)
        return []

    def get_valid_reactants(self, template_index):
        if template_index not in self.valid_reactants_cache:
            self.valid_reactants_cache[template_index] = self._compute_valid_reactants(
                template_index
            )
        compatible_reactants = self.valid_reactants_cache.get(template_index, [])
        logger.info(
            "Found %s compatible second reactants for template: %s",
            len(compatible_reactants),
            self.templates[template_index]["name"],
        )
        return compatible_reactants

    def _compute_valid_reactants(self, template_index):
        template = self.templates[template_index]
        return [
            reactant
            for reactant in self.reactants.keys()
            if self._match_template(reactant, template["smarts"])["second"]
        ]

    def get_mask(self, reactant):
        if reactant not in self.template_mask_cache:
            self.template_mask_cache[reactant] = self._compute_mask(reactant)
        valid_templates = self.template_mask_cache[reactant]
        logger.info(
            "Found %s compatible templates for reactant: %s",
            int(valid_templates.sum().item()),
            reactant,
        )
        return valid_templates

    def _compute_mask(self, reactant):
        """Generate a tensor mask indicating which templates are valid for the given reactant."""
        if reactant is None:
            return torch.zeros(len(self.templates))

        mask = [
            int(self._match_template(reactant, t["smarts"])["first"])
            for t in self.templates.values()
        ]
        mask = []

        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        return mask_tensor

    def _compute_mask(self, reactant):
        """Generate a tensor mask indicating which templates are valid for the given reactant."""
        if reactant is None:
            return torch.zeros(len(self.templates))

        mask = []
        for t in self.templates.values():
            match_result = self._match_template(reactant, t["smarts"])["first"]
            if match_result:
                product = self.apply_reaction(reactant, t["smarts"])
                if product:
                    mask.append(1)
                else:
                    mask.append(0)
            else:
                mask.append(0)

        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        return mask_tensor

    def _match_template(self, reactant, template):
        """Check if a reactant matches the reaction template."""
        try:
            reaction = AllChem.ReactionFromSmarts(template)
            reactant_mol = Chem.MolFromSmiles(reactant)

            if reactant_mol is None:
                return {"first": False, "second": False}

            matches = {"first": False, "second": False}
            reactant1_template = reaction.GetReactantTemplate(0)
            matches["first"] = reactant_mol.HasSubstructMatch(
                reactant1_template, useChirality=True
            )

            if reaction.GetNumReactantTemplates() == 2:
                reactant2_template = reaction.GetReactantTemplate(1)
                matches["second"] = reactant_mol.HasSubstructMatch(
                    reactant2_template, useChirality=True
                )

            return matches
        except Exception as e:
            logger.error("Error in matching template: %s", e)
            return {"first": False, "second": False}
