# tests/test_environment.py

import unittest
from unittest.mock import patch, MagicMock
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from src.models.pgfs.environments.environment import Environment


class TestEnvironment(unittest.TestCase):

    def setUp(self):
        """
        Set up the environment for testing.
        """
        with patch('src.models.pgfs.environments.environment.load_precomputed_vectors') as mock_load_precomputed_vectors, \
             patch('src.models.pgfs.environments.environment.load_templates') as mock_load_templates, \
             patch('src.models.pgfs.environments.environment.KNN') as mock_knn:
            
            mock_load_templates.return_value = ["[C:1]>>[C:1]=O", "[C:1]=[O:2].[O:3]>>[C:1](=[O:2])[O:3]"]
            mock_load_precomputed_vectors.return_value = {'CCO': torch.tensor([0.5] * 1024),
                                                          'OCC': torch.tensor([0.6] * 1024)}
            mock_knn_instance = MagicMock()
            mock_knn.return_value = mock_knn_instance

            self.env = Environment('/rds/user/gtj21/hpc-work/designing-new-molecules/data/preprocessed_data/enamine_fingerprints.pkl',
                               '/rds/user/gtj21/hpc-work/designing-new-molecules/data/preprocessed_data/rxn_set_processed.txt')
    
    def test_init(self):
        """
        Test initialization of the Environment class.
        """
        self.assertEqual(self.env.templates, ["[C:1]>>[C:1]=O", "[C:1]=[O:2].[O:3]>>[C:1](=[O:2])[O:3]"])
        self.assertTrue(torch.equal(self.env.precomputed_vectors['CCO'], torch.tensor([0.5] * 1024)))
        self.assertEqual(self.env.k, 1)
        self.assertEqual(self.env.max_steps, 10)
        self.assertEqual(self.env.current_step, 0)
        self.assertIsInstance(self.env.knn, MagicMock)

    def test_one_hot_to_smarts(self):
        """
        Test conversion of one-hot encoded template to SMARTS template.
        """
        one_hot_template = torch.tensor([0, 1])
        self.env.templates = ["template0", "template1"]
        smarts_template = self.env.one_hot_to_smarts(one_hot_template)
        
        self.assertEqual(smarts_template, "template1")

    @patch('src.models.pgfs.environments.environment.Environment.match_template')
    def test_get_valid_reactants(self, mock_match_template):
        """
        Test getting valid reactants for a given template.
        """
        mock_match_template.return_value = True
        template = "[C:1]=[O:2].[O:3]>>[C:1](=[O:2])[O:3]"
        valid_reactants = self.env.get_valid_reactants(template)
        expected_valid_reactants = [torch.tensor([0.5] * 1024), torch.tensor([0.6] * 1024)]

        self.assertEqual(len(valid_reactants), len(expected_valid_reactants))
        for valid_reactant, expected_reactant in zip(valid_reactants, expected_valid_reactants):
            self.assertTrue(torch.equal(valid_reactant, expected_reactant))

    def test_match_template(self):
        """
        Test matching of reactants to a reaction template.
        """
        # Test for unimolecular reaction
        unimolecular_template = "[C:1]>>[C:1]=O"
        unimolecular_reactant = 'CCO'
        unimolecular_reaction = AllChem.ReactionFromSmarts(unimolecular_template)

        with patch('rdkit.Chem.AllChem.ReactionFromSmarts', return_value=unimolecular_reaction):
            with patch('rdkit.Chem.MolFromSmiles', return_value=Chem.MolFromSmiles(unimolecular_reactant)):
                self.assertTrue(self.env.match_template(unimolecular_reactant, unimolecular_template))

        # Test for bimolecular reaction
        bimolecular_template = "[C:1]=[O:2].[O:3]>>[C:1](=[O:2])[O:3]"
        bimolecular_reactant = 'CCO'
        bimolecular_reaction = AllChem.ReactionFromSmarts(bimolecular_template)

        with patch('rdkit.Chem.AllChem.ReactionFromSmarts', return_value=bimolecular_reaction):
            with patch('rdkit.Chem.MolFromSmiles', return_value=Chem.MolFromSmiles(bimolecular_reactant)):
                self.assertTrue(self.env.match_template(bimolecular_reactant, bimolecular_template))
    
    @patch('rdkit.Chem.AllChem.ReactionFromSmarts')
    @patch('rdkit.Chem.MolFromSmiles')
    def test_forward_reaction_unimolecular(self, mock_mol_from_smiles, mock_reaction_from_smarts):
        """ Test the forward reaction for unimolecular reactions. """
        unimolecular_state = "CCO"
        unimolecular_template = "[C:1]>>[C:1]=O"
        unimolecular_product_smiles = 'CC=O'
        unimolecular_product = Chem.MolFromSmiles(unimolecular_product_smiles)

        # Mock the state molecule
        mock_mol_from_smiles.side_effect = [Chem.MolFromSmiles(unimolecular_state), unimolecular_product]

        # Mock the reaction
        unimolecular_reaction = MagicMock()
        unimolecular_reaction.GetNumReactantTemplates.return_value = 1
        unimolecular_reaction.RunReactants.return_value = ((unimolecular_product,),)
        mock_reaction_from_smarts.return_value = unimolecular_reaction

        # Call the forward_reaction method
        products = self.env.forward_reaction(unimolecular_state, unimolecular_template, None)
        self.assertIsNotNone(products, "Products should not be None for unimolecular reaction")
        self.assertGreater(len(products), 0, "There should be at least one product for unimolecular reaction")
        self.assertIn(unimolecular_product,  products)

    @patch('rdkit.Chem.AllChem.ReactionFromSmarts')
    @patch('rdkit.Chem.MolFromSmiles')
    def test_forward_reaction_bimolecular(self, mock_mol_from_smiles, mock_reaction_from_smarts):
        """ Test the forward reaction for bimolecular reactions. """
        bimolecular_state = "CCO"
        bimolecular_template = "[C:1]=[O:2].[O:3]>>[C:1](=[O:2])[O:3]"
        bimolecular_reactant = "OCC"
        bimolecular_product1_smiles = 'CCOC'
        bimolecular_product2_smiles = 'CCOCC'
        bimolecular_product1 = Chem.MolFromSmiles(bimolecular_product1_smiles)
        bimolecular_product2 = Chem.MolFromSmiles(bimolecular_product2_smiles)

        # Mock the state and reactant molecules
        mock_mol_from_smiles.side_effect = [Chem.MolFromSmiles(bimolecular_state), Chem.MolFromSmiles(bimolecular_reactant)]

        # Mock the reaction
        bimolecular_reaction = MagicMock()
        bimolecular_reaction.GetNumReactantTemplates.return_value = 2
        bimolecular_reaction.RunReactants.return_value = ((bimolecular_product1,), (bimolecular_product2,))
        mock_reaction_from_smarts.return_value = bimolecular_reaction

        # Call the forward_reaction method
        products = self.env.forward_reaction(bimolecular_state, bimolecular_template, bimolecular_reactant)
        self.assertIsNotNone(products, "Products should not be None for bimolecular reaction")
        self.assertGreater(len(products), 0, "There should be at least one product for bimolecular reaction")
        self.assertIn(bimolecular_product1, products)
        self.assertIn(bimolecular_product2, products)

    def test_scoring_function(self):
        """
        Test the scoring function using QED.
        """
        mol = Chem.MolFromSmiles('CCO')
        score = self.env.scoring_function(mol)
        self.assertGreater(score, 0.0)

        invalid_mol = None
        score = self.env.scoring_function(invalid_mol)
        self.assertEqual(score, 0.0)

    @patch('src.models.pgfs.environments.environment.Environment.get_valid_reactants')
    @patch('src.models.pgfs.environments.environment.Environment.forward_reaction')
    @patch('src.models.pgfs.environments.environment.Environment.scoring_function')
    @patch('src.models.pgfs.environments.environment.Environment.is_done', return_value=False)
    def test_step(self, mock_is_done, mock_scoring_function, mock_forward_reaction, mock_get_valid_reactants):
        """
        Test the step function of the environment.
        """
        state = 'CCO'
        one_hot_template = torch.tensor([0, 1])
        action = torch.tensor([0.5] * 1024)
        mock_get_valid_reactants.return_value = [torch.tensor([0.5] * 1024)]
        mock_forward_reaction.return_value = [Chem.MolFromSmiles('CCOC')]
        mock_scoring_function.return_value = 0.9

        next_state, reward, done = self.env.step(state, one_hot_template, action)
        
        self.assertEqual(next_state, 'CCOC')
        self.assertEqual(reward, 0.9)
        self.assertFalse(done)

    def test_is_done(self):
        """
        Test if the episode has ended.
        """
        self.env.current_step = self.env.max_steps
        self.assertTrue(self.env.is_done())

        self.env.current_step = 0
        self.assertFalse(self.env.is_done())


if __name__ == '__main__':
    unittest.main()
