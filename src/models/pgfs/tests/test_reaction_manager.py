import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from src.models.pgfs.utility.reaction_manager import ReactionManager


# Helper function to create Morgan fingerprints
def create_mock_fingerprint(smiles, radius=2, nBits=1024):
    fpgen = AllChem.GetMorganGenerator(radius, fpSize=nBits)
    mol = Chem.MolFromSmiles(smiles)
    fp1 = fpgen.GetFingerprint(mol)
    return fp1 if mol else None


@pytest.fixture
def reaction_manager_setup():
    templates = {
        "0": {
            "name": "Pictet-Gams",
            "smarts": "[$(C([CH2,CH3])),CH:10](=[O:11])-[NH+0:9]-[C$([CH](N)(C)(C)),C$([CH2](N)(C)):8]-[C$([C](c)(C)(C)),C$([CH](c)(C)):7]([O$(OC),OH])-[c:6]1[cH:1][c:2][c:3][c:4][c:5]1>>[c:10]-1[n:9][c:8][c:7][c:6]2[c:5][c:4][c:3][c:2][c:1]-12",
            "type": "unimolecular",
        },
        "1": {
            "name": "Pictet-Spengler-6-membered-ring_R2",
            "smarts": "[CH:10](-[CX4:12])=[O:11].[NH3+,NH2]-[C$(C(N)(C)(C)(C)),C$([CH](N)(C)(C)),C$([CH2](N)(C)):8]-[C$(C(c)(C)(C)(C)),C$([CH](c)(C)(C)),C$([CH2](c)(C)):7]-[c:6]1[c:1][c:2][c:3][c:4][cH:5]1>>[c,C:12]-[CH:10]-1-[N]-[C:8]-[C:7]-[c:6]2[c:1][c:2][c:3][c:4][c:5]-12",
            "type": "bimolecular",
        },
    }
    reactants = {
        "CC(=O)NCC(O)c1ccccc1": create_mock_fingerprint("CC(=O)NCC(O)c1ccccc1"),
        "O=CCc1nccn1C(c1ccccc1)(c1ccccc1)c1ccccc1": create_mock_fingerprint(
            "O=CCc1nccn1C(c1ccccc1)(c1ccccc1)c1ccccc1"
        ),
        "C#CCOc1ccc(CCN)cc1": create_mock_fingerprint("C#CCOc1ccc(CCN)cc1"),
    }
    manager = ReactionManager(templates, reactants)
    return manager


def test_initialization(reaction_manager_setup):
    manager = reaction_manager_setup
    assert isinstance(manager, ReactionManager)
    assert "0" in manager.templates
    assert "C#CCOc1ccc(CCN)cc1" in manager.reactants


def test_apply_unimolecular_reaction(reaction_manager_setup):
    manager = reaction_manager_setup
    expected_product = "Cc1nccc2ccccc12"
    actual_product = manager.apply_reaction(
        "CC(=O)NCC(O)c1ccccc1", manager.templates["0"]["smarts"]
    )
    assert actual_product == expected_product


def test_apply_bimolecular_reaction(reaction_manager_setup):
    manager = reaction_manager_setup
    expected_product = "C#CCOc1ccc2c(c1)C(Cc1nccn1C(c1ccccc1)(c1ccccc1)c1ccccc1)NCC2"
    actual_product = manager.apply_reaction(
        "O=CCc1nccn1C(c1ccccc1)(c1ccccc1)c1ccccc1",
        manager.templates["1"]["smarts"],
        "C#CCOc1ccc(CCN)cc1",
    )
    assert actual_product == expected_product


def test_invalid_state_molecule(reaction_manager_setup):
    manager = reaction_manager_setup
    result = manager.apply_reaction("invalid", manager.templates["0"]["smarts"])
    assert result == None, "Should return None when the molecule is invalid"


def test_no_second_reactant_provided(reaction_manager_setup):
    manager = reaction_manager_setup
    result = manager.apply_reaction(
        "O=CCc1nccn1C(c1ccccc1)(c1ccccc1)c1ccccc1", manager.templates["1"]["smarts"]
    )
    assert (
        result == None
    ), "Should return None when no second reactant is provided for a bimolecular reaction"


def test_invalid_second_reactant(reaction_manager_setup):
    manager = reaction_manager_setup
    result = manager.apply_reaction(
        "O=CCc1nccn1C(c1ccccc1)(c1ccccc1)c1ccccc1",
        manager.templates["1"]["smarts"],
        "invalid",
    )
    assert result == None, "Should return None when the second reactant is invalid"


def test_get_valid_reactants_unimolecular(reaction_manager_setup):
    manager = reaction_manager_setup
    template_index = 0
    expected_reactants = ["CC(=O)NCC(O)c1ccccc1"]
    assert manager.get_valid_reactants(template_index) == expected_reactants


def test_get_valid_reactants_bimolecular(reaction_manager_setup):
    manager = reaction_manager_setup
    template_index = 1
    expected_reactants = ["O=CCc1nccn1C(c1ccccc1)(c1ccccc1)c1ccccc1"]
    assert manager.get_valid_reactants(template_index) == expected_reactants


def test_get_template_mask(reaction_manager_setup):
    manager = reaction_manager_setup
    reactant = "CC(=O)NCC(O)c1ccccc1"
    expected_mask_tensor = torch.tensor([[1, 0]], dtype=torch.float32)
    assert torch.equal(manager.get_template_mask(reactant), expected_mask_tensor)


def test_match_template_unimolecular(reaction_manager_setup):
    manager = reaction_manager_setup
    reactant = "CC(=O)NCC(O)c1ccccc1"
    template = manager.templates[0]["smarts"]
    expected_matches = {"first": True, "second": False}
    assert manager.match_template(reactant, template) == expected_matches


def test_match_template_bimolecular(reaction_manager_setup):
    manager = reaction_manager_setup
    reactant1 = "O=CCc1nccn1C(c1ccccc1)(c1ccccc1)c1ccccc1"
    reactant2 = "C#CCOc1ccc(CCN)cc1"
    template = manager.templates[1]["smarts"]
    expected_matches = {"first": False, "second": True}
    assert manager.match_template(reactant2, template) == expected_matches
