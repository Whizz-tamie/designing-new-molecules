import pytest
import os
import pickle
from src.models.pgfs.envs.molecule_design_env import MoleculeDesignEnv
from gymnasium import spaces
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit import Chem


# Fixture for setting up and tearing down the environment
@pytest.fixture
def env_setup(tmpdir):
    reactant_file = tmpdir.join('reactants.pkl')
    template_file = tmpdir.join('templates.pkl')
    
    # Simulate the pickle data for reactants and templates
    reactants = {'C(C)O': AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles('C(C)O'), 2, nBits=1024)}
    templates = {0: {'name': 'Addition', 'smarts': 'C(C)C + O>>C(C)(C)O'}}
    
    with open(reactant_file, 'wb') as rf:
        pickle.dump(reactants, rf)
    with open(template_file, 'wb') as tf:
        pickle.dump(templates, tf)
    
    env = MoleculeDesignEnv(str(reactant_file), str(template_file))
    
    return env

# Test initialization and basic properties
def test_initialization(env_setup):
    env = env_setup
    assert isinstance(env.template_action_space, spaces.Discrete), "Action space should be correctly initialized"
    assert isinstance(env.observation_space, spaces.Box), "Observation space should be correctly initialized"
    assert isinstance(env.templates, dict), "Templates should be stored as a dictionary"
    assert isinstance(env.reactants, dict), "Reactants should be stored as a dictionary"
    assert isinstance(env.reactants['C(C)O'], ExplicitBitVect), "Reactants should be loaded as RDKit fingerprint objects"
    assert 'smarts' in env.templates[0], "Templates should contain 'smarts' key"

# Test reset functionality
def test_reset(env_setup):
    env = env_setup
    observation, info = env.reset()
    assert isinstance(observation, ExplicitBitVect), "Observation should be a numpy array"
    assert 'SMILES' in info, "Info should contain SMILES key"

# Test step functionality
def test_valid_step_and_action_space(env_setup):
    env = env_setup
    env.reset()
    valid_reactants_indices = [0]  # Assume template 0 has one valid reactant at index 0
    env.reaction_manager.get_valid_reactants = lambda idx: valid_reactants_indices
    
    observation, reward, terminated, truncated, info = env.step((0, 0))

    # Check if the reactant_action_space was set correctly
    assert isinstance(env.reactant_action_space, spaces.Discrete)
    assert env.reactant_action_space.n == 1  # Only one valid reactant 'CCN'

    assert env.current_state != None, "State should have changed after a valid step"
    assert env_setup.current_step == 1
    assert len(env_setup.steps_log) == 1
    assert env.reactant_action_space.n == len(valid_reactants_indices), "Reactant action space should match the number of valid reactants"
    assert not terminated, "Terminated should be False if max_steps are not reached"

    assert isinstance(observation, np.ndarray), "Observation should be a numpy array"
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert 'SMILES' in info

# Test handling of invalid actions
@pytest.mark.parametrize("template_index, reactant_index", [(-1, 0), (100, 0), (0, -1), (0, 100)])
def test_invalid_action(env_setup, template_index, reactant_index):
    env = env_setup
    env.reset()
    with pytest.raises(ValueError):
        env.step((template_index, reactant_index))

# Test max_steps termination condition
def test_termination_condition(env_setup):
    env = env_setup
    env.reset()
    for _ in range(env.max_steps):
        env.step((0, 0))  # Assuming 0, 0 is a valid action index
    _, _, terminated, _, _ = env.step((0, 0))
    assert terminated, "Environment should terminate after max_steps"
    assert env_setup.current_step == env.max_steps

def test_render_human(env_setup):
    env = env_setup
    env.reset()
    action = (0, 0)

    # Mock the get_valid_reactants method
    env.reaction_manager.get_valid_reactants = lambda template_index: ['CCN']
    env.step(action)

    save_path = 'test_render_human.png'
    env.render(mode='human', save_path=save_path)

    # Check if the file was created
    assert os.path.exists(save_path)

    # Clean up the file after the test
    os.remove(save_path)

def test_render_console(env_setup, capsys):
    env = env_setup
    env.reset()
    action = (0, 0)

    # Mock the get_valid_reactants method
    env.reaction_manager.get_valid_reactants = lambda template_index: ['CCN']
    env.step(action)

    env.render(mode='console')
    captured = capsys.readouterr()

    for i, (old_state, template_index, reactant, new_state) in enumerate(env.steps_log):
        assert f"Step {i+1}:" in captured.out
        assert f"Initial State (SMILES): {old_state}" in captured.out
        assert f"Reaction Name: {env.templates[template_index]['name']}" in captured.out
        assert f"Second Reactant (if applicable): {reactant}" in captured.out
        assert f"New State (SMILES): {new_state}" in captured.out
 

# Run these tests
if __name__ == "__main__":
    pytest.main([__file__])