# chem/precompute_vector.py

import numpy as np
import torch
import pickle
import uuid
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
import logging

# Set up logging
logging.basicConfig(filename='/rds/user/gtj21/hpc-work/designing-new-molecules/logs/prevec_progress.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def reactant_to_vector(reactant, representation="ECFP"):
    """
    Convert a reactant to a feature vector based on the selected representation.

    Args:
        reactant (str): Reactant molecule in SMILES format.
        representation (str): Type of molecular representation ('ECFP', 'MACCS', 'MolDSet').

    Returns:
        list: Feature vector representation of the reactant.
    """
    mol = Chem.MolFromSmiles(reactant)
    if not mol:
        raise ValueError(f"Invalid SMILES string: {reactant}")
    
    if representation == 'ECFP':
        fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
        fp = fpgen.GetFingerprint(mol)
        return fp
    
    elif representation == 'MACCS':
        return MACCSkeys.GenMACCSKeys(mol)
    
    elif representation == 'MolDSet':
        descriptor_names = [
            'MaxEStateIndex', 'MinEStateIndex', 'MinAbsEStateIndex', 'qed', 'MolWt', 'FpDensityMorgan1', 'BalabanJ',
            'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA7', 'SlogP_VSA3',
            'SlogP_VSA4', 'SlogP_VSA5', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6',
            'FractionCSP3', 'MolLogP', 'Kappa2', 'PEOE_VSA2', 'SMR_VSA5', 'SMR_VSA6', 'EState_VSA7', 'Chi4v', 'SMR_VSA10',
            'SlogP_VSA6', 'EState_VSA8', 'EState_VSA9', 'VSA_EState9'
        ]
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
        return calculator.CalcDescriptors(mol)
    else:
        raise ValueError("Unknown representation type")

def precompute_vectors(smiles, output_file, representation="ECFP"):
    """
    Precompute vectors for a list of reactants and save to a file.

    Args:
        reactants (list): List of reactants in SMILES format.
        output_file (str): Path to the output file.
        representation (str): Type of molecular representation ('ECFP', 'MACCS', 'MolDSet').
        batch_size (int): Number of reactants to process in each batch.
    """
    vectors = {}
    total_reactants = len(smiles)
    logging.info(f"Starting to process {total_reactants} reactants")

    for i, smile in enumerate(smiles):
        try:
            smiles_fpt = convert_to_tensor(reactant_to_vector(smile, representation))
            vectors[uuid.uuid4().hex] = {"smiles": smile, "vector": smiles_fpt}
            if (i + 1) % 1000 == 0:
                logging.info(f"Processed {i + 1}/{total_reactants} reactants")
        except ValueError as e:
            logging.error(f"Error processing reactant {smile}: {e}")

    # Save all vectors to the pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(vectors, f)
    logging.info(f"Finished processing all reactants and saved to {output_file}")

def convert_to_tensor(vector):
    """
    Convert an RDKit ExplicitBitVect to a PyTorch tensor.

    Args:
        vector (ExplicitBitVect): RDKit fingerprint vector.

    Returns:
        torch.Tensor: Corresponding PyTorch tensor.
    """
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(vector, arr)
    return torch.tensor(arr, dtype=torch.float32)

def load_precomputed_vectors(file_path):
    """
    Load precomputed vectors from a file.

    Args:
        file_path (str): Path to the file with precomputed vectors.

    Returns:
        dict: Dictionary of reactants and their feature vectors.
    """
    with open(file_path, 'rb') as f:
        vectors = pickle.load(f)
    return vectors

def load_templates(file_path):
    """
    Load SMARTS templates from a file.

    Args:
        file_path (str): Path to the file with SMARTS templates.

    Returns:
        list: List of SMARTS templates.
    """
    templates = pd.read_csv(file_path, delimiter="|")["Smarts"].tolist()
    return templates

if __name__ == "__main__":
    # Load the CSV file
    input_file = "/rds/user/gtj21/hpc-work/designing-new-molecules/data/preprocessed_data/enamine_building_blocks.csv"
    building_blocks = pd.read_csv(input_file)['SMILES'].tolist()

    # Define the output file path
    output_file = '/rds/user/gtj21/hpc-work/designing-new-molecules/data/preprocessed_data/enamine_fpt_uuid.pkl'

    # Run the precompute function
    precompute_vectors(building_blocks, output_file, representation='ECFP')
