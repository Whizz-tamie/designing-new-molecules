import os
import logging
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import SDWriter
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging to a file
log_dir = '/rds/user/gtj21/hpc-work/designing-new-molecules/logs'
log_file_path = os.path.join(log_dir, 'chembl_processing.log')
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        #logging.StreamHandler()  # To keep logging to console as well
    ]
)

logger = logging.getLogger(__name__)

# Initialize ChEMBL molecule client
molecule = new_client.molecule

# Define the query to fetch molecules with relevant properties
molecules = molecule.filter(molecule_properties__isnull=False, molecule_type='Small molecule').only(['molecule_chembl_id', 'molecule_structures', 'molecule_properties'])

# Function to process each molecule
def process_molecule(mol):
    try:
        chembl_id = mol['molecule_chembl_id']
        smiles = mol['molecule_structures'].get('canonical_smiles', "")
        molfile = mol['molecule_structures'].get('molfile', "")
        properties = mol['molecule_properties'] if mol['molecule_properties'] else {}
        mol_wt = properties.get('full_mwt')
        logp = properties.get('alogp')
        qed = properties.get('qed_weighted')
        
        # Create RDKit molecule object
        rdkit_mol = Chem.MolFromMolBlock(molfile)
        if rdkit_mol:
            rdkit_mol.SetProp('ChEMBL_ID', chembl_id)
            rdkit_mol.SetProp('SMILES', smiles)
            if mol_wt is not None:
                rdkit_mol.SetProp('Molecular_Weight', str(mol_wt))
            if logp is not None:
                rdkit_mol.SetProp('LogP', str(logp))
            if qed is not None:
                rdkit_mol.SetProp('QED', str(qed))
        
        logger.info(f"Processed molecule {chembl_id}")
        return rdkit_mol
    except Exception as e:
        logger.error(f"Error processing molecule {mol['molecule_chembl_id']}: {e}")
        return None

# Save each molecule to the SDF file incrementally
def save_molecule_incrementally(molecule, writer):
    rdkit_mol = process_molecule(molecule)
    if rdkit_mol:
        writer.write(rdkit_mol)

# Main execution
if __name__ == "__main__":
    output_sdf_path = '/rds/user/gtj21/hpc-work/designing-new-molecules/data/chembl_molecule_data.sdf'
    os.makedirs(os.path.dirname(output_sdf_path), exist_ok=True)
    
    with SDWriter(output_sdf_path) as writer:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(save_molecule_incrementally, mol, writer) for mol in molecules]
            for future in as_completed(futures):
                future.result()  # This will re-raise any exceptions encountered during processing