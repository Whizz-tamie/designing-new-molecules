from rdkit import Chem
from rdkit.Chem import AllChem
import csv
import concurrent.futures
import pandas as pd


def match_template(reactant: str, template: str) -> dict:
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
            match_first = reactant_mol.HasSubstructMatch(reactant1_template)
        elif num_reactants == 2:
            reactant1_template = reaction.GetReactantTemplate(0)
            reactant2_template = reaction.GetReactantTemplate(1)
            match_first = reactant_mol.HasSubstructMatch(reactant1_template)
            match_second = reactant_mol.HasSubstructMatch(reactant2_template)

        return {"first": match_first, "second": match_second}
    except Exception as e:
        print(f"Error in matching template: {e}")
        return {"first": False, "second": False}


def check_reactant_compatibility(reactant, templates):
    first_compatible = []
    second_compatible = []

    for template in templates:
        result = match_template(reactant, template)
        if result['first']:
            first_compatible.append(template)
        if result['second']:
            second_compatible.append(template)

    return {
        'reactant': reactant,
        'first_compatible_templates': len(first_compatible),
        'second_compatible_templates': len(second_compatible)
    }


def generate_compatibility_file(reactants, templates, output_file):
    results = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(check_reactant_compatibility, reactant, templates): reactant for reactant in reactants}

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)

    # Sort results based on the original order of reactants
    results.sort(key=lambda x: reactants.index(x['reactant']))

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['reactant', 'first_compatible_templates', 'second_compatible_templates']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(result)

if __name__ == "__main__":
    r_tant = pd.read_csv("/rds/user/gtj21/hpc-work/designing-new-molecules/data/preprocessed_data/enamine_building_blocks.csv")
    # Load the file
    file_path = '/rds/user/gtj21/hpc-work/designing-new-molecules/data/preprocessed_data/reactions_R2_filtered.txt'
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Example usage:
    reactants = r_tant.SMILES.to_list() # reactants
    templates = [line.split("|")[1] for line in lines]
    output_file = '/rds/user/gtj21/hpc-work/designing-new-molecules/data/reactant_compatibility_R2_filtered.csv'

    generate_compatibility_file(reactants, templates, output_file)
