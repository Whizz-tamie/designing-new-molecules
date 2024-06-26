import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


# Function to check compatibility of reactants with templates
def check_compatibility(reactants, templates):
    compatibility_counts = {}
    
    for template_line in templates:
        name, reaction = template_line.strip().split('|')
        reaction = AllChem.ReactionFromSmarts(reaction)
        num_reactants = reaction.GetNumReactantTemplates()
        
        # Initialize compatibility counts
        if name not in compatibility_counts:
            compatibility_counts[name] = {'first': 0, 'second': 0, 'type':"unimolecular" if num_reactants == 1 else "bimolecular"}
        
        # Check compatibility for unimolecular and bimolecular templates
        if num_reactants == 1:
            reactant_template = reaction.GetReactantTemplate(0)
            for reactant in reactants:
                reactant_mol = Chem.MolFromSmiles(reactant)
                if reactant_mol.HasSubstructMatch(reactant_template):
                    compatibility_counts[name]['first'] += 1
        elif num_reactants == 2:
            reactant1_template = reaction.GetReactantTemplate(0)
            reactant2_template = reaction.GetReactantTemplate(1)
            for reactant in reactants:
                reactant_mol = Chem.MolFromSmiles(reactant)
                if reactant_mol.HasSubstructMatch(reactant1_template):
                    compatibility_counts[name]['first'] += 1
                if reactant_mol.HasSubstructMatch(reactant2_template):
                    compatibility_counts[name]['second'] += 1
    
    return compatibility_counts

# Function to save compatibility counts to a CSV file
def save_compatibility_counts(file_path, compatibility_counts):
    rows = []
    for template, counts in compatibility_counts.items():
        rows.append({
            'Template': template,
            'Type': counts['type'],
            'First Reactants Compatible': counts['first'],
            'Second Reactants Compatible': counts['second']
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    r_tant = pd.read_csv("/rds/user/gtj21/hpc-work/designing-new-molecules/data/preprocessed_data/enamine_building_blocks.csv")
    # Load the file
    file_path = '/rds/user/gtj21/hpc-work/designing-new-molecules/data/preprocessed_data/reactions_R2.txt'
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Example usage:
    reactants = r_tant.SMILES.to_list() # Add reactants
    templates = lines
    output_file = '/rds/user/gtj21/hpc-work/designing-new-molecules/data/template_compatibility_R2.csv'

    # Check compatibility and get counts
    compatibility_counts = check_compatibility(reactants, templates)
    # Save the results to a CSV file
    save_compatibility_counts(output_file, compatibility_counts)

