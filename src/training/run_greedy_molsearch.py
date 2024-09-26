import pickle

import pandas as pd
from rdkit.Chem import AllChem

from src.models.gsearch.greedy_molsearch import GreedyMolSearch


def get_reaction_type(smarts):
    try:
        # Create a reaction object from the SMARTS string
        reaction = AllChem.ReactionFromSmarts(smarts)

        # Get the number of reactant templates
        num_reactants = reaction.GetNumReactantTemplates()

        if num_reactants == 1:
            return "unimolecular"
        elif num_reactants == 2:
            return "bimolecular"
        else:
            pass
    except Exception as e:
        print(f"Error processing SMARTS '{smarts}': {e}")
        return "invalid"


if __name__ == "__main__":
    # Load 2000 validation set reactants
    enamine_val = "data/preprocessed_data/enamine_val.pkl"
    with open(enamine_val, "rb") as f:
        val = pickle.load(f)

    reactants = [info["smiles"] for info in val.values()]

    # Load reaction templates
    file_path = "data/preprocessed_data/reactions_R2.txt"
    templates = pd.read_csv(
        file_path, delimiter="|", header=None, names=["Reaction", "Smarts"]
    )
    templates["Type"] = templates["Smarts"].apply(get_reaction_type)

    search = GreedyMolSearch(
        reactants,
        templates,
        "greedy_search_clogp",
        "src/models/gsearch/checkpoint_clogp.pkl",
        "src/models/gsearch/results/Gmolsearch_results_clogp.csv",
        max_steps=5,
        max_reactions=8000000,
    )
    search.run_molsearch()
