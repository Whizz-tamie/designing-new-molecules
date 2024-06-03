import pandas as pd
import random
from src.models.random_molsearch import RandomMolSearch

# Load reactants and sample 2000
enamine_df = pd.read_csv("/rds/user/gtj21/hpc-work/designing-new-molecules/data/preprocessed_data/enamine_building_blocks.csv")
reactants = random.sample(enamine_df.SMILES.to_list(), 2000)

# Load reaction templates
templates = pd.read_csv("/rds/user/gtj21/hpc-work/designing-new-molecules/data/preprocessed_data/rxn_set_processed.txt", delimiter="|")

search = RandomMolSearch(reactants, templates, max_steps=20, max_reactions=400000, max_attempts=800000)
search.run_molsearch()