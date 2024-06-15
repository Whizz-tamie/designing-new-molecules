import requests
import urllib.parse

def get_compound_name(smiles):
    name = __get_name_from_smiles(smiles)
    if not name:
        name = __get_iupac_name(smiles)
    return name

def __get_name_from_smiles(smiles):
    try:
        # URL-encode the SMILES string
        encoded_smiles = urllib.parse.quote(smiles)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smiles}/property/IUPACName/JSON"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Extract the IUPAC name
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                return data['PropertyTable']['Properties'][0]['IUPACName']
            return None
        else:
            #print(f"Error fetching name for SMILES {smiles}: HTTP {response.status_code}")
            return None
    except Exception as e:
        #print(f"Error fetching name for SMILES {smiles}: {e}")
        return None

def __get_iupac_name(smiles):
        return ("Molecule not in PubChem Database")