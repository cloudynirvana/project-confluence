import requests
import os
import json

def fetch_alphafold_structure(uniprot_id: str, output_dir: str = "structures"):
    """
    Fetches the pre-computed AlphaFold 3D structure for a given UniProt ID
    from the EMBL-EBI REST API and saves it locally as a PDB file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            print(f"No AlphaFold prediction found for UniProt ID: {uniprot_id}")
            return None
            
        prediction = data[0]
        pdb_url = prediction.get("pdbUrl")
        
        if not pdb_url:
            print(f"PDB download URL not found in API response for {uniprot_id}")
            return None
            
        print(f"Downloading structure from: {pdb_url}")
        pdb_response = requests.get(pdb_url)
        pdb_response.raise_for_status()
        
        file_path = os.path.join(output_dir, f"{uniprot_id}_alphafold.pdb")
        with open(file_path, "wb") as f:
            f.write(pdb_response.content)
            
        print(f"Successfully saved AlphaFold structure to {file_path}")
        return file_path
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch data from AlphaFold API: {e}")
        return None

if __name__ == "__main__":
    test_uniprot_id = "P04637"
    fetch_alphafold_structure(test_uniprot_id)
