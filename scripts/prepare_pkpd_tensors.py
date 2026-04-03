import pandas as pd
import torch
import os

def prepare_pkpd_tensors():
    """
    Zero-Friction Workflow:
    1. Directly downloads the raw PK/PD dataset.
    2. Filters down to the exact input/output variables (Time, Dose, Concentration).
    3. Converts irregular patient timelines into standardized PyTorch Tensors.
    4. Saves them locally for your neural_ode.py to ingest immediately.
    """
    url = "https://raw.githubusercontent.com/dpastoor/PKPDdatasets/master/data/sd_oral_richpk.csv"
    output_path = "data/pkpd_tensors.pt"
    
    print("Initiating automated pipeline...")
    print(f"Fetching highly irregular clinical data from {url}")
    
    try:
        # Load the raw dataset
        df = pd.read_csv(url)
        
        # Clean the dataset: Remove empty concentration readings (NaNs) and isolate required columns
        df_clean = df[['ID', 'Time', 'Dose', 'Conc']].dropna(subset=['Conc']).copy()
        
        print(f"Dataset successfully cleaned. Total observations: {len(df_clean)}")
        print("Transforming clinical patient data into PyTorch Neural ODE structures...")
        
        # Group by patient ID so we have distinct time-series sequences
        patients = df_clean['ID'].unique()
        
        patient_tensors = []
        for patient_id in patients:
            # Isolate the data for this specific patient
            p_data = df_clean[df_clean['ID'] == patient_id]
            
            # Create a localized tensor for this patient [Time, Dose, Concentration]
            # Since the data is irregular, the Time column will tell the Neural ODE exactly when the gaps occur
            tensor_seq = torch.tensor(p_data[['Time', 'Dose', 'Conc']].values, dtype=torch.float32)
            patient_tensors.append({
                'patient_id': patient_id,
                'series_tensor': tensor_seq
            })
            
        print(f"Successfully tensorized sequences for {len(patients)} distinct patients.")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the dictionary of tensors directly to disk
        torch.save(patient_tensors, output_path)
        print(f"\nSUCCESS. The fully prepped biological complexity data is saved at: {output_path}")
        print("Your neural_ode.py script can now run `torch.load('data/pkpd_tensors.pt')` and start training instantly.")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        print("Please ensure you have an active internet connection and pip install pandas torch.")

if __name__ == "__main__":
    prepare_pkpd_tensors()
