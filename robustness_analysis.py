"""
Robustness Analysis
===================
Maps per-drug dose sensitivity curves. Sweeps the dosage of each 
intervention and computes the resulting basin curvature to test
robustness and find the optimal therapeutic window.
Outputs to results/robustness_analysis.json
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tnbc_ode import TNBCODESystem
from geometric_optimization import GeometricOptimizer
from intervention import InterventionMapper

def main():
    print("Running Robustness Analysis (Dose Sensitivity)...")
    
    n = 10
    mapper = InterventionMapper(n)
    lib = mapper.intervention_library
    optimizer = GeometricOptimizer(n)
    
    A_cancer = TNBCODESystem.pan_cancer_generators()["TNBC"]
    # We will use min_curvature as our sensitivity metric
    base_curvature = optimizer.compute_basin_curvature(A_cancer)
    
    # Dose multipliers to test (from 0.0x to 3.0x standard dose)
    dose_multipliers = np.linspace(0.0, 3.0, 31)
    
    results = {
        "base_curvature": float(base_curvature),
        "dose_points": dose_multipliers.tolist(),
        "curves": {}
    }
    
    for drug in lib:
        print(f"  Mapping curve for {drug.name}...")
        curve = []
        
        for mult in dose_multipliers:
            # We scale the expected_effect matrix by the dose multiplier
            A_eff = A_cancer + drug.expected_effect * mult
            
            try:
                c = optimizer.compute_basin_curvature(A_eff)
                curve.append(float(c))
            except Exception as e:
                # If curvature calculation fails (e.g. matrix becomes unstable)
                curve.append(None)
                
        results["curves"][drug.name] = curve
        
    out_path = 'results/robustness_analysis.json'
    os.makedirs('results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Analysis complete. Curves mapped for {len(lib)} drugs.")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
