"""
Combination Sweep Analysis
==========================
Computes synergy and antagonism between all pairs of interventions
in the library by evaluating the resulting basin curvature.
Outputs to results/combination_sweep.json
"""

import sys
import os
import json
import numpy as np
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tnbc_ode import TNBCODESystem
from geometric_optimization import GeometricOptimizer
from intervention import InterventionMapper

def main():
    print("Running Combination Sweep...")
    
    # 1. Setup
    n = 10
    mapper = InterventionMapper(n)
    lib = mapper.intervention_library
    optimizer = GeometricOptimizer(n)
    
    A_cancer = TNBCODESystem.pan_cancer_generators()["TNBC"]
    base_curvature = optimizer.compute_basin_curvature(A_cancer)
    
    # 2. Evaluate individual drug effects
    indiv_effects = {}
    for drug in lib:
        # Evaluate at standard dose multiplier = 1.0
        A_eff = A_cancer + drug.expected_effect
        c = optimizer.compute_basin_curvature(A_eff)
        # Effect is how much curvature is reduced (positive is good)
        effect = base_curvature - c
        indiv_effects[drug.name] = effect
        
    # 3. Evaluate combinations
    results = []
    synergy_pairs = []
    antagonism_pairs = []
    
    for d1, d2 in combinations(lib, 2):
        # We assume independent additive effects on the generator matrix.
        # But the eigenvalues/curvature are non-linear w.r.t the matrix entries!
        A_eff = A_cancer + d1.expected_effect + d2.expected_effect
        c_combo = optimizer.compute_basin_curvature(A_eff)
        
        effect_combo = base_curvature - c_combo
        effect_expected = indiv_effects[d1.name] + indiv_effects[d2.name]
        
        # Synergy index: Actual vs Expected
        # If effect_expected is exactly 0, handle division by zero
        if effect_expected == 0:
            synergy_index = 0.0
        else:
            synergy_index = effect_combo / effect_expected
            
        diff = effect_combo - effect_expected
        
        pair_data = {
            "drug1": d1.name,
            "drug2": d2.name,
            "effect_1": float(indiv_effects[d1.name]),
            "effect_2": float(indiv_effects[d2.name]),
            "effect_expected": float(effect_expected),
            "effect_combo": float(effect_combo),
            "synergy_index": float(synergy_index),
            "difference": float(diff)
        }
        
        results.append(pair_data)
        
        # Arbitrary thresholds for meaningful synergy/antagonism
        if diff > 0.05 and synergy_index > 1.1:
            synergy_pairs.append(pair_data)
        elif diff < -0.05 and synergy_index < 0.9:
            antagonism_pairs.append(pair_data)
            
    # Sort for best/worst
    synergy_pairs.sort(key=lambda x: x["difference"], reverse=True)
    antagonism_pairs.sort(key=lambda x: x["difference"])
    
    # 4. Save results
    output = {
        "base_curvature": float(base_curvature),
        "individual_effects": {k: float(v) for k, v in indiv_effects.items()},
        "top_synergies": synergy_pairs,
        "top_antagonisms": antagonism_pairs,
        "all_combinations": results
    }
    
    os.makedirs('results', exist_ok=True)
    out_path = 'results/combination_sweep.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
        
    print(f"Sweep complete. Found {len(synergy_pairs)} synergistic and {len(antagonism_pairs)} antagonistic pairs.")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
