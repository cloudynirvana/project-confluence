"""
CCLE Metabolomics Calibration (Simulated)
=========================================
Validates that the GeneratorExtractor can recover the correct attractor
geometry from noisy, high-dimensional metabolomics time-series data,
mimicking public datasets like CCLE or TCGA.
Outputs to results/ccle_calibration.json
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tnbc_ode import TNBCODESystem
from generator import GeneratorExtractor, simulate_dynamics
from geometric_optimization import GeometricOptimizer

def main():
    print("Running CCLE/TCGA Metabolomics Calibration (Simulated)...")

    n = 10
    optimizer = GeometricOptimizer(n)
    
    # 1. Ground Truth Generator (Representing true CCLE biology)
    A_true = TNBCODESystem.pan_cancer_generators()["TNBC"]
    true_curvature = optimizer.compute_basin_curvature(A_true)
    print(f"  Ground Truth Curvature: {true_curvature:.4f}")

    # 2. Simulate Noisy CCLE Time-Series Data
    # 50 patients/replicates, 20 timepoints each
    np.random.seed(42)
    n_replicates = 50
    n_timepoints = 20
    time_points = np.linspace(0, 10, n_timepoints)
    
    all_X = []
    all_dX = []
    
    # We will pool data from multiple replicates to improve extraction
    for i in range(n_replicates):
        initial_state = np.random.randn(n) * 2.0
        # Add 10% measurement noise (typical for Mass Spec)
        X_sim = simulate_dynamics(A_true, initial_state, time_points, noise_std=0.2)
        all_X.append(X_sim)

    # 3. Extract Generator from Data
    extractor = GeneratorExtractor(regularization=0.5)
    
    # For a pooled dataset, we can just concatenate the mid-points and derivatives
    # But generator.py's extract() expects a single continuous timeseries.
    # To adapt, we'll manually compute derivatives and fit.
    
    X_mid_pool = []
    dX_mid_pool = []
    
    for X in all_X:
        dX = extractor._compute_derivatives(X, time_points)
        X_mid_pool.append(X[1:-1, :])
        dX_mid_pool.append(dX)
        
    X_mid_flat = np.vstack(X_mid_pool)
    dX_mid_flat = np.vstack(dX_mid_pool)
    
    from sklearn.linear_model import Ridge
    A_extracted = np.zeros((n, n))
    ridge = Ridge(alpha=0.5, fit_intercept=False)
    for j in range(n):
        ridge.fit(X_mid_flat, dX_mid_flat[:, j])
        A_extracted[:, j] = ridge.coef_
    A_extracted = A_extracted.T
    
    # 4. Compare Extracted vs True Geometry
    ext_curvature = optimizer.compute_basin_curvature(A_extracted)
    print(f"  Extracted Curvature:  {ext_curvature:.4f}")
    
    # Compute relative error in matrix norm
    norm_diff = np.linalg.norm(A_true - A_extracted, 'fro')
    norm_true = np.linalg.norm(A_true, 'fro')
    matrix_error = norm_diff / norm_true
    
    # Compare eigenvalues
    val_true, _ = np.linalg.eig(A_true)
    val_ext, _ = np.linalg.eig(A_extracted)
    
    val_true_real = np.sort(val_true.real)
    val_ext_real = np.sort(val_ext.real)
    eig_error = np.mean(np.abs(val_true_real - val_ext_real))
    
    print(f"  Matrix Frobenius Error: {matrix_error:.4f} ({matrix_error*100:.1f}%)")
    print(f"  Eigenvalue MAE:         {eig_error:.4f}")
    
    # 5. Save results
    results = {
        "dataset_name": "Simulated_CCLE_TNBC",
        "n_replicates": n_replicates,
        "n_timepoints": n_timepoints,
        "noise_std": 0.2,
        "true_curvature": float(true_curvature),
        "extracted_curvature": float(ext_curvature),
        "curvature_error": float(abs(true_curvature - ext_curvature)),
        "matrix_frobenius_error": float(matrix_error),
        "eigenvalue_mae": float(eig_error),
        "calibration_status": "SUCCESS" if matrix_error < 0.2 else "FAILED"
    }

    out_path = 'results/ccle_calibration.json'
    os.makedirs('results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nCalibration Status: {results['calibration_status']}")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
