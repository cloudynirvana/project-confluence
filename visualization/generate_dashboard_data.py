"""
Dashboard Data Generator
========================

Runs all pan-cancer generators through the geometric analysis pipeline
and outputs JSON data for the interactive visualization dashboard.

Usage:
    python visualization/generate_dashboard_data.py
"""

import sys
import os
import json
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tnbc_ode import TNBCODESystem, METABOLITES
from geometric_optimization import GeometricOptimizer, TherapeuticProtocolOptimizer
from coherence import CoherenceAnalyzer
from intervention import InterventionMapper


def analyze_cancer_type(name: str, A_cancer: np.ndarray, A_healthy: np.ndarray):
    """Run full geometric + coherence analysis on a cancer generator."""
    n = A_cancer.shape[0]
    optimizer = GeometricOptimizer(n)
    coherence = CoherenceAnalyzer()

    # Geometric analysis
    geo = optimizer.analyze_geometry(A_cancer)
    geo_healthy = optimizer.analyze_geometry(A_healthy)

    # Eigenvalue analysis
    evals = np.linalg.eigvals(A_cancer)
    evals_healthy = np.linalg.eigvals(A_healthy)

    # Coherence analysis
    coh = coherence.analyze(A_cancer, A_healthy)

    # Kramers escape rate (baseline, no drugs)
    escape_rate = optimizer.compute_kramers_escape_rate(A_cancer, noise_variance=0.1)

    return {
        "name": name,
        "eigenvalues": {
            "real": evals.real.tolist(),
            "imag": evals.imag.tolist(),
        },
        "eigenvalues_healthy": {
            "real": evals_healthy.real.tolist(),
            "imag": evals_healthy.imag.tolist(),
        },
        "geometry": {
            "curvature": float(geo.curvature),
            "anisotropy": float(geo.anisotropy),
            "volume": float(geo.volume),
        },
        "geometry_healthy": {
            "curvature": float(geo_healthy.curvature),
            "anisotropy": float(geo_healthy.anisotropy),
            "volume": float(geo_healthy.volume),
        },
        "coherence": {
            "overall_score": float(coh["overall_score"]),
            "stability_score": float(coh["stability"]["stability_score"]),
            "lyapunov": float(coh["stability"]["lyapunov_exponent"]),
            "contraction_rate": float(coh["stability"]["contraction_rate"]),
            "symmetry": float(coh["coupling"]["symmetry_score"]),
            "balance": float(coh["coupling"]["balance"]),
            "coupling_density": float(coh["coupling"]["density"]),
        },
        "escape_rate_baseline": float(escape_rate),
        "generator_matrix": A_cancer.tolist(),
    }


def analyze_drug_effects(cancer_data: dict, A_healthy: np.ndarray):
    """Compute drug effectiveness heatmap data across all cancer types."""
    mapper = InterventionMapper()
    optimizer = GeometricOptimizer(10)

    drug_names = []
    cancer_names = []
    effectiveness = []

    for drug in mapper.intervention_library:
        if drug.category == "geometric_deepener":
            continue  # Skip negative controls
        drug_names.append(drug.name)

    for cancer in cancer_data:
        cancer_names.append(cancer["name"])
        A = np.array(cancer["generator_matrix"])
        row = []
        for drug in mapper.intervention_library:
            if drug.category == "geometric_deepener":
                continue
            # Apply drug at mid-dose
            mid_dose = (drug.dosage_range[0] + drug.dosage_range[1]) / 2.0
            A_treated = A + drug.expected_effect * (mid_dose / drug.dosage_range[1])

            # Measure curvature reduction
            orig_curv = optimizer.compute_basin_curvature(A)
            new_curv = optimizer.compute_basin_curvature(A_treated)

            # Effectiveness = curvature reduction (positive = good)
            if abs(orig_curv) > 1e-10:
                eff = (orig_curv - new_curv) / abs(orig_curv)
            else:
                eff = 0.0
            row.append(float(eff))
        effectiveness.append(row)

    return {
        "drug_names": drug_names,
        "cancer_names": cancer_names,
        "effectiveness": effectiveness,
    }


def generate_protocol_data():
    """Generate the Geometric Achievement Protocol timeline data."""
    return {
        "phases": [
            {
                "name": "Phase 1: Flatten",
                "start_day": 0,
                "end_day": 25,
                "drugs": ["DCA", "Metformin"],
                "color": "#3b82f6",
                "description": "Minimize basin curvature via metabolic intervention",
            },
            {
                "name": "Phase 2: Heat",
                "start_day": 20,
                "end_day": 25,
                "drugs": ["Hyperthermia"],
                "color": "#f59e0b",
                "description": "Inject entropic noise to destabilize attractor",
            },
            {
                "name": "Phase 3: Push",
                "start_day": 25,
                "end_day": 60,
                "drugs": ["Anti-PD-1", "DCA", "Metformin"],
                "color": "#10b981",
                "description": "Apply immune vector force toward healthy state",
            },
        ]
    }


def main():
    print("Generating dashboard data...")

    A_healthy = TNBCODESystem.healthy_generator()
    generators = TNBCODESystem.pan_cancer_generators()

    # Analyze each cancer type
    cancer_data = []
    for name, A_cancer in generators.items():
        print(f"  Analyzing {name}...")
        result = analyze_cancer_type(name, A_cancer, A_healthy)
        cancer_data.append(result)

    # Drug effectiveness heatmap
    print("  Computing drug effectiveness matrix...")
    drug_data = analyze_drug_effects(cancer_data, A_healthy)

    # Protocol timeline
    protocol_data = generate_protocol_data()

    # Bifurcation scan for TNBC
    print("  Running TNBC bifurcation scan...")
    bifurcation = TNBCODESystem.bifurcation_scan(n_points=30)

    # Assemble output
    dashboard_data = {
        "cancer_types": cancer_data,
        "drug_effectiveness": drug_data,
        "protocol": protocol_data,
        "bifurcation": bifurcation,
        "metabolites": METABOLITES,
        "healthy_geometry": {
            "curvature": cancer_data[0]["geometry_healthy"]["curvature"],
            "anisotropy": cancer_data[0]["geometry_healthy"]["anisotropy"],
            "volume": cancer_data[0]["geometry_healthy"]["volume"],
        },
    }

    out_path = os.path.join(os.path.dirname(__file__), "dashboard_data.json")
    with open(out_path, "w") as f:
        json.dump(dashboard_data, f, indent=2)

    print(f"  Dashboard data written to {out_path}")
    print(f"  {len(cancer_data)} cancer types analyzed")
    print(f"  {len(drug_data['drug_names'])} drugs evaluated")
    print("Done!")


if __name__ == "__main__":
    main()
