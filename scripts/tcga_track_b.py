# -*- coding: utf-8 -*-
"""
TCGA Track B Ingestion — Project Confluence
==========================================

Minimal end-to-end ingestion for real longitudinal cohorts:
  - Load cohort JSON
  - Map variables to 15D state
  - Resample to uniform dt
  - Compute Phi profiles
  - Correlate Phi-distance with survival (if provided)
"""

import argparse
import json
import os
import sys
from typing import Optional
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.track_b import (
    load_track_b_cohort,
    build_trajectory,
    compute_phi_for_patient,
    healthy_reference_phi,
    infer_neural_ode,
)
from models.ode_system import ComplexAttractorODE, ExtendedParams


def _spearman_rank_correlation(x, y):
    n = len(x)
    if n < 3:
        return 0.0
    rank_x = np.argsort(np.argsort(x)).astype(float)
    rank_y = np.argsort(np.argsort(y)).astype(float)
    d = rank_x - rank_y
    rho = 1 - 6 * np.sum(d**2) / (n * (n**2 - 1))
    return float(rho)


def run_track_b(input_path: str, output_path: str, use_neural: bool, resample_dt: Optional[float]):
    cohort = load_track_b_cohort(input_path)
    ode = ComplexAttractorODE(params=ExtendedParams(), use_nonlinear=True, use_immune=True, use_microenv=True)
    baseline = ode.healthy_initial_state()
    healthy_phi = healthy_reference_phi(dt=0.5)

    results = []
    for patient in cohort.patients:
        traj, t_grid, dt = build_trajectory(
            patient,
            variables=cohort.variables,
            baseline=baseline,
            resample_dt=resample_dt,
        )

        traj_used = traj
        if use_neural:
            inferred = infer_neural_ode(traj, t_grid)
            if inferred is not None:
                traj_used = inferred

        phi = compute_phi_for_patient(traj_used, dt)
        phi_vec = np.array(phi["phi_vector"])
        phi_dist = float(np.linalg.norm(phi_vec - healthy_phi))

        results.append({
            "patient_id": patient.patient_id,
            "phi_vector": phi["phi_vector"],
            "phi_magnitude": phi["phi_magnitude"],
            "phi_distance": phi_dist,
            "coherence": phi["coherence"],
            "archetype": phi["archetype"],
            "archetype_confidence": phi["archetype_confidence"],
            "survival_days": patient.survival_days,
        })

    survival = [r["survival_days"] for r in results if r["survival_days"] is not None]
    distances = [r["phi_distance"] for r in results if r["survival_days"] is not None]
    rho = _spearman_rank_correlation(distances, survival) if len(survival) >= 3 else None

    output = {
        "pipeline": "TCGA Track B Ingestion (Longitudinal Cohort)",
        "cohort_id": cohort.cohort_id,
        "disease": cohort.disease,
        "n_patients": len(results),
        "time_unit": cohort.time_unit,
        "use_neural_ode": bool(use_neural),
        "spearman_rho": rho,
        "patients": results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("=" * 70)
    print("  PROJECT CONFLUENCE -- TCGA TRACK B INGESTION")
    print("=" * 70)
    print(f"  Cohort: {cohort.cohort_id} | Disease: {cohort.disease}")
    print(f"  Patients: {len(results)} | Time unit: {cohort.time_unit}")
    print(f"  Output: {output_path}")
    if rho is not None:
        print(f"  Spearman rho (Phi-distance vs survival): {rho:.4f}")
    else:
        print("  Spearman rho: not computed (need >= 3 patients with survival)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/track_b/mock_cohort.json",
        help="Path to Track B cohort JSON",
    )
    parser.add_argument(
        "--output",
        default="results/tcga_val/track_b_metrics.json",
        help="Output metrics JSON path",
    )
    parser.add_argument(
        "--use-neural-ode",
        action="store_true",
        help="Run neural ODE reconstruction if torchdiffeq is available",
    )
    parser.add_argument(
        "--resample-dt",
        type=float,
        default=None,
        help="Optional fixed dt for resampling (overrides median spacing)",
    )
    args = parser.parse_args()

    run_track_b(args.input, args.output, args.use_neural_ode, args.resample_dt)


if __name__ == "__main__":
    main()
