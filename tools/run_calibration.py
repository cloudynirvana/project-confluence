"""
Run Real Data Calibration — Project Confluence
================================================

Calibrates all 10 cancer generator matrices against literature-derived
metabolomics profiles, producing refined generators and a detailed report.

Usage:
    python tools/run_calibration.py
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tnbc_ode import TNBCODESystem, METABOLITES
from generator_calibrator import GeneratorCalibrator
from calibration_data import get_all_profiles, get_real_profiles, get_flux_data, print_summary


def main():
    print("=" * 65)
    print("  PROJECT CONFLUENCE — REAL DATA CALIBRATION")
    print("  Grounding theoretical generators in published metabolomics")
    print("=" * 65)
    
    # 1. Print data summary
    print_summary()
    
    # 2. Load theoretical generators and real profiles
    generators = TNBCODESystem.pan_cancer_generators()
    all_profiles = get_all_profiles()
    mean_profiles = get_real_profiles()
    
    # 3. Run calibration for each cancer type
    calibrator = GeneratorCalibrator(max_iterations=300, verbose=True)
    
    refined_generators = {}
    reports = {}
    
    print("\n" + "=" * 65)
    print("  CALIBRATING GENERATORS")
    print("=" * 65)
    
    for cancer_type in generators:
        if cancer_type not in all_profiles:
            print(f"\n⚠ {cancer_type}: No real data available, keeping theoretical")
            refined_generators[cancer_type] = generators[cancer_type]
            continue
        
        print(f"\n{'─' * 65}")
        print(f"  {cancer_type}")
        print(f"{'─' * 65}")
        
        profiles = all_profiles[cancer_type]
        A_ref, report = calibrator.refine_generator(
            A_prior=generators[cancer_type],
            real_profiles=profiles,
            cancer_type=cancer_type,
            alpha=0.3,  # Allow ±30% entry deviation
        )
        
        refined_generators[cancer_type] = A_ref
        reports[cancer_type] = report
    
    # 4. Summary comparison
    print("\n" + "=" * 65)
    print("  CALIBRATION SUMMARY")
    print("=" * 65)
    print(f"\n{'Cancer':<10} {'RMSE Before':>12} {'RMSE After':>12} {'ΔR²':>8} {'Entries Δ':>10} {'Stable':>7}")
    print("─" * 65)
    
    for ctype, report in reports.items():
        delta_r2 = report.r_squared_after - report.r_squared_before
        stable_str = "✓" if report.stable_after else "✗"
        print(f"{ctype:<10} {report.rmse_before:>12.4f} {report.rmse_after:>12.4f} "
              f"{delta_r2:>+8.4f} {report.entries_changed:>10d} {stable_str:>7}")
    
    # 5. Most-changed metabolite axes per cancer type
    print(f"\n{'Cancer':<10} Top-3 Metabolite Errors (Before → After)")
    print("─" * 65)
    for ctype, report in reports.items():
        # Sort by error reduction
        changes = []
        for met in METABOLITES:
            before = report.per_metabolite_error_before.get(met, 0)
            after = report.per_metabolite_error_after.get(met, 0)
            changes.append((met, before, after, before - after))
        changes.sort(key=lambda x: x[3], reverse=True)
        top3 = changes[:3]
        desc = ", ".join(f"{m}: {b:.3f}→{a:.3f}" for m, b, a, _ in top3)
        print(f"{ctype:<10} {desc}")
    
    # 6. Save results
    os.makedirs('results/data', exist_ok=True)
    
    # Save as JSON (for human review)
    summary_json = {}
    for ctype, report in reports.items():
        summary_json[ctype] = {
            "n_samples": report.n_samples,
            "rmse_before": report.rmse_before,
            "rmse_after": report.rmse_after,
            "r_squared_before": report.r_squared_before,
            "r_squared_after": report.r_squared_after,
            "entries_changed": report.entries_changed,
            "frobenius_distance": report.frobenius_distance,
            "stable_after": report.stable_after,
        }
    
    with open('results/data/calibration_results.json', 'w') as f:
        json.dump(summary_json, f, indent=2)
    print(f"\n✓ Results saved to results/data/calibration_results.json")
    
    # Save refined generators as numpy arrays
    np.savez(
        'results/data/refined_generators.npz',
        **{f"{k}_refined": v for k, v in refined_generators.items()},
        **{f"{k}_original": v for k, v in generators.items()},
    )
    print(f"✓ Generator matrices saved to results/data/refined_generators.npz")
    
    # 7. Eigenvalue comparison
    print(f"\n{'Cancer':<10} {'λ_min (Original)':>18} {'λ_min (Refined)':>18} {'Δ Depth':>10}")
    print("─" * 65)
    for ctype in reports:
        orig_eigs = np.linalg.eigvals(generators[ctype])
        ref_eigs = np.linalg.eigvals(refined_generators[ctype])
        orig_depth = float(np.min(np.abs(orig_eigs.real)))
        ref_depth = float(np.min(np.abs(ref_eigs.real)))
        delta = ref_depth - orig_depth
        print(f"{ctype:<10} {orig_depth:>18.4f} {ref_depth:>18.4f} {delta:>+10.4f}")
    
    print(f"\n{'=' * 65}")
    print(f"  CALIBRATION COMPLETE")
    print(f"{'=' * 65}")
    
    return refined_generators, reports


if __name__ == "__main__":
    main()
