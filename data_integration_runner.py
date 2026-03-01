"""
Real Data Integration Runner — Project Confluence
====================================================

Connects the calibration data (literature-derived profiles), 
generator calibrator (Bayesian refinement), and drug validator
into a single executable pipeline.

Usage:
    python data_integration_runner.py
    python data_integration_runner.py --cancer TNBC --verbose
    python data_integration_runner.py --compare-only
"""

import sys
import os
import argparse
import numpy as np
from typing import Dict, List, Tuple

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tnbc_ode import TNBCODESystem, GENERATOR_METADATA
from geometric_optimization import GeometricOptimizer
from coherence import CoherenceAnalyzer
from intervention import InterventionMapper
from calibration_data import get_real_profiles, get_all_profiles, get_ic50_data, get_flux_data
from generator_calibrator import GeneratorCalibrator, CalibrationReport

# Cancer types that map between calibration_data (Ovarian=HGSOC, CML=CRC)
CANCER_NAME_MAP = {
    "TNBC": "TNBC", "PDAC": "PDAC", "NSCLC": "NSCLC",
    "GBM": "GBM", "Melanoma": "Melanoma", "CML": "CRC",
    "Ovarian": "HGSOC", "AML": "AML", "mCRPC": "mCRPC", "HCC": "HCC",
}


def calibrate_generators(
    cancers: List[str] = None,
    alpha: float = 0.3,
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, CalibrationReport]]:
    """
    Calibrate all generators against real metabolomics profiles.
    
    Returns:
        (refined_generators, calibration_reports)
    """
    generators = TNBCODESystem.pan_cancer_generators()
    all_profiles = get_all_profiles()  # {cancer: (n, 10)}
    calibrator = GeneratorCalibrator(max_iterations=500, verbose=verbose)

    target_cancers = cancers or list(generators.keys())
    
    refined = {}
    reports = {}
    
    for cancer in target_cancers:
        # Map generator name to calibration data name
        data_name = CANCER_NAME_MAP.get(cancer, cancer)
        
        if data_name not in all_profiles:
            if verbose:
                print(f"⚠️ No calibration data for {cancer} (mapped to {data_name}), skipping.")
            refined[cancer] = generators[cancer]
            continue
        
        A_prior = generators[cancer]
        real_data = all_profiles[data_name]
        
        if verbose:
            print(f"\n{'═' * 60}")
            print(f"CALIBRATING: {cancer} ({real_data.shape[0]} cell line profiles)")
            print(f"{'═' * 60}")
        
        A_refined, report = calibrator.refine_generator(
            A_prior=A_prior,
            real_profiles=real_data,
            cancer_type=cancer,
            alpha=alpha,
        )
        
        refined[cancer] = A_refined
        reports[cancer] = report
    
    return refined, reports


def compare_generators(
    original: Dict[str, np.ndarray],
    refined: Dict[str, np.ndarray],
    reports: Dict[str, CalibrationReport],
) -> str:
    """Generate a comparison report between original and refined generators."""
    optimizer = GeometricOptimizer(10)
    coherence = CoherenceAnalyzer()
    
    lines = [
        "# Real Data Integration — Calibration Results",
        "",
        "> Generator matrices refined using literature-derived metabolomics profiles",
        "> from 30 cell lines across 10 cancer types.",
        "",
        "## Calibration Summary",
        "",
        "| Cancer | R² Before | R² After | RMSE Before | RMSE After | Entries Changed | Frobenius Dist |",
        "|--------|-----------|----------|-------------|------------|-----------------|----------------|",
    ]
    
    for cancer in original:
        if cancer in reports:
            r = reports[cancer]
            lines.append(
                f"| {cancer} | {r.r_squared_before:.4f} | {r.r_squared_after:.4f} "
                f"| {r.rmse_before:.4f} | {r.rmse_after:.4f} "
                f"| {r.entries_changed} | {r.frobenius_distance:.4f} |"
            )
        else:
            lines.append(f"| {cancer} | - | - | - | - | 0 | 0.0 |")
    
    # Basin curvature comparison
    lines.extend([
        "",
        "## Basin Curvature Changes",
        "",
        "| Cancer | Original Curvature | Calibrated Curvature | Delta |",
        "|--------|-------------------|---------------------|-------|",
    ])
    
    for cancer in original:
        c_orig = optimizer.compute_basin_curvature(original[cancer])
        c_ref = optimizer.compute_basin_curvature(refined.get(cancer, original[cancer]))
        delta = c_ref - c_orig
        lines.append(f"| {cancer} | {c_orig:.4f} | {c_ref:.4f} | {delta:+.4f} |")
    
    # Coherence comparison
    lines.extend([
        "",
        "## Coherence Changes",
        "",
        "| Cancer | Original | Calibrated | Delta |",
        "|--------|----------|------------|-------|",
    ])
    
    for cancer in original:
        coherence.analyze(original[cancer])
        co_orig = coherence._compute_overall_score()
        coherence.analyze(refined.get(cancer, original[cancer]))
        co_ref = coherence._compute_overall_score()
        delta = co_ref - co_orig
        lines.append(f"| {cancer} | {co_orig:.4f} | {co_ref:.4f} | {delta:+.4f} |")
    
    return "\n".join(lines)


def run_comparison_simulations(
    original_generators: Dict[str, np.ndarray],
    refined_generators: Dict[str, np.ndarray],
    n_trials: int = 30,
) -> str:
    """Run simulations with both generator sets and compare."""
    from confluence_runner import (
        compute_seriousness, select_drugs, compute_phase_timing,
        run_monte_carlo,
    )
    
    A_healthy = TNBCODESystem.healthy_generator()
    mapper = InterventionMapper()
    
    lines = [
        "",
        "## Simulation Comparison (Original vs Calibrated)",
        "",
        "| Cancer | Orig Cure Rate | Cal Cure Rate | Orig Escape | Cal Escape | Δ Cure Rate |",
        "|--------|---------------|---------------|-------------|------------|-------------|",
    ]
    
    for cancer in original_generators:
        results = {}
        
        for label, gens in [("orig", original_generators), ("cal", refined_generators)]:
            A = gens[cancer]
            try:
                seriousness = compute_seriousness(cancer, A, A_healthy)
                drugs = select_drugs(mapper, A, A_healthy, cancer)
                timing = compute_phase_timing(seriousness)
                dists, cure_rate, ci_lo, ci_hi = run_monte_carlo(
                    A, A_healthy, drugs, timing, n_trials=n_trials
                )
                results[label] = {
                    "cure_rate": cure_rate,
                    "escape": float(np.mean(dists)),
                }
            except Exception as e:
                results[label] = {"cure_rate": 0, "escape": 999, "error": str(e)}
        
        orig = results["orig"]
        cal = results["cal"]
        delta_cr = (cal["cure_rate"] - orig["cure_rate"]) * 100
        
        lines.append(
            f"| {cancer} "
            f"| {orig['cure_rate']*100:.1f}% "
            f"| {cal['cure_rate']*100:.1f}% "
            f"| {orig['escape']:.3f} "
            f"| {cal['escape']:.3f} "
            f"| {delta_cr:+.1f}% |"
        )
    
    return "\n".join(lines)


def validate_drugs() -> str:
    """Cross-reference SAEM drug effects with real IC50 data."""
    ic50_data = get_ic50_data()
    mapper = InterventionMapper()
    lib = {inv.name: inv for inv in mapper.intervention_library}
    
    lines = [
        "",
        "## Drug Validation Against IC50 Data",
        "",
        "| Drug | Cell Line | IC50 | SAEM Effect Magnitude | In Library? |",
        "|------|-----------|------|----------------------|-------------|",
    ]
    
    # Map IC50 drug names to SAEM names
    drug_map = {
        "DCA": "Dichloroacetate (DCA)",
        "CB-839": "CB-839 (Telaglenastat)",
        "Metformin": "Metformin",
        "2-DG": "2-Deoxyglucose (2-DG)",
    }
    
    for ic50_name, cell_lines in ic50_data.items():
        saem_name = drug_map.get(ic50_name)
        in_lib = "✅" if saem_name and saem_name in lib else "❌"
        effect_mag = f"{float(np.linalg.norm(lib[saem_name].expected_effect)):.4f}" if saem_name and saem_name in lib else "-"
        
        for cell_line, ic50 in list(cell_lines.items())[:3]:  # top 3
            unit = "nM" if ic50 < 0.001 else ("µM" if ic50 < 1 else "mM")
            val = ic50 * 1e6 if unit == "nM" else (ic50 * 1e3 if unit == "µM" else ic50)
            lines.append(
                f"| {ic50_name} | {cell_line} | {val:.1f} {unit} | {effect_mag} | {in_lib} |"
            )
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="SAEM Real Data Integration")
    parser.add_argument("--cancer", type=str, help="Single cancer type to calibrate")
    parser.add_argument("--alpha", type=float, default=0.3, help="Max entry deviation (default 0.3)")
    parser.add_argument("--compare-only", action="store_true", help="Skip calibration, just compare")
    parser.add_argument("--simulate", action="store_true", help="Run simulations after calibration")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output", type=str, default="results/data_integration_report.md")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PROJECT CONFLUENCE — Real Data Integration Pipeline")
    print("=" * 60)
    
    cancers = [args.cancer] if args.cancer else None
    
    # Step 1: Get original generators
    original_generators = TNBCODESystem.pan_cancer_generators()
    print(f"\n✅ Loaded {len(original_generators)} original generators")
    
    # Step 2: Show calibration data summary
    profiles = get_real_profiles()
    all_profiles = get_all_profiles()
    print(f"✅ Loaded calibration data for {len(profiles)} cancer types")
    for ctype, prof in all_profiles.items():
        print(f"   {ctype}: {prof.shape[0]} cell lines")
    
    # Step 3: Calibrate
    if args.compare_only:
        refined_generators = original_generators
        reports = {}
        print("\n⏭️ Skipping calibration (--compare-only)")
    else:
        print("\n🔬 Starting Bayesian generator calibration...")
        refined_generators, reports = calibrate_generators(
            cancers=cancers,
            alpha=args.alpha,
            verbose=args.verbose,
        )
        print(f"\n✅ Calibration complete! {len(reports)} generators refined.")
    
    # Step 4: Generate comparison report
    report_text = compare_generators(original_generators, refined_generators, reports)
    
    # Step 5: Drug validation
    report_text += validate_drugs()
    
    # Step 6: Run simulations (optional)
    if args.simulate:
        print("\n🧬 Running comparison simulations...")
        report_text += run_comparison_simulations(
            original_generators, refined_generators, n_trials=30
        )
    
    # Step 7: Save report
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n📄 Report saved to: {args.output}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("INTEGRATION SUMMARY")
    print("=" * 60)
    
    if reports:
        r2_improvements = []
        for cancer, r in reports.items():
            improvement = r.r_squared_after - r.r_squared_before
            r2_improvements.append(improvement)
            emoji = "📈" if improvement > 0 else "📉"
            print(f"  {emoji} {cancer}: R² {r.r_squared_before:.4f} → {r.r_squared_after:.4f} (Δ={improvement:+.4f})")
        
        print(f"\n  Mean R² improvement: {np.mean(r2_improvements):+.4f}")
    
    print(f"\n🏁 Done! Full report at: {args.output}")


if __name__ == "__main__":
    main()
