#!/usr/bin/env python3
"""
Calibration Audit — Project Confluence
========================================

Audits the complexity calibration path for biological plausibility,
cross-disease coherence, and stability under perturbation.

Usage:
    python scripts/calibration_audit.py --all --mock
    python scripts/calibration_audit.py --disease TNBC --mock
    python scripts/calibration_audit.py --optimize --mock --iterations 20
"""

import sys
import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from models.complexity_calibrator import (
    GlobalCalibrator,
    BiologicalAuditor,
    CalibrationWeights,
    SISCalibrator,
    ParameterMapper,
    run_full_calibration,
    SIGN_CONSTRAINTS,
    PLAUSIBILITY_BOUNDS,
)
from models.alphafold_client import DISEASE_PANELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_audit(
    diseases: list,
    use_mock: bool = True,
    optimize: bool = False,
    n_iterations: int = 20,
    output_dir: str = "results/calibration",
) -> dict:
    """Run full calibration audit."""

    print(f"\n{'='*70}")
    print(f"  Complexity Calibration Audit")
    print(f"  Diseases: {', '.join(diseases)}")
    print(f"  Mode: {'mock' if use_mock else 'live'}")
    print(f"{'='*70}\n")

    if optimize:
        print("▸ Running global optimization...")
        report = run_full_calibration(
            diseases=diseases,
            n_iterations=n_iterations,
            use_mock=use_mock,
        )
    else:
        # Run with default weights
        calibrator = GlobalCalibrator(diseases=diseases, use_mock=use_mock)
        weights = CalibrationWeights()
        results = {}
        for disease in diseases:
            try:
                result = calibrator.calibrate_disease(disease, weights)
                results[disease] = result
                print(f"  ✓ {disease}: calibration_score={result.calibration_score:.4f}")
            except Exception as e:
                print(f"  ✗ {disease}: FAILED — {e}")

        auditor = BiologicalAuditor()
        audit = auditor.audit(results)

        # Cross-panel coherence
        coherence = calibrator.cross_panel_coherence(results)

        report = {
            "calibration_summary": {
                "diseases": diseases,
                "n_iterations": 0,
                "optimized_weights": weights.to_dict(),
                "audit_grade": audit["grade"],
            },
            "audit": audit,
            "per_disease_results": {
                d: r.to_dict() for d, r in results.items()
            },
        }

    # ── Print Report ───────────────────────────────────────────────
    audit = report.get("audit", {})

    print(f"\n{'─'*70}")
    print(f"  AUDIT GRADE: {audit.get('grade', '?')}")
    print(f"{'─'*70}")

    print(f"\n  Sign Violations:    {audit.get('total_sign_violations', 0)}")
    for v in audit.get("sign_violations", []):
        print(f"    ⚠ {v}")

    print(f"\n  Bound Violations:   {audit.get('total_bound_violations', 0)}")
    for v in audit.get("bound_violations", []):
        print(f"    ⚠ {v}")

    print(f"\n  Avg Stability:      {audit.get('avg_stability', 0):.4f}")
    print(f"  Avg Coherence:      {audit.get('avg_cross_panel_coherence', 0):.4f}")

    # Per-disease
    per_disease = audit.get("per_disease", {})
    if per_disease:
        print(f"\n  {'Disease':<15} {'Cal.Score':<12} {'Φ-base':<10} {'Φ-cal':<10} {'Sign':<6} {'Stab'}")
        print(f"  {'─'*55}")
        for d, info in per_disease.items():
            print(f"  {d:<15} {info['calibration_score']:<12.4f} "
                  f"{info['baseline_phi_dist']:<10.4f} {info['calibrated_phi_dist']:<10.4f} "
                  f"{info['n_sign_violations']:<6} {info['avg_stability']:.4f}")

    # Cross-panel coherence
    coherence = audit.get("cross_panel_coherence", {})
    if coherence:
        print(f"\n  Cross-Panel Gene Coherence:")
        print(f"  {'Gene':<12} {'Diseases':<15} {'Sign OK?':<10} {'Mag CV':<10} {'Score'}")
        print(f"  {'─'*55}")
        for gene, info in sorted(coherence.items(), key=lambda x: x[1]["coherence_score"]):
            ds = "+".join(info["diseases"][:3])
            print(f"  {gene:<12} {ds:<15} {'✓' if info['sign_consistent'] else '✗':<10} "
                  f"{info['magnitude_cv']:<10.4f} {info['coherence_score']:.4f}")

    # SIS weights
    weights_data = report.get("calibration_summary", {}).get("optimized_weights", {})
    if weights_data:
        sis_w = weights_data.get("sis_weights", [])
        labels = ["pLDDT", "Stability", "Pocket", "ActiveSite", "Druggability"]
        if sis_w:
            print(f"\n  SIS Feature Weights:")
            for label, w in zip(labels, sis_w):
                bar = "█" * int(w * 40)
                print(f"    {label:<14} {w:.4f} {bar}")

    print()

    # ── Save Report ────────────────────────────────────────────────
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_path / f"calibration_audit_{ts}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved: {report_file}\n")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Complexity Calibration Audit"
    )
    parser.add_argument("--disease", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--optimize", action="store_true",
                       help="Run global weight optimization")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="results/calibration")
    args = parser.parse_args()

    # Pick diseases
    calibratable = ["TNBC", "GBM", "Alzheimers", "Parkinsons", "ALS", "Diabetes", "Lupus"]

    if args.all:
        diseases = calibratable
    elif args.disease:
        diseases = [args.disease]
    else:
        diseases = ["TNBC", "GBM", "Alzheimers"]

    run_audit(
        diseases=diseases,
        use_mock=not args.live,
        optimize=args.optimize,
        n_iterations=args.iterations,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
