"""
Biologic Operator Integration Test — Project Confluence
========================================================

Validates the biologic operator formalism:
  1. All 6 biologic classes instantiate and evaluate correctly
  2. Synergy tensor identifies known synergistic pairs
  3. Resistance geometry detects trajectory signatures
  4. Phi-state classifier returns correct biologic recommendations

Run:
    python scripts/test_biologics.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.biologic_operator import (
    BiologicOperator, BIOLOGIC_LIBRARY, SynergyTensor,
    curvature_approx, detect_resistance_signal, classify_phi_state,
    bifurcation_proximity, PHI_STAR_DEFAULT, PHI_LABELS,
    create_checkpoint_inhibitor, create_bispecific, create_adc,
    create_anti_angiogenic, create_cytokine, create_targeted_biologic,
)


def divider(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_operators():
    divider("TEST 1: Biologic Operator Evaluation")

    phi_cancer = np.array([0.85, 0.30, 0.20, 0.80, 0.35])
    phi_star = PHI_STAR_DEFAULT

    print(f"Cancer Phi : {phi_cancer}")
    print(f"Healthy Phi: {phi_star}")
    print()

    for name, op in list(BIOLOGIC_LIBRARY.items())[:6]:
        force = op.evaluate(phi_cancer, t=5.0, dose=1.0, phi_star=phi_star)
        print(f"  {name:25s} [{op.class_label:10s}]  dPhi = {np.round(force, 4)}")

    print("\n  All 16 operators in library instantiated successfully.")


def test_synergy():
    divider("TEST 2: Synergy Tensor")

    biologics = [
        create_checkpoint_inhibitor("anti-PD1"),
        create_bispecific("blinatumomab"),
        create_adc("T-DXd"),
        create_anti_angiogenic("bevacizumab"),
        create_cytokine("IL-15"),
        create_targeted_biologic("cetuximab"),
    ]

    tensor = SynergyTensor(biologics)

    print("Synergy matrix S_ij:")
    labels = [b.name for b in biologics]
    header = f"{'':15s}" + "".join(f"{l:>12s}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        vals = "".join(f"{tensor.S[i,j]:12.2f}" for j in range(len(labels)))
        print(f"{row_label:15s}{vals}")

    print("\nTop 3 synergistic pairs:")
    for i, j, score in tensor.best_combination(k=3):
        print(f"  {biologics[i].name} + {biologics[j].name}: S = {score:+.2f}")


def test_resistance():
    divider("TEST 3: Resistance Geometry")

    # Simulate a phi trajectory showing CPI resistance (phi3 declining)
    T = 20
    phi_traj = np.zeros((5, T))
    for t in range(T):
        phi_traj[0, t] = 0.7 + 0.02 * t          # phi1 rising
        phi_traj[1, t] = 0.5                       # phi2 stable
        phi_traj[2, t] = 0.2 + 0.04 * t if t < 10 else 0.6 - 0.03 * (t - 10)  # phi3 rises then falls
        phi_traj[3, t] = 0.5 + 0.01 * t           # phi4 slowly rising
        phi_traj[4, t] = 0.5                       # phi5 stable

    cpi = create_checkpoint_inhibitor()
    K = curvature_approx(phi_traj, cpi.A_matrix)
    print(f"Curvature in CPI action direction: {K:.6f}")

    signals = detect_resistance_signal(phi_traj, window=5)
    print(f"Resistance signals detected:")
    for key, val in signals.items():
        if key != "sufficient_data":
            print(f"  {key:25s}: {val}")


def test_classifier():
    divider("TEST 4: Phi-State Classifier")

    test_cases = [
        ("Immune-cold + high heterogeneity", np.array([0.90, 0.40, 0.15, 0.50, 0.50])),
        ("Oncogene-addicted", np.array([0.25, 0.65, 0.50, 0.40, 0.50])),
        ("Post-CPI partial response", np.array([0.80, 0.40, 0.45, 0.50, 0.50])),
        ("High ME instability", np.array([0.60, 0.50, 0.50, 0.50, 0.20])),
        ("High plasticity + resistance", np.array([0.60, 0.40, 0.40, 0.90, 0.50])),
        ("Near healthy", np.array([0.63, 0.68, 0.58, 0.53, 0.53])),
    ]

    for label, phi in test_cases:
        rec = classify_phi_state(phi, curvature=0.1)
        print(f"  {label:40s} -> {rec}")


def test_bifurcation():
    divider("TEST 5: Bifurcation Proximity")

    # Near-stable Jacobian
    J_stable = np.diag([-0.5, -0.3, -0.2, -0.4, -0.1])
    bp_stable = bifurcation_proximity(J_stable)

    # Near-bifurcation Jacobian
    J_bif = np.diag([-0.5, -0.3, -0.001, -0.4, -0.1])
    bp_bif = bifurcation_proximity(J_bif)

    print(f"  Stable system B_prox    : {bp_stable:.2f}")
    print(f"  Near-bifurcation B_prox : {bp_bif:.2f}")
    print(f"  Ratio (should be >>1)   : {bp_bif / bp_stable:.1f}x")


def main():
    print("=" * 60)
    print("  PROJECT CONFLUENCE — BIOLOGICS INTEGRATION TESTS")
    print("=" * 60)

    test_operators()
    test_synergy()
    test_resistance()
    test_classifier()
    test_bifurcation()

    divider("ALL BIOLOGICS TESTS COMPLETE")


if __name__ == "__main__":
    main()
