"""
Geometric Pathway Integration Test — Project Confluence
========================================================

End-to-end validation of the three geometric calibration modules:
  1. FreidlinWentzellOptimizer  (Minimum Action Path)
  2. FisherManifoldAnalyzer     (Stiff / Sloppy decomposition)
  3. NetworkCurvatureAnalyzer   (Forman-Ricci bottlenecks)
  4. Convergent Target Ranking  (integrated in TherapeuticProtocolOptimizer)

Run:
    python scripts/test_pathways.py
"""

import sys
import os
import time
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ode_system import (
    ComplexAttractorODE, TNBCParams, ExtendedParams,
    GlioblastomaParams, STATE_NAMES, METABOLITE_NAMES, TNBCODESystem,
)
from models.geometric_pathways import FreidlinWentzellOptimizer
from models.fisher_geometry import FisherManifoldAnalyzer
from models.network_curvature import NetworkCurvatureAnalyzer
from models.geometric_optimization import TherapeuticProtocolOptimizer


def divider(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════
# TEST 1: Minimum Action Pathway (String Method)
# ══════════════════════════════════════════════════════════════
def test_map():
    divider("TEST 1: Minimum Action Pathway (MAP)")

    # Set up disease and healthy ODE systems
    healthy_sys = ComplexAttractorODE(params=ExtendedParams())
    tnbc_sys = ComplexAttractorODE(params=TNBCParams())

    # Compute attractors
    print("Computing healthy attractor (300-day settle)...")
    opt = FreidlinWentzellOptimizer(tnbc_sys, dt=0.2)
    z_healthy = opt.get_attractor("healthy")
    print(f"  z_healthy[:5] = {z_healthy[:5].round(3)}")

    # Compute TNBC attractor using the TNBC system
    print("Computing TNBC attractor...")
    res_tnbc = tnbc_sys.solve(t_span=(0, 300), dt_eval=1.0)
    z_tnbc = res_tnbc["z"][:, -1]
    print(f"  z_tnbc[:5]    = {z_tnbc[:5].round(3)}")

    # Distance between attractors
    dist = np.linalg.norm(z_tnbc - z_healthy)
    print(f"  Euclidean distance = {dist:.4f}")

    # --- Subspace MAP (lazy: only metabolic core variables 0-4) ---
    print("\nRunning String Method (subspace: Glucose, Lactate, Pyruvate, ATP, NADH)...")
    t0 = time.perf_counter()
    path, action, history = opt.compute_minimum_action_path(
        z_tnbc, z_healthy,
        n_images=30, max_iter=80, tau=0.003,
        active_indices=[0, 1, 2, 3, 4],
    )
    elapsed = time.perf_counter() - t0
    print(f"  Action (total cost) = {action:.4f}")
    print(f"  Converged in {len(history)} iterations ({elapsed:.1f}s)")

    # Energy profile
    energy = opt.compute_energy_profile(path)
    print(f"  Max quasi-potential = {energy.max():.4f} at image {np.argmax(energy)}")

    # Saddle point
    saddle_idx, z_saddle, saddle_energy = opt.get_saddle_point(path)
    print(f"  Saddle point: image {saddle_idx}, energy = {saddle_energy:.4f}")
    print(f"  Saddle state[:5] = {z_saddle[:5].round(3)}")

    # Realignment targets
    targets = opt.get_realignment_targets(path, state_names=STATE_NAMES)
    print("\n  Top 5 realignment targets (by displacement along MAP):")
    for t in targets[:5]:
        print(f"    {t['name']:20s}  displacement = {t['displacement']:.4f}")

    return path, action, targets


# ══════════════════════════════════════════════════════════════
# TEST 2: Fisher Information Geometry
# ══════════════════════════════════════════════════════════════
def test_fisher():
    divider("TEST 2: Fisher Information Geometry")

    tnbc_sys = ComplexAttractorODE(params=TNBCParams())
    fisher = FisherManifoldAnalyzer(
        tnbc_sys, TNBCParams(),
        t_span=(0, 20), dt=2.0,
        observable_indices=[0, 1, 3, 9, 10],  # Glucose, Lactate, ATP, ROS, I_eff
    )

    print(f"Parameter count: {fisher.dim_p}")
    print("Computing FIM (threaded, short horizon for test speed)...")

    t0 = time.perf_counter()
    fim = fisher.compute_fim(perturbation=1e-3, max_workers=4)
    elapsed = time.perf_counter() - t0
    print(f"  FIM shape: {fim.shape}")
    print(f"  Computed in {elapsed:.1f}s")
    print(f"  FIM condition number: {np.linalg.cond(fim):.2e}")

    # Stiff / sloppy decomposition
    analysis = fisher.identify_stiff_sloppy(fim)
    print(fisher.generate_report(analysis))

    return analysis


# ══════════════════════════════════════════════════════════════
# TEST 3: Network Curvature (Forman-Ricci)
# ══════════════════════════════════════════════════════════════
def test_curvature():
    divider("TEST 3: Network Curvature (Forman-Ricci)")

    # Build graphs for healthy and TNBC generator matrices
    net = NetworkCurvatureAnalyzer()

    A_healthy = TNBCODESystem.healthy_generator()
    A_tnbc = TNBCODESystem.tnbc_generator()

    g_healthy = net.build_graph(A_healthy)
    g_tnbc = net.build_graph(A_tnbc)

    # Compute curvatures
    g_healthy = net.compute_forman_ricci(g_healthy)
    g_tnbc = net.compute_forman_ricci(g_tnbc)

    print("HEALTHY NETWORK:")
    print(net.generate_report(g_healthy, top_k=3))

    print("\nTNBC NETWORK:")
    print(net.generate_report(g_tnbc, top_k=5))

    # Curvature shift
    shifts = net.curvature_difference(g_tnbc, g_healthy)
    print("\nLargest curvature shifts (TNBC vs Healthy):")
    for s in shifts[:5]:
        print(
            f"  {s['source_name']:10s} → {s['target_name']:10s}  "
            f"shift = {s['shift']:+.3f}  ({s['interpretation']})"
        )

    bottlenecks = net.identify_bottlenecks(g_tnbc, top_k=5)
    return bottlenecks


# ══════════════════════════════════════════════════════════════
# TEST 4: Convergent Target Ranking
# ══════════════════════════════════════════════════════════════
def test_convergent_ranking(map_targets, fim_analysis, ricci_bottlenecks):
    divider("TEST 4: Convergent Target Ranking")

    protocol_opt = TherapeuticProtocolOptimizer(n_metabolites=15)

    # Adapt formats for the ranking function
    map_input = [(t["index"], t["displacement"]) for t in map_targets]
    fim_input = fim_analysis["stiff"]
    ricci_input = ricci_bottlenecks

    ranking = protocol_opt.convergent_target_ranking(map_input, fim_input, ricci_input)

    print("UNIFIED DRUG TARGET PRIORITY LIST:")
    print("-" * 45)
    for i, entry in enumerate(ranking[:10]):
        print(f"  {i+1:2d}. {entry['target']:30s}  score = {entry['score']:.4f}")
    print("-" * 45)

    return ranking


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  PROJECT CONFLUENCE — GEOMETRIC CALIBRATION TESTS")
    print("=" * 60)

    # Run all tests
    path, action, map_targets = test_map()
    fim_analysis = test_fisher()
    ricci_bottlenecks = test_curvature()
    ranking = test_convergent_ranking(map_targets, fim_analysis, ricci_bottlenecks)

    divider("ALL TESTS COMPLETE")
    print(f"  MAP action (TNBC → Healthy)    : {action:.4f}")
    print(f"  FIM condition number            : {fim_analysis['condition_number']:.2e}")
    print(f"  #1 convergent target            : {ranking[0]['target']}")
    print(f"  Total targets identified        : {len(ranking)}")


if __name__ == "__main__":
    main()
