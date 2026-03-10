# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
End-to-End Validation — Project Confluence
============================================

Full pipeline test:
  1. Simulate HEALTHY dynamics (15D complex attractor)
  2. Simulate CANCER dynamics (TNBC pathological attractor)
  3. Profile BOTH with ComplexityProfiler -> compare Phi vectors
  4. Apply therapeutic protocol -> measure Phi improvement
  5. Report results with ML archetype classification

This script answers: "Can we detect disease, quantify it, and restore complexity?"
"""

import sys
import os
import json
import time
import argparse
import warnings
import numpy as np

# Suppress numpy runtime warnings (divide-by-zero in correlation)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Parse args early
parser = argparse.ArgumentParser(description="Project Confluence E2E Validation")
parser.add_argument("--verbose", "-v", action="store_true", help="Show full per-step details")
args, _ = parser.parse_known_args()
VERBOSE = args.verbose

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ode_system import (
    ComplexAttractorODE, ExtendedParams, TNBCODESystem,
    simulate_trajectory, METABOLITE_NAMES, STATE_NAMES,
    TrajectoryAnalyzer,
)
from models.complexity_profiler import (
    ComplexityProfiler, classify_archetype, PhiProfile,
    mse_mean, correlation_dimension, largest_lyapunov_exponent,
    power_spectral_slope, coherence_metric,
)

print("=" * 70)
print("  PROJECT CONFLUENCE — END-TO-END DISEASE SIMULATION & CURE TEST")
print("=" * 70)

profiler = ComplexityProfiler()

# ─────────────────────────────────────────────────────────────────────────
# STEP 1: HEALTHY DYNAMICS — The Target Attractor
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STEP 1: Simulating HEALTHY dynamics (15D complex attractor)")
print("─" * 70)

ode_healthy = ComplexAttractorODE(
    use_nonlinear=True, use_immune=True, use_microenv=True
)
z0_healthy = ode_healthy.healthy_initial_state()
print(f"  Initial state (15D): {z0_healthy.round(2)}")

start = time.perf_counter()
result_healthy = ode_healthy.solve(z0=z0_healthy, t_span=(0, 200), dt_eval=0.5)
t_healthy_sim = time.perf_counter() - start

print(f"  Solver: {'SUCCESS' if result_healthy['success'] else 'FAILED'}")
print(f"  Runtime: {t_healthy_sim:.2f}s | Timepoints: {result_healthy['n_timepoints']}")
print(f"  Bounded: {TrajectoryAnalyzer.is_bounded(result_healthy['z'])}")

# Profile healthy trajectory
phi_healthy = profiler.profile(result_healthy["z"], dt=0.5)
print(f"\n  ── Healthy Φ Vector ──")
print(f"  Φ_temporal      = {phi_healthy.Phi_temporal:.4f}")
print(f"  Φ_spatial        = {phi_healthy.Phi_spatial:.4f}")
print(f"  Φ_functional     = {phi_healthy.Phi_functional:.4f}")
print(f"  Φ_informational  = {phi_healthy.Phi_informational:.4f}")
print(f"  Φ_coupling       = {phi_healthy.Phi_coupling:.4f}")
print(f"  |Φ| (magnitude)  = {phi_healthy.phi_magnitude:.4f}")
print(f"  C (coherence)    = {phi_healthy.coherence_C:.4f}")
print(f"  Archetype: {phi_healthy.archetype} (conf={phi_healthy.archetype_confidence:.2f})")

# ─────────────────────────────────────────────────────────────────────────
# STEP 2: CANCER DYNAMICS — The Pathological Attractor
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STEP 2: Simulating TNBC CANCER dynamics (pathological attractor)")
print("─" * 70)

# Create TNBC-perturbed initial conditions (Warburg phenotype)
z0_cancer = z0_healthy.copy()
z0_cancer[0] *= 2.5   # High glucose uptake (Warburg)
z0_cancer[1] *= 4.0   # High lactate (acid environment)
z0_cancer[2] *= 0.5   # Pyruvate diversion
z0_cancer[3] *= 0.5   # ATP crisis
z0_cancer[9] *= 3.0   # ROS overproduction
z0_cancer[10] *= 0.3  # Immune evasion (low effectors)
z0_cancer[11] *= 2.0  # High Tregs (immunosuppression)
z0_cancer[12] *= 3.0  # Immune exhaustion
z0_cancer[13] *= 2.5  # Stromal barrier (fibrosis)
z0_cancer[14] *= 0.4  # Poor vasculature

# Use cancer-perturbed parameters
cancer_params = ExtendedParams(
    glucose_uptake=-1.25,       # 2.5x healthy
    glycolysis_flux=0.80,       # 2x healthy
    lactate_clearance=-0.32,    # Reduced clearance
    glutamine_utilization=-0.80,# Glutamine addiction
    ros_clearance=-0.45,        # Compromised ROS defense
    r_prime=0.03,               # Weakened immune priming
    k_exhaust=0.12,             # Faster exhaustion
    r_fibrosis=0.06,            # Accelerated fibrosis
)

ode_cancer = ComplexAttractorODE(
    params=cancer_params,
    use_nonlinear=True, use_immune=True, use_microenv=True
)

start = time.perf_counter()
result_cancer = ode_cancer.solve(z0=z0_cancer, t_span=(0, 200), dt_eval=0.5)
t_cancer_sim = time.perf_counter() - start

print(f"  Solver: {'SUCCESS' if result_cancer['success'] else 'FAILED'}")
print(f"  Runtime: {t_cancer_sim:.2f}s | Timepoints: {result_cancer['n_timepoints']}")
print(f"  Bounded: {TrajectoryAnalyzer.is_bounded(result_cancer['z'])}")

# Profile cancer trajectory
phi_cancer = profiler.profile(result_cancer["z"], dt=0.5)
print(f"\n  ── TNBC Cancer Φ Vector ──")
print(f"  Φ_temporal      = {phi_cancer.Phi_temporal:.4f}")
print(f"  Φ_spatial        = {phi_cancer.Phi_spatial:.4f}")
print(f"  Φ_functional     = {phi_cancer.Phi_functional:.4f}")
print(f"  Φ_informational  = {phi_cancer.Phi_informational:.4f}")
print(f"  Φ_coupling       = {phi_cancer.Phi_coupling:.4f}")
print(f"  |Φ| (magnitude)  = {phi_cancer.phi_magnitude:.4f}")
print(f"  C (coherence)    = {phi_cancer.coherence_C:.4f}")
print(f"  Archetype: {phi_cancer.archetype} (conf={phi_cancer.archetype_confidence:.2f})")

# ─────────────────────────────────────────────────────────────────────────
# STEP 3: COMPARE — Can We DETECT Disease?
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STEP 3: DISEASE DETECTION — Healthy vs. Cancer Φ Comparison")
print("─" * 70)

phi_diff = phi_cancer.phi_vector - phi_healthy.phi_vector
dims = ["Φ_temporal", "Φ_spatial", "Φ_functional", "Φ_informational", "Φ_coupling"]

print(f"\n  {'Dimension':<20} {'Healthy':>10} {'Cancer':>10} {'Delta':>10} {'Direction':>12}")
print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*12}")
for i, name in enumerate(dims):
    h_val = phi_healthy.phi_vector[i]
    c_val = phi_cancer.phi_vector[i]
    delta = c_val - h_val
    direction = "^ Elevated" if delta > 0.05 else ("v Depleted" if delta < -0.05 else "~ Similar")
    print(f"  {name:<20} {h_val:>10.4f} {c_val:>10.4f} {delta:>+10.4f} {direction:>12}")

print(f"\n  |Phi| magnitude: Healthy={phi_healthy.phi_magnitude:.4f}  Cancer={phi_cancer.phi_magnitude:.4f}")
print(f"  Coherence C:   Healthy={phi_healthy.coherence_C:.4f}  Cancer={phi_cancer.coherence_C:.4f}")
print(f"  Archetype shift: {phi_healthy.archetype} -> {phi_cancer.archetype}")

# Disease detection test
phi_distance = np.linalg.norm(phi_diff)
detection_success = phi_distance > 0.1
print(f"\n  Phi-space distance: {phi_distance:.4f}")
print(f"  [PASS] DISEASE DETECTED" if detection_success else "  [FAIL] DETECTION FAILED")

# ─────────────────────────────────────────────────────────────────────────
# STEP 4: CURE TEST — Apply Therapeutic Protocol & Measure Restoration
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STEP 4: CURE TEST — Simulating therapeutic intervention")
print("─" * 70)

# Simulate treatment: shift cancer parameters BACK toward healthy
# This represents what the RADO engine would prescribe:
#   - DCA (PDK inhibitor): restores OXPHOS -> improves pyruvate_to_atp
#   - Metformin: AMPK activation -> reduces glucose uptake
#   - Anti-PD1: immune restoration -> better r_prime, lower k_exhaust

treated_params = ExtendedParams(
    # Metabolic — partially restored by DCA + Metformin
    glucose_uptake=-0.75,         # DCA + Metformin reduce from -1.25 toward -0.50
    glycolysis_flux=0.55,         # 2-DG partially blocks glycolysis
    lactate_clearance=-0.55,      # Improving but not fully restored
    glutamine_utilization=-0.55,  # CB-839 partially blocks glutaminolysis
    ros_clearance=-0.70,          # Vitamin C + GPX4 restoration
    # Immune — restored by Anti-PD1
    r_prime=0.06,                 # Anti-PD1 restores priming
    k_exhaust=0.07,               # Reduced exhaustion rate
    r_rescue=0.05,                # Better rescue
    # Microenvironment — slower impact
    r_fibrosis=0.03,              # Reduced but not eliminated
)

ode_treated = ComplexAttractorODE(
    params=treated_params,
    use_nonlinear=True, use_immune=True, use_microenv=True
)

# Start from the cancer state (day 200 of cancer sim) — NOT healthy ICs
z0_treated = result_cancer["z"][:, -1].copy()  # End state of cancer
print(f"  Starting treatment from cancer end-state")
print(f"  Protocol: DCA + Metformin + Anti-PD1 + CB-839 + Vitamin C")

start = time.perf_counter()
result_treated = ode_treated.solve(z0=z0_treated, t_span=(0, 200), dt_eval=0.5)
t_treated_sim = time.perf_counter() - start

print(f"  Solver: {'SUCCESS' if result_treated['success'] else 'FAILED'}")
print(f"  Runtime: {t_treated_sim:.2f}s")
print(f"  Bounded: {TrajectoryAnalyzer.is_bounded(result_treated['z'])}")

# Profile treated trajectory
phi_treated = profiler.profile(result_treated["z"], dt=0.5)
print(f"\n  ── Post-Treatment Phi Vector ──")
print(f"  Phi_temporal      = {phi_treated.Phi_temporal:.4f}")
print(f"  Phi_spatial        = {phi_treated.Phi_spatial:.4f}")
print(f"  Phi_functional     = {phi_treated.Phi_functional:.4f}")
print(f"  Phi_informational  = {phi_treated.Phi_informational:.4f}")
print(f"  Phi_coupling       = {phi_treated.Phi_coupling:.4f}")
print(f"  |Phi| (magnitude)  = {phi_treated.phi_magnitude:.4f}")
print(f"  C (coherence)    = {phi_treated.coherence_C:.4f}")
print(f"  Archetype: {phi_treated.archetype} (conf={phi_treated.archetype_confidence:.2f})")

# ─────────────────────────────────────────────────────────────────────────
# STEP 5: FINAL SCORECARD
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  FINAL SCORECARD — Complexity Restoration Assessment")
print("═" * 70)

print(f"\n  {'Metric':<25} {'Healthy':>10} {'Cancer':>10} {'Treated':>10} {'Restored?':>12}")
print(f"  {'═'*25} {'═'*10} {'═'*10} {'═'*10} {'═'*12}")

for i, name in enumerate(dims):
    h = phi_healthy.phi_vector[i]
    c = phi_cancer.phi_vector[i]
    t = phi_treated.phi_vector[i]
    # Did treatment move toward healthy?
    distance_before = abs(c - h)
    distance_after = abs(t - h)
    if distance_before < 0.01:
        restored = "n/a"
    elif distance_after < distance_before:
        pct = (1 - distance_after / distance_before) * 100
        restored = f"[OK] {pct:.0f}%"
    else:
        restored = "[X] Worse"
    print(f"  {name:<25} {h:>10.4f} {c:>10.4f} {t:>10.4f} {restored:>12}")

print(f"\n  {'Coherence C':<25} {phi_healthy.coherence_C:>10.4f} {phi_cancer.coherence_C:>10.4f} {phi_treated.coherence_C:>10.4f}")
print(f"  {'|Phi| magnitude':<25} {phi_healthy.phi_magnitude:>10.4f} {phi_cancer.phi_magnitude:>10.4f} {phi_treated.phi_magnitude:>10.4f}")
print(f"\n  Archetype trajectory: {phi_healthy.archetype} -> {phi_cancer.archetype} -> {phi_treated.archetype}")

# Overall restoration score
dist_cancer_to_healthy = np.linalg.norm(phi_cancer.phi_vector - phi_healthy.phi_vector)
dist_treated_to_healthy = np.linalg.norm(phi_treated.phi_vector - phi_healthy.phi_vector)

if dist_cancer_to_healthy > 0.01:
    restoration_pct = (1 - dist_treated_to_healthy / dist_cancer_to_healthy) * 100
else:
    restoration_pct = 100.0

print(f"\n  +-------------------------------------------------------+")
print(f"  |  OVERALL COMPLEXITY RESTORATION: {restoration_pct:>6.1f}%             |")
print(f"  |  Phi-distance: Cancer->Healthy = {dist_cancer_to_healthy:.4f}              |")
print(f"  |  Phi-distance: Treated->Healthy = {dist_treated_to_healthy:.4f}             |")
if restoration_pct > 50:
    print(f"  |  [PASS] THERAPEUTIC HYPOTHESIS SUPPORTED             |")
elif restoration_pct > 0:
    print(f"  |  [WARN] PARTIAL RESTORATION -- refine protocol       |")
else:
    print(f"  |  [FAIL] RESTORATION FAILED -- rethink approach       |")
print(f"  +-------------------------------------------------------+")

# ─────────────────────────────────────────────────────────────────────────
# STEP 6: METABOLIC STATE COMPARISON
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STEP 6: Metabolic State Comparison (Final Timepoint)")
print("─" * 70)

z_h_final = result_healthy["z"][:10, -1]
z_c_final = result_cancer["z"][:10, -1]
z_t_final = result_treated["z"][:10, -1]

print(f"\n  {'Metabolite':<12} {'Healthy':>10} {'Cancer':>10} {'Treated':>10} {'Status':>12}")
print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*12}")
for i, name in enumerate(METABOLITE_NAMES):
    h, c, t = z_h_final[i], z_c_final[i], z_t_final[i]
    dist_before = abs(c - h)
    dist_after = abs(t - h)
    if dist_before < 0.01:
        status = "= Stable"
    elif dist_after < dist_before:
        status = "^ Better"
    else:
        status = "v Worse"
    print(f"  {name:<12} {h:>10.3f} {c:>10.3f} {t:>10.3f} {status:>12}")

# Immune state
print(f"\n  {'Immune Var':<16} {'Healthy':>10} {'Cancer':>10} {'Treated':>10}")
print(f"  {'─'*16} {'─'*10} {'─'*10} {'─'*10}")
for i, name in enumerate(["I_eff", "I_reg", "I_exhaust", "σ_stromal", "ν_vascular"]):
    h = result_healthy["z"][10+i, -1]
    c = result_cancer["z"][10+i, -1]
    t = result_treated["z"][10+i, -1]
    print(f"  {name:<16} {h:>10.4f} {c:>10.4f} {t:>10.4f}")

print(f"\n{'═' * 70}")
print(f"  Total simulation time: {t_healthy_sim + t_cancer_sim + t_treated_sim:.2f}s")
print(f"  Framework: Project Confluence v1.0")
print(f"{'═' * 70}")
