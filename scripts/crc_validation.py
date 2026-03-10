# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
End-to-End Validation — Colorectal Cancer (CRC)
===============================================

Full pipeline test for Colorectal Cancer (CRC) on Project Confluence:
  1. Simulate HEALTHY dynamics (15D complex attractor)
  2. Simulate CANCER dynamics (CRC pathological attractor)
  3. Profile BOTH with ComplexityProfiler -> compare Phi vectors
  4. Apply therapeutic protocol -> measure Phi improvement
  5. Report results with ML archetype classification

CRC Pathophysiology modeled:
  - Strong Warburg effect (elevated glycolysis/glucose uptake)
  - Gut microbiome dysbiosis & mucosal permeability (stromal barrier alterations)
  - Severe immune suppression in the tumor microenvironment
  - High lactate & hypoxia-induced angiogenesis
"""

import os
import time
import argparse
import warnings
import numpy as np

# Suppress numpy runtime warnings (divide-by-zero in correlation)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Parse args early
parser = argparse.ArgumentParser(description="Project Confluence E2E Validation - CRC")
parser.add_argument("--verbose", "-v", action="store_true", help="Show full per-step details")
args, _ = parser.parse_known_args()
VERBOSE = args.verbose

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ode_system import (
    ComplexAttractorODE, ExtendedParams,
    TrajectoryAnalyzer, METABOLITE_NAMES, STATE_NAMES
)
from models.complexity_profiler import ComplexityProfiler

print("=" * 70)
print("  PROJECT CONFLUENCE — CRC DISEASE SIMULATION & CURE TEST")
print("=" * 70)

profiler = ComplexityProfiler()

# ─────────────────────────────────────────────────────────────────────────
# STEP 1: HEALTHY DYNAMICS — The Target Attractor
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STEP 1: Simulating HEALTHY dynamics (15D complex attractor)")
print("─" * 70)

ode_healthy = ComplexAttractorODE(use_nonlinear=True, use_immune=True, use_microenv=True)
z0_healthy = ode_healthy.healthy_initial_state()

start = time.perf_counter()
result_healthy = ode_healthy.solve(z0=z0_healthy, t_span=(0, 200), dt_eval=0.5)
t_healthy_sim = time.perf_counter() - start

print(f"  Solver: {'SUCCESS' if result_healthy['success'] else 'FAILED'}")
print(f"  Runtime: {t_healthy_sim:.2f}s | Timepoints: {result_healthy['n_timepoints']}")

# Profile healthy trajectory
phi_healthy = profiler.profile(result_healthy["z"], dt=0.5)
print(f"\n  ── Healthy Φ Vector ──")
print(f"  |Φ| (magnitude)  = {phi_healthy.phi_magnitude:.4f}")
print(f"  C (coherence)    = {phi_healthy.coherence_C:.4f}")
print(f"  Archetype        = {phi_healthy.archetype}")

# ─────────────────────────────────────────────────────────────────────────
# STEP 2: CANCER DYNAMICS — The CRC Pathological Attractor
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STEP 2: Simulating COLORECTAL CANCER dynamics (pathological attractor)")
print("─" * 70)

# Create CRC-perturbed initial conditions
z0_crc = z0_healthy.copy()
z0_crc[0] *= 2.0   # Elevated Glucose uptake
z0_crc[1] *= 3.0   # High lactate (acidic TME)
z0_crc[9] *= 2.5   # ROS overproduction (inflammation)
z0_crc[10] *= 0.4  # Immune evasion (low effectors)
z0_crc[11] *= 2.5  # High Tregs (immunosuppression prominent in CRC)
z0_crc[12] *= 2.5  # Immune exhaustion
z0_crc[13] *= 3.0  # Desmoplastic stroma (fibrosis/barrier)
z0_crc[14] *= 3.0  # High angiogenesis (VEGF drive)

# Use CRC-perturbed parameters
crc_params = ExtendedParams(
    glucose_uptake=-1.00,       # 2x healthy
    glycolysis_flux=0.72,       # 1.8x healthy
    lactate_clearance=-0.40,    # 0.5x healthy clearance
    glutamine_utilization=-0.60,# 1.5x healthy
    ros_clearance=-0.495,       # 0.55x healthy
    r_prime=0.04,               # Weakened immune priming
    k_exhaust=0.15,             # Faster exhaustion
    r_fibrosis=0.08,            # Accelerated fibrosis (stromal barrier)
    r_angio=0.08,               # High angiogenesis drive
)

ode_crc = ComplexAttractorODE(
    params=crc_params,
    use_nonlinear=True, use_immune=True, use_microenv=True
)

start = time.perf_counter()
result_crc = ode_crc.solve(z0=z0_crc, t_span=(0, 200), dt_eval=0.5)
t_crc_sim = time.perf_counter() - start

print(f"  Solver: {'SUCCESS' if result_crc['success'] else 'FAILED'}")
print(f"  Runtime: {t_crc_sim:.2f}s | Timepoints: {result_crc['n_timepoints']}")

# Profile CRC trajectory
phi_crc = profiler.profile(result_crc["z"], dt=0.5)
print(f"\n  ── CRC Cancer Φ Vector ──")
print(f"  |Φ| (magnitude)  = {phi_crc.phi_magnitude:.4f}")
print(f"  C (coherence)    = {phi_crc.coherence_C:.4f}")
print(f"  Archetype        = {phi_crc.archetype}")

# ─────────────────────────────────────────────────────────────────────────
# STEP 3: COMPARE — Can We DETECT Disease?
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STEP 3: DISEASE DETECTION — Healthy vs. CRC Φ Comparison")
print("─" * 70)

phi_diff = phi_crc.phi_vector - phi_healthy.phi_vector
dims = ["Φ_temporal", "Φ_spatial", "Φ_functional", "Φ_informational", "Φ_coupling"]

print(f"\n  {'Dimension':<20} {'Healthy':>10} {'Cancer':>10} {'Delta':>10} {'Direction':>12}")
print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*12}")
for i, name in enumerate(dims):
    h_val = phi_healthy.phi_vector[i]
    c_val = phi_crc.phi_vector[i]
    delta = c_val - h_val
    direction = "^ Elevated" if delta > 0.05 else ("v Depleted" if delta < -0.05 else "~ Similar")
    print(f"  {name:<20} {h_val:>10.4f} {c_val:>10.4f} {delta:>+10.4f} {direction:>12}")

phi_distance = np.linalg.norm(phi_diff)
detection_success = phi_distance > 0.1
print(f"\n  Phi-space distance: {phi_distance:.4f}")
print(f"  [PASS] DISEASE DETECTED" if detection_success else "  [FAIL] DETECTION FAILED")

# ─────────────────────────────────────────────────────────────────────────
# STEP 4: CURE TEST — Apply Therapeutic Protocol & Measure Restoration
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("STEP 4: CURE TEST — Simulating CRC therapeutic intervention")
print("─" * 70)

# Simulate treatment: FOLFIRI + Bevacizumab (Anti-VEGF) + Anti-PD1
treated_params = ExtendedParams(
    # Metabolic — slightly restored by chemo
    glucose_uptake=-0.80,         
    glycolysis_flux=0.60,         
    lactate_clearance=-0.50,      
    ros_clearance=-0.60,          
    # Immune — restored by Anti-PD1
    r_prime=0.10,                 # Anti-PD1 restores priming
    k_exhaust=0.07,               # Reduced exhaustion rate
    # Microenvironment — Bevacizumab directly targets angiogenesis
    r_angio=0.01,                 # Strong anti-VEGF effect
    k_vascular_prune=0.08,        # Accelerated pruning
    r_fibrosis=0.04,              # Reduced but not eliminated
)

ode_treated = ComplexAttractorODE(
    params=treated_params,
    use_nonlinear=True, use_immune=True, use_microenv=True
)

# Start from the cancer state (day 200 of CRC sim)
z0_treated = result_crc["z"][:, -1].copy()
print(f"  Starting treatment from CRC end-state")
print(f"  Protocol: FOLFIRI + Bevacizumab + Anti-PD1 (Trimodal target)")

start = time.perf_counter()
result_treated = ode_treated.solve(z0=z0_treated, t_span=(0, 200), dt_eval=0.5)
t_treated_sim = time.perf_counter() - start

print(f"  Solver: {'SUCCESS' if result_treated['success'] else 'FAILED'}")

# Profile treated trajectory
phi_treated = profiler.profile(result_treated["z"], dt=0.5)

# ─────────────────────────────────────────────────────────────────────────
# STEP 5: FINAL SCORECARD
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  FINAL SCORECARD — Complexity Restoration Assessment")
print("═" * 70)

print(f"\n  {'Metric':<25} {'Healthy':>10} {'CRC':>10} {'Treated':>10} {'Restored?':>12}")
print(f"  {'═'*25} {'═'*10} {'═'*10} {'═'*10} {'═'*12}")

for i, name in enumerate(dims):
    h, c, t = phi_healthy.phi_vector[i], phi_crc.phi_vector[i], phi_treated.phi_vector[i]
    dist_before = abs(c - h)
    dist_after = abs(t - h)
    if dist_before < 0.01:
        restored = "n/a"
    elif dist_after < dist_before:
        pct = (1 - dist_after / dist_before) * 100
        restored = f"[OK] {pct:.0f}%"
    else:
        restored = "[X] Worse"
    print(f"  {name:<25} {h:>10.4f} {c:>10.4f} {t:>10.4f} {restored:>12}")

dist_cancer_to_healthy = np.linalg.norm(phi_crc.phi_vector - phi_healthy.phi_vector)
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
else:
    print(f"  |  [WARN] PARTIAL RESTORATION -- refine protocol       |")
print(f"  +-------------------------------------------------------+")
