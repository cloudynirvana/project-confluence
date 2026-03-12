# -*- coding: utf-8 -*-
"""
Disease Attractor Proof of Concept (POC)
========================================
Validating multi-disease dynamics (9 models)
in Project Confluence.
"""

import sys
import os
import time
import io

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ode_system import (
    ComplexAttractorODE, ExtendedParams, TNBCParams,
    AlzheimersParams, ParkinsonsParams, DiabetesParams, NephroblastomaParams,
    ALSParams, LupusParams, GlioblastomaParams,
    TrajectoryAnalyzer,
)
from models.complexity_profiler import ComplexityProfiler

print("=====================================================================")
print(" PROJECT CONFLUENCE - MULTI-DISEASE ATTRACTOR Divergence Test")
print("=====================================================================\n")

profiler = ComplexityProfiler()

# Define the models we want to test
disease_params = {
    "Healthy": ExtendedParams(),
    "TNBC": TNBCParams(),
    "Alzheimers": AlzheimersParams(),
    "Parkinsons": ParkinsonsParams(),
    "Diabetes": DiabetesParams(),
    "Nephroblastoma": NephroblastomaParams(),
    "ALS": ALSParams(),
    "Lupus": LupusParams(),
    "Glioblastoma": GlioblastomaParams(),
}

results = {}

for disease, params in disease_params.items():
    print(f"--- Simulating {disease} Dynamics ---")
    
    # Instantiate the full nonlinear ODE with the specific disease parameters
    ode = ComplexAttractorODE(params=params, use_nonlinear=True, use_immune=True, use_microenv=True)
    z0 = ode.healthy_initial_state()
    
    start = time.perf_counter()
    sol = ode.solve(z0=z0, t_span=(0, 200), dt_eval=0.5)
    t_sim = time.perf_counter() - start
    
    print(f"  Solver: {'SUCCESS' if sol['success'] else 'FAILED'} in {t_sim:.2f}s")
    
    # Profile the resulting trajectory
    phi = profiler.profile(sol["z"], dt=0.5)
    print(f"  |Phi| (magnitude) = {phi.phi_magnitude:.4f}")
    print(f"  Coherence C       = {phi.coherence_C:.4f}")
    print(f"  Archetype         = {phi.archetype} (conf={phi.archetype_confidence:.2f})\n")
    
    results[disease] = phi

# Compare all diseases to Healthy
print("=====================================================================")
print(" DISEASE DETECTION: Vector Distance from Healthy Baseline")
print("=====================================================================")

for disease in disease_params.keys():
    if disease == "Healthy":
        continue
    phi_healthy = results["Healthy"].phi_vector
    phi_disease = results[disease].phi_vector
    dist = sum((p - h)**2 for p, h in zip(phi_disease, phi_healthy))**0.5
    print(f"  Distance (Healthy -> {disease:<15}): {dist:.4f}")

print("\n=====================================================================")
print(" DIFFERENTIAL DIAGNOSIS: Distance between Cancers")
print("=====================================================================")
phi_tnbc = results["TNBC"].phi_vector
phi_nephro = results["Nephroblastoma"].phi_vector
dist_cancer = sum((p - h)**2 for p, h in zip(phi_tnbc, phi_nephro))**0.5
print(f"  Distance (TNBC -> Nephroblastoma)  : {dist_cancer:.4f}")

# =====================================================================
# PER-DIMENSION Phi DIVERGENCE (TNBC vs Nephroblastoma)
# =====================================================================
print("\n=====================================================================")
print(" PER-DIMENSION Phi DIVERGENCE: TNBC vs Nephroblastoma")
print("=====================================================================")

dims = ["Phi_temporal", "Phi_spatial", "Phi_functional", "Phi_informational", "Phi_coupling"]
phi_healthy_vec = results["Healthy"].phi_vector

print(f"\n  {'Dimension':<20} {'Healthy':>10} {'TNBC':>10} {'Nephro':>10} {'D(TNBC-Nephro)':>16}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*16}")
for i, dim in enumerate(dims):
    h = phi_healthy_vec[i]
    t = phi_tnbc[i]
    n = phi_nephro[i]
    delta = abs(t - n)
    print(f"  {dim:<20} {h:>10.4f} {t:>10.4f} {n:>10.4f} {delta:>16.4f}")

# =====================================================================
# THERAPEUTIC SIMULATION: IGF2R Inhibition on Nephroblastoma
# =====================================================================
print("\n=====================================================================")
print(" THERAPEUTIC SIMULATION: IGF2R Antibody on Nephroblastoma")
print("=====================================================================")
print("  Intervention: Reducing IGF2_signaling from 0.75 -> 0.30 (60% inhibition)")
print("  Rationale: Simulates IGF2R antibody (e.g., cixutumumab) effect\n")

import numpy as np

# Create treated nephroblastoma params
nephro_treated_params = NephroblastomaParams()
nephro_treated_params.IGF2_signaling = 0.30  # 60% reduction

ode_treated = ComplexAttractorODE(
    params=nephro_treated_params,
    use_nonlinear=True, use_immune=True, use_microenv=True
)
z0_treated = ode_treated.healthy_initial_state()

start = time.perf_counter()
sol_treated = ode_treated.solve(z0=z0_treated, t_span=(0, 200), dt_eval=0.5)
t_treated_sim = time.perf_counter() - start

print(f"  Solver: {'SUCCESS' if sol_treated['success'] else 'FAILED'} in {t_treated_sim:.2f}s")

phi_treated = profiler.profile(sol_treated["z"], dt=0.5)
print(f"  |Phi| (magnitude) = {phi_treated.phi_magnitude:.4f}")
print(f"  Coherence C       = {phi_treated.coherence_C:.4f}")
print(f"  Archetype         = {phi_treated.archetype} (conf={phi_treated.archetype_confidence:.2f})\n")

# Measure Phi-shift toward healthy attractor
phi_nephro_vec = np.array(results["Nephroblastoma"].phi_vector)
phi_healthy_np = np.array(results["Healthy"].phi_vector)
phi_treated_vec = np.array(phi_treated.phi_vector)

pre_treatment_dist = np.linalg.norm(phi_nephro_vec - phi_healthy_np)
post_treatment_dist = np.linalg.norm(phi_treated_vec - phi_healthy_np)

if pre_treatment_dist > 1e-6:
    restoration_pct = (1.0 - post_treatment_dist / pre_treatment_dist) * 100.0
else:
    restoration_pct = 100.0

print("  +-------------------------------------------------------------+")
print(f"  |  PRE-TREATMENT  Phi-distance to healthy:  {pre_treatment_dist:.4f}            |")
print(f"  |  POST-TREATMENT Phi-distance to healthy:  {post_treatment_dist:.4f}            |")
print(f"  |  COMPLEXITY RESTORATION:                {restoration_pct:>6.1f}%            |")
print("  +-------------------------------------------------------------+")

# Per-dimension shift
print(f"\n  {'Dimension':<20} {'Untreated':>10} {'Treated':>10} {'Healthy':>10} {'Shifted?':>10}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for i, dim in enumerate(dims):
    u = phi_nephro_vec[i]
    t = phi_treated_vec[i]
    h = phi_healthy_np[i]
    dist_before = abs(u - h)
    dist_after = abs(t - h)
    shifted = "[+] YES" if dist_after < dist_before else "[-] NO"
    print(f"  {dim:<20} {u:>10.4f} {t:>10.4f} {h:>10.4f} {shifted:>10}")

# =====================================================================
# COMBINATION THERAPY: IGF2R + WT1 mRNA Restoration
# =====================================================================
print("\n=====================================================================")
print(" COMBINATION THERAPY: IGF2R Inhibition + WT1 mRNA Restoration")
print("=====================================================================")
print("  Adding: WT1_activity restored from 0.20 -> 0.55 (mRNA therapy)\n")

nephro_combo_params = NephroblastomaParams()
nephro_combo_params.IGF2_signaling = 0.30  # IGF2R antibody
nephro_combo_params.WT1_activity = 0.55    # WT1 mRNA restoration

ode_combo = ComplexAttractorODE(
    params=nephro_combo_params,
    use_nonlinear=True, use_immune=True, use_microenv=True
)
z0_combo = ode_combo.healthy_initial_state()

start = time.perf_counter()
sol_combo = ode_combo.solve(z0=z0_combo, t_span=(0, 200), dt_eval=0.5)
t_combo_sim = time.perf_counter() - start

print(f"  Solver: {'SUCCESS' if sol_combo['success'] else 'FAILED'} in {t_combo_sim:.2f}s")

phi_combo = profiler.profile(sol_combo["z"], dt=0.5)
phi_combo_vec = np.array(phi_combo.phi_vector)

combo_dist = np.linalg.norm(phi_combo_vec - phi_healthy_np)
combo_restoration_pct = (1.0 - combo_dist / pre_treatment_dist) * 100.0

print(f"\n  +-------------------------------------------------------------+")
print(f"  |  UNTREATED       Phi-distance to healthy:  {pre_treatment_dist:.4f}            |")
print(f"  |  IGF2R ONLY      Phi-distance to healthy:  {post_treatment_dist:.4f}            |")
print(f"  |  IGF2R + WT1     Phi-distance to healthy:  {combo_dist:.4f}            |")
print(f"  |                                                             |")
print(f"  |  MONO RESTORATION:  {restoration_pct:>6.1f}%                              |")
print(f"  |  COMBO RESTORATION: {combo_restoration_pct:>6.1f}%                              |")
synergy = combo_restoration_pct - restoration_pct
print(f"  |  SYNERGY GAIN:      {synergy:>6.1f}%                              |")
print(f"  +-------------------------------------------------------------+")

print(f"\n{'=' * 70}")
print(f"  Project Confluence v1.0 -- Multi-Disease Attractor Validation")
print(f"  Diseases tested: {len(disease_params)}")
print(f"  Therapeutic simulations: 2 (mono + combo)")
print(f"{'=' * 70}")

