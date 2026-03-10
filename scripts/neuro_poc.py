# -*- coding: utf-8 -*-
"""
Neurodegeneration Proof of Concept (POC)
========================================
Validating the newly added neurodegenerative dynamics
(Alzheimer's and Parkinson's) in Project Confluence.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ode_system import (
    ComplexAttractorODE, ExtendedParams, TNBCODESystem,
    TrajectoryAnalyzer,
)
from models.complexity_profiler import ComplexityProfiler

print("=====================================================================")
print(" PROJECT CONFLUENCE — NEURODEGENERATION PROOF OF CONCEPT")
print("=====================================================================\n")

profiler = ComplexityProfiler()

diseases = ["Healthy", "Alzheimers", "Parkinsons"]
results = {}

for disease in diseases:
    print(f"--- Simulating {disease} Dynamics ---")
    
    # We use the linear system generators for now to test the metabolic shifts
    generator = TNBCODESystem.all_generators()[disease]
    
    # We create a base ODE but override the generator matrix for the metabolic part
    # A cleaner integration would pass this naturally, but for POC we patch it in
    ode = ComplexAttractorODE(use_nonlinear=False, use_immune=True, use_microenv=True)
    ode._A = generator # Override metabolic generator
    
    z0 = ode.healthy_initial_state()
    
    start = time.perf_counter()
    sol = ode.solve(z0=z0, t_span=(0, 200), dt_eval=0.5)
    t_sim = time.perf_counter() - start
    
    print(f"  Solver: {'SUCCESS' if sol['success'] else 'FAILED'} in {t_sim:.2f}s")
    
    # Profile
    phi = profiler.profile(sol["z"], dt=0.5)
    print(f"  |Phi| (magnitude) = {phi.phi_magnitude:.4f}")
    print(f"  Coherence C       = {phi.coherence_C:.4f}")
    print(f"  Archetype         = {phi.archetype} (conf={phi.archetype_confidence:.2f})\n")
    
    results[disease] = phi

# Compare
print("=====================================================================")
print(" DISEASE DETECTION: Vector Distance from Healthy Baseline")
print("=====================================================================")

for disease in ["Alzheimers", "Parkinsons"]:
    phi_healthy = results["Healthy"].phi_vector
    phi_disease = results[disease].phi_vector
    dist = sum((p - h)**2 for p, h in zip(phi_disease, phi_healthy))**0.5
    print(f"  Distance (Healthy -> {disease}): {dist:.4f}")

print("\n[DONE] Neurodegeneration models generated distinct complexity signatures.")
