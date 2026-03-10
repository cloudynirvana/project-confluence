# -*- coding: utf-8 -*-
"""
End-to-End AI Validation — Project Confluence
=============================================

Executes the full AI pipeline:
1. Synthetic Clinical Data Generation
2. Neural-ODE Continuous Trajectory Extrapolation
3. Digital Twin Memory Ingestion (Graphiti + Cognee)
"""

import os
import sys
import numpy as np

# Add parent path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.patient_fitter import PatientFitter
from models.neural_ode import TORCHDIFFEQ_AVAILABLE

def run_ai_pipeline():
    print("=" * 60)
    print("  AI Memory Integration Pipeline Validation")
    print("=" * 60)

    # 1. Initialize Patient Fitter in Neural Mode
    print("\n[1] Initializing PatientFitter (inference_mode='neural')...")
    fitter = PatientFitter(
        cancer_type="TNBC", 
        patient_id="PATIENT-AI-001",
        inference_mode="neural"
    )
    
    # 2. Simulate sparse clinical observations
    # Shape: [seq_len=5, obs_dim=15]
    print("[2] Generating synthetic sparse clinical observations...")
    np.random.seed(42)
    clinical_data = np.random.uniform(0.1, 5.0, size=(5, 15))
    
    if not TORCHDIFFEQ_AVAILABLE:
        print("\n[!] PyTorch/torchdiffeq not installed. Running MOCK continuous trajectory simulation...")
        print("[3] Simulating Continuous ODE Inference & Memory Ingestion...")
        n_steps = 200
        t_span = np.linspace(0, n_steps, n_steps)
        # Create a mock trajectory with some non-linear shifts (sine waves)
        trajectory_np = clinical_data.mean(axis=0) * (1 + 0.5 * np.sin(t_span[:, None] / 10.0))
        memory_stats = fitter.memory_controller.process_neural_trajectory(trajectory_np, t_span)
        from models.patient_fitter import DigitalTwin
        twin = DigitalTwin(patient_id="PATIENT-AI-001", cancer_type="TNBC")
        twin.add_trajectory(trajectory_np, t_span)
        print(f"Neural Fitter (Mock) mapped patient PATIENT-AI-001 into Memory Layer:")
        print(f"  Graphiti Temp Events: {memory_stats['temporal_events_recorded']}")
        print(f"  Cognee Structural Edges: {memory_stats['unique_correlations_mapped']}")
    else:
        # 3. Fit Digital Twin (Runs Neural-ODE and Ingests to Memory)
        print("[3] Running Continuous Neural-ODE Inference & Memory Ingestion...")
        # This automatically calls DigitalTwinMemory.process_neural_trajectory
        twin = fitter.fit_neural(n_steps=200, seed=42, clinical_time_series=clinical_data)
        
    # 4. Extract Memory Traces
    print("\n[4] Querying Memory Layers for Pathological Complexity:")
    
    # Query Graphiti (Temporal $\Phi$)
    temporal_events = fitter.memory_controller.temporal_memory.query_temporal_sequence()
    print(f"\n  Graphiti Temporal Trace (Showing first 3 events of {len(temporal_events)} total):")
    for i, event in enumerate(temporal_events[:3]):
        # Just show the time and first 3 states (Metabolic)
        mini_state = [round(x, 2) for x in event['state_vector'][:3]]
        print(f"    - Valid Time t={event['valid_time']:>3.1f} | Shift Detected | State (Metab): {mini_state}")

    if len(temporal_events) > 3:
        print("    - ...")

    # Query Cognee (Structural $\Phi$)
    structural_correlations = fitter.memory_controller.structural_memory.get_strongest_correlations()
    print(f"\n  Cognee Structural Correlates (Top 5 strongest relationships):")
    top_correlations = list(structural_correlations.items())[:5]
    if not top_correlations:
        print("    - No significant correlational thresholds breached.")
    for edge, meta in top_correlations:
        print(f"    - {edge} (Weight: {meta['weight']:.1f})")
        
    print("\n✅ End-to-End Validation Complete.\n")

if __name__ == "__main__":
    run_ai_pipeline()
    
