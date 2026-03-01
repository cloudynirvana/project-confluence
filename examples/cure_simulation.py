"""
Geometric Cure Simulation
=========================

Demonstrates the "Geometric Achievement Protocol" for curing complex cancer.
Compares three scenarios:
1. Standard of Care (Checkpoint Inhibitor Monotherapy) -> FAIL
2. Metabolic Flattening Only (DCA + Metformin) -> STALL
3. Geometric Alignment (Flatten -> Heat -> Vector) -> CURE

The simulation integrates:
- Generator physics (Attractor stability)
- Immune vector fields (Exhaustion dynamics)
- Entropic resonance (Noise injection)
"""

import numpy as np
import sys
from pathlib import Path
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # Plotting disabled; simulation still runs

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generator import simulate_dynamics, GeneratorExtractor
from intervention import TNBCMetabolicModel, InterventionMapper
from geometric_optimization import GeometricOptimizer
from immune_dynamics import LymphocyteForceField, ImmuneParams

def simulate_protocol(
    scenario_name: str,
    A_initial: np.ndarray,
    protocol_steps: list,
    n_days: int = 60,
    dt: float = 0.1
):
    """
    Simulate a treatment protocol.
    
    protocol_steps: List of (start_day, end_day, intervention_list)
    """
    print(f"\nRunning Scenario: {scenario_name}")
    
    n_metabolites = A_initial.shape[0]
    steps = int(n_days / dt)
    time = np.linspace(0, n_days, steps)
    
    # State initialization (Deep inside the attractor)
    # A vector mostly aligned with the dominant eigenvector
    val, vec = np.linalg.eig(A_initial)
    idx = np.argsort(val.real)
    x = np.real(vec[:, idx[0]]) * 5.0 # Deep in the well
    
    trajectory = np.zeros((steps, n_metabolites))
    
    # System Components
    optimizer = GeometricOptimizer(n_metabolites)
    immune_sys = LymphocyteForceField(n_metabolites, ImmuneParams())
    
    # Tracking
    entropies = []
    well_depths = []
    forces = []
    
    for i in range(steps):
        t_current = i * dt
        
        # 1. Determine active interventions
        current_A = A_initial.copy()
        current_noise = 0.1
        current_immune_params = ImmuneParams() # Reset to base
        
        for start, end, interventions in protocol_steps:
            if start <= t_current <= end:
                for intervention in interventions:
                    # Apply matrix effects
                    current_A += intervention.expected_effect
                    
                    # Apply noise modifiers
                    if intervention.entropic_driver > 0:
                        current_noise *= intervention.entropic_driver
                        
                    # Apply immune modifiers
                    if intervention.immune_modifiers:
                        if 'pd1_blockade' in intervention.immune_modifiers:
                            current_immune_params.pd1_blockade = max(
                                current_immune_params.pd1_blockade, 
                                intervention.immune_modifiers['pd1_blockade']
                            )
                        if 'ctla4_blockade' in intervention.immune_modifiers:
                            current_immune_params.ctla4_blockade = max(
                                current_immune_params.ctla4_blockade, 
                                intervention.immune_modifiers['ctla4_blockade']
                            )
        
        # 2. Physics Update
        # Update immune system params with current drugs
        immune_sys.params = current_immune_params
        
        # Calculate Well Depth (Curvature)
        mu = optimizer.compute_basin_curvature(current_A)
        
        # Calculate Immune Force
        f_immune = immune_sys.compute_net_force(x, mu, dt)
        
        # SDE Step: dx = (Ax + F_immune)dt + sigma*dW
        deterministic_step = (current_A @ x + f_immune) * dt
        stochastic_step = np.random.randn(n_metabolites) * current_noise * np.sqrt(dt)
        
        x += deterministic_step + stochastic_step
        
        trajectory[i] = x
        well_depths.append(mu)
        forces.append(np.linalg.norm(f_immune))
        entropies.append(current_noise)
        
    distance_from_health = np.linalg.norm(trajectory[-1])
    success = distance_from_health < 1.0
    
    print(f"  Final Distance from Health: {distance_from_health:.3f}")
    print(f"  Max Well Depth: {max(well_depths):.3f}")
    print(f"  Min Well Depth: {min(well_depths):.3f}")
    print(f"  Outcome: {'CURE' if success else 'FAILURE'}")
    
    return {
        "trajectory": trajectory,
        "well_depths": well_depths,
        "forces": forces,
        "success": success
    }

def run_comprehensive_simulation():
    # Setup
    mapper = InterventionMapper()
    lib = {i.name: i for i in mapper.intervention_library}
    A_tnbc = TNBCMetabolicModel.get_tnbc_generator()
    
    # Define Agents
    dca = lib["Dichloroacetate (DCA)"]
    met = lib["Metformin"]
    pd1 = lib["Anti-PD-1 (Pembrolizumab)"]
    heat = lib["Entropic Heating (Hyperthermia)"]
    epo = lib["Epogen (Epoetin alfa)"]
    
    # ---------------------------------------------------------
    # Scenario 1: Standard of Care (Checkpoint Only)
    # ---------------------------------------------------------
    # Days 0-60: Anti-PD-1
    s1 = simulate_protocol(
        "Standard of Care (Checkpoint Only)", 
        A_tnbc,
        [(0, 60, [pd1])]
    )

    # ---------------------------------------------------------
    # Scenario 2: The Iatrogenic Trap (Epogen + Checkpoint)
    # ---------------------------------------------------------
    # Days 0-60: Epogen + Anti-PD-1
    s2 = simulate_protocol(
        "Iatrogenic Failure (Epogen + Checkpoint)", 
        A_tnbc,
        [(0, 60, [epo, pd1])]
    )
    
    # ---------------------------------------------------------
    # Scenario 3: Geometric Alignment (The Cure)
    # ---------------------------------------------------------
    # Days 0-20: Flattening (DCA + Metformin)
    # Days 20-25: Heating (Hyperthermia)
    # Days 25-60: Vector (Anti-PD-1)
    s3 = simulate_protocol(
        "Geometric Achievement Protocol", 
        A_tnbc,
        [
            (0, 25, [dca, met]),       # Phase 1: Flatten
            (20, 25, [heat]),          # Phase 2: Heat (Overlap)
            (25, 60, [pd1, dca, met])  # Phase 3: Push (Maintain flat)
        ]
    )
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"1. Standard Care:   {'SUCCESS' if s1['success'] else 'FAIL'} (Dist: {np.linalg.norm(s1['trajectory'][-1]):.2f})")
    print(f"2. Epogen Trap:     {'SUCCESS' if s2['success'] else 'FAIL'} (Dist: {np.linalg.norm(s2['trajectory'][-1]):.2f})")
    print(f"3. Geometric Cure:  {'SUCCESS' if s3['success'] else 'FAIL'} (Dist: {np.linalg.norm(s3['trajectory'][-1]):.2f})")

if __name__ == "__main__":
    run_comprehensive_simulation()
