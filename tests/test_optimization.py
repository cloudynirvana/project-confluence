import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from intervention import InterventionMapper
from tnbc_ode import TNBCODESystem, simulate_treatment_protocol
from geometric_optimization import TherapeuticProtocolOptimizer

def test_protocol_generation():
    print("Testing protocol generation with GeometricOptimizer...")
    mapper = InterventionMapper()
    
    # 1. Setup our dummy intervention "libraries" for the optimizer interface
    metabolic_drugs = []
    entropic_drivers = []
    immune_rectifiers = []
    
    for inv in mapper.intervention_library:
        if inv.category == "curvature_reducer":
            metabolic_drugs.append((inv.name, inv.expected_effect, inv.dosage_range))
        elif inv.category == "entropic_driver":
            entropic_drivers.append((inv.name, inv.expected_effect, inv.dosage_range, inv.entropic_driver))
        elif inv.category == "vector_rectifier":
            # Taking the max force multiplier out of the immune modifiers
            force_mult = max(inv.immune_modifiers.values()) if inv.immune_modifiers else 0.1
            immune_rectifiers.append((inv.name, inv.expected_effect, inv.dosage_range, force_mult))
    
    # 2. Get the base TNBC generator state
    A_cancer = TNBCODESystem.tnbc_generator()
    
    # 3. Optimize sequence
    optimizer = TherapeuticProtocolOptimizer(n_metabolites=10)
    protocol = optimizer.generate_optimal_sequence(
        A_cancer,
        metabolic_drugs,
        entropic_drivers,
        immune_rectifiers,
        toxicity_penalty=0.05
    )
    
    # Assertions
    assert len(protocol) == 3, "Protocol should have 3 phases (Flatten, Heat, Push)"
    assert protocol[0].day_start == 0, "Phase 1 must start at day 0"
    assert protocol[1].day_start > protocol[0].day_start, "Phase 2 must occur after phase 1"
    
    print("  ✓ Basic Sequence Generated Successfully.")
    
    # 4. Check if Kramers rate strictly increased across protocol
    rates = [p.expected_escape_rate for p in protocol]
    if rates[0] < rates[1] and rates[1] < rates[2]:
         print("  ✓ Kramers Escape Rate Increased: Phase 1: {:.1e}, Phase 2: {:.1e}, Phase 3: {:.1e}".format(rates[0], rates[1], rates[2]))
    else:
         print("  ✗ Warning: Kramers rate was not monotonically increasing.")
         
    # 5. Monte Carlo testing
    print("\nTesting Monte Carlo Patient Variance...")
    final_A = A_cancer
    for phase in protocol:
        for name, dose in phase.interventions:
             for inv in mapper.intervention_library:
                  if inv.name == name:
                       final_A = final_A + (inv.expected_effect * dose)
                       break
                       
    mc_results = optimizer.evaluate_robustness_monte_carlo(final_A, base_noise=0.1, base_force=0.5, n_trials=50)
    print(f"  ✓ Protocol Robustness: {mc_results['robustness_score']*100:.1f}%")
    print(f"  ✓ Mean Endpoint Coherence Score: {mc_results['mean_coherence_score']:.3f}")
    assert mc_results['robustness_score'] > 0, "Protocol failed on all parameter permutations"

if __name__ == "__main__":
    test_protocol_generation()
    print("\nAll Core Tests Passed.\n")
