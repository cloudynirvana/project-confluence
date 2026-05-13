"""
Test script for geometric pathway optimization and target discovery.
Validates the MAP implementation and tests the integration.
"""

import numpy as np
import sys
import os

# Ensure models are in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.geometric_pathways import FreidlinWentzellOptimizer
from models.fisher_geometry import FisherManifoldAnalyzer
from models.network_curvature import NetworkCurvatureAnalyzer
from models.geometric_optimization import TherapeuticProtocolOptimizer
from models.ode_system import ComplexAttractorODE, TNBCParams, ExtendedParams

def test_map_and_ranking():
    print("Initializing ODE Systems...")
    # Initialize systems
    healthy_sys = ComplexAttractorODE(params=ExtendedParams())
    tnbc_sys = ComplexAttractorODE(params=TNBCParams())
    
    # 1. MAP Path Finding (Lazy Subspace)
    print("\n--- Component 1: Minimum Action Pathway ---")
    opt = FreidlinWentzellOptimizer(tnbc_sys)
    z_healthy = opt.get_attractor("healthy")
    z_tnbc = opt.get_attractor("TNBC")
    
    # Run optimizer on active subsystem (e.g., core metabolic variables 0-3)
    path, action = opt.compute_minimum_action_path(z_tnbc, z_healthy, n_images=10, max_iter=50, active_indices=[0, 1, 2, 3])
    print(f"MAP Action (Cost): {action:.4f}")
    saddle_idx, saddle_state = opt.get_saddle_point(path)
    print(f"Saddle Point index: {saddle_idx}")
    
    map_targets = opt.get_realignment_targets(path)
    print(f"Top MAP Targets (Index, Gradient): {map_targets[:3]}")
    
    # 2. Fisher Information (Stiff/Sloppy)
    print("\n--- Component 2: Fisher Information Geometry ---")
    fisher = FisherManifoldAnalyzer(tnbc_sys, TNBCParams(), t_span=(0, 10))
    # In a real run, this would be computed using compute_fim(max_workers=4)
    # Here we mock a tiny FIM for test speed
    fim_mock = np.eye(fisher.dim_p) 
    fim_mock[0,0] = 100.0 # Make param 0 artificially stiff
    
    manifold_info = fisher.identify_stiff_sloppy(fim_mock)
    fim_targets = manifold_info['stiff']
    print(f"Top Stiff Parameters: {[t['dominant_param'] for t in fim_targets[:3]]}")
    
    # 3. Network Curvature (Ricci Bottlenecks)
    print("\n--- Component 3: Network Curvature ---")
    net_opt = NetworkCurvatureAnalyzer()
    A_tnbc = tnbc_sys.get_metabolic_generator()
    graph = net_opt.build_graph(A_tnbc)
    bottlenecks = net_opt.identify_bottlenecks(graph)
    print(f"Top Structural Bottlenecks (Edges):")
    for b in bottlenecks[:3]:
        print(f"  {b['source']} -> {b['target']} (Curvature: {b['curvature']:.2f})")
        
    # 4. Integration
    print("\n--- Component 4: Target Integration ---")
    protocol_opt = TherapeuticProtocolOptimizer(n_metabolites=15)
    
    # Mock some data structure matching
    ricci_targets = bottlenecks
    
    unified_ranking = protocol_opt.convergent_target_ranking(map_targets, fim_targets, ricci_targets)
    print("Convergent Target Priority List:")
    for i, target in enumerate(unified_ranking[:5]):
        print(f"  {i+1}. {target['target']} (Score: {target['score']:.2f})")

if __name__ == "__main__":
    test_map_and_ranking()
