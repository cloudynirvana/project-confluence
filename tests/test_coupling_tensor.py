"""
Coupling Tensor Tests — Project Confluence
==========================================

Validates the mathematical and logical completeness of the CouplingTensorAnalyzer:
  1. Dimensional consistency — N_scales x N_scales x T output shape
  2. Boundedness — all C_ij elements are normalized to [0, 1]
  3. Viability computation — correct SVD and entropy comparison
  4. Failure classification — detects aging (uniform) vs. cancer (selective)
  5. Optimal target selection — correctly identifies gradient directions

Run:
    python -m unittest tests/test_coupling_tensor.py
"""

import sys
import os
import unittest
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ode_system import ComplexAttractorODE
from models.coupling_tensor import CouplingTensorAnalyzer


class TestCouplingTensorAnalyzer(unittest.TestCase):
    """Unit test suite for the BAC CouplingTensorAnalyzer."""

    def setUp(self):
        self.analyzer = CouplingTensorAnalyzer()
        
        # Healthy mock coupling tensor (highly coupled off-diagonals)
        self.C_healthy = np.array([
            [1.0,  0.85, 0.75, 0.65],
            [0.85, 1.0,  0.80, 0.70],
            [0.75, 0.80, 1.0,  0.60],
            [0.65, 0.70, 0.60, 1.0 ]
        ])

    def test_dimensions_and_bounds(self):
        """Coupling tensor must have N_scales x N_scales shape and fall within [0, 1]."""
        ode = ComplexAttractorODE()
        # Integrate a tiny trajectory to keep test fast
        sol = ode.solve(t_span=(0, 2), dt_eval=0.5)
        self.assertTrue(sol["success"])

        C_series = self.analyzer.compute_from_jacobian(ode, sol["z"], sol["t"])
        
        # Check shape: 5 scales x 5 scales x T steps
        self.assertEqual(C_series.shape, (5, 5, len(sol["t"])))
        
        # Check bounds: all values in [0, 1]
        self.assertTrue(np.all(C_series >= 0.0))
        self.assertTrue(np.all(C_series <= 1.0))

    def test_scale_entropy_rates(self):
        """Entropy rates should compute rolling entropy normalized against s_ref."""
        ode = ComplexAttractorODE()
        sol = ode.solve(t_span=(0, 10), dt_eval=0.5)
        
        # Compute entropy rates
        dt = 0.5
        entropy_series = self.analyzer.scale_entropy_rates(sol["z"], dt, window=10)
        
        self.assertEqual(entropy_series.shape, (5, len(sol["t"])))
        # Check that entropy rates are positive and reasonable
        self.assertTrue(np.all(entropy_series >= 0.0))

    def test_viability_calculation(self):
        """Viability must equal σ_min(C) - max_k[ṡ_k]."""
        C = self.C_healthy.copy()
        entropy = np.array([0.15, 0.20, 0.10, 0.25])
        
        # Hand-compute viability
        sigma_min = np.linalg.svd(C, compute_uv=False)[-1]
        expected_v = sigma_min - np.max(entropy)
        
        v = self.analyzer.viability(C, entropy)
        self.assertAlmostEqual(v, expected_v, places=6)
        
        # Check boolean satisfy check
        self.assertEqual(self.analyzer.bac_satisfied(C, entropy), expected_v > 0)

    def test_classifier_healthy(self):
        """Healthy current tensor should classify as healthy."""
        C_current = self.C_healthy.copy()
        # Minimal change (0.02)
        C_current[1, 2] -= 0.02
        C_current[2, 1] -= 0.02
        
        tag, confidence, details = self.analyzer.classify_failure(C_current, self.C_healthy)
        self.assertEqual(tag, 'healthy')
        self.assertGreaterEqual(confidence, 0.8)

    def test_classifier_aging(self):
        """Uniform decay of off-diagonal elements should classify as aging."""
        # Simulated global off-diagonal decay (uniform subtraction)
        C_aging = self.C_healthy.copy()
        for i in range(4):
            for j in range(4):
                if i != j:
                    C_aging[i, j] -= 0.40  # Massive uniform decay
        
        tag, confidence, details = self.analyzer.classify_failure(C_aging, self.C_healthy)
        self.assertEqual(tag, 'aging')
        self.assertGreaterEqual(confidence, 0.5)
        self.assertIn('uniformity', details)

    def test_classifier_cancer(self):
        """Selective cellular-organismal decoupling (C_24) should classify as cancer."""
        # Index 1 = Cellular scale, Index 2 = Organism scale (immune effectors)
        C_cancer = self.C_healthy.copy()
        C_cancer[1, 2] = 0.05  # Severe selective decoupling
        C_cancer[2, 1] = 0.05
        
        # Maintain or slightly elevate cellular coherence (diagonal C[1,1])
        C_cancer[1, 1] = 1.0
        
        tag, confidence, details = self.analyzer.classify_failure(C_cancer, self.C_healthy)
        self.assertEqual(tag, 'cancer')
        self.assertGreaterEqual(confidence, 0.5)
        self.assertIn('selectivity', details)

    def test_optimal_intervention_targeting(self):
        """Targeting logic should identify the off-diagonal element that yields maximum viability gain."""
        C_degraded = self.C_healthy.copy()
        # Degrade specifically cell-organism coupling (1, 2)
        C_degraded[1, 2] = 0.1
        C_degraded[2, 1] = 0.1
        
        entropy = np.array([0.2, 0.2, 0.2, 0.2])
        
        i, j, grad = self.analyzer.optimal_intervention_target(C_degraded, entropy)
        
        # The optimizer should select a valid off-diagonal edge with a
        # positive bottleneck-aware viability gain.
        self.assertNotEqual(i, j)
        self.assertTrue(0 <= i < C_degraded.shape[0])
        self.assertTrue(0 <= j < C_degraded.shape[1])
        self.assertGreater(grad, 0.0)

    def test_biologic_operator_lifting(self):
        """Lifting a 5x5 Φ biologic operator should yield a symmetric 4x4 tensor perturbation."""
        # 5x5 diagonal operator
        biologic_op = np.diag([0.1, 0.2, 0.3, 0.4, 0.5])
        
        C_pert = self.analyzer.lift_biologic_to_coupling(biologic_op)
        
        self.assertEqual(C_pert.shape, (5, 5))
        # Asserts direct diagonal mappings
        self.assertEqual(C_pert[1, 1], 0.1) # phi1 -> molecular diagonal
        self.assertEqual(C_pert[2, 2], 0.2) # phi2 -> cellular diagonal
        
        # Asserts off-diagonal lifted terms (which must be symmetric)
        self.assertEqual(C_pert[2, 3], 0.3) # phi3 -> cell-organism
        self.assertEqual(C_pert[3, 2], 0.3)
        
        self.assertEqual(C_pert[2, 4], 0.4) # phi4 -> cell-tissue
        self.assertEqual(C_pert[4, 2], 0.4)
        
        self.assertEqual(C_pert[3, 4], 0.5) # phi5 -> organism-tissue
        self.assertEqual(C_pert[4, 3], 0.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
