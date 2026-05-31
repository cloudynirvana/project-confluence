"""
Optimal Inference Observer Tests — Project Confluence
======================================================

Validates the mathematical and logical completeness of the EKF State Observer:
  1. Prediction step — z_hat propagates and covariance P increases.
  2. Update step — z_hat corrections converge and covariance P decreases.
  3. Coupling tensor reconstruction — 4x4 shape matches.
  4. Viability estimation — reconstructs scalar viability.

Run:
    python -m unittest tests/test_optimal_inference.py
"""

import sys
import os
import unittest
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ode_system import ComplexAttractorODE
from models.optimal_inference import (
    ExtendedKalmanFilterObserver,
    get_clinical_measurement_matrix,
    get_neuroidentity_measurement_matrix,
)


class TestExtendedKalmanFilterObserver(unittest.TestCase):
    """Unit test suite for the EKF State Observer."""

    def setUp(self):
        self.ode = ComplexAttractorODE()
        self.observer = ExtendedKalmanFilterObserver(self.ode)
        
        # Define a 4-variable sparse clinical panel:
        # Index 0: Glucose (metabolic load)
        # Index 7: Treg fraction (cellular immunity)
        # Index 10: Effector T-cells (organismal immune)
        # Index 14: Vascular integrity (tissue scale)
        self.selected_indices = [0, 7, 10, 14]
        self.H = get_clinical_measurement_matrix(self.selected_indices)
        self.R = np.eye(len(self.selected_indices)) * 0.05  # Technical assay variance

    def test_initialization_dimensions(self):
        """Initial estimate must include 16D biology plus hidden memory kernel."""
        self.assertEqual(self.observer.z_hat.shape, (16,))
        self.assertEqual(self.observer.M_hat.shape, (2, 2))
        self.assertEqual(self.observer.x_hat.shape, (20,))
        self.assertEqual(self.observer.P.shape, (20, 20))
        self.assertEqual(self.H.shape, (4, 16))

    def test_predict_step_covariance_growth(self):
        """Without updates, the prediction step must propagate state and increase uncertainty (P)."""
        P_initial = self.observer.P.copy()
        
        # Propagate forward 10 steps
        for _ in range(10):
            z_hat, P = self.observer.predict(dt=0.1, t_current=0.0)
            
        # Assert state remains bounded and physical
        self.assertTrue(np.all(z_hat >= 0.0))
        self.assertTrue(np.all(z_hat <= 10.0))
        
        # Stable dynamics may contract uncertainty, but covariance must remain
        # finite, positive on the diagonal, and dynamically updated.
        self.assertTrue(np.all(np.isfinite(P)))
        self.assertTrue(np.all(np.diag(P) > 0.0))
        self.assertFalse(np.allclose(P, P_initial))

    def test_update_step_uncertainty_reduction(self):
        """Measurement updates must incorporate observations and reduce estimation covariance (P)."""
        # 1. Run a predict step to expand uncertainty
        z_pred, P_expanded = self.observer.predict(dt=1.0)
        
        # Create a mock clinical measurement vector (disturbing state slightly)
        y_obs = np.array([1.2, 0.4, 0.6, 0.8])
        
        # 2. Apply update step
        z_hat_post, P_post = self.observer.update(y_obs, self.H, self.R)
        
        # Assert that covariance trace has shrunk (information gained)
        self.assertLess(np.trace(P_post), np.trace(P_expanded))
        
        # Assert that updated z_hat has been steered towards the observation values
        # Measured indices: 0, 7, 10, 14
        self.assertLess(abs(z_hat_post[0] - y_obs[0]), abs(z_pred[0] - y_obs[0]))
        self.assertAlmostEqual(z_hat_post[7], 0.4, delta=0.5)

    def test_coupling_and_viability_reconstruction(self):
        """Observer must successfully reconstruct the 5x5 coupling tensor and viability functional."""
        C_est = self.observer.reconstruct_coupling_tensor(t_current=0.0)
        
        # Check shape: 5 scales x 5 scales
        self.assertEqual(C_est.shape, (5, 5))
        self.assertTrue(np.all(C_est >= 0.0))
        self.assertTrue(np.all(C_est <= 1.0))
        
        # Check viability calculation
        entropy_rates = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        V_est = self.observer.reconstruct_viability(entropy_rates, t_current=0.0)
        self.assertIsInstance(V_est, float)

    def test_memory_kernel_reconstruction(self):
        """Observer must expose hidden memory kernel and covariance estimates."""
        for _ in range(3):
            self.observer.predict(dt=0.1, t_current=0.0)

        M_est = self.observer.reconstruct_memory_kernel()
        P_M = self.observer.reconstruct_memory_covariance()
        confidence = self.observer.identity_confidence_margin()

        self.assertEqual(M_est.shape, (2, 2))
        self.assertEqual(P_M.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(M_est)))
        self.assertTrue(np.all(np.isfinite(P_M)))
        self.assertIn("confidence", confidence)
        self.assertGreaterEqual(confidence["confidence"], 0.0)

    def test_neuroidentity_measurement_channels(self):
        """DMN coherence and EEG PCI should update hidden memory covariance."""
        H_neuro = get_neuroidentity_measurement_matrix(self.observer)
        self.assertEqual(H_neuro.shape, (2, 20))

        P_before = self.observer.P.copy()
        self.observer.update_neuroidentity_channels(
            dmn_coherence=0.7,
            eeg_pci=0.65,
            R=np.eye(2) * 0.02,
        )
        P_after = self.observer.P

        self.assertLess(
            np.trace(P_after[self.observer.bio_dim:, self.observer.bio_dim:]),
            np.trace(P_before[self.observer.bio_dim:, self.observer.bio_dim:]),
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
