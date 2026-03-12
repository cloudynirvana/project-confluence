"""
Track B Ingestion Tests — Project Confluence
============================================
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.track_b import load_track_b_cohort, build_trajectory, compute_phi_for_patient
from models.ode_system import ComplexAttractorODE, ExtendedParams


class TestTrackBIngestion(unittest.TestCase):
    def setUp(self):
        self.data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "track_b", "mock_cohort.json")
        )

    def test_load_mock_cohort(self):
        cohort = load_track_b_cohort(self.data_path)
        self.assertEqual(cohort.disease, "TNBC")
        self.assertGreaterEqual(len(cohort.patients), 2)
        self.assertEqual(len(cohort.variables), 15)

    def test_build_trajectory_and_phi(self):
        cohort = load_track_b_cohort(self.data_path)
        patient = cohort.patients[0]
        ode = ComplexAttractorODE(params=ExtendedParams(), use_nonlinear=True, use_immune=True, use_microenv=True)
        baseline = ode.healthy_initial_state()
        traj, t_grid, dt = build_trajectory(patient, cohort.variables, baseline)

        self.assertEqual(traj.shape[0], 15)
        self.assertEqual(traj.shape[1], len(t_grid))
        self.assertTrue(dt > 0)

        phi = compute_phi_for_patient(traj, dt)
        self.assertEqual(len(phi["phi_vector"]), 5)
        self.assertTrue(all(np.isfinite(v) for v in phi["phi_vector"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
