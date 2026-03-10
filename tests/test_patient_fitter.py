"""
Patient Fitter Tests — Project Confluence
==========================================

Smoke tests for Bayesian calibration (minimal MCMC to verify infrastructure).

Run:
    python -m pytest tests/test_patient_fitter.py -v
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.patient_fitter import (
    PatientFitter,
    DigitalTwin,
    DEFAULT_PARAM_BOUNDS,
    CCLE_CURVATURE_TARGETS,
)
from models.ode_system import ODEParams


class TestParameterSpace(unittest.TestCase):
    """Test that the parameter space is well-defined."""

    def test_bounds_valid(self):
        """All bounds must have lower < upper."""
        for b in DEFAULT_PARAM_BOUNDS:
            self.assertLess(b.lower, b.upper,
                            f"{b.name}: lower ({b.lower}) >= upper ({b.upper})")

    def test_default_params_within_bounds(self):
        """ODEParams defaults must fall within prior bounds."""
        params = ODEParams()
        for b in DEFAULT_PARAM_BOUNDS:
            val = getattr(params, b.name)
            self.assertGreaterEqual(val, b.lower,
                                     f"{b.name}: default {val} < lower bound {b.lower}")
            self.assertLessEqual(val, b.upper,
                                  f"{b.name}: default {val} > upper bound {b.upper}")

    def test_targets_exist_for_all_cancers(self):
        """Calibration targets must exist for all 10 cancer types."""
        expected = {"TNBC", "PDAC", "NSCLC", "Melanoma", "GBM",
                    "CRC", "HGSOC", "mCRPC", "AML", "HCC"}
        self.assertEqual(set(CCLE_CURVATURE_TARGETS.keys()), expected)


class TestPriorLikelihood(unittest.TestCase):
    """Test prior and likelihood functions."""

    def setUp(self):
        self.fitter = PatientFitter(cancer_type="TNBC")

    def test_prior_inside_bounds(self):
        """Parameters inside bounds should have log-prior = 0."""
        theta = np.array([
            0.5 * (b.lower + b.upper) for b in DEFAULT_PARAM_BOUNDS
        ])
        self.assertEqual(self.fitter.log_prior(theta), 0.0)

    def test_prior_outside_bounds(self):
        """Parameters outside bounds should have log-prior = -inf."""
        theta = np.array([
            b.lower - 1.0 for b in DEFAULT_PARAM_BOUNDS
        ])
        self.assertEqual(self.fitter.log_prior(theta), -np.inf)

    def test_likelihood_finite(self):
        """Likelihood at default params should be finite."""
        params = ODEParams()
        theta = np.array([
            getattr(params, b.name) for b in DEFAULT_PARAM_BOUNDS
        ])
        ll = self.fitter.log_likelihood(theta)
        self.assertTrue(np.isfinite(ll),
                        f"Likelihood at defaults is not finite: {ll}")

    def test_posterior_equals_prior_plus_likelihood(self):
        """log P(θ|D) = log P(θ) + log P(D|θ)."""
        params = ODEParams()
        theta = np.array([
            getattr(params, b.name) for b in DEFAULT_PARAM_BOUNDS
        ])
        lp = self.fitter.log_prior(theta)
        ll = self.fitter.log_likelihood(theta)
        posterior = self.fitter.log_probability(theta)
        self.assertAlmostEqual(posterior, lp + ll, places=10)


class TestDigitalTwin(unittest.TestCase):
    """Test the DigitalTwin output class."""

    def test_identifiability_report(self):
        """Identifiability report should classify all parameters."""
        twin = DigitalTwin(
            cancer_type="TNBC",
            samples=np.random.randn(100, len(DEFAULT_PARAM_BOUNDS)),
            log_probs=np.random.randn(100),
            param_names=[b.name for b in DEFAULT_PARAM_BOUNDS],
            param_bounds=DEFAULT_PARAM_BOUNDS,
        )
        report = twin.identifiability_report()
        self.assertEqual(len(report), len(DEFAULT_PARAM_BOUNDS))

    def test_json_export(self):
        """DigitalTwin should export to valid JSON."""
        twin = DigitalTwin(
            cancer_type="TNBC",
            patient_id="test-001",
            samples=np.random.randn(100, len(DEFAULT_PARAM_BOUNDS)),
            log_probs=np.random.randn(100),
            param_names=[b.name for b in DEFAULT_PARAM_BOUNDS],
            param_bounds=DEFAULT_PARAM_BOUNDS,
        )
        import json
        data = json.loads(twin.to_json())
        self.assertIn("patient_id", data)
        self.assertIn("parameters", data)
        self.assertIn("diagnostics", data)


class TestMCMCSmokeTest(unittest.TestCase):
    """Minimal MCMC run to verify infrastructure works."""

    def test_sampler_runs(self):
        """MCMC with minimal walkers/steps should not crash."""
        try:
            import emcee
        except ImportError:
            self.skipTest("emcee not installed")

        fitter = PatientFitter(cancer_type="TNBC")
        twin = fitter.fit(
            n_walkers=34,
            n_steps=20,
            n_burnin=10,
            seed=42,
            progress=False,
        )

        self.assertIsInstance(twin, DigitalTwin)
        self.assertGreater(len(twin.samples), 0)
        self.assertGreater(twin.acceptance_fraction, 0.0)


class TestProfileLikelihood(unittest.TestCase):
    """Test profile likelihood computation."""

    def test_profile_returns_arrays(self):
        """Profile likelihood should return grid + values arrays."""
        fitter = PatientFitter(cancer_type="TNBC")
        grid, profile_ll = fitter.run_profile_likelihood(
            param_index=0, n_points=5, n_inner_evals=3, seed=42,
        )

        self.assertEqual(len(grid), 5)
        self.assertEqual(len(profile_ll), 5)
        self.assertTrue(np.all(np.isfinite(grid)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
