"""
Complexity Profiler Tests — Project Confluence
================================================

Tests for the 5D Φ vector computation and archetype classification.

Run:
    python -m pytest tests/test_complexity_profiler.py -v
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.complexity_profiler import (
    sample_entropy,
    multiscale_entropy,
    correlation_dimension,
    largest_lyapunov_exponent,
    power_spectral_slope,
    coupling_score,
    coherence_metric,
    classify_archetype,
    ComplexityProfiler,
    PhiProfile,
)


class TestSampleEntropy(unittest.TestCase):
    """Test sample entropy computation."""

    def test_constant_signal_zero_entropy(self):
        """Constant signal should have zero entropy."""
        signal = np.ones(200)
        se = sample_entropy(signal)
        self.assertAlmostEqual(se, 0.0, places=5)

    def test_random_signal_positive_entropy(self):
        """Random signal should have positive entropy."""
        rng = np.random.RandomState(42)
        signal = rng.randn(500)
        se = sample_entropy(signal)
        self.assertGreater(se, 0.0)

    def test_short_signal_handled(self):
        """Very short signal should not crash."""
        se = sample_entropy(np.array([1.0, 2.0]))
        self.assertEqual(se, 0.0)


class TestMultiscaleEntropy(unittest.TestCase):
    """Test MSE computation."""

    def test_output_shape(self):
        """MSE should return array of length max_scale."""
        rng = np.random.RandomState(42)
        signal = rng.randn(1000)
        mse = multiscale_entropy(signal, max_scale=10)
        self.assertEqual(len(mse), 10)


class TestCorrelationDimension(unittest.TestCase):
    """Test D₂ estimation."""

    def test_returns_nonnegative(self):
        """D₂ should be non-negative."""
        rng = np.random.RandomState(42)
        signal = rng.randn(500)
        d2 = correlation_dimension(signal, emb_dim=5, max_points=200)
        self.assertGreaterEqual(d2, 0.0)

    def test_short_signal_fallback(self):
        """Very short signal should return 0, not crash."""
        d2 = correlation_dimension(np.array([1.0, 2.0, 3.0]), emb_dim=2)
        self.assertEqual(d2, 0.0)


class TestLyapunovExponent(unittest.TestCase):
    """Test λ_max estimation."""

    def test_returns_finite(self):
        """Lyapunov exponent of random signal should be finite."""
        rng = np.random.RandomState(42)
        trajectory = rng.randn(5, 200)
        lyap = largest_lyapunov_exponent(trajectory, dt=0.5,
                                          emb_dim=4, tau=1)
        self.assertTrue(np.isfinite(lyap))


class TestPowerSpectralSlope(unittest.TestCase):
    """Test β estimation."""

    def test_white_noise_near_zero(self):
        """White noise should have β ≈ 0."""
        rng = np.random.RandomState(42)
        signal = rng.randn(2000)
        beta = power_spectral_slope(signal)
        self.assertAlmostEqual(beta, 0.0, delta=0.5)

    def test_returns_finite(self):
        """Should return finite value for valid signal."""
        rng = np.random.RandomState(42)
        signal = rng.randn(500)
        beta = power_spectral_slope(signal)
        self.assertTrue(np.isfinite(beta))


class TestCouplingScore(unittest.TestCase):
    """Test inter-system coupling."""

    def test_identical_signals_high_coupling(self):
        """Identical signals should show high coupling."""
        trajectory = np.tile(np.sin(np.linspace(0, 10, 200)), (13, 1))
        c = coupling_score(trajectory, group_a=[0, 1, 2], group_b=[10, 11, 12])
        self.assertGreater(c, 0.9)


class TestCoherenceMetric(unittest.TestCase):
    """Test composite C metric."""

    def test_healthy_reference_scores_high(self):
        """Healthy reference values should score near 1.0."""
        result = coherence_metric(D2=4.5, mse_mean_val=0.8,
                                   lyap_max=0.05, beta=1.0)
        self.assertGreater(result["C"], 0.7)

    def test_pathological_scores_lower(self):
        """Extreme values should score lower than healthy."""
        result = coherence_metric(D2=0.5, mse_mean_val=0.1,
                                   lyap_max=2.0, beta=3.0)
        healthy = coherence_metric(D2=4.5, mse_mean_val=0.8,
                                    lyap_max=0.05, beta=1.0)
        self.assertLess(result["C"], healthy["C"])


class TestArchetypeClassifier(unittest.TestCase):
    """Test pathology archetype classification."""

    def test_healthy_classified_correctly(self):
        """Healthy Φ vector should classify as Healthy Complex."""
        phi = PhiProfile(Phi_temporal=0.65, Phi_spatial=0.6,
                         Phi_functional=0.7, Phi_informational=0.65,
                         Phi_coupling=0.55)
        archetype, confidence = classify_archetype(phi, use_ml=False)
        self.assertEqual(archetype, "Healthy Complex")
        self.assertGreater(confidence, 0.5)

    def test_chaotic_classified(self):
        """Chaotic Φ vector should classify as Chaotic/Decoupled."""
        phi = PhiProfile(Phi_temporal=0.8, Phi_spatial=0.7,
                         Phi_functional=0.3, Phi_informational=0.8,
                         Phi_coupling=0.15)
        archetype, _ = classify_archetype(phi, use_ml=False)
        self.assertEqual(archetype, "Chaotic/Decoupled")


class TestComplexityProfiler(unittest.TestCase):
    """Integration test for full profiling pipeline."""

    def test_profile_produces_valid_phi(self):
        """Profiling a trajectory should produce a valid PhiProfile."""
        rng = np.random.RandomState(42)
        trajectory = rng.randn(15, 300) * 0.5 + 1.0
        profiler = ComplexityProfiler()
        phi = profiler.profile(trajectory, dt=0.5)

        self.assertIsInstance(phi, PhiProfile)
        self.assertEqual(len(phi.phi_vector), 5)
        self.assertTrue(all(0 <= v <= 1 for v in phi.phi_vector))
        self.assertIn(phi.archetype,
                      ["Chaotic/Decoupled", "Rigid/Locked",
                       "Collapsed/Exhausted", "Healthy Complex",
                       "Warburg Metabolic", "Immune Evasion",
                       "Transitional/Pre-disease", "Mixed Pathology"])

    def test_phi_json_export(self):
        """PhiProfile should export to valid JSON."""
        phi = PhiProfile(Phi_temporal=0.5, Phi_spatial=0.4,
                         Phi_functional=0.6, Phi_informational=0.5,
                         Phi_coupling=0.45, archetype="Healthy Complex",
                         coherence_C=0.72)
        import json
        data = json.loads(phi.to_json())
        self.assertIn("phi_vector", data)
        self.assertIn("summary", data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
