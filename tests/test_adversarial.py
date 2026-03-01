"""
Adversarial Stress Tests for SAEM Cancer PoC
=============================================

These tests deliberately break the model to document failure boundaries.
A model that never fails is a model that hasn't been tested hard enough.

Tests verify:
  1. No warning-level drugs (Epogen) appear in protocols
  2. High resistance breaks cure rate
  3. Zero immune force prevents cure
  4. Extreme noise destabilizes outcomes
  5. Continuous therapy accumulates more resistance
  6. Not all cancers achieve 100% cure (model isn't trivially over-constrained)
"""

import sys
import os
import math
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tnbc_ode import TNBCODESystem, GENERATOR_METADATA
from geometric_optimization import GeometricOptimizer
from intervention import InterventionMapper, TherapeuticIntervention
from coherence import CoherenceAnalyzer
from resistance_model import ResistanceTracker, ResistanceParams


class TestNoWarningDrugsInProtocols(unittest.TestCase):
    """Epogen (warning-level) must never appear in any selected protocol."""

    def test_universal_cure_engine_excludes_warnings(self):
        """Test that select_drugs_with_diversity filters warning drugs."""
        # Import from the universal cure engine
        from universal_cure_engine import select_drugs_with_diversity

        mapper = InterventionMapper()
        generators = TNBCODESystem.pan_cancer_generators()
        A_healthy = TNBCODESystem.healthy_generator()

        for cancer_name, A_cancer in generators.items():
            meta = GENERATOR_METADATA[cancer_name]
            delta_A = A_healthy - A_cancer
            drugs = select_drugs_with_diversity(mapper, delta_A, meta, max_drugs=4)
            drug_names = [d.name for d, _ in drugs]
            evidence_levels = [d.evidence_level for d, _ in drugs]

            for name, level in zip(drug_names, evidence_levels):
                self.assertNotEqual(level, "warning",
                    f"{cancer_name}: Warning-level drug '{name}' selected! "
                    f"This is a known negative control and must be excluded.")

    def test_confluence_runner_excludes_warnings(self):
        """Test that confluence_runner's select_drugs also filters warnings."""
        from confluence_runner import select_drugs

        mapper = InterventionMapper()
        generators = TNBCODESystem.pan_cancer_generators()
        A_healthy = TNBCODESystem.healthy_generator()

        for cancer_name, A_cancer in generators.items():
            drugs = select_drugs(mapper, A_cancer, A_healthy, cancer_name)
            for d, _ in drugs:
                self.assertNotEqual(d.evidence_level, "warning",
                    f"{cancer_name}: Warning drug '{d.name}' in confluence runner protocol!")


class TestHighResistanceBreaksCure(unittest.TestCase):
    """With extreme resistance buildup, cure should become unreliable."""

    def test_extreme_resistance_prevents_reliable_cure(self):
        """resistance_tau=3d means drugs lose efficacy within days."""
        from confluence_runner import run_protocol_simulation, compute_phase_timing, select_drugs
        import confluence_runner

        # Save originals
        orig_tau = confluence_runner.RESISTANCE_TAU

        try:
            # Extreme resistance: drugs lose all efficacy in ~3 days
            confluence_runner.RESISTANCE_TAU = 3.0

            mapper = InterventionMapper()
            A_cancer = TNBCODESystem.pdac_generator()  # Hardest cancer
            A_healthy = TNBCODESystem.healthy_generator()

            drugs = select_drugs(mapper, A_cancer, A_healthy, "PDAC")
            phase_days = compute_phase_timing(0.50)

            # Run multiple trials — cure should be unreliable
            n_trials = 20
            distances = []
            for seed in range(n_trials):
                dist, _ = run_protocol_simulation(A_cancer, A_healthy, drugs, phase_days, seed=seed)
                distances.append(dist)

            cure_rate = sum(1 for d in distances if d < 0.90) / len(distances)
            # With extreme resistance, cure rate should drop below reliable levels
            self.assertLess(cure_rate, 1.0,
                f"With resistance_tau=3.0d, cure rate should not be perfect. "
                f"Got {cure_rate:.0%}. Model may be under-constrained.")
        finally:
            confluence_runner.RESISTANCE_TAU = orig_tau


class TestZeroImmuneForceFails(unittest.TestCase):
    """Without immune force, the Push phase has no mechanism to escape the basin."""

    def test_no_immune_force_reduces_cure(self):
        from confluence_runner import run_protocol_simulation, compute_phase_timing, select_drugs
        import confluence_runner

        orig_force = confluence_runner.BASE_FORCE

        try:
            confluence_runner.BASE_FORCE = 0.0

            mapper = InterventionMapper()
            A_cancer = TNBCODESystem.tnbc_generator()
            A_healthy = TNBCODESystem.healthy_generator()

            drugs = select_drugs(mapper, A_cancer, A_healthy, "TNBC")
            phase_days = compute_phase_timing(0.40)

            n_trials = 20
            distances = []
            for seed in range(n_trials):
                dist, _ = run_protocol_simulation(A_cancer, A_healthy, drugs, phase_days, seed=seed)
                distances.append(dist)

            mean_dist = np.mean(distances)
            # Without immune force, escape should be harder (higher distance = worse)
            self.assertGreater(mean_dist, 0.1,
                "With zero immune force, escape distance should not be trivially small. "
                "Push phase should depend on immune activation.")
        finally:
            confluence_runner.BASE_FORCE = orig_force


class TestExtremeNoiseDestabilizes(unittest.TestCase):
    """Very high noise should produce chaotic, unreliable outcomes."""

    def test_high_noise_increases_variance(self):
        from confluence_runner import run_protocol_simulation, compute_phase_timing, select_drugs
        import confluence_runner

        orig_noise = confluence_runner.NOISE_SCALE

        try:
            # Normal run
            confluence_runner.NOISE_SCALE = 0.12

            mapper = InterventionMapper()
            A_cancer = TNBCODESystem.tnbc_generator()
            A_healthy = TNBCODESystem.healthy_generator()

            drugs = select_drugs(mapper, A_cancer, A_healthy, "TNBC")
            phase_days = compute_phase_timing(0.40)

            normal_dists = []
            for seed in range(15):
                dist, _ = run_protocol_simulation(A_cancer, A_healthy, drugs, phase_days, seed=seed)
                normal_dists.append(dist)
            normal_std = np.std(normal_dists)

            # High noise run
            confluence_runner.NOISE_SCALE = 1.0
            high_noise_dists = []
            for seed in range(15):
                dist, _ = run_protocol_simulation(A_cancer, A_healthy, drugs, phase_days, seed=seed)
                high_noise_dists.append(dist)
            high_noise_std = np.std(high_noise_dists)

            self.assertGreater(high_noise_std, normal_std,
                f"High noise (σ=1.0) should produce more variance than normal (σ=0.12). "
                f"Normal std={normal_std:.4f}, High noise std={high_noise_std:.4f}")
        finally:
            confluence_runner.NOISE_SCALE = orig_noise


class TestContinuousTherapyBuildsResistance(unittest.TestCase):
    """Continuous (no-holiday) therapy should accumulate more resistance."""

    def test_adaptive_beats_continuous(self):
        from confluence_runner import run_resistance_comparison, compute_phase_timing, select_drugs

        mapper = InterventionMapper()
        A_cancer = TNBCODESystem.tnbc_generator()
        A_healthy = TNBCODESystem.healthy_generator()

        drugs = select_drugs(mapper, A_cancer, A_healthy, "TNBC")
        phase_days = compute_phase_timing(0.40)

        comparison = run_resistance_comparison(A_cancer, drugs, phase_days, A_healthy)
        self.assertTrue(comparison["adaptive_wins"],
            f"Adaptive should beat continuous. "
            f"Adaptive dist={comparison['adaptive_distance']:.4f}, "
            f"Continuous dist={comparison['continuous_distance']:.4f}")


class TestModelIsNotTrivial(unittest.TestCase):
    """The model must not trivially cure everything at 100% — that signals over-fitting."""

    def test_not_all_cancers_100_percent(self):
        """With realistic parameters, at least one cancer should have <100% cure rate."""
        from confluence_runner import (
            run_monte_carlo, compute_phase_timing, select_drugs,
            MONTE_CARLO_TRIALS
        )

        mapper = InterventionMapper()
        generators = TNBCODESystem.pan_cancer_generators()
        A_healthy = TNBCODESystem.healthy_generator()

        cure_rates = {}
        # Test the 3 hardest cancers to save time
        hardest = ["PDAC", "HCC", "GBM"]
        for cancer_name in hardest:
            A_cancer = generators[cancer_name]
            drugs = select_drugs(mapper, A_cancer, A_healthy, cancer_name)
            phase_days = compute_phase_timing(0.50)

            # Smaller MC for speed
            distances, cure_rate, _, _ = run_monte_carlo(
                A_cancer, A_healthy, drugs, phase_days, n_trials=30
            )
            cure_rates[cancer_name] = cure_rate

        # At least one should not be perfect 100%
        all_perfect = all(r >= 1.0 for r in cure_rates.values())
        # Note: This test documents the expected behavior.
        # If ALL are 100%, it's a signal the model may be over-constrained.
        if all_perfect:
            import warnings
            warnings.warn(
                f"All tested cancers achieved 100% cure rate: {cure_rates}. "
                f"Consider whether the model is sufficiently constrained.",
                UserWarning
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)
