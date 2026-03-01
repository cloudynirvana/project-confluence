"""
Test Suite for SAEM Universal Cure Framework
=============================================

Covers:
  1. Generator validation (shape, bounds, non-degeneracy)
  2. Intervention mapping diversity
  3. Resistance model monotonicity
  4. Report generation completeness
  5. Non-uniform escape distances (regression test)
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tnbc_ode import (
    TNBCODESystem, validate_generator, validate_all_generators,
    GENERATOR_METADATA, GeneratorMetadata,
)
from geometric_optimization import GeometricOptimizer
from intervention import InterventionMapper
from coherence import CoherenceAnalyzer


class TestGeneratorValidation(unittest.TestCase):
    """Test that all 10 generators pass validation: 10×10, bounded, non-degenerate."""

    def setUp(self):
        self.generators = TNBCODESystem.pan_cancer_generators()

    def test_count(self):
        """We must have exactly 10 generators."""
        self.assertEqual(len(self.generators), 10)

    def test_all_10x10(self):
        for name, A in self.generators.items():
            self.assertEqual(A.shape, (10, 10), f"{name} shape is {A.shape}")

    def test_bounded(self):
        for name, A in self.generators.items():
            max_val = float(np.max(np.abs(A)))
            self.assertLess(max_val, 5.0, f"{name} has entry |{max_val:.3f}| >= 5.0")

    def test_non_degenerate(self):
        """No two generators should be nearly identical (diff < 0.01)."""
        names = list(self.generators.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i+1:]:
                diff = float(np.linalg.norm(self.generators[n1] - self.generators[n2]))
                self.assertGreater(diff, 0.01,
                    f"{n1} and {n2} are nearly identical (diff={diff:.6f})")

    def test_validate_all(self):
        """Full validator should report no issues."""
        results = validate_all_generators()
        for name, issues in results.items():
            self.assertEqual(len(issues), 0, f"{name}: {issues}")

    def test_metadata_exists_for_all(self):
        """Every generator must have a metadata entry."""
        for name in self.generators:
            self.assertIn(name, GENERATOR_METADATA, f"Missing metadata for {name}")

    def test_metadata_fields(self):
        """Metadata must have valid confidence and non-empty tags."""
        for name, meta in GENERATOR_METADATA.items():
            self.assertIn(meta.confidence, ('high', 'medium', 'low'),
                f"{name}: invalid confidence '{meta.confidence}'")
            self.assertGreater(len(meta.tags), 0, f"{name}: empty tags")
            self.assertGreater(len(meta.evidence_notes), 0, f"{name}: empty evidence_notes")

    def test_distinct_curvatures(self):
        """Each generator should have a distinct basin curvature."""
        optimizer = GeometricOptimizer(10)
        curvatures = {}
        for name, A in self.generators.items():
            curvatures[name] = optimizer.compute_basin_curvature(A)

        values = list(curvatures.values())
        # Check that not all identical
        self.assertGreater(max(values) - min(values), 0.01,
            f"Curvatures too uniform: {curvatures}")


class TestInterventionDiversity(unittest.TestCase):
    """Test that drug selection produces diverse regimens across cancer types."""

    def setUp(self):
        self.generators = TNBCODESystem.pan_cancer_generators()
        self.mapper = InterventionMapper()
        self.A_healthy = TNBCODESystem.healthy_generator()

    def test_not_all_same_top_drug(self):
        """Top drug should not be identical across all 10 cancers."""
        top_drugs = []
        for name, A_cancer in self.generators.items():
            delta_A = self.A_healthy - A_cancer
            matched = self.mapper.map_correction_to_interventions(delta_A, max_interventions=1)
            if matched:
                top_drugs.append(matched[0][0].name)

        unique = len(set(top_drugs))
        self.assertGreater(unique, 1,
            f"All cancers selected the same top drug: {top_drugs[0]}")

    def test_minimum_library_size(self):
        """Intervention library should have at least 10 drugs."""
        self.assertGreaterEqual(len(self.mapper.intervention_library), 10)


class TestResistanceMonotonicity(unittest.TestCase):
    """Test that resistance accumulates monotonically under continuous exposure."""

    def test_continuous_curvature_restores(self):
        """Under continuous therapy, resistance should make late-stage curvature closer to untreated."""
        import math

        A_cancer = TNBCODESystem.tnbc_generator()
        n = 10
        optimizer = GeometricOptimizer(n)
        mapper = InterventionMapper(n)
        lib = {i.name: i for i in mapper.intervention_library}

        drug = lib["Dichloroacetate (DCA)"]
        delta_static = drug.expected_effect
        tau = 15.0

        untreated_curvature = optimizer.compute_basin_curvature(A_cancer)

        curvatures = []
        for day in range(60):
            R = 1.0 - math.exp(-day / tau)
            A_eff = A_cancer + delta_static * (1.0 - R)
            c = optimizer.compute_basin_curvature(A_eff)
            curvatures.append(c)

        # Day 55 curvature should be closer to untreated than day 5
        # (resistance restores the basin toward its original depth)
        early_diff = abs(curvatures[5] - untreated_curvature)
        late_diff = abs(curvatures[55] - untreated_curvature)
        self.assertLess(late_diff, early_diff,
            f"Resistance should restore basin: late diff={late_diff:.4f} should be < early diff={early_diff:.4f}")


class TestReportGeneration(unittest.TestCase):
    """Test that the report contains all required sections."""

    def test_report_file_created(self):
        """After running the engine, the report file must exist."""
        report_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'universal_cure_proof.md')
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()

            required_sections = [
                "Seriousness Ranking",
                "Per-Cancer Tailored Protocols",
                "Cure Metrics with Uncertainty",
                "Resistance Comparison",
                "Sensitivity Analysis",
                "Coherence Restoration",
                "Failure Modes",
                "Final Verdict",
            ]
            for section in required_sections:
                self.assertIn(section, content,
                    f"Report missing required section: '{section}'")
        else:
            self.skipTest("Report not yet generated — run universal_cure_engine.py first")


class TestNonUniformOutcomes(unittest.TestCase):
    """Regression test: escape distances must not be identical across cancers."""

    def test_escape_distances_vary(self):
        """Run a quick single-seed simulation for each cancer and verify distances differ."""
        generators = TNBCODESystem.pan_cancer_generators()
        A_healthy = TNBCODESystem.healthy_generator()
        n = 10
        optimizer = GeometricOptimizer(n)
        mapper = InterventionMapper(n)
        lib = {i.name: i for i in mapper.intervention_library}

        distances = {}
        for name, A_cancer in generators.items():
            # Simplified single-step protocol
            val, vec = np.linalg.eig(A_cancer)
            idx = np.argsort(val.real)
            x = np.real(vec[:, idx[0]]) * 5.0

            # Apply DCA + Metformin for 25 days
            A_eff = A_cancer + lib["Dichloroacetate (DCA)"].expected_effect + lib["Metformin"].expected_effect
            dt = 0.1
            rng = np.random.default_rng(42)
            for step in range(250):
                x += A_eff @ x * dt + rng.standard_normal(n) * 0.1 * np.sqrt(dt)

            distances[name] = float(np.linalg.norm(x))

        dist_values = list(distances.values())
        dist_range = max(dist_values) - min(dist_values)
        self.assertGreater(dist_range, 0.01,
            f"Escape distances too uniform (range={dist_range:.6f}): {distances}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
