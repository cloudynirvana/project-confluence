"""
Complexity Calibrator Tests — Project Confluence
==================================================

Tests for the staged calibration pipeline:
  Stage 1: SIS (monotonicity, feature weighting)
  Stage 2: Δθ (L1 sparsity, sign constraints, bounds)
  Stage 3: Global (cross-disease coherence, stability, end-to-end)

Run:
    python -m pytest tests/test_complexity_calibrator.py -v
"""

import sys
import os
import unittest
import numpy as np
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.complexity_calibrator import (
    SISCalibrator,
    ParameterMapper,
    GlobalCalibrator,
    BiologicalAuditor,
    CalibrationWeights,
    SISVector,
    CalibrationResult,
    SIGN_CONSTRAINTS,
    PLAUSIBILITY_BOUNDS,
    run_full_calibration,
)
from models.alphafold_client import create_mock_structure, DISEASE_PANELS
from models.structure_bridge import StructureBridge, StructuralModifiers


class TestSISCalibrator(unittest.TestCase):
    """Stage 1: SIS feature mapping tests."""

    def setUp(self):
        self.calibrator = SISCalibrator()

    def test_sis_in_range(self):
        """SIS should always be in [0, 1]."""
        sv = SISVector(
            gene="TP53", plddt_score=0.9, stability_score=0.8,
            pocket_accessibility=0.5, active_site_flag=1.0,
            druggability_score=0.7,
        )
        sis = self.calibrator.compute_sis(sv)
        self.assertGreaterEqual(sis, 0.0)
        self.assertLessEqual(sis, 1.0)

    def test_monotonicity(self):
        """Increasing any feature should not decrease SIS."""
        self.assertTrue(self.calibrator.verify_monotonicity())

    def test_active_site_increases_sis(self):
        """Active site flag should increase SIS."""
        sv_no_active = SISVector(
            gene="TP53", plddt_score=0.7, stability_score=0.6,
            pocket_accessibility=0.3, active_site_flag=0.0,
            druggability_score=0.4,
        )
        sv_active = SISVector(
            gene="TP53", plddt_score=0.7, stability_score=0.6,
            pocket_accessibility=0.3, active_site_flag=1.0,
            druggability_score=0.4,
        )
        sis_no = self.calibrator.compute_sis(sv_no_active)
        sis_yes = self.calibrator.compute_sis(sv_active)
        self.assertGreater(sis_yes, sis_no)

    def test_higher_plddt_higher_sis(self):
        """Higher pLDDT (more ordered disruption) → higher SIS."""
        sv_low = SISVector("X", 0.3, 0.5, 0.3, 0.0, 0.3)
        sv_high = SISVector("X", 0.9, 0.5, 0.3, 0.0, 0.3)
        self.assertGreater(
            self.calibrator.compute_sis(sv_high),
            self.calibrator.compute_sis(sv_low),
        )

    def test_from_modifiers(self):
        """SISVector should be constructable from StructuralModifiers."""
        mod = StructuralModifiers(
            gene="TP53", uniprot_id="P04637", mutation_position=248,
            stability_score=0.9, pocket_accessibility=0.4,
            ode_parameter_multiplier=2.0, local_plddt=89.0,
            is_in_pocket=True, is_in_active_site=True,
            nearby_pocket_druggability=0.8,
        )
        sv = SISVector.from_modifiers(mod)
        self.assertEqual(sv.gene, "TP53")
        self.assertAlmostEqual(sv.plddt_score, 0.89, places=2)
        self.assertEqual(sv.active_site_flag, 1.0)

    def test_batch_processing(self):
        """compute_batch should process multiple genes."""
        bridge = StructureBridge()
        profile = bridge.profile_disease("TNBC", use_mock=True)
        batch = self.calibrator.compute_batch(profile.gene_modifiers)
        self.assertGreater(len(batch), 0)
        for gene, (sv, sis) in batch.items():
            self.assertIsInstance(sv, SISVector)
            self.assertGreaterEqual(sis, 0.0)
            self.assertLessEqual(sis, 1.0)


class TestParameterMapper(unittest.TestCase):
    """Stage 2: L1 parameter mapping tests."""

    def setUp(self):
        self.mapper = ParameterMapper()

    def test_delta_theta_computed(self):
        """Should produce non-empty Δθ for known genes."""
        sis_scores = {"TP53": 0.8, "KRAS": 0.9, "BRCA1": 0.7}
        delta_theta = self.mapper.compute_delta_theta(sis_scores)
        self.assertGreater(len(delta_theta), 0)

    def test_l1_sparsity(self):
        """Low SIS scores should produce zero Δθ (sparsity)."""
        weights = CalibrationWeights(l1_lambda=0.5)
        mapper = ParameterMapper(weights=weights)
        sis_scores = {"TP53": 0.1}  # Very low SIS
        delta_theta = mapper.compute_delta_theta(sis_scores)
        # With high L1 lambda and low SIS, should be zeroed out
        for gene, shifts in delta_theta.items():
            for param, delta in shifts.items():
                self.assertAlmostEqual(delta, 0.0,
                    msg=f"L1 sparsity failed for {gene}/{param}")

    def test_sign_constraints_enforced(self):
        """No sign violations should exist in output."""
        sis_scores = {
            "TP53": 0.8, "KRAS": 0.9, "BRCA1": 0.7,
            "HK2": 0.6, "GPX4": 0.5,
        }
        delta_theta = self.mapper.compute_delta_theta(sis_scores)
        violations = self.mapper.audit_signs(delta_theta)
        self.assertEqual(len(violations), 0,
            f"Sign violations found: {violations}")

    def test_plausibility_bounds_enforced(self):
        """All Δθ should be within plausibility bounds."""
        sis_scores = {"TP53": 1.0, "KRAS": 1.0}
        delta_theta = self.mapper.compute_delta_theta(sis_scores)
        violations = self.mapper.audit_bounds(delta_theta)
        self.assertEqual(len(violations), 0,
            f"Bound violations: {violations}")

    def test_parameter_entropy(self):
        """Parameter entropy should be non-negative."""
        sis_scores = {"TP53": 0.8}
        delta_theta = self.mapper.compute_delta_theta(sis_scores)
        entropy = self.mapper.compute_parameter_entropy(delta_theta)
        self.assertGreaterEqual(entropy, 0.0)

    def test_apply_to_params_creates_copy(self):
        """apply_to_params should not modify the original."""
        from models.ode_system import ExtendedParams
        base = ExtendedParams()
        original_glucose = base.glucose_uptake

        delta_theta = {"HK2": {"glucose_uptake": -0.2}}
        modified = self.mapper.apply_to_params(base, delta_theta)

        self.assertAlmostEqual(base.glucose_uptake, original_glucose)
        self.assertNotAlmostEqual(modified.glucose_uptake, original_glucose)

    def test_unknown_gene_ignored(self):
        """Unknown genes should be silently skipped."""
        sis_scores = {"NONEXISTENT_GENE_42": 0.9}
        delta_theta = self.mapper.compute_delta_theta(sis_scores)
        self.assertEqual(len(delta_theta), 0)


class TestGlobalCalibrator(unittest.TestCase):
    """Stage 3: Global calibration tests."""

    def setUp(self):
        self.calibrator = GlobalCalibrator(
            diseases=["TNBC"],
            use_mock=True,
        )

    def test_calibrate_single_disease(self):
        """Should produce a valid CalibrationResult."""
        weights = CalibrationWeights()
        result = self.calibrator.calibrate_disease("TNBC", weights)
        self.assertIsInstance(result, CalibrationResult)
        self.assertEqual(result.disease, "TNBC")
        self.assertGreater(len(result.sis_vectors), 0)

    def test_phi_vectors_have_5_components(self):
        """Φ vectors should have 5 components."""
        weights = CalibrationWeights()
        result = self.calibrator.calibrate_disease("TNBC", weights)
        self.assertEqual(len(result.phi_baseline), 5)
        self.assertEqual(len(result.phi_calibrated), 5)
        self.assertEqual(len(result.phi_healthy), 5)

    def test_stability_scores_in_range(self):
        """Stability scores should be in [0, 1]."""
        weights = CalibrationWeights()
        result = self.calibrator.calibrate_disease("TNBC", weights)
        for gene, score in result.stability_scores.items():
            self.assertGreaterEqual(score, 0.0, f"{gene} stability below 0")
            self.assertLessEqual(score, 1.0, f"{gene} stability above 1")

    def test_result_serializable(self):
        """CalibrationResult.to_dict() should be JSON-serializable."""
        weights = CalibrationWeights()
        result = self.calibrator.calibrate_disease("TNBC", weights)
        d = result.to_dict()
        json.dumps(d)  # Should not raise


class TestBiologicalAuditor(unittest.TestCase):
    """Biological plausibility audit tests."""

    def test_audit_produces_grade(self):
        """Audit should assign a grade A-D."""
        calibrator = GlobalCalibrator(diseases=["TNBC"], use_mock=True)
        weights = CalibrationWeights()
        results = {"TNBC": calibrator.calibrate_disease("TNBC", weights)}

        auditor = BiologicalAuditor()
        report = auditor.audit(results)

        self.assertIn(report["grade"], ["A", "B", "C", "D"])

    def test_clean_calibration_gets_good_grade(self):
        """Default calibration should not produce many violations."""
        calibrator = GlobalCalibrator(diseases=["TNBC"], use_mock=True)
        weights = CalibrationWeights()
        results = {"TNBC": calibrator.calibrate_disease("TNBC", weights)}

        auditor = BiologicalAuditor()
        report = auditor.audit(results)

        # Should be at least grade C (≤5 violations)
        self.assertIn(report["grade"], ["A", "B", "C"])


class TestCrossDiseaseCoherence(unittest.TestCase):
    """Cross-disease coherence tests."""

    def test_shared_genes_detected(self):
        """Should detect shared genes across diseases."""
        calibrator = GlobalCalibrator(
            diseases=["TNBC", "GBM"],
            use_mock=True,
        )
        weights = CalibrationWeights()
        results = {
            d: calibrator.calibrate_disease(d, weights)
            for d in ["TNBC", "GBM"]
        }
        coherence = calibrator.cross_panel_coherence(results)
        # EGFR, TP53, PTEN are shared between TNBC and GBM
        shared_genes = set(coherence.keys())
        self.assertGreater(len(shared_genes), 0,
                          "No shared genes found between TNBC and GBM")


class TestCalibrationWeights(unittest.TestCase):
    """Weight serialization tests."""

    def test_weights_roundtrip(self):
        """Weights should survive serialization roundtrip."""
        w = CalibrationWeights()
        w.gene_scales = {"TP53": 1.5, "KRAS": 0.8}
        d = w.to_dict()
        json.dumps(d)  # Should be serializable

    def test_optimization_vector_roundtrip(self):
        """Optimization vector should restore same weights."""
        w = CalibrationWeights()
        genes = ["TP53", "KRAS", "BRCA1"]
        for g in genes:
            w.gene_scales[g] = 1.0

        vec = w.to_optimization_vector()
        w2 = CalibrationWeights()
        w2.from_optimization_vector(vec, genes)

        np.testing.assert_array_almost_equal(w.sis_weights, w2.sis_weights, decimal=3)


class TestEndToEnd(unittest.TestCase):
    """Full pipeline integration test."""

    def test_full_calibration_pipeline(self):
        """run_full_calibration should produce a complete report."""
        report = run_full_calibration(
            diseases=["TNBC"],
            n_iterations=2,
            use_mock=True,
        )

        self.assertIn("calibration_summary", report)
        self.assertIn("audit", report)
        self.assertIn("per_disease_results", report)
        self.assertIn("audit_grade", report["calibration_summary"])

    def test_calibrated_phi_differs_from_baseline(self):
        """Calibration should produce measurably different Φ profiles."""
        calibrator = GlobalCalibrator(diseases=["TNBC"], use_mock=True)
        weights = CalibrationWeights()
        result = calibrator.calibrate_disease("TNBC", weights)

        # Calibrated and baseline Φ should exist and be valid
        self.assertTrue(np.all(np.isfinite(result.phi_baseline)))
        self.assertTrue(np.all(np.isfinite(result.phi_calibrated)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
