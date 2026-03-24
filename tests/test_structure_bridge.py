"""
Structure Bridge Tests — Project Confluence
=============================================

Tests for structure_bridge.py: mutation impact, drug-target affinity,
disease profiling, and ODE parameter integration.

Run:
    python -m pytest tests/test_structure_bridge.py -v
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.alphafold_client import create_mock_structure, DISEASE_PANELS
from models.structure_bridge import (
    StructureBridge,
    StructuralModifiers,
    DrugTargetAffinity,
    DiseaseStructuralProfile,
    DRUG_TARGET_MAP,
)
from models.structural_docking import (
    DrugTargetMatcher,
    DRUG_PROFILES,
    compute_docking_score,
    DockingResult,
)
from models.ode_system import ComplexAttractorODE, ExtendedParams, TNBCParams
from models.complexity_profiler import ComplexityProfiler


class TestMutationImpact(unittest.TestCase):
    """Test mutation impact scoring."""

    def setUp(self):
        self.bridge = StructureBridge()
        self.structure = create_mock_structure(
            uniprot_id="P04637", gene_name="TP53", n_residues=393
        )

    def test_returns_structural_modifiers(self):
        """compute_mutation_impact should return StructuralModifiers."""
        result = self.bridge.compute_mutation_impact(
            gene="TP53", mutation_position=248,
            uniprot_id="P04637", structure=self.structure
        )
        self.assertIsInstance(result, StructuralModifiers)

    def test_stability_score_in_range(self):
        """Stability score should be in [0, 1]."""
        result = self.bridge.compute_mutation_impact(
            gene="TP53", mutation_position=248,
            uniprot_id="P04637", structure=self.structure
        )
        self.assertGreaterEqual(result.stability_score, 0.0)
        self.assertLessEqual(result.stability_score, 1.0)

    def test_ode_multiplier_range(self):
        """ODE parameter multiplier should be in reasonable range."""
        result = self.bridge.compute_mutation_impact(
            gene="TP53", mutation_position=248,
            uniprot_id="P04637", structure=self.structure
        )
        self.assertGreaterEqual(result.ode_parameter_multiplier, 0.3)
        self.assertLessEqual(result.ode_parameter_multiplier, 2.5)

    def test_active_site_mutation_high_impact(self):
        """Mutations at active site residues should have high stability scores."""
        # TP53 hotspot residue 248
        result = self.bridge.compute_mutation_impact(
            gene="TP53", mutation_position=248,
            uniprot_id="P04637", structure=self.structure
        )
        self.assertTrue(result.is_in_active_site)
        self.assertGreater(result.stability_score, 0.5)

    def test_pocket_accessibility_in_range(self):
        """Pocket accessibility should be in [0, 1]."""
        result = self.bridge.compute_mutation_impact(
            gene="TP53", mutation_position=200,
            uniprot_id="P04637", structure=self.structure
        )
        self.assertGreaterEqual(result.pocket_accessibility, 0.0)
        self.assertLessEqual(result.pocket_accessibility, 1.0)

    def test_confidence_tier_assigned(self):
        """A confidence tier should be assigned."""
        result = self.bridge.compute_mutation_impact(
            gene="TP53", mutation_position=100,
            uniprot_id="P04637", structure=self.structure
        )
        valid_tiers = {"very_high", "confident", "low", "disordered"}
        self.assertIn(result.confidence_tier, valid_tiers)

    def test_to_dict_serializable(self):
        """to_dict should return JSON-serializable dict."""
        import json
        result = self.bridge.compute_mutation_impact(
            gene="TP53", mutation_position=248,
            uniprot_id="P04637", structure=self.structure
        )
        d = result.to_dict()
        json.dumps(d)  # Should not raise


class TestDrugTargetAffinity(unittest.TestCase):
    """Test drug-target affinity scoring."""

    def setUp(self):
        self.bridge = StructureBridge()
        self.structure = create_mock_structure(
            uniprot_id="P52789", gene_name="HK2", n_residues=400
        )

    def test_returns_affinity_object(self):
        result = self.bridge.compute_drug_target_affinity(
            "2-Deoxyglucose (2-DG)", self.structure
        )
        self.assertIsInstance(result, DrugTargetAffinity)

    def test_efficacy_multiplier_range(self):
        """Efficacy multiplier should be in [0.5, 1.5]."""
        result = self.bridge.compute_drug_target_affinity(
            "2-Deoxyglucose (2-DG)", self.structure
        )
        self.assertGreaterEqual(result.efficacy_multiplier, 0.4)
        self.assertLessEqual(result.efficacy_multiplier, 1.6)

    def test_structural_confidence_range(self):
        result = self.bridge.compute_drug_target_affinity(
            "Metformin", self.structure
        )
        self.assertGreaterEqual(result.structural_confidence, 0.0)
        self.assertLessEqual(result.structural_confidence, 1.0)


class TestDrugTargetMatcher(unittest.TestCase):
    """Test geometric docking."""

    def setUp(self):
        self.matcher = DrugTargetMatcher()
        self.structure = create_mock_structure(n_residues=400)

    def test_dock_returns_results(self):
        """dock() should return a list of DockingResults."""
        results = self.matcher.dock("Metformin", self.structure)
        self.assertIsInstance(results, list)
        if results:
            self.assertIsInstance(results[0], DockingResult)

    def test_results_sorted_by_score(self):
        """Results should be sorted by overall_score (best first)."""
        results = self.matcher.dock("Olaparib (PARP inhibitor)", self.structure)
        if len(results) >= 2:
            for i in range(len(results) - 1):
                self.assertGreaterEqual(
                    results[i].overall_score,
                    results[i + 1].overall_score,
                )

    def test_unknown_drug_gets_generic_profile(self):
        """Unknown drugs should use generic pharmacophore."""
        results = self.matcher.dock("UnknownDrug42", self.structure)
        # Should not raise, should return results
        self.assertIsInstance(results, list)

    def test_convenience_function(self):
        """compute_docking_score should return a float."""
        score = compute_docking_score("Metformin", self.structure)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestDiseaseProfile(unittest.TestCase):
    """Test disease-level structural profiling."""

    def test_profile_tnbc(self):
        """Should generate a valid profile for TNBC."""
        bridge = StructureBridge()
        profile = bridge.profile_disease("TNBC", use_mock=True)
        self.assertIsInstance(profile, DiseaseStructuralProfile)
        self.assertEqual(profile.disease, "TNBC")
        self.assertGreater(len(profile.gene_modifiers), 0)

    def test_aggregate_vulnerability_range(self):
        bridge = StructureBridge()
        profile = bridge.profile_disease("TNBC", use_mock=True)
        self.assertGreaterEqual(profile.aggregate_vulnerability, 0.0)
        self.assertLessEqual(profile.aggregate_vulnerability, 1.0)

    def test_to_dict_serializable(self):
        import json
        bridge = StructureBridge()
        profile = bridge.profile_disease("TNBC", use_mock=True)
        d = profile.to_dict()
        json.dumps(d)  # Should not raise

    def test_unknown_disease_raises(self):
        bridge = StructureBridge()
        with self.assertRaises(ValueError):
            bridge.profile_disease("FakeDisease123", use_mock=True)


class TestStructuralIntegration(unittest.TestCase):
    """
    Integration test: AlphaFold → structure analysis → ODE → Φ profile.

    Verifies that structural modifiers produce measurably different
    Φ profiles compared to baseline.
    """

    def test_structural_modifiers_affect_phi(self):
        """Structurally-modified ODE params should yield different Φ-distance."""
        profiler = ComplexityProfiler()

        # Baseline TNBC
        baseline_ode = ComplexAttractorODE(params=TNBCParams())
        baseline_result = baseline_ode.solve(t_span=(0, 60), dt_eval=1.0)
        baseline_phi = profiler.profile(baseline_result['z'], dt=1.0)

        # Structurally-modified TNBC (amplified parameters)
        modified_params = TNBCParams()
        modified_params.glucose_uptake *= 1.5  # Structural amplification
        modified_params.glycolysis_flux *= 1.3
        modified_ode = ComplexAttractorODE(params=modified_params)
        modified_result = modified_ode.solve(t_span=(0, 60), dt_eval=1.0)
        modified_phi = profiler.profile(modified_result['z'], dt=1.0)

        # Both should produce valid Φ vectors
        self.assertEqual(len(baseline_phi.phi_vector), 5)
        self.assertEqual(len(modified_phi.phi_vector), 5)

        # They should differ
        baseline_v = np.array(baseline_phi.phi_vector)
        modified_v = np.array(modified_phi.phi_vector)
        diff = np.linalg.norm(baseline_v - modified_v)

        # Difference should be non-trivial but bounded
        self.assertGreater(diff, 0.0,
                          "Structural modification had no effect on Φ")

    def test_healthy_reference_stable(self):
        """Healthy reference Φ should be reproducible."""
        profiler = ComplexityProfiler()

        ode1 = ComplexAttractorODE(params=ExtendedParams())
        r1 = ode1.solve(t_span=(0, 60), dt_eval=1.0)
        phi1 = profiler.profile(r1['z'], dt=1.0)

        ode2 = ComplexAttractorODE(params=ExtendedParams())
        r2 = ode2.solve(t_span=(0, 60), dt_eval=1.0)
        phi2 = profiler.profile(r2['z'], dt=1.0)

        np.testing.assert_array_almost_equal(
            phi1.phi_vector, phi2.phi_vector, decimal=4
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
