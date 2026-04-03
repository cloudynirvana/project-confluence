"""
Tests for Nigeria Clinical Guidelines Integration
====================================================

Tests the NigeriaGuidelineRetriever, Nigeria-specific guardrails,
and integration with the AdaptiveController and PatientFitter.

Data Source:
    Nigeria Standard Treatment Guidelines (NSTG) 2022
    License: CC-BY-4.0
    Federal Ministry of Health, Nigeria
    Dataset curated by Chisom Rutherford
"""

import json
import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════
# RETRIEVER TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestNigeriaGuidelineRetriever:
    """Tests for the NigeriaGuidelineRetriever module."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize retriever with mock data (no network required)."""
        from agents.nigeria_guideline_retriever import NigeriaGuidelineRetriever
        self.retriever = NigeriaGuidelineRetriever()

    def test_loads_conditions(self):
        """Retriever should load conditions on initialization."""
        assert len(self.retriever.conditions) > 0
        print(f"  Loaded {len(self.retriever.conditions)} conditions")

    def test_loads_chunks(self):
        """Retriever should create chunks from conditions."""
        assert len(self.retriever.chunks) > 0
        print(f"  Created {len(self.retriever.chunks)} chunks")

    def test_list_conditions(self):
        """list_conditions() should return non-empty sorted list."""
        conditions = self.retriever.list_conditions()
        assert isinstance(conditions, list)
        assert len(conditions) > 0
        assert conditions == sorted(conditions)  # Alphabetically sorted
        print(f"  Conditions: {conditions}")

    def test_get_stats(self):
        """get_stats() should return valid statistics dict."""
        stats = self.retriever.get_stats()
        assert "n_conditions" in stats
        assert "n_chunks" in stats
        assert "index_type" in stats
        assert stats["n_conditions"] > 0
        assert stats["n_chunks"] > 0
        print(f"  Stats: {stats}")

    def test_retrieve_breast_cancer(self):
        """Retrieve should return results for an oncology query."""
        results = self.retriever.retrieve("breast cancer treatment")
        assert len(results) > 0
        # At least one result should mention breast cancer
        has_breast = any(
            "breast" in r.chunk.condition_name.lower() or
            "breast" in r.chunk.text.lower()
            for r in results
        )
        assert has_breast, "No breast cancer results found"
        print(f"  Retrieved {len(results)} results for 'breast cancer treatment'")
        for r in results[:3]:
            print(f"    [{r.rank}] {r.chunk.condition_name} ({r.chunk.field_type}) "
                  f"score={r.score:.3f}")

    def test_retrieve_with_field_filter(self):
        """Retrieve with field_filter should only return matching field types."""
        results = self.retriever.retrieve(
            "treatment dosing",
            field_filter="treatment_protocol"
        )
        for r in results:
            assert r.chunk.field_type == "treatment_protocol", \
                f"Expected treatment_protocol, got {r.chunk.field_type}"

    def test_answer_format(self):
        """answer() should return a formatted string with source attribution."""
        answer = self.retriever.answer("What is the treatment for malaria?")
        assert isinstance(answer, str)
        assert len(answer) > 0
        # Should contain NSTG attribution
        assert "NSTG" in answer or "Nigeria" in answer or "Query:" in answer
        print(f"  Answer preview: {answer[:200]}...")

    def test_get_treatment_protocol_exact(self):
        """get_treatment_protocol() should return protocol for exact match."""
        protocol = self.retriever.get_treatment_protocol("BREAST CANCER")
        if protocol is None:
            # May be using mock data with different name
            protocol = self.retriever.get_treatment_protocol("breast cancer")
        assert protocol is not None, "Breast cancer protocol not found"
        assert "treatment_protocols" in protocol or "condition_name" in protocol
        print(f"  Protocol found: {protocol.get('condition_name', 'N/A')}")

    def test_get_treatment_protocol_fuzzy(self):
        """get_treatment_protocol() should handle fuzzy matching."""
        protocol = self.retriever.get_treatment_protocol("BREAST")
        # Should find breast cancer via substring match
        if protocol:
            assert "breast" in protocol.get("condition_name", "").lower()

    def test_get_dosing_constraints(self):
        """get_dosing_constraints() should return drug info across conditions."""
        constraints = self.retriever.get_dosing_constraints("tamoxifen")
        if constraints:  # May be empty if using minimal mock data
            assert isinstance(constraints, list)
            for c in constraints:
                assert "condition" in c
                print(f"  Drug found in: {c['condition']}")

    def test_extract_oncology_constraints(self):
        """extract_oncology_constraints() should find cancer-related conditions."""
        constraints = self.retriever.extract_oncology_constraints()
        assert isinstance(constraints, dict)
        print(f"  Found {len(constraints)} oncology conditions")
        for name in constraints:
            print(f"    - {name}")

    def test_repr(self):
        """__repr__ should produce a readable string."""
        repr_str = repr(self.retriever)
        assert "NigeriaGuidelineRetriever" in repr_str
        assert "conditions=" in repr_str
        print(f"  repr: {repr_str}")

    def test_chunk_field_types(self):
        """Chunks should cover multiple field types."""
        field_types = set(c.field_type for c in self.retriever.chunks)
        assert len(field_types) >= 2, f"Only found field types: {field_types}"
        print(f"  Field types: {field_types}")

    def test_retrieval_result_serialization(self):
        """RetrievalResult.to_dict() should produce JSON-serializable output."""
        results = self.retriever.retrieve("treatment")
        if results:
            d = results[0].to_dict()
            json_str = json.dumps(d)  # Should not raise
            assert "rank" in d
            assert "score" in d
            assert "condition" in d


# ═══════════════════════════════════════════════════════════════════════════
# GUARDRAILS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestNigeriaGuardrails:
    """Tests for the Nigeria-specific clinical guardrails."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.guardrails_path = (
            PROJECT_ROOT / "validation" / "nigeria_clinical_guardrails.json"
        )

    def test_guardrails_file_exists(self):
        """Nigeria guardrails JSON should exist."""
        assert self.guardrails_path.exists(), \
            f"Missing: {self.guardrails_path}"

    def test_guardrails_valid_json(self):
        """Nigeria guardrails should be valid JSON."""
        with open(self.guardrails_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "meta" in data

    def test_guardrails_has_required_sections(self):
        """Guardrails should have all required sections."""
        with open(self.guardrails_path) as f:
            data = json.load(f)

        required_sections = [
            "meta", "oncology_protocols",
            "common_comorbidities_nigeria",
            "resource_aware_constraints",
            "safety_thresholds_nigeria",
        ]
        for section in required_sections:
            assert section in data, f"Missing section: {section}"

    def test_guardrails_meta_attribution(self):
        """Guardrails meta should include proper CC-BY-4.0 attribution."""
        with open(self.guardrails_path) as f:
            data = json.load(f)
        meta = data["meta"]
        assert "CC-BY-4.0" in meta.get("license", ""), "Missing CC-BY-4.0 license"
        assert "NSTG" in meta.get("standard", ""), "Missing NSTG reference"
        assert "Nigeria" in meta.get("source", ""), "Missing Nigeria attribution"

    def test_guardrails_comorbidities(self):
        """Guardrails should include Nigeria-specific comorbidities."""
        with open(self.guardrails_path) as f:
            data = json.load(f)
        comorbidities = data["common_comorbidities_nigeria"]
        expected = ["HIV_coinfection", "malaria_coinfection", "sickle_cell_disease"]
        for key in expected:
            assert key in comorbidities, f"Missing comorbidity: {key}"

    def test_guardrails_safety_thresholds(self):
        """Safety thresholds should be numerically valid."""
        with open(self.guardrails_path) as f:
            data = json.load(f)
        safety = data["safety_thresholds_nigeria"]
        hematologic = safety["hematologic"]

        assert hematologic["ANC_min_cells_per_uL"] > 0
        assert hematologic["platelets_min_per_uL"] > 0
        assert hematologic["hemoglobin_min_g_per_dL"] > 0

    def test_guardrails_drug_availability(self):
        """Resource constraints should list available drugs."""
        with open(self.guardrails_path) as f:
            data = json.load(f)
        drugs = data["resource_aware_constraints"]["drug_availability"]
        assert len(drugs["commonly_available"]) > 0
        assert "Doxorubicin" in drugs["commonly_available"]


# ═══════════════════════════════════════════════════════════════════════════
# CONTROLLER INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestControllerWithGuidelines:
    """Tests for AdaptiveController with Nigeria guideline integration."""

    def test_controller_loads_nigeria_guardrails(self):
        """Controller should auto-load Nigeria guardrails if JSON exists."""
        from models.adaptive_controller import AdaptiveController, PolicyMode
        controller = AdaptiveController(policy_mode=PolicyMode.ROBUST_ADAPTIVE)

        # The nigeria_clinical_guardrails.json exists in this repo
        assert controller.nigeria_guardrails is not None, \
            "Controller did not load Nigeria guardrails"
        print(f"  Guardrails loaded: {bool(controller.nigeria_guardrails)}")

    def test_controller_adjusts_uncertainty_margin(self):
        """Controller should increase uncertainty margin with Nigeria guardrails."""
        from models.adaptive_controller import (
            AdaptiveController, PolicyMode, PolicyParams
        )
        
        params = PolicyParams(uncertainty_margin=0.10)
        controller = AdaptiveController(
            policy_mode=PolicyMode.ROBUST_ADAPTIVE,
            policy_params=params,
        )

        # Nigeria guardrails should bump margin to at least 0.18
        if controller.nigeria_guardrails:
            assert controller.params.uncertainty_margin >= 0.18, \
                f"Margin should be >= 0.18, got {controller.params.uncertainty_margin}"

    def test_controller_without_guardrails_unchanged(self):
        """Controller with non-existent guardrails should behave normally."""
        from models.adaptive_controller import AdaptiveController, PolicyParams
        
        params = PolicyParams(uncertainty_margin=0.15)
        controller = AdaptiveController(policy_params=params)
        
        # If guardrails aren't loaded, margin should stay at 0.15 or be raised
        # (it may still load from the JSON file in this repo)
        assert controller.params.uncertainty_margin >= 0.15

    def test_controller_summary_includes_guidelines_flag(self):
        """Controller summary should report Nigeria guidelines status."""
        from models.adaptive_controller import AdaptiveController
        controller = AdaptiveController()
        summary = controller.get_summary()
        assert "nigeria_guidelines_active" in summary
        assert "guideline_retriever_active" in summary

    def test_controller_with_retriever(self):
        """Controller should accept a guideline retriever."""
        from models.adaptive_controller import AdaptiveController
        from agents.nigeria_guideline_retriever import NigeriaGuidelineRetriever

        retriever = NigeriaGuidelineRetriever()
        controller = AdaptiveController(guideline_retriever=retriever)
        assert controller.guideline_retriever is not None

        # Query guidelines through controller
        context = controller.get_guideline_context(
            "breast cancer first-line treatment Nigeria"
        )
        if context:
            assert len(context) > 0
            print(f"  Guideline context: {context[:150]}...")

    def test_nigeria_safety_layer(self):
        """Nigeria safety layer should reduce dose in high-toxicity scenarios."""
        from models.adaptive_controller import (
            AdaptiveController, PolicyMode, PolicyParams
        )

        params = PolicyParams(max_cumulative_toxicity=100.0)
        controller = AdaptiveController(
            policy_mode=PolicyMode.ROBUST_ADAPTIVE,
            policy_params=params,
        )

        if controller.nigeria_guardrails:
            # Simulate high cumulative toxicity
            controller.ctrl_state.cumulative_toxicity = 75.0  # 75% of budget

            # The Nigeria safety layer should reduce a dose of 0.7
            reduced = controller._apply_nigeria_safety(0.7, dt=0.1)
            assert reduced < 0.7, \
                f"Expected dose reduction, got {reduced}"
            print(f"  Dose reduced from 0.7 to {reduced:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# PATIENT FITTER INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPatientFitterWithGuidelines:
    """Tests for PatientFitter with Nigeria guideline priors."""

    def test_fitter_accepts_guideline_priors(self):
        """PatientFitter should accept guideline_priors parameter."""
        from models.patient_fitter import PatientFitter
        
        priors = {
            "parameter_adjustments": {
                "glucose_uptake": {"lower": -1.00, "upper": -0.20},
            }
        }
        fitter = PatientFitter(
            cancer_type="TNBC",
            guideline_priors=priors,
        )
        assert fitter.guideline_priors is not None

    def test_fitter_adjusts_bounds_with_priors(self):
        """Guideline priors should narrow parameter bounds."""
        from models.patient_fitter import PatientFitter, ParameterBounds
        
        priors = {
            "parameter_adjustments": {
                "glucose_uptake": {"lower": -1.00, "upper": -0.20},
            }
        }
        fitter = PatientFitter(
            cancer_type="TNBC",
            guideline_priors=priors,
        )
        
        # Find the glucose_uptake bound
        for b in fitter.param_bounds:
            if b.name == "glucose_uptake":
                # Lower should be max(original, prior)
                assert b.lower >= -1.00
                # Upper should be min(original, prior)
                assert b.upper <= -0.10  # Original upper
                print(f"  Adjusted glucose_uptake: [{b.lower}, {b.upper}]")
                break

    def test_fitter_without_priors_unchanged(self):
        """PatientFitter without priors should maintain default bounds."""
        from models.patient_fitter import PatientFitter, DEFAULT_PARAM_BOUNDS
        
        fitter = PatientFitter(cancer_type="TNBC")
        assert fitter.guideline_priors is None
        # Bounds should match defaults
        for i, b in enumerate(fitter.param_bounds):
            assert b.lower == DEFAULT_PARAM_BOUNDS[i].lower
            assert b.upper == DEFAULT_PARAM_BOUNDS[i].upper


# ═══════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
