"""Disease Profile gates, export, and JSON Schema.

Research-only: these tests refuse LDHA/OnCo knowledge as ODE parameters.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from confluence.onco.schemas import OncoRef, refuse_knowledge_as_parameter
from confluence.profiles.disease_profile import (
    RESEARCH_DISCLAIMER,
    SCHEMA_PATH,
    AdmittedHypothesis,
    CandidateMechanism,
    Citation,
    DiseaseProfile,
    admit_hypotheses,
    build_disease_profile,
    export_disease_profile,
    filename_for,
    load_json_schema,
    refuse_ldha_onco_as_parameter,
    schema_required_fields,
)

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - optional in slim envs
    Draft202012Validator = None


def _validate_against_schema(payload: dict) -> None:
    schema = load_json_schema()
    required = schema["required"]
    for key in required:
        assert key in payload, f"missing required field {key}"
    DiseaseProfile.model_validate(payload)
    if Draft202012Validator is not None:
        Draft202012Validator(schema).validate(payload)


def test_ldha_onco_knowledge_cannot_enter_admitted_hypotheses_as_parameter():
    ref = OncoRef(
        onco_id="ldha",
        kind="target",
        name="LDHA",
        retrieved_at="2026-09-20T00:00:00Z",
    )
    refused = refuse_knowledge_as_parameter(ref, "p_lactate")
    assert refused.provenance == "forbidden"

    poison = CandidateMechanism(
        statement=(
            "Set confluence_v2_15d p_lactate from OnCo LDHA knowledge and "
            "confidence.probability / Idea maturity."
        ),
        evidence_class="knowledge",
        citation_ids=["4"],
        falsifier="Should never be admitted.",
    )
    assert refuse_ldha_onco_as_parameter(poison.statement)
    assert admit_hypotheses([poison]) == []

    profile = build_disease_profile(disease_id="tnbc", extra_candidates=[poison])
    for hyp in profile.admitted_hypotheses:
        assert hyp.layer == "hypothesis"
        assert hyp.parameter_status == "forbidden_to_enter_theta"
        assert not refuse_ldha_onco_as_parameter(hyp.statement)
        lowered = hyp.statement.lower()
        assert "p_lactate" not in lowered
        assert "confidence.probability" not in lowered
        assert "maturity" not in lowered
    assert any("p_lactate" in c.statement for c in profile.candidate_mechanisms)
    assert "OnCo confidence.probability" in profile.non_parameters
    assert "OnCo Idea maturity" in profile.non_parameters


def test_export_includes_citations_array():
    profile = build_disease_profile(disease_id="tnbc")
    payload = export_disease_profile(profile)
    assert isinstance(payload["citations"], list)
    assert len(payload["citations"]) >= 1
    for entry in payload["citations"]:
        assert "id" in entry and "text" in entry
        doi = entry.get("doi")
        if doi:
            assert doi.startswith("10.")
            assert "example" not in doi
            assert "xxxxx" not in doi.lower()
    assert payload["disclaimer"] == RESEARCH_DISCLAIMER
    assert filename_for(profile) == "disease-profile-tnbc.json"


def test_schema_validates_exported_profile():
    assert SCHEMA_PATH.is_file()
    schema = load_json_schema()
    assert schema["required"] == list(schema_required_fields())
    profile = build_disease_profile(
        role_id="family",
        role_label="Technical family member",
        disease_id="tnbc",
        profile_id="dp-tnbc-test",
        created_at="2026-09-20T00:00:00Z",
    )
    assert profile.asker_role == "patient_advocate"
    clinician = build_disease_profile(role_id="clinician", role_label="Clinician / trialist")
    assert clinician.asker_role == "clinician"
    payload = export_disease_profile(profile)
    _validate_against_schema(payload)
    roundtrip = DiseaseProfile.model_validate(payload)
    assert roundtrip.citations
    assert roundtrip.schema_version == "1.0.0"


def test_fabricated_doi_rejected():
    with pytest.raises(Exception):
        Citation(id="x", text="fake", doi="not-a-doi")


def test_admitted_hypothesis_cannot_be_a_parameter_layer():
    with pytest.raises(Exception):
        AdmittedHypothesis(
            statement="ok",
            layer="parameter",  # type: ignore[arg-type]
            evidence_class="review_level",
            falsifier="x",
            gates_passed=["Knowledge≠Evidence"],
        )


def test_thinking_lab_export_matches_python_disclaimer_and_filename():
    js = Path(__file__).resolve().parents[1] / "evidence" / "thinking.js"
    html = Path(__file__).resolve().parents[1] / "evidence" / "thinking.html"
    text = js.read_text(encoding="utf-8")
    page = html.read_text(encoding="utf-8")
    assert RESEARCH_DISCLAIMER in text
    assert "disease-profile-" in text
    assert "buildDiseaseProfile" in text
    assert "profile-export" in page
    assert "download-profile" in page


def test_schema_file_is_json():
    raw = json.loads(Path(SCHEMA_PATH).read_text(encoding="utf-8"))
    assert raw["properties"]["asker_role"]["enum"] == [
        "patient_advocate",
        "clinician",
        "researcher",
        "student",
        "other",
    ]
    assert "citations" in raw["properties"]
    assert "non_parameters" in raw["properties"]
