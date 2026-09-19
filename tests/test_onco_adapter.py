from __future__ import annotations

from pathlib import Path

import pytest

from confluence.onco.attribution import ATTRIBUTION, MissingAttributionError, assert_attribution
from confluence.onco.client import OncoClient
from confluence.onco.mapper import bind, classify_legacy_gene_map_row, load_bindings, wired_claims
from confluence.onco.schemas import (
    EvidenceObject,
    OncoRef,
    ParameterObject,
    refuse_knowledge_as_parameter,
)

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "onco" / "fixtures"


def test_attribution_required():
    assert_attribution(ATTRIBUTION)
    with pytest.raises(MissingAttributionError):
        assert_attribution("no licence line here")


def test_bind_ldha_and_tnbc():
    hits = bind("ldha")
    assert hits
    assert hits[0].annotate == "p_lactate"
    assert "L" in hits[0].states
    assert hits[0].u_slot is None

    disease = bind("tnbc")
    assert disease
    assert disease[0].onco_kind == "cancer"


def test_wired_claims_are_not_auto_supported():
    rows = wired_claims()
    assert len(rows) >= 10
    assert {row["id"] for row in rows} >= {"E-L-switch", "E-L-exh", "E-MCT1-clear"}
    for row in rows:
        assert row["status"] == "wired"
        assert row.get("evidence_level") in {"concept", "cell_line", "clinical"}


def test_evidence_object_defaults_not_promoted():
    ev = EvidenceObject(
        id="E-demo",
        claim="demo",
        mechanism_family="warburg",
        falsifier="if the sweep does not move L",
    )
    assert ev.promoted_to_rhs is False
    assert ev.status == "candidate"
    assert "OnCo" in ev.attribution


def test_client_reads_local_fixtures():
    client = OncoClient(api_root=str(FIXTURES))
    meta = client.get_meta()
    assert meta["buildDate"] == "2026-09-18"
    entity = client.get_entity("ldha")
    ref = client.as_ref(entity, meta)
    assert ref.onco_id == "ldha"
    assert ref.kind == "target"
    assert ref.attribution == ATTRIBUTION


def test_bindings_file_loads():
    data = load_bindings()
    assert "targets" in data
    assert data["attribution"].startswith("Data from OnCo")


def test_onco_page_cannot_become_a_parameter():
    ref = OncoRef(
        onco_id="ldha",
        kind="target",
        name="LDHA",
        retrieved_at="2026-09-18T00:00:00Z",
    )
    refused = refuse_knowledge_as_parameter(ref, "p_lactate")
    assert refused.provenance == "forbidden"
    assert refused.identifiability == "unidentified"
    assert "knowledge" in refused.notes.lower() or "not a value" in refused.notes.lower()


def test_legacy_parameter_map_cannot_promote_knowledge():
    import json

    sliver = json.loads(
        (FIXTURES / "legacy_gene_map_sliver.json").read_text(encoding="utf-8")
    )
    row = sliver["mappings"]["LDHA"]
    tagged = classify_legacy_gene_map_row("LDHA", row)
    assert tagged["source_type"] == "legacy_mapping"
    assert tagged["provenance"] == "assumed"
    assert tagged["identifiability"] == "unidentified"
    assert tagged["legacy_parameter"] == "pyruvate_to_lactate"
    assert tagged["not_v2_symbol"] is True

    bindings = bind("ldha")
    assert bindings
    from confluence.onco.mapper import Binding

    assert all(isinstance(b, Binding) for b in bindings)
    refused = refuse_knowledge_as_parameter(
        OncoRef(onco_id="ldha", kind="target", name="LDHA", retrieved_at="2026-09-18T00:00:00Z"),
        "p_lactate",
    )
    assert refused.provenance == "forbidden"
    refused_legacy = refuse_knowledge_as_parameter(
        OncoRef(onco_id="ldha", kind="target", name="LDHA", retrieved_at="2026-09-18T00:00:00Z"),
        "pyruvate_to_lactate",
    )
    assert refused_legacy.provenance == "forbidden"


def test_assumed_parameter_is_honest():
    p = ParameterObject(
        id="P-p_lactate",
        symbol="p_lactate",
        model_id="confluence_v2_15d",
        provenance="assumed",
        identifiability="unidentified",
        point_value=0.22,
        bounds=[0.0, 2.0],
        notes="v2 default; not identified from OnCo",
    )
    assert p.layer == "parameter"
    assert p.provenance != "identified"
