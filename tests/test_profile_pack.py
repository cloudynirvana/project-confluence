"""Complex Disease Profile pack, batch builder, and HypothesisObject."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from confluence.profiles.disease_profile import (
    RESEARCH_DISCLAIMER,
    DiseaseProfile,
    refuse_ldha_onco_as_parameter,
)
from confluence.profiles.hypothesis_object import HypothesisObject, load_hypothesis
from confluence.profiles.pack import (
    DEFAULT_TABLE,
    RowRefused,
    build_pack,
    profile_from_row,
)

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "profiles" / "cases"
HYP = ROOT / "data" / "hypotheses" / "tnbc_lactate_immune_exclusion.yaml"
SCRIPT = ROOT / "scripts" / "build_disease_profile_pack.py"

REQUIRED = {
    "tnbc_metabolic_immune.json",
    "gbm_invasive_niche.json",
    "pdac_stromal_barrier.json",
    "dormant_occult.json",
}


def test_pack_table_builds_four_gated_profiles():
    profiles, refused = build_pack(DEFAULT_TABLE, include_refuse_examples=False)
    assert refused == []
    names = {name for name, _ in profiles}
    assert names == REQUIRED
    for name, profile in profiles:
        payload = profile.model_dump(mode="json")
        DiseaseProfile.model_validate(payload)
        assert payload["disclaimer"] == RESEARCH_DISCLAIMER
        assert payload["citations"]
        for hyp in profile.admitted_hypotheses:
            assert hyp.layer == "hypothesis"
            assert hyp.parameter_status == "forbidden_to_enter_theta"
            assert not refuse_ldha_onco_as_parameter(hyp.statement)
            assert "p_lactate" not in hyp.statement.lower()
        assert any(refuse_ldha_onco_as_parameter(c.statement) for c in profile.candidate_mechanisms)
        assert "OnCo confidence.probability" in profile.non_parameters


def test_batch_script_dry_run_refuses_theta_row(tmp_path):
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["accepted"] == 4
    assert data["refused_examples"]
    assert any("p_lactate" in r or "forbidden" in r or "parameter" in r for r in data["refused_examples"])


def test_ldha_poison_row_cannot_be_built():
    table = yaml.safe_load(DEFAULT_TABLE.read_text(encoding="utf-8"))
    poison = table["refuse_examples"][0]
    with pytest.raises(RowRefused):
        profile_from_row(poison)


def test_hypothesis_object_loads_named_dataset():
    raw = yaml.safe_load(HYP.read_text(encoding="utf-8"))
    obj = load_hypothesis(raw)
    assert obj.id == "H-TNBC-LAC-EXCL-001"
    assert obj.status == "proposed"
    assert "TCGA-BRCA" in obj.named_public_dataset
    assert obj.disease_profile_ref == "tnbc_metabolic_immune"
    assert "cds" in obj.non_claims.lower()
    assert obj.disclaimer == RESEARCH_DISCLAIMER


def test_pack_summary_is_scholar_safe():
    text = (CASES / "SUMMARY.md").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "in-silico" in lowered or "research" in lowered
    assert "not" in lowered and ("clinical" in lowered or "cds" in lowered)
    assert "10.3322/caac.70090" in text
    assert "personalized medicine as cds" in lowered or "not clinical decision support" in lowered
    assert "cure patients" not in lowered


def test_hypothesis_object_refuses_missing_dataset_name():
    with pytest.raises(Exception):
        HypothesisObject(
            id="H-bad",
            disease_profile_ref="tnbc_metabolic_immune",
            statement="x",
            evidence_class="review_level",
            named_public_dataset="   ",
            falsifier="y",
            non_claims="Not a medical device and not clinical CDS.",
        )
