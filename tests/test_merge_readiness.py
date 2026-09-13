"""Merge-readiness package and the external clinical-validation gate."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_disclaimer_and_gate_docs_exist() -> None:
    assert (REPO / "DISCLAIMER.md").is_file()
    assert (REPO / "docs" / "MERGE_READINESS.md").is_file()
    assert (REPO / "docs" / "AWAITING_CLINICAL_VALIDATION.md").is_file()
    assert (REPO / ".github" / "workflows" / "pytest.yml").is_file()


def test_merge_readiness_states_research_not_product() -> None:
    text = (REPO / "docs" / "MERGE_READINESS.md").read_text(encoding="utf-8")
    assert "research codebase" in text.lower()
    assert "not a medical product" in text.lower()
    assert 'pip install -e ".[dev]"' in text
    assert "pytest" in text
    assert "DISCLAIMER" in text
    assert "F-256" in text
    assert "unitless" in text.lower()
    assert "demo_blender" in text
    assert "not a cure" in text.lower()


def test_awaiting_gate_is_external_not_self_validation() -> None:
    text = (REPO / "docs" / "AWAITING_CLINICAL_VALIDATION.md").read_text(encoding="utf-8")
    assert "IRB" in text
    assert "wet-lab" in text.lower() or "preclinical" in text.lower()
    assert "outside this repository" in text.lower() or "outside this repo" in text.lower()
    assert "does **not** validate itself as a cure" in text or "does not validate itself as a cure" in text.lower()
    assert "Phase I" in text or "trials" in text.lower()


def test_readme_install_and_no_cure_claim() -> None:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert 'pip install -e ".[dev]"' in text
    assert "docs/MERGE_READINESS.md" in text
    assert "docs/AWAITING_CLINICAL_VALIDATION.md" in text
    assert "docs/demo/blender" in text
    assert "viz complete" in text.lower()
    lowered = text.lower()
    assert "not a medical device" in lowered or "not a medical product" in lowered
    for phrase in (
        "this is a cure",
        "we cured",
        "fda-approved product",
        "phase ii result",
        "clinically validated treatment",
    ):
        assert phrase not in lowered


def test_ci_skips_slow_and_flybody() -> None:
    yml = (REPO / ".github" / "workflows" / "pytest.yml").read_text(encoding="utf-8")
    assert "not slow" in yml
    assert "not flybody" in yml
    assert 'pip install -e ".[dev]"' in yml
    assert "tests/test_clinical_endpoints.py" in yml
    assert "tests/test_merge_readiness.py" in yml
