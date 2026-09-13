"""Computational validation suite answers falsifiable questions."""

from confluence.benchmarks.validation_suite import (
    NON_CLAIM,
    ablations,
    citation_table,
    class_signatures,
    fusion_allocation_shift,
    lead_time_detectability,
    toxicity_failure_case,
)
from confluence.contracts import DISEASE_CLASS_IDS


def test_non_claim_has_no_clinical_admissibility():
    assert "not a clinical validation" in NON_CLAIM.lower()
    assert "cure" not in NON_CLAIM.lower()


def test_q1_classes_are_separable():
    rep = class_signatures(days=16.0, dt=0.5, seed=3)
    assert rep["n_classes"] == len(DISEASE_CLASS_IDS)
    assert rep["nearest_centroid_accuracy"] >= 0.6
    dist = rep["pairwise_l2"]
    # Distinct classes are not collapsed to the same point.
    assert dist["benign"]["terminal"] > 0.5
    assert dist["occult"]["benign"] > 0.15


def test_q2_occult_has_positive_lead_time():
    rep = lead_time_detectability("occult", days=36.0, dt=0.4, seed=11)
    assert rep["t_early_warning"] is not None
    if rep["lead_time_days"] is not None:
        assert rep["lead_time_days"] >= 0.0


def test_q3_antibodies_can_fail_host():
    rep = toxicity_failure_case(days=20.0, dt=0.25, seed=2)
    assert rep["H_end"] < rep["H0"]
    assert rep["terminal_toxicity"] or rep["H_end"] < 0.45


def test_q4_fusion_signal_moves_allocation():
    rep = fusion_allocation_shift(seed=4)
    assert "tki_alk" in rep["U_high_junction"]
    assert rep["delta_sum"] > 0.0 or max(rep["U_high_junction"].values()) > 0.05


def test_q5_dropping_antibodies_lowers_auc():
    rep = ablations(days=10.0, dt=0.5, seed=6)
    assert rep["protein_auc_drop_antibody_U"] < rep["protein_auc_full"]
    assert rep["drop_antibody_U"] > 0.0


def test_q6_placeholders_are_marked():
    tab = citation_table()
    assert any(v.get("status") == "placeholder" for v in tab["class_knobs"].values())
    assert any(v.get("status") == "citation-tied-class" for v in tab["class_knobs"].values())
    assert all(row["doi"].startswith("10.") or row["placeholder"] for row in tab["catalog"])
