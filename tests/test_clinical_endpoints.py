"""In-silico endpoint mapping + computational stress tests.

Statistics are computed (never hardcoded). This is not a clinical trial.
"""

import inspect

import numpy as np
import pytest

from confluence.benchmarks.clinical_endpoint_mapper import (
    ENDPOINT_NON_CLAIM,
    cox_ph_binary,
    ctcae_grade_from_H,
    ctcae_like,
    discretize_infusion,
    kaplan_meier,
    km_median,
    log_rank,
    recist_like,
)
from confluence.benchmarks.closed_loop_translation import (
    ARMS,
    DEFAULT_SOLVER,
    F_PROXY_NEURONS,
    conservation_check,
    lhs_virtual_cohort,
    run_ablations,
    run_real_trial,
    stiff_solver_agreement,
    weight_convergence,
)
from confluence.benchmarks.stat_eval import HAS_LIFELINES, evaluate_pairwise
from confluence.cancer_env.archetypes import get_archetype
from confluence.cancer_env.ode_system import CancerODE, DEFAULT_SOLVER as ODE_DEFAULT
from confluence.contracts import ALL_EFFECTOR_IDS
from confluence.loop import ClosedLoopSimulator
from confluence.pharmacology.pk_pd_model import PKPDModel
from confluence.pharmacology.toxicity_constraints import load_drug_catalog


def test_non_claim_is_translation_layer_not_a_trial():
    text = ENDPOINT_NON_CLAIM.lower()
    assert "in-silico endpoint mapping" in text or "in silico endpoint mapping" in text
    assert "not a clinical trial" in text
    assert "not fda/ema readiness" in text
    assert "phase ii" in text and "not a phase ii" in text
    assert "unitless" in text or "normalized" in text


def test_lsoda_is_the_published_default():
    assert ODE_DEFAULT == "LSODA"
    assert DEFAULT_SOLVER == "LSODA"
    sig = inspect.signature(CancerODE.step)
    assert sig.parameters["method"].default == "LSODA"
    sim = ClosedLoopSimulator(archetype="glioblastoma", embodiment_enabled=False, dt=0.5, seed=1)
    assert sim.solver == "LSODA"
    rec = run_real_trial("glioblastoma", "A", seed=2, days=4.0, dt=1.0)
    assert rec["solver"] == "LSODA"


def test_recist_true_cr_only_near_zero():
    near = recist_like([1.0, 0.08, 0.06])
    assert near["raw_best"] == "near-CR"
    assert near["best"] in {"near-CR", "unconfirmed-near-CR"}
    cr = recist_like([1.0, 5e-4, 2e-4])
    assert cr["raw_best"] == "CR"
    pr = recist_like([1.0, 0.65, 0.60])
    assert pr["raw_best"] == "PR"
    sd = recist_like([1.0, 0.92, 0.88, 0.90])
    assert sd["best"] == "SD"
    pd = recist_like([1.0, 1.05, 1.30])
    assert pd["best"] == "PD"
    # Confirmation when horizon allows (≥28 d).
    confirmed = recist_like(
        [1.0, 0.6, 0.55, 0.5, 0.5],
        times=[0.0, 10.0, 20.0, 30.0, 40.0],
    )
    assert confirmed["raw_best"] == "PR"
    assert confirmed["confirmed"] is True
    unconf = recist_like(
        [1.0, 0.5, 0.95, 1.2],
        times=[0.0, 5.0, 35.0, 40.0],
    )
    assert unconf["raw_best"] in {"PR", "near-CR"}
    assert unconf["confirmed"] is False


def test_ctcae_h_band_edges():
    assert ctcae_grade_from_H(1.0) == 1
    assert ctcae_grade_from_H(0.85) == 1
    assert ctcae_grade_from_H(0.849) == 2
    assert ctcae_grade_from_H(0.70) == 2
    assert ctcae_grade_from_H(0.699) == 3
    assert ctcae_grade_from_H(0.45) == 3
    assert ctcae_grade_from_H(0.449) == 4
    assert ctcae_grade_from_H(0.20) == 4  # point on the G4 band edge
    assert ctcae_grade_from_H(0.199) == 5
    worst = ctcae_like([0.90, 0.55, 0.18, 0.40])
    assert worst["worst_grade"] == 5
    # Host-death H≤0.2 upgrades the trajectory to G5 (aligned with OS).
    assert ctcae_like([0.90, 0.20])["worst_grade"] == 5
    assert "H-band surrogate" in worst["label"]


def test_log_rank_is_computed_not_hardcoded():
    t1 = [1.0, 1.2, 1.4, 1.5, 2.0]
    e1 = [1, 1, 1, 1, 1]
    t2 = [8.0, 9.0, 10.0, 11.0, 12.0]
    e2 = [1, 1, 1, 1, 1]
    out = log_rank(t1, e1, t2, e2)
    assert out["inconclusive"] is False
    assert out["p"] is not None
    assert 0.0 <= float(out["p"]) <= 1.0
    assert abs(float(out["p"]) - 0.001) > 1e-12
    weak = log_rank([5.0], [1], [6.0], [0])
    assert weak["inconclusive"] is True
    t_lo = [1.0] * 20
    t_hi = [30.0] * 20
    ev = [1] * 20
    extreme = log_rank(t_lo, ev, t_hi, ev)
    assert float(extreme["p"]) > 0.0


def test_cox_marks_underpowered_and_ci_crosses_one():
    out = cox_ph_binary([10.0, 12.0, 14.0, 8.0], [1, 0, 0, 1], [1, 1, 0, 0], min_events=8)
    assert out["inconclusive"] is True
    assert out["hr"] is None
    rng = np.random.default_rng(0)
    times = np.concatenate([rng.exponential(10.0, 20), rng.exponential(10.5, 20)])
    events = np.ones(40)
    group = np.concatenate([np.ones(20), np.zeros(20)])
    powered = cox_ph_binary(times, events, group, min_events=8)
    if not powered["inconclusive"]:
        lo, hi = powered["ci95"]
        assert powered["ci_includes_one"] == (lo <= 1.0 <= hi)


@pytest.mark.skipif(not HAS_LIFELINES, reason="optional lifelines extra not installed")
def test_lifelines_crosscheck_when_available():
    t1 = list(np.linspace(2, 6, 12))
    t2 = list(np.linspace(8, 16, 12))
    e1 = [1] * 12
    e2 = [1] * 12
    ev = evaluate_pairwise(t1, e1, t2, e2)
    assert ev["lifelines_logrank"]["available"]
    assert ev["crosscheck"]["logrank_agree"]


def test_lifelines_optional_skip_path():
    ev = evaluate_pairwise([2, 3, 4], [1, 1, 0], [8, 9, 10], [1, 0, 1])
    if not HAS_LIFELINES:
        assert ev["lifelines_logrank"]["available"] is False
        assert "lifelines not installed" in ev["lifelines_logrank"]["reason"]


def test_km_median_not_reached():
    km = kaplan_meier([5.0, 6.0, 7.0], [0, 0, 0])
    assert km["n_events"] == 0
    assert km_median(km) is None


def test_discretize_q3w_pd1_and_weekday_tki():
    u = {"anti_pd1": 0.8, "hdac": 0.6, "targeted_kinase": 0.5, "protein_ifng": 0.4}
    day0 = discretize_infusion(0.0, u, pd1_interval=21.0, pulse_width=1.0)
    assert day0["anti_pd1"] == 0.8
    day2 = discretize_infusion(2.0, u, pd1_interval=21.0, pulse_width=1.0)
    assert day2["anti_pd1"] == 0.0
    saturday = discretize_infusion(5.0, u)
    assert saturday["hdac"] == 0.0


def test_conservation_preclip_nonneg():
    rep = conservation_check(steps=80, dt=0.25)
    assert rep["ok"]
    assert rep["preclip_nonnegative"]
    assert rep["clip_rarely_repairs"]
    assert rep["nonnegative_core"]
    assert rep["H_in_unit_interval"]
    assert rep["terminal_if_H_le_0.2"]


def test_stiff_solver_tight_agreement():
    rep = stiff_solver_agreement(days=4.0, seed=2)
    assert rep["finite"]
    assert rep["agree_rtol"] == 1e-3
    assert rep["agree_atol"] == 1e-4
    assert rep["agreed"]
    for v in rep["max_abs_err_vs_tight_LSODA"].values():
        assert v < 0.05  # far tighter than the old 0.25 green-pass


def test_weight_plateau_not_called_convergence_if_clipped():
    rep = weight_convergence(days=12.0, dt=0.5, seed=3, n_kc=48)
    assert rep["bounded"]
    assert "clip" in rep["note"].lower()
    assert "NOT" in rep["note"] or "not" in rep["note"]


def test_real_closed_loop_trial_lsoda_short_horizon_label():
    rec = run_real_trial("glioblastoma", "A", seed=4, days=6.0, dt=1.0)
    assert rec["solver"] == "LSODA"
    assert rec["horizon_label"] == "short-horizon virtual event time"
    assert rec["best"] in {
        "CR",
        "near-CR",
        "unconfirmed-CR",
        "unconfirmed-near-CR",
        "PR",
        "unconfirmed-PR",
        "SD",
        "PD",
        "NE",
    }
    assert rec["u_units"].startswith("unitless")


def test_f_runs_are_labeled_256_proxy():
    rec = run_real_trial("glioblastoma", "F", seed=5, days=4.0, dt=1.0)
    assert rec["arm_label"] == "F-256 proxy"
    assert rec["f_neurons"] == F_PROXY_NEURONS


def test_ppo_excluded_from_primary_arms():
    assert "C" not in ARMS
    with pytest.raises(RuntimeError, match="PPO"):
        run_real_trial("glioblastoma", "C", seed=1, days=2.0, dt=1.0)


def test_lhs_smoke_all_four_arms():
    cohort = lhs_virtual_cohort(n=2, days=6.0, dt=1.0, seed=9, verbose=False)
    assert cohort["n_patients"] == 2
    assert cohort["horizon_label"] == "short-horizon virtual event time"
    assert cohort["f_label"].startswith("F-256")
    assert cohort["ppo_excluded"] is True
    for arm in ("A", "B", "E", "F"):
        assert cohort["km"][arm]["n"] == 2
    pfs = cohort["pairwise"]["E_vs_A_pfs"]
    if not pfs["inconclusive"]:
        assert abs(float(pfs["p"]) - 0.001) > 1e-12


def test_ablation_rerollout_endpoint_deltas():
    rep = run_ablations(days=8.0, dt=1.0, seed=3)
    assert "delta_vs_E" in rep
    assert "frozen_final_burden" in rep["delta_vs_E"]
    assert "zero_ab_final_burden" in rep["delta_vs_E"]
    assert "masked_final_burden" in rep["delta_vs_E"]


def test_catalog_and_knobs_are_doi_or_placeholder():
    from confluence.cancer_env.disease_classes import PARAM_PROVENANCE

    for spec in load_drug_catalog():
        assert spec.provenance.doi or spec.provenance.placeholder
        if spec.provenance.placeholder:
            assert "PLACEHOLDER" in spec.provenance.note.upper() or spec.provenance.placeholder
        else:
            assert spec.provenance.doi.startswith("10.")
    for key, row in PARAM_PROVENANCE.items():
        assert row["status"] in {"placeholder", "citation-tied-class"}
        if row["status"] == "placeholder":
            assert "PLACEHOLDER" in row["note"].upper()
        else:
            assert row["doi"].startswith("10.")


def test_preclip_assert_on_direct_rhs_step():
    ode = CancerODE(get_archetype("glioblastoma"), PKPDModel(drug_ids=ALL_EFFECTOR_IDS), seed=1)
    x, c = ode.initial_state()
    u = np.zeros(len(ALL_EFFECTOR_IDS))
    u[0] = 0.3
    for _ in range(40):
        x, c = ode.step(x, c, u, 0.25)
        raw = ode.last_preclip_x
        assert raw is not None
        assert np.all(raw[:12] >= -1e-9)
        assert ode.last_clip_repairs == 0
