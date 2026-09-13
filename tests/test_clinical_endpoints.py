"""In-silico endpoint mapping + computational stress tests.

Statistics are computed (never hardcoded). This is not a clinical trial.
"""

import numpy as np

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
    conservation_check,
    lhs_virtual_cohort,
    run_real_trial,
    stiff_solver_agreement,
    weight_convergence,
)


def test_non_claim_is_translation_layer_not_a_trial():
    text = ENDPOINT_NON_CLAIM.lower()
    assert "in-silico endpoint mapping" in text or "in silico endpoint mapping" in text
    assert "not a clinical trial" in text
    assert "not fda/ema readiness" in text
    assert "phase ii" in text and "not a phase ii" in text


def test_recist_cr_pr_sd_pd():
    cr = recist_like([1.0, 0.08, 0.06])
    assert cr["best"] == "CR"
    pr = recist_like([1.0, 0.65, 0.60])
    assert pr["best"] == "PR"
    sd = recist_like([1.0, 0.92, 0.88, 0.90])
    assert sd["best"] == "SD"
    pd = recist_like([1.0, 1.05, 1.30])
    assert pd["best"] == "PD"
    assert pd["pd"] is True


def test_ctcae_bands_from_H():
    assert ctcae_grade_from_H(0.92) == 0
    assert ctcae_grade_from_H(0.80) == 1
    assert ctcae_grade_from_H(0.60) == 2
    assert ctcae_grade_from_H(0.45) == 3
    assert ctcae_grade_from_H(0.30) == 4
    assert ctcae_grade_from_H(0.15) == 5
    worst = ctcae_like([0.90, 0.55, 0.18, 0.40])
    assert worst["worst_grade"] == 5
    assert worst["min_H"] == 0.18


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
    again = log_rank(t1, e1, t2, e2)
    assert again["p"] == out["p"]
    weak = log_rank([5.0], [1], [6.0], [0])
    assert weak["inconclusive"] is True
    assert weak["p"] is None
    # Large separation must not underflow to a fake-looking 0.0.
    t_lo = [1.0] * 20
    t_hi = [30.0] * 20
    ev = [1] * 20
    extreme = log_rank(t_lo, ev, t_hi, ev)
    assert extreme["p"] is not None
    assert float(extreme["p"]) > 0.0
    assert float(extreme["p"]) < 1e-6


def test_cox_marks_underpowered_when_few_events():
    out = cox_ph_binary(
        [10.0, 12.0, 14.0, 8.0],
        [1, 0, 0, 1],
        [1, 1, 0, 0],
        min_events=8,
    )
    assert out["inconclusive"] is True
    assert out["hr"] is None
    assert "underpowered" in str(out["reason"]).lower()


def test_cox_hr_and_ci_when_powered():
    rng = np.random.default_rng(0)
    t_ctrl = rng.exponential(8.0, size=20)
    t_trt = rng.exponential(16.0, size=20)
    times = np.concatenate([t_trt, t_ctrl])
    events = np.ones(40)
    group = np.concatenate([np.ones(20), np.zeros(20)])
    out = cox_ph_binary(times, events, group, min_events=8)
    assert out["inconclusive"] is False
    assert out["hr"] is not None
    lo, hi = out["ci95"]
    assert lo < out["hr"] < hi
    # Protective arm should not be a fake hardcoded 0.001-style p/HR.
    assert out["hr"] != 0.001


def test_km_median_not_reached():
    km = kaplan_meier([5.0, 6.0, 7.0], [0, 0, 0])
    assert km["n_events"] == 0
    assert km_median(km) is None


def test_discretize_q3w_pd1_and_weekday_tki():
    u = {"anti_pd1": 0.8, "hdac": 0.6, "targeted_kinase": 0.5, "protein_ifng": 0.4}
    day0 = discretize_infusion(0.0, u, pd1_interval=21.0, pulse_width=1.0)
    assert day0["anti_pd1"] == 0.8
    assert day0["hdac"] == 0.6
    day2 = discretize_infusion(2.0, u, pd1_interval=21.0, pulse_width=1.0)
    assert day2["anti_pd1"] == 0.0
    saturday = discretize_infusion(5.0, u)  # day-of-week 5 → holiday
    assert saturday["hdac"] == 0.0
    assert saturday["targeted_kinase"] == 0.0
    assert saturday["protein_ifng"] == 0.4  # other biologics stay continuous


def test_conservation_nonneg_H_carrying_terminal():
    rep = conservation_check(steps=80, dt=0.25)
    assert rep["ok"]
    assert rep["nonnegative_core"]
    assert rep["H_in_unit_interval"]
    assert rep["carrying_ok"]
    assert rep["terminal_if_H_le_0.2"]


def test_stiff_solver_finite_and_agrees():
    rep = stiff_solver_agreement(days=4.0, seed=2)
    assert rep["finite"]
    assert not any(np.isnan(v) or np.isinf(v) for v in rep["max_abs_err_vs_tight_LSODA"].values())
    assert rep["dt_endstate_abs_err"] is None or np.isfinite(rep["dt_endstate_abs_err"])
    assert rep["agreed"]


def test_weight_convergence_bounded():
    rep = weight_convergence(days=12.0, dt=0.5, seed=3, n_kc=48)
    assert rep["bounded"]
    assert rep["no_runaway"]
    assert rep["w_max"] is not None and rep["w_max"] < 80.0


def test_real_closed_loop_trial_is_not_a_toy_euler():
    rec = run_real_trial("glioblastoma", "A", seed=4, days=6.0, dt=0.5)
    assert rec["times"]
    assert rec["best"] in {"CR", "PR", "SD", "PD", "NE"}
    assert rec["ctcae_worst_grade"] in range(6)
    assert rec["os_time"] > 0.0
    assert np.all(np.asarray(rec["health"]) >= 0.0)
    assert np.all(np.asarray(rec["health"]) <= 1.0)
    assert np.all(np.asarray(rec["burden"]) >= -1e-9)


def test_lhs_smoke_all_four_arms():
    cohort = lhs_virtual_cohort(n=2, days=6.0, dt=0.75, seed=9, verbose=False)
    assert cohort["n_patients"] == 2
    assert cohort["n_trials"] == 8
    for arm in ("A", "B", "E", "F"):
        assert cohort["km"][arm]["n"] == 2
        assert sum(cohort["recist_counts"][arm].values()) == 2
    # Pairwise stats are computed; underpowered cohorts may be inconclusive.
    pfs = cohort["pairwise"]["E_vs_A_pfs"]
    assert "inconclusive" in pfs
    if not pfs["inconclusive"]:
        assert pfs["p"] is not None
        assert abs(float(pfs["p"]) - 0.001) > 1e-12
    cox = cohort["pairwise"]["E_vs_A_cox_os"]
    if cox["inconclusive"]:
        assert cox["hr"] is None
    else:
        lo, hi = cox["ci95"]
        assert lo <= cox["hr"] <= hi
