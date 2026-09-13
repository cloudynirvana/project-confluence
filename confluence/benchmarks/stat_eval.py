"""Survival statistics: custom estimators plus optional lifelines cross-check.

p-values and HRs are never hardcoded. When ``lifelines`` is installed,
Kaplan–Meier, log-rank, and Cox PH are recomputed on the same arrays.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from confluence.benchmarks.clinical_endpoint_mapper import (
    cox_ph_binary,
    kaplan_meier,
    log_rank,
)

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import pandas as pd

    HAS_LIFELINES = True
except Exception:  # pragma: no cover - optional extra
    HAS_LIFELINES = False
    CoxPHFitter = None  # type: ignore
    KaplanMeierFitter = None  # type: ignore
    logrank_test = None  # type: ignore
    pd = None  # type: ignore


def ci_includes_one(ci95: Tuple[Optional[float], Optional[float]]) -> Optional[bool]:
    lo, hi = ci95
    if lo is None or hi is None:
        return None
    return bool(lo <= 1.0 <= hi)


def lifelines_km(times: Sequence[float], events: Sequence[float]) -> Dict[str, Any]:
    if not HAS_LIFELINES:
        return {"available": False, "reason": "lifelines not installed"}
    kmf = KaplanMeierFitter()
    kmf.fit(np.asarray(times, dtype=float), np.asarray(events, dtype=float))
    sf = kmf.survival_function_
    return {
        "available": True,
        "t": [float(x) for x in sf.index.to_list()],
        "s": [float(x) for x in sf.iloc[:, 0].to_list()],
        "median": None if kmf.median_survival_time_ is None or np.isnan(kmf.median_survival_time_) else float(kmf.median_survival_time_),
    }


def lifelines_logrank(
    t1: Sequence[float],
    e1: Sequence[float],
    t2: Sequence[float],
    e2: Sequence[float],
) -> Dict[str, Any]:
    if not HAS_LIFELINES:
        return {"available": False, "reason": "lifelines not installed"}
    res = logrank_test(
        np.asarray(t1, dtype=float),
        np.asarray(t2, dtype=float),
        event_observed_A=np.asarray(e1, dtype=float),
        event_observed_B=np.asarray(e2, dtype=float),
    )
    p = float(res.p_value)
    return {"available": True, "p": p, "test_statistic": float(res.test_statistic), "method": "lifelines.logrank_test"}


def lifelines_cox(
    times: Sequence[float],
    events: Sequence[float],
    group: Sequence[float],
) -> Dict[str, Any]:
    if not HAS_LIFELINES:
        return {"available": False, "reason": "lifelines not installed"}
    df = pd.DataFrame(
        {
            "T": np.asarray(times, dtype=float),
            "E": np.asarray(events, dtype=float),
            "g": np.asarray(group, dtype=float),
        }
    )
    if int(df["E"].sum()) < 8 or df["g"].nunique() < 2:
        return {
            "available": True,
            "inconclusive": True,
            "hr": None,
            "ci95": (None, None),
            "reason": "underpowered for Cox",
        }
    cph = CoxPHFitter()
    cph.fit(df, duration_col="T", event_col="E")
    hr = float(np.exp(cph.params_["g"]))
    ci = cph.confidence_intervals_.loc["g"]
    lo, hi = float(np.exp(ci.iloc[0])), float(np.exp(ci.iloc[1]))
    return {
        "available": True,
        "hr": hr,
        "ci95": (lo, hi),
        "ci_includes_one": bool(lo <= 1.0 <= hi),
        "inconclusive": bool(lo <= 1.0 <= hi),
        "method": "lifelines.CoxPHFitter",
    }


def evaluate_pairwise(
    t1: Sequence[float],
    e1: Sequence[float],
    t2: Sequence[float],
    e2: Sequence[float],
    *,
    min_cox_events: int = 8,
) -> Dict[str, Any]:
    """Custom KM / log-rank / Cox plus optional lifelines cross-check."""
    custom_lr = log_rank(t1, e1, t2, e2)
    times = list(t1) + list(t2)
    events = list(e1) + list(e2)
    group = [1.0] * len(list(t1)) + [0.0] * len(list(t2))
    custom_cox = cox_ph_binary(times, events, group, min_events=min_cox_events)
    if custom_cox.get("ci95"):
        custom_cox = dict(custom_cox)
        custom_cox["ci_includes_one"] = ci_includes_one(tuple(custom_cox["ci95"]))  # type: ignore[arg-type]
        if custom_cox.get("hr") is not None and custom_cox["ci_includes_one"]:
            custom_cox["no_demonstrated_difference"] = True
    ll_lr = lifelines_logrank(t1, e1, t2, e2)
    ll_cox = lifelines_cox(times, events, group)
    cross: Dict[str, Any] = {"lifelines": HAS_LIFELINES}
    if HAS_LIFELINES and not custom_lr.get("inconclusive") and ll_lr.get("available") and ll_lr.get("p") is not None:
        cross["logrank_p_abs_diff"] = abs(float(custom_lr["p"]) - float(ll_lr["p"]))
        cross["logrank_agree"] = bool(cross["logrank_p_abs_diff"] < 0.05 or (float(custom_lr["p"]) < 1e-8 and float(ll_lr["p"]) < 1e-8))
    if HAS_LIFELINES and not custom_cox.get("inconclusive") and ll_cox.get("hr") is not None:
        cross["cox_hr_rel_diff"] = abs(float(custom_cox["hr"]) - float(ll_cox["hr"])) / max(float(custom_cox["hr"]), 1e-9)
        cross["cox_agree"] = bool(cross["cox_hr_rel_diff"] < 0.25)
    return {
        "custom_logrank": custom_lr,
        "custom_cox": custom_cox,
        "lifelines_logrank": ll_lr,
        "lifelines_cox": ll_cox,
        "crosscheck": cross,
        "custom_km_arm1": kaplan_meier(t1, e1),
        "custom_km_arm2": kaplan_meier(t2, e2),
        "lifelines_km_arm1": lifelines_km(t1, e1),
        "lifelines_km_arm2": lifelines_km(t2, e2),
    }
