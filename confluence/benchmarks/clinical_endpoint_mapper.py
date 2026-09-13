"""In-silico oncology endpoint mapping (computational–clinical translation layer).

Maps real Confluence closed-loop trajectories (latent burden, H, U) onto
RECIST 1.1-*like* response, CTCAE v5.0-*like* grades, Kaplan–Meier OS/PFS,
log-rank, and a univariate Cox HR. Also discretizes continuous U(t) into
simulated Q2W/Q3W anti-PD-1 pulses and daily TKI/HDAC with holidays.

This is **not** a clinical trial, not RECIST/CTCAE adjudication, not FDA/EMA
readiness, and not a Phase II readout. Endpoints are research scores for
scientific scrutiny of the ODE + controller.

Statistics are computed (Mantel–Haenszel log-rank; univariate Cox via
Newton). p-values and HRs are never hardcoded. If underpowered, the Cox
block is marked inconclusive.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm


ENDPOINT_NON_CLAIM = (
    "In-silico endpoint mapping / computational–clinical translation layer. "
    "RECIST 1.1-like and CTCAE v5.0-like labels are simulated mappings from "
    "ODE burden and host-health H. Kaplan–Meier / log-rank / Cox are computed "
    "on a virtual cohort. This is not a clinical trial, not FDA/EMA readiness, "
    "and not a Phase II result."
)

# CTCAE v5.0-*like* bands on host health H ∈ [0, 1].
# G5 matches the model's terminal toxicity threshold (H ≤ 0.2).
CTCAE_H_BANDS = (
    (0.85, 1.01, 0, "none / G0"),
    (0.70, 0.85, 1, "G1"),
    (0.55, 0.70, 2, "G2"),
    (0.40, 0.55, 3, "G3"),
    (0.20, 0.40, 4, "G4"),
    (-0.01, 0.20, 5, "G5 / terminal"),
)

Q2W_DAYS = 14.0
Q3W_DAYS = 21.0
PULSE_WIDTH_DAYS = 1.0
WEEKDAY_ON = 5  # daily TKI/HDAC: 5 days on, 2 off

PD1_LIKE = ("anti_pd1", "protein_anti_pd1")
DAILY_LIKE = ("targeted_kinase", "hdac", "tki_imatinib_like", "tki_alk", "mct1")


def recist_like(
    burden: Sequence[float],
    baseline: Optional[float] = None,
    cr_frac: float = 0.10,
    pr_frac: float = 0.30,
    pd_frac: float = 0.20,
    pd_abs: float = 0.05,
) -> Dict[str, object]:
    """Best overall response from a burden trajectory (RECIST 1.1-like).

    CR: best burden ≤ ``cr_frac`` of baseline (disappearance analog).
    PR: ≥30% decrease from baseline.
    PD: ≥20% increase over nadir and ≥ ``pd_abs`` absolute.
    SD: otherwise.
    """
    y = np.asarray(burden, dtype=float).ravel()
    if y.size == 0:
        return {"best": "NE", "nadir": None, "pct_from_baseline": None, "pd": False}
    b0 = float(baseline if baseline is not None else y[0])
    b0 = max(b0, 1e-8)
    nadir = float(np.min(y))
    best_drop = (b0 - nadir) / b0
    pd = False
    running_nadir = y[0]
    for val in y:
        running_nadir = min(running_nadir, float(val))
        if val >= running_nadir * (1.0 + pd_frac) and (val - running_nadir) >= pd_abs:
            pd = True
            break
    if nadir <= cr_frac * b0:
        best = "CR"
    elif best_drop >= pr_frac:
        best = "PR"
    elif pd:
        best = "PD"
    else:
        best = "SD"
    return {
        "best": best,
        "nadir": nadir,
        "baseline": b0,
        "pct_from_baseline": float(100.0 * (nadir - b0) / b0),
        "pd": pd,
        "non_claim": ENDPOINT_NON_CLAIM,
    }


def ctcae_grade_from_H(h: float) -> int:
    """Worst-compatible grade for a single H sample."""
    h = float(h)
    for lo, hi, grade, _name in CTCAE_H_BANDS:
        if lo <= h < hi:
            return int(grade)
    return 5 if h < 0.2 else 0


def ctcae_like(health: Sequence[float]) -> Dict[str, object]:
    """Worst CTCAE v5.0-like grade along H(t). G5 iff any H ≤ 0.2."""
    h = np.asarray(health, dtype=float).ravel()
    if h.size == 0:
        return {"worst_grade": None, "grades": []}
    grades = [ctcae_grade_from_H(v) for v in h]
    worst = int(max(grades))
    if np.any(h <= 0.2):
        worst = max(worst, 5)
    return {
        "worst_grade": worst,
        "grades": grades,
        "min_H": float(np.min(h)),
        "bands": [(lo, hi, g, n) for lo, hi, g, n in CTCAE_H_BANDS],
        "non_claim": ENDPOINT_NON_CLAIM,
    }


def pfs_os_times(
    times: Sequence[float],
    burden: Sequence[float],
    health: Sequence[float],
    baseline: Optional[float] = None,
    horizon: Optional[float] = None,
) -> Dict[str, float]:
    """Simulated OS / PFS times. OS: H≤0.2. PFS: RECIST-like PD or OS."""
    t = np.asarray(times, dtype=float).ravel()
    y = np.asarray(burden, dtype=float).ravel()
    h = np.asarray(health, dtype=float).ravel()
    n = min(t.size, y.size, h.size)
    t, y, h = t[:n], y[:n], h[:n]
    horizon = float(horizon if horizon is not None else (t[-1] if n else 0.0))
    b0 = float(baseline if baseline is not None else (y[0] if n else 1.0))
    os_t, os_e = horizon, 0.0
    for ti, hi in zip(t, h):
        if hi <= 0.2:
            os_t, os_e = float(ti), 1.0
            break
    pfs_t, pfs_e = os_t, os_e
    running_nadir = y[0] if n else b0
    for ti, yi, hi in zip(t, y, h):
        running_nadir = min(running_nadir, float(yi))
        pd = yi >= running_nadir * 1.20 and (yi - running_nadir) >= 0.05
        if hi <= 0.2 or pd:
            pfs_t, pfs_e = float(ti), 1.0
            break
    return {
        "os_time": float(os_t),
        "os_event": float(os_e),
        "pfs_time": float(pfs_t),
        "pfs_event": float(pfs_e),
        "horizon": horizon,
    }


def kaplan_meier(
    times: Sequence[float], events: Sequence[float]
) -> Dict[str, List[float]]:
    """Product-limit KM. ``events`` is 1=event, 0=censored."""
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=float)
    order = np.argsort(t, kind="mergesort")
    t, e = t[order], e[order]
    uniq = np.unique(t[e > 0.5])
    surv = 1.0
    xs = [0.0]
    ys = [1.0]
    n = t.size
    for ut in uniq:
        at_risk = float(np.sum(t >= ut))
        deaths = float(np.sum((t == ut) & (e > 0.5)))
        if at_risk <= 0:
            continue
        surv *= 1.0 - deaths / at_risk
        xs.append(float(ut))
        ys.append(float(max(surv, 0.0)))
        n = at_risk
    if xs[-1] < (float(t[-1]) if t.size else 0.0):
        xs.append(float(t[-1]))
        ys.append(ys[-1])
    return {"t": xs, "s": ys, "n": int(t.size), "n_events": int(np.sum(e > 0.5))}


def km_median(km: Dict[str, List[float]]) -> Optional[float]:
    """First time the product-limit curve is ≤ 0.5; None if not reached."""
    for ti, si in zip(km.get("t", []), km.get("s", [])):
        if float(si) <= 0.5:
            return float(ti)
    return None


def log_rank(
    t1: Sequence[float],
    e1: Sequence[float],
    t2: Sequence[float],
    e2: Sequence[float],
) -> Dict[str, object]:
    """Two-sample Mantel–Haenszel log-rank. p from N(0,1); never hardcoded."""
    a_t = np.asarray(t1, dtype=float)
    a_e = np.asarray(e1, dtype=float)
    b_t = np.asarray(t2, dtype=float)
    b_e = np.asarray(e2, dtype=float)
    event_times = np.unique(np.concatenate([a_t[a_e > 0.5], b_t[b_e > 0.5]]))
    o_e = 0.0
    var = 0.0
    n_ev = 0
    for ut in event_times:
        n1 = float(np.sum(a_t >= ut))
        n2 = float(np.sum(b_t >= ut))
        n = n1 + n2
        d1 = float(np.sum((a_t == ut) & (a_e > 0.5)))
        d2 = float(np.sum((b_t == ut) & (b_e > 0.5)))
        d = d1 + d2
        if n <= 1 or d <= 0:
            continue
        n_ev += int(d)
        exp1 = n1 * d / n
        o_e += d1 - exp1
        var += n1 * n2 * d * (n - d) / (n * n * (n - 1.0))
    if var <= 1e-18 or n_ev < 2:
        return {
            "stat": 0.0,
            "p": None,
            "n_events": n_ev,
            "inconclusive": True,
            "reason": "too few events for log-rank",
        }
    z = o_e / np.sqrt(var)
    # Survival function (not 1-cdf) so large |z| does not underflow to p=0.0.
    p = float(2.0 * norm.sf(abs(z)))
    return {
        "stat": float(z * z),
        "z": float(z),
        "p": p,
        "n_events": n_ev,
        "inconclusive": False,
        "method": "Mantel-Haenszel log-rank",
    }


def cox_ph_binary(
    times: Sequence[float],
    events: Sequence[float],
    group: Sequence[float],
    min_events: int = 8,
) -> Dict[str, object]:
    """Univariate Cox PH for a binary covariate (0/1). Newton on partial LL.

    Returns HR and 95% CI. Marks inconclusive when events are sparse.
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=float)
    x = np.asarray(group, dtype=float)
    n_ev = int(np.sum(e > 0.5))
    if n_ev < min_events or np.unique(x).size < 2:
        return {
            "hr": None,
            "ci95": (None, None),
            "beta": None,
            "n_events": n_ev,
            "inconclusive": True,
            "reason": f"underpowered for Cox (n_events={n_ev}, need ≥{min_events})",
            "method": "univariate Cox PH (partial likelihood)",
        }

    def _negll(beta: float) -> Tuple[float, float, float]:
        ll = 0.0
        g = 0.0
        h = 0.0
        order = np.argsort(-t, kind="mergesort")  # descending time for risk sets
        # Use event-time loop (Breslow ties).
        for ut in np.unique(t[e > 0.5]):
            dying = np.where((t == ut) & (e > 0.5))[0]
            risk = np.where(t >= ut)[0]
            if risk.size == 0 or dying.size == 0:
                continue
            xb = np.exp(np.clip(beta * x[risk], -20, 20))
            s0 = float(np.sum(xb))
            s1 = float(np.sum(x[risk] * xb))
            s2 = float(np.sum((x[risk] ** 2) * xb))
            for i in dying:
                ll += beta * x[i] - np.log(max(s0, 1e-15))
                g += x[i] - s1 / max(s0, 1e-15)
                h += -(s2 / max(s0, 1e-15) - (s1 / max(s0, 1e-15)) ** 2)
        return -ll, -g, max(-h, 1e-12)

    beta = 0.0
    for _ in range(25):
        nll, grad, hess = _negll(beta)
        step = grad / hess
        beta -= float(np.clip(step, -1.5, 1.5))
        if abs(step) < 1e-6:
            break
    _, _, hess = _negll(beta)
    se = float(np.sqrt(1.0 / max(hess, 1e-12)))
    hr = float(np.exp(beta))
    lo = float(np.exp(beta - 1.96 * se))
    hi = float(np.exp(beta + 1.96 * se))
    return {
        "hr": hr,
        "ci95": (lo, hi),
        "beta": float(beta),
        "se": se,
        "n_events": n_ev,
        "inconclusive": False,
        "method": "univariate Cox PH (Breslow, Newton)",
    }


def discretize_infusion(
    t_days: float,
    u: Dict[str, float],
    pd1_interval: float = Q3W_DAYS,
    pulse_width: float = PULSE_WIDTH_DAYS,
) -> Dict[str, float]:
    """Map continuous U → simulated Q2W/Q3W PD-1 pulses + 5/2 daily TKI/HDAC.

    Documented as a **simulated regimen**, not a labeled schedule.
    """
    out = {k: float(v) for k, v in u.items()}
    cycle = float(t_days) % pd1_interval
    pd1_on = cycle < pulse_width
    dow = int(np.floor(t_days)) % 7
    daily_on = dow < WEEKDAY_ON
    for key in list(out):
        if key in PD1_LIKE or key.endswith("anti_pd1"):
            out[key] = out[key] if pd1_on else 0.0
        elif key in DAILY_LIKE:
            out[key] = out[key] if daily_on else 0.0
    return out


REGIMEN_DOC = {
    "anti_pd1 / protein_anti_pd1": f"Q3W pulse ({Q3W_DAYS:.0f} d), width {PULSE_WIDTH_DAYS:.0f} d — pembrolizumab-class analog, simulated",
    "targeted_kinase / tki_* / hdac / mct1": "daily 5 days on / 2 days holiday — simulated oral TKI/HDAC",
    "other biologics": "left continuous (infusion analog) unless overridden",
    "non_claim": ENDPOINT_NON_CLAIM,
}


def summarize_arm(
    times: Sequence[float],
    burden: Sequence[float],
    health: Sequence[float],
) -> Dict[str, object]:
    rec = recist_like(burden)
    ctc = ctcae_like(health)
    surv = pfs_os_times(times, burden, health)
    return {**rec, **{f"ctcae_{k}": v for k, v in ctc.items() if k != "grades"}, **surv}
