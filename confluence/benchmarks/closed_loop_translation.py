"""Real closed-loop computational–clinical translation (not a toy Euler script).

Every trajectory comes from ``ClosedLoopSimulator`` → ``CancerODE.step`` /
``integrate`` (scipy LSODA/Radau/RK45) + observation layer + PK/PD +
controllers A / B / E / F.

    python -m confluence.benchmarks.closed_loop_translation

Default: master seed 17, N=100, horizon 180 d, LSODA, identical noise
across A / B / E / F-256 proxy. Writes JSON + 4-panel + plain-text report.

PPO/C is excluded from primary figures (untrained stub).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import qmc

from confluence.benchmarks.clinical_endpoint_mapper import (
    ENDPOINT_NON_CLAIM,
    OS_PFS_MIN_HORIZON_DAYS,
    REGIMEN_DOC,
    discretize_infusion,
    kaplan_meier,
    km_median,
    summarize_arm,
)
from confluence.benchmarks.stat_eval import evaluate_pairwise
from confluence.cancer_env.archetypes import get_archetype
from confluence.cancer_env.ode_system import DEFAULT_SOLVER, CancerODE
from confluence.contracts import ALL_EFFECTOR_IDS, PROTEIN_CHANNEL_IDS, InterventionAction, ObservationRecord
from confluence.controllers import make_controller
from confluence.controllers.base import BaseController, ControllerContext
from confluence.loop import ClosedLoopSimulator
from confluence.pharmacology.pk_pd_model import PKPDModel


ARMS = ("A", "B", "E", "F")  # PPO/C excluded — untrained stub
F_PROXY_NEURONS = 256
F_FULL_NEURONS = 166700
MASTER_SEED = 17
DEFAULT_DAYS = 180.0
DEFAULT_DT = 2.0
DEFAULT_N = 100
RECIST_KEYS = ("CR", "near-CR", "unconfirmed-CR", "unconfirmed-near-CR", "PR", "unconfirmed-PR", "SD", "PD", "NE")

# Stiff-agreement gates (relative + absolute). Do not green-pass at 0.25/0.35.
STIFF_RTOL = 1e-3
STIFF_ATOL = 1e-4


class DiscretizedController(BaseController):
    """Wrap A/B/E/F and pulse U onto a simulated clinical calendar."""

    def __init__(self, inner: BaseController, pd1_interval: float = 21.0):
        super().__init__(drug_ids=inner.drug_ids)
        self.inner = inner
        self.pd1_interval = pd1_interval
        self.letter = getattr(inner, "letter", "?") + "Δ"
        self.name = getattr(inner, "name", "wrapped") + " (discretized)"
        self.uses_connectome = getattr(inner, "uses_connectome", False)

    def reset(self) -> None:
        if hasattr(self.inner, "reset"):
            self.inner.reset()

    def decide(self, observation: ObservationRecord, context: ControllerContext) -> InterventionAction:
        raw = self.inner.decide(observation, context)
        pulsed = discretize_infusion(observation.t, raw.infusion, pd1_interval=self.pd1_interval)
        if getattr(self, "zero_antibody", False):
            for key in PROTEIN_CHANNEL_IDS:
                if key in pulsed:
                    pulsed[key] = 0.0
        return raw.model_copy(
            update={
                "infusion": pulsed,
                "notes": (raw.notes or "") + " · simulated Q3W PD-1 / 5-2 TKI · U unitless [0,1]",
            }
        )

    def connectome_telemetry(self):
        if hasattr(self.inner, "connectome_telemetry"):
            return self.inner.connectome_telemetry()
        return None


def _controller(
    name: str,
    seed: int,
    discretize: bool = True,
    n_kc: int = 64,
    *,
    freeze_plasticity: bool = False,
    f_neurons: int = F_PROXY_NEURONS,
    zero_antibody: bool = False,
):
    if name.upper() == "C":
        raise RuntimeError("PPO stub (C) is excluded from primary A/B/E/F figures unless trained")
    kwargs: Dict[str, Any] = {}
    if name in {"D", "E"}:
        kwargs = {"n_kc": n_kc, "seed": seed, "plastic": not freeze_plasticity}
    if name == "F":
        kwargs = {"n_neurons": int(f_neurons), "seed": seed}
    inner = make_controller(name, **kwargs)
    if name == "F" and int(f_neurons) < F_FULL_NEURONS:
        inner.letter = "F-256"
        inner.name = f"F-{int(f_neurons)} proxy (not {F_FULL_NEURONS})"
    if freeze_plasticity and hasattr(inner, "network"):
        inner.network.config.plastic = False
        if hasattr(inner.network, "plasticity"):
            inner.network.plasticity.params.eta = 0.0
    wrapped = DiscretizedController(inner) if discretize else inner
    if discretize:
        wrapped.zero_antibody = zero_antibody
    return wrapped


def run_real_trial(
    archetype: str,
    controller: str,
    seed: int,
    days: float = DEFAULT_DAYS,
    dt: float = DEFAULT_DT,
    discretize: bool = True,
    param_scale: Optional[Dict[str, float]] = None,
    half_life_scale: float = 1.0,
    solver: str = DEFAULT_SOLVER,
    freeze_plasticity: bool = False,
    mask_fusion_y: bool = False,
    zero_antibody: bool = False,
    f_neurons: int = F_PROXY_NEURONS,
) -> Dict[str, Any]:
    """One virtual patient on the real closed loop (embodiment off, LSODA)."""
    params = get_archetype(archetype)
    if param_scale:
        for key, fac in param_scale.items():
            if hasattr(params, key):
                setattr(params, key, float(getattr(params, key)) * float(fac))
    pk = PKPDModel(drug_ids=ALL_EFFECTOR_IDS)
    if abs(half_life_scale - 1.0) > 1e-9:
        pk.k_el = pk.k_el / float(half_life_scale)
    ctrl = _controller(
        controller,
        seed,
        discretize=discretize,
        freeze_plasticity=freeze_plasticity,
        f_neurons=f_neurons,
        zero_antibody=zero_antibody,
    )
    sim = ClosedLoopSimulator(
        archetype=archetype,
        controller=ctrl,
        dt=dt,
        seed=seed,
        embodiment_enabled=False,
        solver=solver,
    )
    sim.params = params
    sim.pk = pk
    sim.ode = CancerODE(params, pk, seed=seed)
    sim.x, sim.c = sim.ode.initial_state()
    sim.observer.set_params(params)
    if mask_fusion_y:
        inner_obs = sim.observer.observe

        def _masked(state, noisy: bool = True):
            rec = inner_obs(state, noisy=noisy)
            return rec.model_copy(
                update={
                    "fusion_allele_fraction": 0.0,
                    "junction_neoantigen": 0.0,
                }
            )

        sim.observer.observe = _masked  # type: ignore[method-assign]
    times, burdens, health, wnorms = [], [], [], []
    baseline = None
    for _ in range(int(round(days / dt))):
        frame = sim.step(run_cancer=True, run_embodiment=False)
        if baseline is None:
            baseline = frame.latent.tumor_burden
        times.append(frame.t)
        burdens.append(frame.latent.tumor_burden)
        health.append(frame.latent.H)
        tel = frame.connectome or {}
        if tel.get("plasticity_norm") is not None:
            wnorms.append(float(tel["plasticity_norm"]))
        if frame.terminal:
            break
    rec = summarize_arm(times, burdens, health)
    rec.update(
        {
            "archetype": archetype,
            "controller": controller,
            "seed": seed,
            "times": times,
            "burden": burdens,
            "health": health,
            "wnorms": wnorms,
            "baseline": baseline,
            "solver": solver,
            "arm_label": "F-256 proxy" if controller == "F" and f_neurons < F_FULL_NEURONS else controller,
            "f_neurons": f_neurons if controller == "F" else None,
            "freeze_plasticity": freeze_plasticity,
            "mask_fusion_y": mask_fusion_y,
            "zero_antibody": zero_antibody,
            "u_units": "unitless normalized [0, 1] (not mg/kg)",
        }
    )
    return rec


def stiff_solver_agreement(
    archetype: str = "glioblastoma",
    days: float = 8.0,
    seed: int = 1,
) -> Dict[str, Any]:
    """LSODA / Radau / RK45 + tight vs loose tol on the real CancerODE RHS."""
    ode = CancerODE(get_archetype(archetype), PKPDModel(drug_ids=ALL_EFFECTOR_IDS), seed=seed)
    x0, c0 = ode.initial_state()
    u = np.zeros(len(ALL_EFFECTOR_IDS))
    u[0] = 0.35

    def u_of_t(_t):
        return u

    methods = ("LSODA", "Radau", "RK45")
    tols = ((1e-4, 1e-6), (1e-8, 1e-10))
    traj: Dict[str, np.ndarray] = {}
    finite = True
    for method in methods:
        for rtol, atol in tols:
            key = f"{method}_rtol{rtol:g}"
            out = ode.integrate(
                x0, c0, u_of_t, (0.0, days), n_eval=80, method=method, rtol=rtol, atol=atol, events=False
            )
            traj[key] = out["x"]
            finite = finite and bool(out["success"] and out["finite"] and np.all(np.isfinite(out["x"])))
    # Effective Δt ∈ {0.001, 0.05} via max_step (same CancerODE.rhs).
    dts = (0.001, 0.05)
    ends = {}
    for dt in dts:
        out = ode.integrate(
            x0, c0, u_of_t, (0.0, days), n_eval=80, method="LSODA", max_step=dt, events=False
        )
        ends[str(dt)] = out["x"][:, -1].copy()
        finite = finite and bool(out["success"] and out["finite"] and np.all(np.isfinite(out["x"])))
    ref = traj.get("LSODA_rtol1e-08")
    max_abs: Dict[str, float] = {}
    max_rel: Dict[str, float] = {}
    mixed_ok = True
    if ref is not None:
        for key, arr in traj.items():
            n = min(ref.shape[1], arr.shape[1])
            a, b = arr[:, :n], ref[:, :n]
            abs_e = float(np.max(np.abs(a - b)))
            rel_e = float(np.max(np.abs(a - b) / (STIFF_ATOL + STIFF_RTOL * np.abs(b))))
            max_abs[key] = abs_e
            max_rel[key] = rel_e
            mixed_ok = mixed_ok and bool(np.all(np.abs(a - b) <= STIFF_ATOL + STIFF_RTOL * np.abs(b)))
    dt_abs = dt_rel = None
    dt_ok = True
    if all(k in ends for k in ("0.001", "0.05")):
        a, b = ends["0.001"], ends["0.05"]
        dt_abs = float(np.max(np.abs(a - b)))
        dt_rel = float(np.max(np.abs(a - b) / (STIFF_ATOL + STIFF_RTOL * np.abs(b))))
        dt_ok = bool(np.all(np.abs(a - b) <= STIFF_ATOL + STIFF_RTOL * np.abs(b)))
    return {
        "finite": finite,
        "rhs": "CancerODE.rhs (no shadow vector field)",
        "max_abs_err_vs_tight_LSODA": max_abs,
        "max_mixed_rel_err_vs_tight_LSODA": max_rel,
        "dt_endstate_abs_err": dt_abs,
        "dt_endstate_mixed_rel_err": dt_rel,
        "agree_rtol": STIFF_RTOL,
        "agree_atol": STIFF_ATOL,
        "agreed": bool(finite and mixed_ok and dt_ok),
        "non_claim": ENDPOINT_NON_CLAIM,
    }


def conservation_check(archetype: str = "glioblastoma", steps: int = 200, dt: float = 0.2) -> Dict[str, Any]:
    ode = CancerODE(get_archetype(archetype), PKPDModel(drug_ids=ALL_EFFECTOR_IDS), seed=3)
    x, c = ode.initial_state()
    u = np.array([0.2, 0.1, 0.05, 0.05, 0.2] + [0.0] * (len(ALL_EFFECTOR_IDS) - 5))
    ok = True
    terminal_implies = True
    k = float(ode.params.k_carry)
    repairs = 0
    preclip_neg = 0
    for _ in range(steps):
        x, c = ode.step(x, c, u, dt, method=DEFAULT_SOLVER)
        raw = ode.last_preclip_x
        repairs += int(ode.last_clip_repairs)
        if raw is not None:
            if np.any(raw[:12] < -1e-9):
                preclip_neg += 1
                ok = False
                break
        if not np.all(np.isfinite(x)) or np.any(x[:12] < -1e-9):
            ok = False
            break
        if not (0.0 <= x[10] <= 1.0):
            ok = False
            break
        burden = float(x[0] + x[1] + x[11])
        if burden > k * 1.35:
            ok = False
            break
        if x[10] <= 0.2:
            terminal_implies = True
            break
    return {
        "ok": ok,
        "H_in_unit_interval": bool(0.0 <= float(x[10]) <= 1.0),
        "nonnegative_core": bool(np.all(x[:12] >= -1e-9)),
        "preclip_nonnegative": preclip_neg == 0,
        "clip_repairs_total": repairs,
        "clip_rarely_repairs": repairs <= max(1, steps // 20),
        "carrying_ok": bool(float(x[0] + x[1] + x[11]) <= k * 1.35),
        "terminal_if_H_le_0.2": bool(x[10] > 0.2 or terminal_implies),
        "non_claim": ENDPOINT_NON_CLAIM,
    }


def _arm_distribution(items: Sequence[Dict[str, Any]], km_os: Dict[str, Any], km_pfs: Dict[str, Any]) -> Dict[str, Any]:
    """Controller performance distribution (not a single cherry-picked seed)."""
    if not items:
        return {"n": 0}
    finals = np.array([r["burden"][-1] for r in items if r.get("burden")], dtype=float)
    min_h = np.array([min(r["health"]) for r in items if r.get("health")], dtype=float)
    os_t = np.array([r["os_time"] for r in items], dtype=float)
    return {
        "n": len(items),
        "os_event_rate": float(np.mean([r["os_event"] for r in items])),
        "pfs_event_rate": float(np.mean([r["pfs_event"] for r in items])),
        "km_os_median": km_median(km_os),
        "km_pfs_median": km_median(km_pfs),
        "os_time_median_incl_censor": float(np.median(os_t)),
        "final_burden_median": float(np.median(finals)) if finals.size else None,
        "final_burden_iqr": (
            [float(np.percentile(finals, 25)), float(np.percentile(finals, 75))]
            if finals.size
            else None
        ),
        "min_H_median": float(np.median(min_h)) if min_h.size else None,
        "min_H_iqr": (
            [float(np.percentile(min_h, 25)), float(np.percentile(min_h, 75))]
            if min_h.size
            else None
        ),
    }


def lhs_virtual_cohort(
    n: int = DEFAULT_N,
    archetype: str = "glioblastoma",
    days: float = DEFAULT_DAYS,
    dt: float = DEFAULT_DT,
    seed: int = MASTER_SEED,
    discretize: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """LHS over r, σ_I (κ_immune), t½ (±25–40%). Each virtual patient × A/B/E/F."""
    rng = np.random.default_rng(seed)
    samp = qmc.LatinHypercube(d=3, seed=seed).random(n)
    # Map [0,1] → [0.60, 1.40] (±40%) for r, σ_I, t½.
    scales = 0.60 + 0.80 * samp
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        if verbose and (i == 0 or (i + 1) % 10 == 0 or i + 1 == n):
            print(f"  LHS virtual patient {i + 1}/{n} × {len(ARMS)} arms", flush=True)
        patient_seed = int(rng.integers(1, 10_000))
        scale = {
            "r_s": float(scales[i, 0]),
            "kappa_immune": float(scales[i, 1]),
            "rho_immune": float(scales[i, 1]),
        }
        t_half = float(scales[i, 2])
        for arm in ARMS:
            rec = run_real_trial(
                archetype,
                arm,
                seed=patient_seed,
                days=days,
                dt=dt,
                discretize=discretize,
                param_scale=scale,
                half_life_scale=t_half,
            )
            rec["patient_id"] = i
            rec["lhs_r_s"] = scale["r_s"]
            rec["lhs_sigma_I"] = scale["kappa_immune"]
            rec["lhs_t_half"] = t_half
            rows.append(rec)
    by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ARMS}
    for rec in rows:
        by_arm[rec["controller"]].append(rec)
    km = {}
    recist_counts = {}
    ctcae_counts = {}
    for arm, items in by_arm.items():
        os_t = [r["os_time"] for r in items]
        os_e = [r["os_event"] for r in items]
        pfs_t = [r["pfs_time"] for r in items]
        pfs_e = [r["pfs_event"] for r in items]
        km[arm] = {
            "os": kaplan_meier(os_t, os_e),
            "pfs": kaplan_meier(pfs_t, pfs_e),
            "n": len(items),
        }
        recist_counts[arm] = {k: 0 for k in RECIST_KEYS}
        ctcae_counts[arm] = {str(g): 0 for g in range(6)}
        for r in items:
            recist_counts[arm][str(r.get("best") or "NE")] += 1
            ctcae_counts[arm][str(int(r.get("ctcae_worst_grade") or 0))] += 1
    pairwise = {}
    ref = by_arm["A"]
    for arm in ("B", "E", "F"):
        oth = by_arm[arm]
        ev = evaluate_pairwise(
            [r["os_time"] for r in oth],
            [r["os_event"] for r in oth],
            [r["os_time"] for r in ref],
            [r["os_event"] for r in ref],
        )
        pairwise[f"{arm}_vs_A_os"] = ev["custom_logrank"]
        pairwise[f"{arm}_vs_A_cox_os"] = ev["custom_cox"]
        pairwise[f"{arm}_vs_A_os_stat"] = ev
        ev_p = evaluate_pairwise(
            [r["pfs_time"] for r in oth],
            [r["pfs_event"] for r in oth],
            [r["pfs_time"] for r in ref],
            [r["pfs_event"] for r in ref],
        )
        pairwise[f"{arm}_vs_A_pfs"] = ev_p["custom_logrank"]
        pairwise[f"{arm}_vs_A_pfs_stat"] = ev_p
    # Example trajectory = median-burden E patient (or first).
    example = by_arm["E"][0] if by_arm["E"] else rows[0]
    if by_arm["E"]:
        e_final = [r["burden"][-1] for r in by_arm["E"] if r.get("burden")]
        if e_final:
            target = float(np.median(e_final))
            example = min(by_arm["E"], key=lambda r: abs(r["burden"][-1] - target) if r.get("burden") else 1e9)
    distribution = {
        arm: _arm_distribution(by_arm[arm], km[arm]["os"], km[arm]["pfs"]) for arm in ARMS if by_arm[arm]
    }
    return {
        "n_patients": n,
        "n_trials": len(rows),
        "design": "LHS virtual patients × arms A/B/E/F (same draw on every controller)",
        "n": n,
        "horizon_label": (
            "OS/PFS-mapped virtual event time"
            if days + 1e-9 >= OS_PFS_MIN_HORIZON_DAYS
            else "short-horizon virtual event time"
        ),
        "ppo_excluded": True,
        "f_label": f"F-{F_PROXY_NEURONS} proxy (not {F_FULL_NEURONS})",
        "solver": DEFAULT_SOLVER,
        "master_seed": seed,
        "archetype": archetype,
        "days": days,
        "km": km,
        "distribution": distribution,
        "recist_counts": recist_counts,
        "ctcae_counts": ctcae_counts,
        "pairwise": pairwise,
        "regimen": REGIMEN_DOC,
        "example": {
            "controller": example["controller"],
            "times": example["times"],
            "burden": example["burden"],
            "health": example["health"],
        },
        "rows_light": [
            {
                "controller": r["controller"],
                "best": r["best"],
                "ctcae_worst_grade": r.get("ctcae_worst_grade"),
                "os_time": r["os_time"],
                "os_event": r["os_event"],
                "pfs_time": r["pfs_time"],
                "pfs_event": r["pfs_event"],
                "patient_id": r.get("patient_id"),
                "lhs_r_s": r.get("lhs_r_s"),
                "lhs_sigma_I": r.get("lhs_sigma_I"),
                "lhs_t_half": r.get("lhs_t_half"),
            }
            for r in rows
        ],
        "non_claim": ENDPOINT_NON_CLAIM,
    }


def full_brain_166k_smoke(days: float = 2.0, dt: float = 1.0, seed: int = 1) -> Dict[str, Any]:
    """Optional F-166700 smoke. Not the primary figure arm."""
    rec = run_real_trial(
        "glioblastoma",
        "F",
        seed=seed,
        days=days,
        dt=dt,
        f_neurons=F_FULL_NEURONS,
    )
    rec["arm_label"] = f"F-{F_FULL_NEURONS}"
    return rec


def weight_convergence(
    days: float = 40.0, dt: float = 0.4, seed: int = 5, n_kc: int = 96
) -> Dict[str, Any]:
    """||W_KC→MBON||_F along a real E loop; must stay bounded (no runaway)."""
    ctrl = make_controller("E", n_kc=n_kc, seed=seed)
    sim = ClosedLoopSimulator(
        archetype="glioblastoma",
        controller=ctrl,
        dt=dt,
        seed=seed,
        embodiment_enabled=False,
    )
    norms = []
    for _ in range(int(round(days / dt))):
        frame = sim.step(run_cancer=True, run_embodiment=False)
        w = float((frame.connectome or {}).get("plasticity_norm") or 0.0)
        norms.append(w)
        if frame.terminal:
            break
    arr = np.asarray(norms, dtype=float)
    tail = arr[-max(8, arr.size // 5) :]
    return {
        "n_steps": int(arr.size),
        "w0": float(arr[0]) if arr.size else None,
        "w_end": float(arr[-1]) if arr.size else None,
        "w_max": float(np.max(arr)) if arr.size else None,
        "tail_std": float(np.std(tail)) if tail.size else None,
        "bounded": bool(arr.size and np.max(arr) < 80.0 and np.all(np.isfinite(arr))),
        "plateau": bool(tail.size and float(np.std(tail)) < 0.15 * max(float(np.mean(tail)), 1e-6)),
        "sat_at_hard_clip": bool(arr.size and float(np.max(arr)) >= 1.49),
        "note": (
            "Plateau = tail ||W||_F is stationary. Hard clip at w_max is NOT "
            "called convergence."
        ),
        "non_claim": ENDPOINT_NON_CLAIM,
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def four_panel_figure(cohort: Dict[str, Any], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), constrained_layout=True)
    # KM OS
    ax = axes[0, 0]
    for arm, color in zip(ARMS, ("#666666", "#d4a054", "#2a9d8f", "#e06b5c")):
        km = cohort["km"][arm]["os"]
        ax.step(km["t"], km["s"], where="post", label=f"{arm}  n={km['n']}", color=color, lw=1.8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("simulated days")
    ax.set_ylabel("OS (in-silico mapping)")
    ax.set_title("Kaplan-Meier OS (virtual cohort)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.25)
    # RECIST
    ax = axes[0, 1]
    cats = ["CR", "near-CR", "PR", "SD", "PD"]
    x = np.arange(len(ARMS))
    width = 0.18
    colors = ("#4c8bf5", "#7ec8e3", "#3ecf8e", "#c0c0c0", "#d45d5d")
    for i, cat in enumerate(cats):
        vals = [cohort["recist_counts"][a].get(cat, 0) for a in ARMS]
        ax.bar(x + i * width, vals, width, label=cat, color=colors[i])
    ax.set_xticks(x + 2.0 * width)
    ax.set_xticklabels(list(ARMS))
    ax.set_ylabel("virtual patients")
    ax.set_title("RECIST 1.1-like (CR only if burden≈0)")
    ax.legend(fontsize=8)
    # CTCAE
    ax = axes[1, 0]
    grades = [str(g) for g in range(6)]
    cmap = ("#9ad1b3", "#d4e157", "#ffc107", "#ff8a65", "#e53935", "#4a148c")
    for i, g in enumerate(grades):
        vals = [cohort["ctcae_counts"][a].get(g, 0) for a in ARMS]
        ax.bar(x + i * 0.12, vals, 0.12, label=f"G{g}", color=cmap[i])
    ax.set_xticks(x + 0.3)
    ax.set_xticklabels(list(ARMS))
    ax.set_ylabel("virtual patients")
    ax.set_title("H-band surrogate / CTCAE-like worst grade")
    ax.legend(fontsize=7, ncol=3)
    # Trajectory
    ax = axes[1, 1]
    ex = cohort["example"]
    ax.plot(ex["times"], ex["burden"], color="#e06b5c", label="tumor burden")
    ax.plot(ex["times"], ex["health"], color="#2a9d8f", label="host health H")
    ax.axhline(0.2, color="#aa3333", ls="--", lw=0.9, label="H=0.2 terminal")
    ax.set_xlabel("simulated days")
    ax.set_title(f"Example trajectory (arm {ex['controller']})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.suptitle(
        "IN SILICO ENDPOINT MAPPING  |  not a clinical trial  |  not FDA/EMA readiness",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run_ablations(
    days: float = 24.0,
    dt: float = 1.0,
    seed: int = MASTER_SEED,
) -> Dict[str, Any]:
    """Full re-rollout under frozen plasticity / masked fusion Y / zero Ab.

    Endpoint deltas (OS event, RECIST, min H), not just U AUC.
    """
    base = run_real_trial("glioblastoma", "E", seed=seed, days=days, dt=dt)
    frozen = run_real_trial("glioblastoma", "E", seed=seed, days=days, dt=dt, freeze_plasticity=True)
    masked = run_real_trial("glioblastoma", "E", seed=seed, days=days, dt=dt, mask_fusion_y=True)
    no_ab = run_real_trial("glioblastoma", "E", seed=seed, days=days, dt=dt, zero_antibody=True)

    def _pack(rec):
        return {
            "best": rec.get("best"),
            "os_event": rec.get("os_event"),
            "os_time": rec.get("os_time"),
            "pfs_time": rec.get("pfs_time"),
            "ctcae_worst_grade": rec.get("ctcae_worst_grade"),
            "min_H": rec.get("ctcae_min_H"),
            "final_burden": rec["burden"][-1] if rec.get("burden") else None,
        }

    def _delta(a, b, key):
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            return None
        try:
            return float(vb) - float(va)
        except (TypeError, ValueError):
            return None

    base_p, fro_p, mask_p, ab_p = map(_pack, (base, frozen, masked, no_ab))
    return {
        "baseline_E": base_p,
        "frozen_plasticity": fro_p,
        "masked_fusion_Y": mask_p,
        "zero_antibody_U": ab_p,
        "delta_vs_E": {
            "frozen_os_event": _delta(base_p, fro_p, "os_event"),
            "frozen_final_burden": _delta(base_p, fro_p, "final_burden"),
            "masked_final_burden": _delta(base_p, mask_p, "final_burden"),
            "zero_ab_final_burden": _delta(base_p, ab_p, "final_burden"),
            "frozen_min_H": _delta(base_p, fro_p, "min_H"),
            "zero_ab_min_H": _delta(base_p, ab_p, "min_H"),
        },
        "non_claim": ENDPOINT_NON_CLAIM,
    }


def write_plaintext_report(report: Dict[str, Any], path: Path) -> None:
    c = report.get("part2_cohort") or {}
    lines = [
        "IN SILICO ENDPOINT MAPPING",
        "computational–clinical translation layer — not a clinical trial",
        "not FDA/EMA readiness — not a Phase II result — not a cure claim",
        "",
        f"master_seed={c.get('master_seed', MASTER_SEED)}  solver={c.get('solver', DEFAULT_SOLVER)}",
        f"n_patients={c.get('n')}  n_trials={c.get('n_trials')}  days={c.get('days')}  {c.get('horizon_label')}",
        f"arms=A (MTD), B (Gatenby), E (plastic MB), {c.get('f_label')}",
        "PPO/C excluded (untrained stub). U is unitless [0,1], not mg/kg.",
        "",
        "PART 1 — computational stress",
        f"  stiff agreed={report['part1_stiff_solver'].get('agreed')}  "
        f"rtol={report['part1_stiff_solver'].get('agree_rtol')}  "
        f"atol={report['part1_stiff_solver'].get('agree_atol')}",
        f"  measured max abs vs tight LSODA: {report['part1_stiff_solver'].get('max_abs_err_vs_tight_LSODA')}",
        f"  conservation ok={report['part1_conservation'].get('ok')}  "
        f"preclip_nonneg={report['part1_conservation'].get('preclip_nonnegative')}  "
        f"clip_repairs={report['part1_conservation'].get('clip_repairs_total')}",
        f"  weights plateau={report['part1_weight_convergence'].get('plateau')}  "
        f"sat_at_hard_clip={report['part1_weight_convergence'].get('sat_at_hard_clip')}  "
        f"(clip is not called convergence)",
        "",
        "PART 2 — mapped endpoints (simulated)",
        f"  RECIST: {c.get('recist_counts')}",
        f"  CTCAE-like H-band: {c.get('ctcae_counts')}",
        f"  pairwise (computed; CI-includes-1 reported honestly):",
    ]
    for key, val in (c.get("pairwise") or {}).items():
        if key.endswith("_stat"):
            continue
        lines.append(f"    {key}: {val}")
    if report.get("part3_ablations"):
        lines += ["", "PART 3 — ablation endpoint deltas", str(report["part3_ablations"].get("delta_vs_E"))]
    lines += ["", ENDPOINT_NON_CLAIM, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_translation(
    n: int = DEFAULT_N,
    out_dir: Optional[Path] = None,
    days: float = DEFAULT_DAYS,
    dt: float = DEFAULT_DT,
    seed: int = MASTER_SEED,
    ablations: bool = True,
) -> Dict[str, Any]:
    report = {
        "layer": "in silico endpoint mapping / computational–clinical translation",
        "clinical_trial": False,
        "fda_ema_readiness": False,
        "phase_ii": False,
        "non_claim": ENDPOINT_NON_CLAIM,
        "part1_stiff_solver": stiff_solver_agreement(),
        "part1_conservation": conservation_check(),
        "part1_weight_convergence": weight_convergence(),
        "part2_cohort": lhs_virtual_cohort(n=n, days=days, dt=dt, seed=seed, verbose=n >= 20),
        "part3_ablations": run_ablations(days=min(days, 28.0), dt=max(dt, 0.5), seed=seed) if ablations else None,
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        four_panel_figure(report["part2_cohort"], out_dir / "four_panel_endpoints.png")
        light = dict(report)
        (out_dir / "translation_report.json").write_text(
            json.dumps(_jsonable(light), indent=2), encoding="utf-8"
        )
        write_plaintext_report(report, out_dir / "IN_SILICO_ENDPOINT_MAPPING.txt")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="In-silico endpoint mapping on the real Confluence closed loop"
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--days", type=float, default=DEFAULT_DAYS)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--out", default="results/validation_translation")
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Short-horizon virtual event time (n=8, 28 d) for CI / laptops",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    n, days, dt = args.n, args.days, args.dt
    if args.quick:
        n, days, dt = 8, 28.0, 1.0
    report = run_translation(n=n, out_dir=Path(args.out), days=days, dt=dt, seed=args.seed)
    print(json.dumps({k: report[k] for k in ("layer", "clinical_trial", "fda_ema_readiness", "non_claim")}, indent=2))
    print("stiff agreed", report["part1_stiff_solver"]["agreed"], "finite", report["part1_stiff_solver"]["finite"])
    print("conservation", report["part1_conservation"]["ok"])
    print("weights bounded", report["part1_weight_convergence"]["bounded"])
    print("cohort n_patients", report["part2_cohort"]["n"], "n_trials", report["part2_cohort"].get("n_trials"))
    print("pairwise (computed; may be inconclusive)", json.dumps(_jsonable(report["part2_cohort"]["pairwise"]), indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
