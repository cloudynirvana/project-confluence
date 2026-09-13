"""Falsifiable computational validation for taxonomy + readiness.

Research scores only. Does not claim clinical admissibility, disease
eradication, or a treatment path.

    python -m confluence.benchmarks.validation_suite --out results/validation_taxonomy
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from confluence.cancer_env.archetypes import get_archetype
from confluence.cancer_env.disease_classes import (
    CLASS_ARCHETYPE_IDS,
    CLASS_PARAM_MAP,
    PARAM_PROVENANCE,
)
from confluence.cancer_env.ode_system import CancerODE
from confluence.cancer_env.readiness import antibody_priority, early_warning_score
from confluence.contracts import ALL_EFFECTOR_IDS, DISEASE_CLASS_IDS, PROTEIN_CHANNEL_IDS
from confluence.controllers import make_controller
from confluence.controllers.base import ControllerContext
from confluence.loop import ClosedLoopSimulator
from confluence.pharmacology.pk_pd_model import PKPDModel
from confluence.pharmacology.toxicity_constraints import load_drug_catalog

NON_CLAIM = (
    "These metrics are computational research scores on an ODE + noisy Y. "
    "They are not a clinical validation, not a diagnostic panel, not a "
    "dosing protocol, and not a claim of disease eradication or clinical benefit."
)


def _features(frame) -> Dict[str, float]:
    L, Y = frame.latent, frame.observation
    return {
        "x_burden": float(L.tumor_burden),
        "x_H": float(L.H),
        "x_I_act": float(L.I_act),
        "x_I_surv": float(L.I_surv),
        "x_A_ready": float(L.A_ready),
        "x_awake": float(L.awake),
        "x_tgfb": float(L.C_tgfb),
        "y_burden": float(Y.tumor_burden),
        "y_competence": float(Y.immune_competence_ratio),
        "y_junction": float(Y.junction_neoantigen),
        "y_occult_af": float(Y.occult_allele_fraction),
        "y_dormancy_exit": float(Y.dormancy_exit),
        "y_surv": float(Y.immune_surveillance),
        "y_ready": float(Y.antibody_readiness),
        "early_warning": early_warning_score(Y),
    }


def _run_open_loop(archetype: str, days: float, dt: float, seed: int, u=None) -> List:
    ode = CancerODE(get_archetype(archetype), PKPDModel(drug_ids=ALL_EFFECTOR_IDS), seed=seed)
    from confluence.cancer_env.observation_layer import ObservationLayer

    obs = ObservationLayer(seed=seed, params=ode.params)
    x, c = ode.initial_state()
    if u is None:
        u = np.zeros(len(ALL_EFFECTOR_IDS))
    frames = []
    t = 0.0
    n = int(round(days / dt))
    for _ in range(n):
        x, c = ode.step(x, c, u, dt)
        t += dt
        lat = ode.to_latent(x, t=t)
        frames.append(type("F", (), {"latent": lat, "observation": obs.observe(lat)})())
    return frames


def class_signatures(days: float = 24.0, dt: float = 0.4, seed: int = 3) -> Dict[str, Any]:
    """Q1: class-separable state + Y signatures."""
    means = {}
    feat_names: Optional[List[str]] = None
    vectors = {}
    for cls in CLASS_ARCHETYPE_IDS:
        frames = _run_open_loop(cls, days, dt, seed)
        rows = [_features(f) for f in frames]
        if feat_names is None:
            feat_names = list(rows[0].keys())
        mat = np.array([[r[k] for k in feat_names] for r in rows], dtype=float)
        vectors[cls] = np.mean(mat, axis=0)
        means[cls] = {k: float(np.mean([r[k] for r in rows])) for k in feat_names}
    keys = list(vectors)
    dist = {a: {b: float(np.linalg.norm(vectors[a] - vectors[b])) for b in keys} for a in keys}
    # Nearest-centroid on mid-horizon snapshots (one per class).
    snapshots = {cls: _features(_run_open_loop(cls, days * 0.5, dt, seed + 1)[-1]) for cls in keys}
    correct = 0
    for cls, snap in snapshots.items():
        v = np.array([snap[k] for k in feat_names], dtype=float)
        pred = min(keys, key=lambda c: float(np.linalg.norm(v - vectors[c])))
        correct += int(pred == cls)
    return {
        "question": "class_separable_state_and_Y",
        "means": means,
        "pairwise_l2": dist,
        "nearest_centroid_accuracy": correct / max(len(keys), 1),
        "n_classes": len(keys),
        "param_map": CLASS_PARAM_MAP,
        "non_claim": NON_CLAIM,
    }


def lead_time_detectability(
    archetype: str = "occult",
    days: float = 50.0,
    dt: float = 0.35,
    seed: int = 11,
    early_thr: float = 0.18,
    burden_thr: float = 0.35,
) -> Dict[str, Any]:
    """Q2: early Y signs before bulk Y_burden blow-up."""
    frames = _run_open_loop(archetype, days, dt, seed)
    t_early = None
    t_bulk = None
    for f in frames:
        y = f.observation
        if t_early is None and early_warning_score(y) >= early_thr:
            t_early = float(y.t)
        if t_bulk is None and y.tumor_burden >= burden_thr:
            t_bulk = float(y.t)
    lead = None
    if t_early is not None and t_bulk is not None:
        lead = float(t_bulk - t_early)
    elif t_early is not None and t_bulk is None:
        lead = float(frames[-1].observation.t - t_early)
    return {
        "question": "dormancy_exit_or_occult_lead_time",
        "archetype": archetype,
        "t_early_warning": t_early,
        "t_bulk_y_burden": t_bulk,
        "lead_time_days": lead,
        "early_thr": early_thr,
        "burden_thr": burden_thr,
        "final_y_burden": float(frames[-1].observation.t and frames[-1].observation.tumor_burden),
        "final_latent_burden": float(frames[-1].latent.tumor_burden),
        "non_claim": NON_CLAIM,
    }


def toxicity_failure_case(days: float = 18.0, dt: float = 0.25, seed: int = 2) -> Dict[str, Any]:
    """Q3: effector PK/PD can harm H (antibodies fail the host)."""
    ids = list(ALL_EFFECTOR_IDS)
    u = np.zeros(len(ids))
    for name in PROTEIN_CHANNEL_IDS:
        u[ids.index(name)] = 1.0
    ode = CancerODE(get_archetype("glioblastoma"), PKPDModel(drug_ids=ALL_EFFECTOR_IDS), seed=seed)
    x, c = ode.initial_state()
    h0 = float(x[10])
    hit = False
    t_hit = None
    t = 0.0
    for _ in range(int(round(days / dt))):
        x, c = ode.step(x, c, u, dt)
        t += dt
        if x[10] <= 0.2:
            hit = True
            t_hit = t
            break
    return {
        "question": "pkpd_can_harm_H",
        "H0": h0,
        "H_end": float(x[10]),
        "terminal_toxicity": hit,
        "t_hit_days": t_hit,
        "forced_channels": list(PROTEIN_CHANNEL_IDS),
        "non_claim": NON_CLAIM,
    }


def fusion_allocation_shift(seed: int = 4) -> Dict[str, Any]:
    """Q4: fusion / neoantigen signal changes therapy allocation."""
    from confluence.contracts import ObservationRecord

    ctrl = make_controller("E", n_kc=128, seed=seed)
    base = ObservationRecord(
        t=0.0,
        tumor_burden=0.28,
        resistance_frequency=0.12,
        lactate=0.30,
        tgfb=0.22,
        immune_competence_ratio=0.35,
        fusion_allele_fraction=0.04,
        junction_neoantigen=0.03,
        occult_allele_fraction=0.05,
        dormancy_exit=0.04,
        immune_surveillance=0.08,
        antibody_readiness=0.06,
    )
    hot = base.model_copy(
        update={
            "fusion_allele_fraction": 0.62,
            "junction_neoantigen": 0.85,
            "occult_allele_fraction": 0.40,
            "immune_surveillance": 0.45,
            "antibody_readiness": 0.40,
            "t": 0.2,
        }
    )
    ctx = ControllerContext(concentrations={k: 0.0 for k in ALL_EFFECTOR_IDS}, dt=0.25)
    u_lo = ctrl.decide(base, ctx).infusion
    u_hi = ctrl.decide(hot, ctx).infusion
    fusionish = ("tki_alk", "tki_imatinib_like", "protein_fusion_mab", "protein_chimeric_engager")
    lo = {k: float(u_lo.get(k, 0.0)) for k in fusionish}
    hi = {k: float(u_hi.get(k, 0.0)) for k in fusionish}
    return {
        "question": "fusion_signal_changes_allocation",
        "U_low_junction": lo,
        "U_high_junction": hi,
        "delta_sum": float(sum(hi[k] - lo[k] for k in fusionish)),
        "non_claim": NON_CLAIM,
    }


def _mask_observation(obs, drop_fusion: bool = False):
    if not drop_fusion:
        return obs
    return obs.model_copy(
        update={
            "fusion_allele_fraction": 0.0,
            "junction_neoantigen": 0.0,
            "occult_allele_fraction": 0.0,
        }
    )


def ablations(days: float = 16.0, dt: float = 0.4, seed: int = 6) -> Dict[str, Any]:
    """Q5: freeze plasticity / drop fusion AF / drop antibody channels."""

    def protein_auc(controller, drop_fusion=False, drop_abs=False, plastic=True) -> float:
        if not plastic and hasattr(controller, "network"):
            controller.network.config.plastic = False
        sim = ClosedLoopSimulator(
            archetype="occult",
            controller=controller,
            dt=dt,
            seed=seed,
            embodiment_enabled=False,
        )
        auc = 0.0
        for _ in range(int(round(days / dt))):
            if drop_fusion:
                # Zero fusion/leak Y before decide by wrapping observer output.
                raw = sim._observe()
                sim.observer  # keep
            frame = sim.step(run_cancer=True, run_embodiment=False)
            u = dict(frame.action.infusion)
            if drop_fusion:
                # Re-decide on a masked observation for the metric (allocation).
                masked = _mask_observation(frame.observation, drop_fusion=True)
                ctx = ControllerContext(concentrations=frame.concentrations, dt=dt)
                u = controller.decide(masked, ctx).infusion
            if drop_abs:
                for pid in PROTEIN_CHANNEL_IDS:
                    u[pid] = 0.0
            auc += sum(float(u.get(pid, 0.0)) for pid in PROTEIN_CHANNEL_IDS) * dt
        return float(auc)

    full = protein_auc(make_controller("E", n_kc=96, seed=seed))
    frozen = protein_auc(make_controller("E", n_kc=96, seed=seed), plastic=False)
    no_fus = protein_auc(make_controller("E", n_kc=96, seed=seed), drop_fusion=True)
    no_ab = protein_auc(make_controller("E", n_kc=96, seed=seed), drop_abs=True)
    return {
        "question": "ablations_drop_performance",
        "protein_auc_full": full,
        "protein_auc_frozen_plasticity": frozen,
        "protein_auc_drop_fusion_Y": no_fus,
        "protein_auc_drop_antibody_U": no_ab,
        "drop_frozen": float(full - frozen),
        "drop_fusion_Y": float(full - no_fus),
        "drop_antibody_U": float(full - no_ab),
        "non_claim": NON_CLAIM,
    }


def citation_table() -> Dict[str, Any]:
    """Q6: citation-tied vs placeholder parameters."""
    catalog = [
        {
            "id": d.id,
            "doi": d.provenance.doi,
            "placeholder": d.provenance.placeholder,
            "note": d.provenance.note,
        }
        for d in load_drug_catalog()
        if d.id in PROTEIN_CHANNEL_IDS or d.drug_class in {"fusion_tki", "fusion_biologic", "immune_surveillance"}
    ]
    return {
        "question": "citation_or_placeholder",
        "class_knobs": PARAM_PROVENANCE,
        "catalog": catalog,
        "non_claim": NON_CLAIM,
    }


def run_suite(out_dir: Optional[Path] = None) -> Dict[str, Any]:
    report = {
        "research_only": True,
        "clinical_admissibility": False,
        "non_claim": NON_CLAIM,
        "disease_classes": list(DISEASE_CLASS_IDS),
        "q1_class_signatures": class_signatures(),
        "q2_lead_time_occult": lead_time_detectability("occult"),
        "q2_lead_time_dormant": lead_time_detectability(
            "dormant", days=40.0, early_thr=0.12, burden_thr=0.28, seed=8
        ),
        "q3_toxicity_failure": toxicity_failure_case(),
        "q4_fusion_allocation": fusion_allocation_shift(),
        "q5_ablations": ablations(),
        "q6_citations": citation_table(),
        "q7_non_claim": NON_CLAIM,
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        _maybe_plots(report, out_dir)
    return report


def _maybe_plots(report: Dict[str, Any], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    means = report["q1_class_signatures"]["means"]
    keys = ["y_burden", "y_junction", "y_occult_af", "y_dormancy_exit", "y_surv", "x_H"]
    classes = list(means)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(keys))
    width = 0.15
    for i, cls in enumerate(classes):
        vals = [means[cls].get(k, 0.0) for k in keys]
        ax.bar(x + i * width, vals, width, label=cls)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylabel("mean (research units)")
    ax.set_title("Class signatures in X / Y (research scores)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "q1_class_signatures.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    labels = ["occult", "dormant"]
    leads = [
        report["q2_lead_time_occult"].get("lead_time_days") or 0.0,
        report["q2_lead_time_dormant"].get("lead_time_days") or 0.0,
    ]
    ax.bar(labels, leads, color=["#7aa2d4", "#d4a054"])
    ax.set_ylabel("lead time (sim days)")
    ax.set_title("Early-warning lead time vs bulk Y (research)")
    fig.tight_layout()
    fig.savefig(out_dir / "q2_lead_time.png", dpi=140)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Taxonomy / readiness computational validation")
    parser.add_argument("--out", default="results/validation_taxonomy")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_suite(Path(args.out))
    print(json.dumps({k: report[k] for k in ("research_only", "clinical_admissibility", "non_claim")}, indent=2))
    print("q1 nearest-centroid acc", report["q1_class_signatures"]["nearest_centroid_accuracy"])
    print("q2 occult lead", report["q2_lead_time_occult"]["lead_time_days"])
    print("q3 terminal", report["q3_toxicity_failure"]["terminal_toxicity"])
    print("q4 delta", report["q4_fusion_allocation"]["delta_sum"])
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
