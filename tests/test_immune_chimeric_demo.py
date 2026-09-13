"""Fly-brain immune rescue + chimeric T-cell engager (research simulation)."""

import numpy as np
import pytest

from confluence.cancer_env.archetypes import get_archetype
from confluence.cancer_env.ode_system import CancerODE
from confluence.contracts import ALL_EFFECTOR_IDS, ObservationRecord, PROTEIN_CHANNEL_IDS
from confluence.controllers import make_controller
from confluence.controllers.base import ControllerContext
from confluence.controllers.immune_prior import apply_immune_secretory_prior
from confluence.loop import ClosedLoopSimulator
from confluence.pharmacology.pk_pd_model import PKPDModel
from confluence.pharmacology.toxicity_constraints import load_drug_catalog


def test_catalog_has_chimeric_engager():
    ids = {d.id for d in load_drug_catalog()}
    assert "protein_chimeric_engager" in ids
    assert "protein_chimeric_engager" in PROTEIN_CHANNEL_IDS


def test_engager_boosts_immune_kill_on_tumor():
    ode = CancerODE(get_archetype("glioblastoma"), PKPDModel(drug_ids=ALL_EFFECTOR_IDS))
    x, _ = ode.initial_state()
    occ0 = {k: 0.0 for k in ALL_EFFECTOR_IDS}
    occ_e = dict(occ0)
    occ_e["protein_chimeric_engager"] = 0.9
    dx0 = ode.rhs_cancer(x, occ0, 0.0)
    dx_e = ode.rhs_cancer(x, occ_e, 0.0)
    assert dx_e[0] < dx0[0]  # T_s more negative
    assert dx_e[11] < dx0[11]  # T_f more negative


def test_ifng_protein_raises_cytokine_production():
    ode = CancerODE(get_archetype("glioblastoma"), PKPDModel(drug_ids=ALL_EFFECTOR_IDS))
    x, _ = ode.initial_state()
    occ0 = {k: 0.0 for k in ALL_EFFECTOR_IDS}
    occ = dict(occ0)
    occ["protein_ifng"] = 1.0
    dx0 = ode.rhs_cancer(x, occ0, 0.0)
    dx = ode.rhs_cancer(x, occ, 0.0)
    assert dx[9] > dx0[9]


def test_immune_prior_lifts_chimeric_and_cytokine_when_suppressed():
    obs = ObservationRecord(
        t=0.0,
        tumor_burden=1.4,
        resistance_frequency=0.2,
        lactate=0.6,
        tgfb=0.5,
        immune_competence_ratio=0.15,
        fusion_allele_fraction=0.45,
        junction_neoantigen=0.4,
    )
    u0 = np.zeros(len(ALL_EFFECTOR_IDS))
    u = apply_immune_secretory_prior(u0, ALL_EFFECTOR_IDS, obs)
    ids = list(ALL_EFFECTOR_IDS)
    assert u[ids.index("protein_chimeric_engager")] > 0.2
    assert u[ids.index("protein_il2")] > 0.2
    assert u[ids.index("protein_ifng")] > 0.15
    assert u[ids.index("tki_alk")] > 0.10


def test_controller_f_emits_chimeric_proteins():
    ctrl = make_controller("F", n_neurons=256, seed=4)
    obs = ObservationRecord(
        t=0.0,
        tumor_burden=1.3,
        resistance_frequency=0.25,
        lactate=0.5,
        tgfb=0.4,
        immune_competence_ratio=0.2,
        fusion_allele_fraction=0.5,
        junction_neoantigen=0.45,
    )
    ctx = ControllerContext(concentrations={k: 0.0 for k in ALL_EFFECTOR_IDS}, dt=0.25)
    act = ctrl.decide(obs, ctx)
    assert act.infusion["protein_chimeric_engager"] > 0.05
    assert act.infusion["protein_il2"] > 0.05


def test_immune_proteins_raise_I_act_in_cold_gbm():
    """Cytokine + chimeric engager rescue I_act when TKI/MTD dump is off."""
    ids = list(ALL_EFFECTOR_IDS)
    ode = CancerODE(get_archetype("glioblastoma"), PKPDModel(drug_ids=ALL_EFFECTOR_IDS))
    x, c = ode.initial_state()
    i0 = float(x[2])
    u = np.zeros(len(ids))
    for name, val in {
        "protein_il2": 0.55,
        "protein_ifng": 0.50,
        "protein_chimeric_engager": 0.45,
        "protein_anti_pd1": 0.45,
        "protein_tgfb_trap": 0.40,
    }.items():
        u[ids.index(name)] = val
    peak = i0
    for _ in range(50):
        x, c = ode.step(x, c, u, dt=0.25)
        peak = max(peak, float(x[2]))
    ode0 = CancerODE(get_archetype("glioblastoma"), PKPDModel(drug_ids=ALL_EFFECTOR_IDS))
    x0, c0 = ode0.initial_state()
    u0 = np.zeros(len(ids))
    for _ in range(50):
        x0, c0 = ode0.step(x0, c0, u0, dt=0.25)
    assert np.isfinite(x[2])
    assert peak > i0
    assert x[2] > i0
    assert x[2] > float(x0[2]) + 0.05


def test_closed_loop_f_manages_tme_with_chimeric_proteins():
    ctrl = make_controller("F", n_neurons=256, seed=8)
    sim = ClosedLoopSimulator(
        archetype="melanoma_persister",
        controller=ctrl,
        dt=0.25,
        seed=8,
        embodiment_enabled=False,
    )
    b0 = sim.history[-1].latent.tumor_burden
    last = None
    for _ in range(48):
        last = sim.step(run_cancer=True, run_embodiment=False)
        if last.terminal:
            break
    assert last is not None
    assert np.isfinite(last.latent.I_act)
    assert last.latent.I_act > 0.15
    assert last.latent.tumor_burden < b0
    assert last.action.infusion.get("protein_chimeric_engager", 0.0) > 0.0
    assert last.latent.H > 0.2


def test_demo_immune_refuses_stub(monkeypatch):
    import confluence.demo_immune as demo

    monkeypatch.setattr(demo, "flybody_available", lambda: False)
    with pytest.raises(SystemExit, match="refuses to ship stub"):
        demo.render_immune_clip(seconds=0.1, fps=2, width=64, height=48)
