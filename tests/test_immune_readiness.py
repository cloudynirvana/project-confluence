"""Immune surveillance + antibody readiness (research scores)."""

import numpy as np

from confluence.cancer_env.archetypes import get_archetype
from confluence.cancer_env.ode_system import CancerODE
from confluence.cancer_env.readiness import antibody_priority, early_warning_score
from confluence.contracts import ALL_EFFECTOR_IDS, ObservationRecord, PROTEIN_CHANNEL_IDS
from confluence.controllers import make_controller
from confluence.controllers.base import ControllerContext
from confluence.loop import ClosedLoopSimulator
from confluence.pharmacology.pk_pd_model import PKPDModel
from confluence.pharmacology.toxicity_constraints import load_drug_catalog


def _obs(**kwargs):
    base = dict(
        t=0.0,
        tumor_burden=0.22,
        resistance_frequency=0.10,
        lactate=0.25,
        tgfb=0.18,
        immune_competence_ratio=0.55,
        fusion_allele_fraction=0.04,
        junction_neoantigen=0.04,
        occult_allele_fraction=0.03,
        dormancy_exit=0.02,
        immune_surveillance=0.05,
        antibody_readiness=0.04,
    )
    base.update(kwargs)
    return ObservationRecord(**base)


def test_catalog_has_readiness_biologics():
    ids = {d.id for d in load_drug_catalog()}
    assert "protein_surveillance_igg" in ids
    assert "protein_fusion_mab" in ids
    assert "protein_surveillance_igg" in PROTEIN_CHANNEL_IDS


def test_early_warning_fires_before_bulk():
    quiet = _obs()
    leak = _obs(
        tumor_burden=0.08,
        immune_competence_ratio=0.18,
        junction_neoantigen=0.55,
        occult_allele_fraction=0.28,
        dormancy_exit=0.35,
        immune_surveillance=0.30,
    )
    assert early_warning_score(leak) > early_warning_score(quiet) + 0.2
    assert antibody_priority(leak) > antibody_priority(quiet)


def test_e_allocates_antibodies_on_early_signs():
    ctrl = make_controller("E", n_kc=96, seed=5)
    ctx = ControllerContext(concentrations={k: 0.0 for k in ALL_EFFECTOR_IDS}, dt=0.25)
    u_quiet = ctrl.decide(_obs(), ctx).infusion
    u_early = ctrl.decide(
        _obs(
            immune_competence_ratio=0.15,
            junction_neoantigen=0.70,
            occult_allele_fraction=0.32,
            dormancy_exit=0.40,
            immune_surveillance=0.40,
            antibody_readiness=0.35,
        ),
        ctx,
    ).infusion
    abs_early = sum(u_early.get(p, 0.0) for p in PROTEIN_CHANNEL_IDS)
    abs_quiet = sum(u_quiet.get(p, 0.0) for p in PROTEIN_CHANNEL_IDS)
    assert u_early["protein_surveillance_igg"] > 0.08
    assert abs_early > abs_quiet


def test_surveillance_igg_raises_I_surv():
    ode = CancerODE(get_archetype("occult"), PKPDModel(drug_ids=ALL_EFFECTOR_IDS))
    x, _ = ode.initial_state()
    occ0 = {k: 0.0 for k in ALL_EFFECTOR_IDS}
    occ = dict(occ0)
    occ["protein_surveillance_igg"] = 0.9
    dx0 = ode.rhs_cancer(x, occ0, 0.0)
    dx = ode.rhs_cancer(x, occ, 0.0)
    assert dx[12] > dx0[12]


def test_forced_antibodies_can_drop_H():
    ids = list(ALL_EFFECTOR_IDS)
    ode = CancerODE(get_archetype("glioblastoma"), PKPDModel(drug_ids=ALL_EFFECTOR_IDS))
    x, c = ode.initial_state()
    h0 = float(x[10])
    u = np.ones(len(ids))
    for _ in range(80):
        x, c = ode.step(x, c, u, dt=0.25)
        if x[10] <= 0.2:
            break
    assert x[10] < h0
    assert x[10] <= 0.25


def test_closed_loop_occult_has_readiness_keys():
    sim = ClosedLoopSimulator(archetype="occult", dt=0.25, seed=2, embodiment_enabled=False)
    frame = sim.step(run_cancer=True, run_embodiment=False)
    assert frame.observation.occult_allele_fraction >= 0.0
    assert frame.latent.I_surv >= 0.0
    assert "protein_surveillance_igg" in frame.action.infusion
