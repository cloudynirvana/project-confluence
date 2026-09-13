"""Chimeric fusion clone, junction observability, and fusion-directed TKIs."""

import numpy as np

from confluence.cancer_env.archetypes import ARCHETYPES, get_archetype
from confluence.cancer_env.observation_layer import ObservationLayer
from confluence.cancer_env.ode_system import DIM, CancerODE
from confluence.contracts import (
    ALL_EFFECTOR_IDS,
    FUSION_CHANNEL_IDS,
    ObservationRecord,
    PROTEIN_CHANNEL_IDS,
)
from confluence.controllers import make_controller
from confluence.controllers.base import ControllerContext
from confluence.loop import ClosedLoopSimulator
from confluence.neural_engine.plasticity import dopamine_signal
from confluence.pharmacology.pk_pd_model import PKPDModel
from confluence.pharmacology.toxicity_constraints import load_drug_catalog


def test_state_keeps_h_at_index_10_and_tf_at_11():
    assert DIM >= 12
    tissue = ("glioblastoma", "pancreatic_pdac", "melanoma_persister")
    for name in tissue:
        ode = CancerODE(get_archetype(name), PKPDModel(drug_ids=ALL_EFFECTOR_IDS))
        x, c = ode.initial_state()
        assert x.shape[0] >= 12
        assert 0.0 <= x[10] <= 1.0
        assert x[11] >= 0.0
        latent = ode.to_latent(x)
        assert latent.T_f == x[11]
        assert latent.fusion_id
        assert latent.tumor_burden == x[0] + x[1] + x[11]


def test_fusion_tki_kills_tf_preferentially():
    ode = CancerODE(get_archetype("glioblastoma"), PKPDModel(drug_ids=ALL_EFFECTOR_IDS))
    x, _ = ode.initial_state()
    occ0 = {k: 0.0 for k in ALL_EFFECTOR_IDS}
    occ_tki = dict(occ0)
    occ_tki["tki_alk"] = 0.9
    dx0 = ode.rhs_cancer(x, occ0, 0.0)
    dx_tki = ode.rhs_cancer(x, occ_tki, 0.0)
    # Fusion clone (index 11) drops more than sensitive clone (index 0).
    d_tf = dx_tki[11] - dx0[11]
    d_ts = dx_tki[0] - dx0[0]
    assert d_tf < d_ts
    assert d_tf < -1e-4


def test_fusion_terms_keep_ode_finite():
    ode = CancerODE(get_archetype("pancreatic_pdac"), PKPDModel(drug_ids=ALL_EFFECTOR_IDS))
    x, c = ode.initial_state()
    u = np.zeros(len(ALL_EFFECTOR_IDS))
    u[list(ALL_EFFECTOR_IDS).index("tki_alk")] = 0.6
    u[0] = 0.2
    for _ in range(400):
        x, c = ode.step(x, c, u, dt=0.15)
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(c))
        assert x[11] >= -1e-9
        assert 0.0 <= x[10] <= 1.0


def test_catalog_has_fusion_tkis_and_proteins():
    catalog = load_drug_catalog()
    ids = {d.id for d in catalog}
    for fid in FUSION_CHANNEL_IDS:
        assert fid in ids
    for pid in PROTEIN_CHANNEL_IDS:
        assert pid in ids
    fusion = [d for d in catalog if d.drug_class == "fusion_tki"]
    assert len(fusion) >= 2
    assert all(d.provenance.doi.startswith("10.") for d in fusion)


def test_junction_observation_is_noisy_proxy():
    ode = CancerODE(get_archetype("melanoma_persister"))
    x, _ = ode.initial_state()
    state = ode.to_latent(x, t=1.0)
    layer = ObservationLayer(seed=4)
    y = layer.observe(state)
    assert 0.0 <= y.fusion_allele_fraction <= 1.0
    assert y.junction_neoantigen >= 0.0
    vec = y.as_vector()
    assert len(vec) >= 7
    assert vec[5] == y.fusion_allele_fraction
    assert y.fusion_id == state.fusion_id
def test_da_rewards_fusion_af_drop():
    da_bad = dopamine_signal(0.1, 0.2, 0.05, delta_fusion=0.2)
    da_good = dopamine_signal(-0.1, 0.2, -0.05, delta_fusion=-0.2)
    assert da_good > da_bad


def test_controller_e_emits_fusion_tkis_and_sees_junction():
    ctrl = make_controller("E", n_kc=64, seed=3)
    assert set(FUSION_CHANNEL_IDS) <= set(ctrl.drug_ids)
    low = ObservationRecord(
        t=0.0,
        tumor_burden=1.0,
        resistance_frequency=0.2,
        lactate=0.4,
        tgfb=0.3,
        immune_competence_ratio=0.5,
        fusion_allele_fraction=0.05,
        junction_neoantigen=0.04,
    )
    high = low.model_copy(update={"fusion_allele_fraction": 0.65, "junction_neoantigen": 0.8, "t": 0.1})
    ctx_low = ControllerContext(concentrations={k: 0.0 for k in ctrl.drug_ids}, dt=0.2, archetype="glioblastoma")
    u_low = ctrl.decide(low, ctx_low).infusion
    u_high = ctrl.decide(high, ctx_low).infusion
    for fid in FUSION_CHANNEL_IDS:
        assert fid in u_low
        assert 0.0 <= u_high[fid] <= 1.0
    # Structured prior: junction-high observation should not collapse TKI to zero.
    assert max(u_high[fid] for fid in FUSION_CHANNEL_IDS) > 0.0


def test_controller_f_includes_fusion_and_protein_channels():
    ctrl = make_controller("F", n_neurons=256, seed=2)
    assert list(ctrl.drug_ids) == list(ALL_EFFECTOR_IDS)
    obs = ObservationRecord(
        t=0.0,
        tumor_burden=1.1,
        resistance_frequency=0.25,
        lactate=0.4,
        tgfb=0.3,
        immune_competence_ratio=0.4,
        fusion_allele_fraction=0.4,
        junction_neoantigen=0.5,
    )
    ctx = ControllerContext(concentrations={k: 0.1 for k in ALL_EFFECTOR_IDS}, dt=0.25, archetype="glioblastoma")
    act = ctrl.decide(obs, ctx)
    for fid in FUSION_CHANNEL_IDS:
        assert fid in act.infusion


def test_closed_loop_fusion_keys_and_a_through_f():
    sim = ClosedLoopSimulator(archetype="glioblastoma", dt=0.25, seed=5)
    frame = sim.step()
    assert hasattr(frame.latent, "T_f")
    assert frame.observation.fusion_allele_fraction >= 0.0
    for fid in FUSION_CHANNEL_IDS:
        assert fid in frame.concentrations
    # A–D still run (5-D U padded).
    for letter in "ABCD":
        sim.set_controller(make_controller(letter))
        last = sim.step()
        assert np.isfinite(last.latent.T_f)
        assert last.latent.tumor_burden >= 0.0
