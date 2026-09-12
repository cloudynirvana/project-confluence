"""Therapeutic protein / biologic PK-PD channels (simulated dosing only)."""

import numpy as np

from confluence.cancer_env.archetypes import get_archetype
from confluence.cancer_env.ode_system import CancerODE
from confluence.contracts import ALL_EFFECTOR_IDS, PROTEIN_CHANNEL_IDS
from confluence.pharmacology.pk_pd_model import PKPDModel
from confluence.pharmacology.toxicity_constraints import load_drug_catalog


def test_catalog_includes_protein_biologics():
    catalog = load_drug_catalog()
    ids = {d.id for d in catalog}
    for pid in PROTEIN_CHANNEL_IDS:
        assert pid in ids
    proteins = [d for d in catalog if d.modality == "protein_biologic"]
    assert len(proteins) >= 4
    assert all(0.0 < d.tox_weight <= 0.85 for d in proteins)


def test_pk_accepts_all_effectors_and_pads_short_u():
    pk = PKPDModel(drug_ids=ALL_EFFECTOR_IDS)
    assert pk.n_drugs == len(ALL_EFFECTOR_IDS)
    u5 = np.array([0.2, 0.1, 0.0, 0.0, 0.3])
    clipped = pk.clip_infusion(u5)
    assert clipped.shape == (pk.n_drugs,)
    assert clipped[0] == 0.2
    assert clipped[5] == 0.0
    c = pk.zeros()
    dc = pk.rhs(c, u5)
    assert dc.shape == (pk.n_drugs,)
    assert np.all(np.isfinite(dc))


def test_default_five_channel_pk_unchanged():
    pk = PKPDModel()
    assert pk.n_drugs == 5
    assert "anti_pd1" in pk.drug_ids
    assert "protein_anti_pd1" not in pk.drug_ids


def test_protein_occupancy_merges_and_ifng_boosts_cytokine():
    ode = CancerODE(get_archetype("glioblastoma"), PKPDModel(drug_ids=ALL_EFFECTOR_IDS))
    x, _ = ode.initial_state()
    occ0 = {k: 0.0 for k in ALL_EFFECTOR_IDS}
    occ_ab = dict(occ0)
    occ_ab["anti_pd1"] = 0.5
    occ_ab["protein_anti_pd1"] = 0.5
    dx0 = ode.rhs_cancer(x, occ0, 0.0)
    dx_sm = ode.rhs_cancer(x, {**occ0, "anti_pd1": 0.5}, 0.0)
    dx_both = ode.rhs_cancer(x, occ_ab, 0.0)
    # Complementary PD-1 occupancy should change immune/exhaustion terms vs small molecule alone.
    assert np.linalg.norm(dx_both - dx_sm) > 1e-8
    occ_ifn = dict(occ0)
    occ_ifn["protein_ifng"] = 1.0
    dx_ifn = ode.rhs_cancer(x, occ_ifn, 0.0)
    # C_ifng production (index 9) rises with the cytokine channel.
    assert dx_ifn[9] > dx0[9]


def test_closed_loop_protein_keys_present_for_controller_e():
    from confluence.loop import ClosedLoopSimulator

    sim = ClosedLoopSimulator(archetype="glioblastoma", dt=0.25, seed=3)
    frame = sim.step()
    for pid in PROTEIN_CHANNEL_IDS:
        assert pid in frame.concentrations
        # Default E does not drive biologics.
        assert frame.action.infusion.get(pid, 0.0) == 0.0
