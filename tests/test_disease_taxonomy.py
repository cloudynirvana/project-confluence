"""Disease-class taxonomy: distinct X and Y signatures."""

import numpy as np

from confluence.cancer_env.archetypes import ARCHETYPES, get_archetype
from confluence.cancer_env.disease_classes import CLASS_ARCHETYPE_IDS, CLASS_PARAM_MAP, DISEASE_CLASS_IDS
from confluence.cancer_env.observation_layer import ObservationLayer
from confluence.cancer_env.ode_system import CORE_DIM, DIM, CancerODE
from confluence.contracts import ALL_EFFECTOR_IDS
from confluence.pharmacology.pk_pd_model import PKPDModel


def test_five_classes_registered():
    for cls in DISEASE_CLASS_IDS:
        assert cls in CLASS_ARCHETYPE_IDS
        assert cls in ARCHETYPES
        p = get_archetype(cls)
        assert p.disease_class == cls
        assert cls in CLASS_PARAM_MAP


def test_core_indices_stable():
    assert DIM == 15
    assert CORE_DIM == 12
    for name in ARCHETYPES:
        ode = CancerODE(get_archetype(name), PKPDModel(drug_ids=ALL_EFFECTOR_IDS))
        x, _ = ode.initial_state()
        assert x.shape == (15,)
        assert 0.0 <= x[10] <= 1.0
        latent = ode.to_latent(x)
        assert latent.T_f == x[11]
        assert latent.I_surv == x[12]
        assert latent.disease_class in DISEASE_CLASS_IDS


def test_benign_stays_growth_limited():
    ode = CancerODE(get_archetype("benign"), PKPDModel(drug_ids=ALL_EFFECTOR_IDS), seed=1)
    x, c = ode.initial_state()
    u = np.zeros(len(ALL_EFFECTOR_IDS))
    b0 = float(x[0] + x[1] + x[11])
    for _ in range(80):
        x, c = ode.step(x, c, u, dt=0.25)
    b1 = float(x[0] + x[1] + x[11])
    assert b1 < 1.2
    assert b1 < b0 + 0.6


def test_occult_y_burden_hidden_while_junction_leaks():
    ode = CancerODE(get_archetype("occult"), PKPDModel(drug_ids=ALL_EFFECTOR_IDS), seed=2)
    x, _ = ode.initial_state()
    lat = ode.to_latent(x)
    y = ObservationLayer(seed=2, params=ode.params).observe(lat, noisy=False)
    assert y.tumor_burden < 0.45 * lat.tumor_burden
    assert y.junction_neoantigen > y.tumor_burden
    assert y.occult_allele_fraction > 0.0


def test_dormant_gate_suppresses_growth_until_awake():
    ode = CancerODE(get_archetype("dormant"), PKPDModel(drug_ids=ALL_EFFECTOR_IDS), seed=0)
    ode.params.awaken_hazard = 0.0
    x, c = ode.initial_state()
    assert x[14] < 0.2
    u = np.zeros(len(ALL_EFFECTOR_IDS))
    occ = {k: 0.0 for k in ALL_EFFECTOR_IDS}
    dx_asleep = ode.rhs_cancer(x, occ, 0.0)
    x_awake = x.copy()
    x_awake[14] = 1.0
    dx_awake = ode.rhs_cancer(x_awake, occ, 0.0)
    # Net tumor growth (T_s + T_r + T_f) is larger when awake.
    g_asleep = dx_asleep[0] + dx_asleep[1] + dx_asleep[11]
    g_awake = dx_awake[0] + dx_awake[1] + dx_awake[11]
    assert g_awake > g_asleep


def test_terminal_starts_near_host_failure():
    p = get_archetype("terminal")
    assert p.x0[10] <= 0.35
    assert p.x0[0] + p.x0[1] + p.x0[11] > 2.5
    ode = CancerODE(p, PKPDModel(drug_ids=ALL_EFFECTOR_IDS))
    x, _ = ode.initial_state()
    assert x[10] <= 0.35
