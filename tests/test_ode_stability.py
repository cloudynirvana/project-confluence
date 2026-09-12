"""ODE stability for the Confluence v2 11-D microenvironment."""

import numpy as np
import pytest

from confluence.cancer_env.archetypes import ARCHETYPES, get_archetype
from confluence.cancer_env.ode_system import CancerODE
from confluence.loop import ClosedLoopSimulator
from confluence.pharmacology.pk_pd_model import PKPDModel


@pytest.mark.parametrize("name", list(ARCHETYPES))
def test_thousand_steps_finite(name):
    ode = CancerODE(get_archetype(name))
    x, c = ode.initial_state()
    u = np.array([0.2, 0.15, 0.1, 0.1, 0.25], dtype=float)
    for _ in range(1000):
        x, c = ode.step(x, c, u, dt=0.1)
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(c))
        assert np.all(x >= -1e-9)
        assert 0.0 <= x[10] <= 1.0


def test_high_dose_hits_terminal_without_nan():
    ode = CancerODE(get_archetype("glioblastoma"))
    x, c = ode.initial_state()
    u = np.ones(5, dtype=float)
    terminal = False
    for _ in range(800):
        x, c = ode.step(x, c, u, dt=0.25)
        assert np.all(np.isfinite(x))
        if x[10] <= 0.2:
            terminal = True
            break
    assert terminal
    assert np.all(np.isfinite(x))


def test_untreated_tumor_does_not_explode():
    ode = CancerODE(get_archetype("melanoma_persister"))
    x, c = ode.initial_state()
    u = np.zeros(5)
    for _ in range(400):
        x, c = ode.step(x, c, u, dt=0.2)
    assert np.all(np.isfinite(x))
    assert x[0] + x[1] < 50.0


def test_pk_decay_and_catalog_size():
    pk = PKPDModel()
    assert pk.n_drugs == 5
    from confluence.pharmacology.toxicity_constraints import load_drug_catalog

    catalog = load_drug_catalog()
    assert len(catalog) >= 6
    c = np.ones(5)
    c1 = c + pk.rhs(c, np.zeros(5)) * 0.1
    assert np.all(c1 < c)


def test_closed_loop_observations_drive_drugs():
    sim = ClosedLoopSimulator(archetype="pancreatic_pdac", dt=0.25, seed=1)
    frame0 = sim.history[-1]
    last = None
    for _ in range(40):
        last = sim.step()
        assert np.isfinite(last.latent.tumor_burden)
        assert set(last.action.infusion) >= {"anti_pd1", "hdac"}
    assert last.t > frame0.t
    assert last.observation.tumor_burden >= 0.0
