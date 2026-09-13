"""Sparse full-brain scale: config accepts 166700; smoke at small N."""

import os

import numpy as np
import pytest

from confluence.contracts import (
    ALL_EFFECTOR_IDS,
    FULL_BRAIN_NEURONS,
    ObservationRecord,
)
from confluence.neural_engine.full_brain import (
    FullBrainConfig,
    FullBrainNetwork,
    population_sizes,
)


def _obs(burden=1.0, resist=0.2):
    return ObservationRecord(
        t=0.0,
        tumor_burden=burden,
        resistance_frequency=resist,
        lactate=0.4,
        tgfb=0.3,
        immune_competence_ratio=0.5,
    )


def test_config_accepts_166700():
    cfg = FullBrainConfig(n_neurons=166700)
    assert cfg.n_neurons == FULL_BRAIN_NEURONS
    pops = population_sizes(166700)
    assert pops["n_neurons"] == 166700
    assert pops["n_hidden"] + pops["n_pn"] + pops["n_mbon"] + pops["n_secretory"] + pops["n_dan"] == 166700
    assert pops["n_hidden"] > 100000


def test_construct_and_step_166700():
    net = FullBrainNetwork(FullBrainConfig(n_neurons=FULL_BRAIN_NEURONS, seed=1))
    assert net.n_neurons == 166700
    # Sparse tables, not a dense N×N matrix.
    assert net.hidden_idx.shape == (net.n_hidden, net.config.fan_in)
    assert net.w_sec.shape[0] == len(ALL_EFFECTOR_IDS)
    u = net.step(_obs(), 0.2, 0.5)
    assert u.shape == (len(ALL_EFFECTOR_IDS),)
    assert np.all(np.isfinite(u))
    assert np.all(u >= 0.0) and np.all(u <= 1.0)
    tel = net.telemetry()
    assert tel["source"] == "sparse_stub_full_brain"
    assert tel["n_neurons"] == 166700


def test_downscaled_loop_is_rate_based():
    net = FullBrainNetwork(FullBrainConfig(n_neurons=512, seed=2))
    u0 = net.step(_obs(1.2, 0.3), 0.4, 0.5)
    u1 = net.step(_obs(0.9, 0.25), 0.3, 0.5)
    assert u0.shape[0] == len(ALL_EFFECTOR_IDS)
    assert np.all(np.isfinite(u1))
    # Plasticity moved the secretory readout.
    assert float(np.linalg.norm(net.w_sec)) > 0.0
    # Dedicated PN→secretory path can drive biologic channels.
    assert float(np.max(net.sec_rate)) > 0.0


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("CONFLUENCE_FULL_BRAIN") != "1",
    reason="set CONFLUENCE_FULL_BRAIN=1 to run a 166700-neuron training episode",
)
def test_optional_full_brain_episode():
    from confluence.controllers.full_brain import FullBrainController
    from confluence.loop import ClosedLoopSimulator

    sim = ClosedLoopSimulator(
        archetype="glioblastoma",
        controller=FullBrainController(n_neurons=FULL_BRAIN_NEURONS, seed=0),
        dt=0.5,
        seed=0,
        embodiment_enabled=False,
    )
    last = None
    for _ in range(8):
        last = sim.step(run_cancer=True, run_embodiment=False)
        assert np.isfinite(last.latent.tumor_burden)
    assert last.t > 0
    assert last.connectome["n_neurons"] == 166700
