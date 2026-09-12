"""Plasticity weights stay bounded under aggressive dopamine."""

import numpy as np

from confluence.neural_engine.network import MushroomBodyNetwork, NetworkConfig
from confluence.neural_engine.plasticity import DopaminePlasticity, dopamine_signal
from confluence.contracts import ObservationRecord


def test_dopamine_formula_signs():
    da_bad = dopamine_signal(delta_burden=0.4, concentrations_sum=2.0, delta_resistance=0.2)
    da_good = dopamine_signal(delta_burden=-0.4, concentrations_sum=0.1, delta_resistance=-0.1)
    assert da_bad < 0
    assert da_good > da_bad


def test_weights_clipped_and_finite():
    plas = DopaminePlasticity()
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.2, size=(8, 64))
    for _ in range(500):
        kc = rng.random(64)
        mbon = rng.random(8)
        da = float(rng.uniform(-5, 5))
        w = plas.step(w, kc, mbon, da, dt=0.5)
        assert np.all(np.isfinite(w))
        assert np.all(w >= plas.params.w_min - 1e-12)
        assert np.all(w <= plas.params.w_max + 1e-12)


def test_decay_pulls_toward_zero_when_da_zero():
    plas = DopaminePlasticity()
    w = np.ones((4, 10)) * 0.8
    for _ in range(80):
        w = plas.step(w, np.zeros(10), np.zeros(4), da=0.0, dt=0.2)
    assert np.max(np.abs(w)) < 0.8


def test_network_step_keeps_weights_finite():
    net = MushroomBodyNetwork(NetworkConfig(n_kc=64, seed=2, plastic=True))
    obs = ObservationRecord(
        t=0.0,
        tumor_burden=1.2,
        resistance_frequency=0.2,
        lactate=0.5,
        tgfb=0.4,
        immune_competence_ratio=0.6,
        host_toxicity_warning=False,
    )
    for i in range(120):
        obs = obs.model_copy(
            update={
                "t": 0.1 * i,
                "tumor_burden": max(0.1, 1.2 - 0.004 * i + 0.05 * np.sin(i)),
                "resistance_frequency": min(0.9, 0.2 + 0.002 * i),
            }
        )
        u = net.step(obs, concentrations_sum=0.8, dt=0.1)
        assert np.all(np.isfinite(u))
        assert np.all(u >= 0.0)
        assert np.all(np.isfinite(net.w_kc_mbon))
        assert np.all(net.w_kc_mbon >= net.plasticity.params.w_min)
        assert np.all(net.w_kc_mbon <= net.plasticity.params.w_max)
