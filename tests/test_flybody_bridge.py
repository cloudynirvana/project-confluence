"""Smoke tests for the optional flybody embodiment bridge."""

import numpy as np
import pytest

from confluence.embodiment.flybody_bridge import (
    WALK_ACTION_DIM,
    FlybodyBridge,
    flybody_available,
    flybody_status,
)
from confluence.loop import ClosedLoopSimulator


def test_status_reports_backend():
    status = flybody_status()
    assert "available" in status
    assert status["repo"].endswith("TuragaLab/flybody")
    assert status["license"] == "Apache-2.0"


def test_stub_bridge_steps_are_finite():
    bridge = FlybodyBridge(task="walk_imitation", prefer_real=False, seed=3)
    assert bridge.backend == "kinematic_stub"
    assert bridge.action_dim == WALK_ACTION_DIM
    u = np.array([0.4, 0.2, 0.3, 0.15, 0.25])
    mbon = np.linspace(0.0, 1.0, 8)
    for _ in range(16):
        tel = bridge.step(u, mbon)
        assert np.all(np.isfinite(tel.action))
        assert np.all(np.isfinite(tel.sensory))
        assert np.all(np.isfinite(tel.joints))
        assert tel.action.size == WALK_ACTION_DIM
        assert tel.action.min() >= -1.0 - 1e-9
        assert tel.action.max() <= 1.0 + 1e-9
    mixed = bridge.mix_observation([1.0, 0.2, 0.4, 0.3, 0.5], alpha=0.25)
    assert mixed.shape == (5,)
    assert np.all(np.isfinite(mixed))


def test_closed_loop_still_runs_with_embodiment_sidecar():
    sim = ClosedLoopSimulator(archetype="glioblastoma", dt=0.25, seed=2)
    last = None
    for _ in range(8):
        last = sim.step(run_cancer=True, run_embodiment=True)
        assert np.isfinite(last.latent.tumor_burden)
        assert last.embodiment is not None
        assert np.isfinite(last.embodiment["action_rms"])
    assert last.t > 0


@pytest.mark.skipif(not flybody_available(), reason="flybody / MuJoCo not installed")
def test_real_flybody_template_steps_finite():
    bridge = FlybodyBridge(task="template", prefer_real=True, seed=1)
    assert bridge.backend == "flybody"
    u = np.ones(5) * 0.2
    mbon = np.zeros(8)
    for _ in range(5):
        tel = bridge.step(u, mbon)
        assert np.all(np.isfinite(tel.action))
        assert tel.action.size >= 1
    rgb = bridge.render_rgb(width=160, height=90)
    assert rgb is not None
    assert rgb.shape == (90, 160, 3)
    assert tel.as_dict().get("frame_jpeg")


def test_stub_does_not_claim_to_be_the_mesh():
    bridge = FlybodyBridge(task="walk_imitation", prefer_real=False, seed=3)
    assert bridge.render_rgb() is None
    assert bridge.render_jpeg() is None
    assert flybody_status()["install"]
