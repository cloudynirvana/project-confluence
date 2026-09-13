"""Cinematic hero renderer stays finite and non-blank."""

import numpy as np
import pytest

from confluence.cinematic import fly_kinematics, render_frame
from confluence.embodiment.flybody_bridge import FlybodyBridge


def test_render_frame_is_dark_lab_rgb():
    rgb = render_frame(
        width=320,
        height=180,
        embodiment={
            "heading": 0.2,
            "xpos": [0.05, 0.0, 0.12],
            "joints": [0.1 * i for i in range(18)],
            "t_fly": 0.4,
        },
        connectome={"da": 0.1, "mbon_rates": [0.2, 0.4, 0.1]},
        latent={"tumor_burden": 1.1, "resistance_frequency": 0.3},
        proteins=["protein_ifng"],
        t_days=2.0,
    )
    assert rgb.shape == (180, 320, 3)
    assert rgb.dtype == np.uint8
    assert rgb.mean() < 80  # dark lab
    assert rgb.max() > 80  # fly / HUD marks are visible


def test_kinematics_six_legs():
    kin = fly_kinematics([0.0] * 18, 0.0, (0.0, 0.0, 0.12), 0.0)
    assert len(kin["legs"]) == 6
    assert kin["thorax"].shape == (3,)


def test_stub_telemetry_includes_wings():
    bridge = FlybodyBridge(prefer_real=False, seed=1)
    tel = bridge.step([0.2, 0.1, 0.1, 0.1, 0.1], [0.1] * 8)
    d = tel.as_dict()
    assert "wings" in d
    assert len(d["wings"]) == 2
    assert "frame_jpeg" not in d


def test_demo_cinematic_refuses_stub(monkeypatch):
    import confluence.demo_cinematic as demo

    monkeypatch.setattr(demo, "flybody_available", lambda: False)
    with pytest.raises(SystemExit, match="refuses to ship stub"):
        demo.render_clip(seconds=0.1, fps=2, width=64, height=48)
