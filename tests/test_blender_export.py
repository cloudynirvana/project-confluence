"""Blender export is logged-sim scientific viz, not generative biology."""

import ast
from pathlib import Path

import pytest

from confluence.demo_blender import main as blender_main
from confluence.embodiment.flybody_bridge import flybody_available
from confluence.viz.blender_export import (
    NON_CLAIM,
    SIMULATION_LABEL,
    export_closed_loop_blender,
    write_sidecar,
)


def test_honesty_is_simulation_research():
    assert "SIMULATION" in SIMULATION_LABEL
    assert "cure" in NON_CLAIM.lower()
    assert "not a cure" in NON_CLAIM.lower()
    assert "generative" in NON_CLAIM.lower()


def test_sidecar_schema_from_logged_values(tmp_path):
    rows = [
        {
            "frame": 0,
            "t_days": 0.2,
            "tumor_burden": 1.1,
            "H": 0.9,
            "fusion_allele_fraction": 0.12,
            "A_ready": 0.08,
            "I_surv": 0.1,
            "I_act": 0.3,
            "resistance_frequency": 0.2,
            "terminal": False,
            "U": {"anti_pd1": 0.4},
            "t_fly": 0.01,
            "xpos": [0.0, 0.0, 0.12],
            "heading": 0.1,
        }
    ]
    paths = write_sidecar(rows, tmp_path)
    payload = (tmp_path / "telemetry.json").read_text(encoding="utf-8")
    assert "SIMULATION / RESEARCH" in payload
    assert "not a cure" in payload.lower()
    assert paths["csv"].exists()
    csv_text = paths["csv"].read_text(encoding="utf-8")
    assert "tumor_burden" in csv_text
    assert "A_ready" in csv_text


def test_allow_stub_rejected():
    with pytest.raises(SystemExit, match="rejected"):
        blender_main(["--allow-stub", "--out", "/tmp/nope"])


def test_bpy_script_is_parseable():
    src = Path("docs/demo/blender/confluence_blender_hud.py").read_text(encoding="utf-8")
    ast.parse(src)
    assert "SIMULATION / RESEARCH" in src
    assert "telemetry.json" in src


def test_export_sidecar_and_optional_mesh(tmp_path):
    report = export_closed_loop_blender(
        tmp_path,
        n_frames=3,
        dt=0.25,
        fps=8,
        width=160,
        height=90,
        seed=2,
        controller="A",
        require_mesh=False,
    )
    assert report["n_frames"] == 3
    assert Path(report["sidecar"]).exists()
    payload = Path(report["sidecar"]).read_text(encoding="utf-8")
    assert "tumor_burden" in payload
    assert "A_ready" in payload
    assert "SIMULATION / RESEARCH" in payload
    if flybody_available() and report["mesh_live"]:
        assert report["n_png"] == 3
        assert (tmp_path / "frames" / "frame_0000.png").exists()
        assert (tmp_path / "pose.json").exists()
    else:
        assert report["n_png"] == 0


@pytest.mark.skipif(not flybody_available(), reason="flybody / MuJoCo not installed")
def test_require_mesh_dumps_fruitfly_pngs(tmp_path):
    report = export_closed_loop_blender(
        tmp_path,
        n_frames=2,
        dt=0.2,
        width=128,
        height=72,
        seed=1,
        controller="E",
        require_mesh=True,
    )
    assert report["mesh_live"] is True
    assert report["n_png"] == 2
    png = tmp_path / "frames" / "frame_0000.png"
    assert png.stat().st_size > 200
