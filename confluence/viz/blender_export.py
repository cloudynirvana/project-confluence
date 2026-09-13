"""Export logged Confluence sims to Blender-friendly scientific assets.

PNG / pose / HUD series come from the real closed loop
(``ClosedLoopSimulator`` + ``CancerODE`` + TuragaLab/flybody ``fruitfly.xml``).
This is **simulation / research**, not a clinical film and not generative
tumor-shrink footage.

    python -m confluence.demo_blender --out docs/demo/blender
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from confluence.cancer_env.ode_system import DEFAULT_SOLVER
from confluence.contracts import ALL_EFFECTOR_IDS
from confluence.controllers import make_controller
from confluence.embodiment.flybody_bridge import (
    MESH_NAME,
    flybody_available,
    flybody_status,
)
from confluence.loop import ClosedLoopSimulator

SIMULATION_LABEL = "SIMULATION / RESEARCH"
NON_CLAIM = (
    "These frames and HUD series are logged from an in-silico closed loop "
    "(CancerODE + PK/PD + controller + optional fruitfly.xml). "
    "They are not a clinical trial, not a cure, not FDA/EMA readiness, "
    "and not generative fantasy biology."
)


def sidecar_row(frame, pose: Optional[Dict[str, Any]], frame_index: int) -> Dict[str, Any]:
    """One HUD sample synced to frame index / sim time. ODE values only."""
    lat = frame.latent
    inf = dict(frame.action.infusion or {})
    row: Dict[str, Any] = {
        "frame": int(frame_index),
        "t_days": float(frame.t),
        "tumor_burden": float(lat.tumor_burden),
        "H": float(lat.H),
        "fusion_allele_fraction": float(lat.fusion_allele_fraction),
        "A_ready": float(lat.A_ready),
        "I_surv": float(lat.I_surv),
        "I_act": float(lat.I_act),
        "resistance_frequency": float(lat.resistance_frequency),
        "terminal": bool(frame.terminal),
        "solver": DEFAULT_SOLVER,
        "label": SIMULATION_LABEL,
        "U": {k: float(inf.get(k, 0.0)) for k in ALL_EFFECTOR_IDS},
    }
    if pose:
        row["t_fly"] = pose.get("t_fly")
        row["xpos"] = pose.get("xpos")
        row["heading"] = pose.get("heading")
        row["qpos_root"] = pose.get("qpos_root")
        row["nq"] = pose.get("nq")
    return row


def write_sidecar(rows: Sequence[Dict[str, Any]], out_dir: Path, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": SIMULATION_LABEL,
        "non_claim": NON_CLAIM,
        "clinical_trial": False,
        "cure_claim": False,
        "generative_biology": False,
        "mesh": MESH_NAME,
        "n_frames": len(rows),
        "fields": [
            "frame",
            "t_days",
            "tumor_burden",
            "H",
            "fusion_allele_fraction",
            "A_ready",
            "I_surv",
            "U",
        ],
        "rows": list(rows),
    }
    if extra:
        payload.update(extra)
    json_path = out_dir / "telemetry.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = out_dir / "telemetry.csv"
    fieldnames = [
        "frame",
        "t_days",
        "t_fly",
        "tumor_burden",
        "H",
        "fusion_allele_fraction",
        "A_ready",
        "I_surv",
        "I_act",
        "resistance_frequency",
        "terminal",
        "heading",
        "xpos_x",
        "xpos_y",
        "xpos_z",
    ]
    u_keys = list(ALL_EFFECTOR_IDS)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames + [f"U_{k}" for k in u_keys])
        w.writeheader()
        for row in rows:
            xpos = row.get("xpos") or [None, None, None]
            flat = {
                "frame": row["frame"],
                "t_days": row["t_days"],
                "t_fly": row.get("t_fly"),
                "tumor_burden": row["tumor_burden"],
                "H": row["H"],
                "fusion_allele_fraction": row["fusion_allele_fraction"],
                "A_ready": row["A_ready"],
                "I_surv": row["I_surv"],
                "I_act": row["I_act"],
                "resistance_frequency": row["resistance_frequency"],
                "terminal": row["terminal"],
                "heading": row.get("heading"),
                "xpos_x": xpos[0] if xpos else None,
                "xpos_y": xpos[1] if xpos else None,
                "xpos_z": xpos[2] if xpos else None,
            }
            for k in u_keys:
                flat[f"U_{k}"] = (row.get("U") or {}).get(k, 0.0)
            w.writerow(flat)
    return {"json": json_path, "csv": csv_path}


def _save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(path, format="PNG")


def _ffmpeg_mp4(frames_dir: Path, out_mp4: Path, fps: int) -> None:
    import subprocess

    pattern = str(frames_dir / "frame_%04d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def export_closed_loop_blender(
    out_dir: Path,
    n_frames: int = 36,
    dt: float = 0.2,
    fps: int = 12,
    width: int = 640,
    height: int = 360,
    seed: int = 3,
    controller: str = "E",
    require_mesh: bool = False,
) -> Dict[str, Any]:
    """Log a real closed-loop run and dump PNG + HUD sidecar + pose.

    PNG dump requires live fruitfly.xml. Sidecar is always written from ODE
    values. Stub / CPG pixels are never saved as mesh frames.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    status = flybody_status()
    mesh_ok = bool(flybody_available())
    if require_mesh and not mesh_ok:
        raise RuntimeError(
            "Blender export refuses stub footage. Install flybody:\n" + status["install"]
        )

    kwargs: Dict[str, Any] = {}
    if controller in {"D", "E", "F"}:
        kwargs["seed"] = seed
    if controller == "E":
        kwargs["n_kc"] = 64
    if controller == "F":
        kwargs["n_neurons"] = 256
    ctrl = make_controller(controller, **kwargs)
    sim = ClosedLoopSimulator(
        archetype="glioblastoma",
        controller=ctrl,
        dt=dt,
        seed=seed,
        embodiment_enabled=True,
        solver=DEFAULT_SOLVER,
    )
    sim.embodiment.camera = "walker/hero"
    sim.embodiment.render_size = (width, height)
    sim.embodiment.render_enabled = False

    live_mesh = sim.embodiment.backend == "flybody" and sim.embodiment._env is not None
    if require_mesh and not live_mesh:
        raise RuntimeError("flybody imported but fruitfly.xml env is not live (check MUJOCO_GL=osmesa)")

    rows: List[Dict[str, Any]] = []
    poses: List[Dict[str, Any]] = []
    pngs: List[str] = []
    last = sim.history[-1]
    for i in range(int(n_frames)):
        last = sim.step(run_cancer=True, run_embodiment=True)
        pose = sim.embodiment.export_pose() if live_mesh else None
        if pose:
            poses.append({"frame": i, **pose})
        rows.append(sidecar_row(last, pose, i))
        if live_mesh:
            rgb = sim.embodiment.render_rgb(width=width, height=height)
            if rgb is None or rgb.shape != (height, width, 3):
                raise RuntimeError(f"real fruitfly.xml render failed at frame {i}")
            png_path = frames_dir / f"frame_{i:04d}.png"
            _save_png(png_path, rgb)
            pngs.append(str(png_path.name))

    extra = {
        "controller": controller if controller != "F" else "F-256 proxy",
        "solver": DEFAULT_SOLVER,
        "dt_days": dt,
        "fps": fps,
        "width": width,
        "height": height,
        "mesh_backend": sim.embodiment.backend,
        "mesh_live": live_mesh,
        "flybody": status,
    }
    paths = write_sidecar(rows, out_dir, extra=extra)
    pose_path = out_dir / "pose.json"
    pose_path.write_text(
        json.dumps(
            {
                "label": SIMULATION_LABEL,
                "non_claim": NON_CLAIM,
                "mesh": MESH_NAME,
                "frames": [
                    {
                        "frame": p["frame"],
                        "t_fly": p.get("t_fly"),
                        "xpos": p.get("xpos"),
                        "heading": p.get("heading"),
                        "qpos_root": p.get("qpos_root"),
                        "nq": p.get("nq"),
                    }
                    for p in poses
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    still = None
    preview_mp4 = None
    if pngs:
        import shutil

        still = out_dir / "still_mesh.png"
        shutil.copyfile(frames_dir / pngs[0], still)
        shutil.copyfile(frames_dir / pngs[len(pngs) // 2], out_dir / "still_mid.png")
        try:
            preview_mp4 = out_dir / "mujoco_preview.mp4"
            _ffmpeg_mp4(frames_dir, preview_mp4, fps=fps)
        except Exception:
            preview_mp4 = None

    manifest = {
        "label": SIMULATION_LABEL,
        "non_claim": NON_CLAIM,
        "clinical_trial": False,
        "cure_claim": False,
        "higgsfield": False,
        "generative_biology": False,
        "one_command": "python3 -m confluence.demo_blender --out docs/demo/blender",
        "open_in_blender": (
            "blender --background --python docs/demo/blender/confluence_blender_hud.py "
            "-- --root docs/demo/blender"
        ),
        "sidecar_json": paths["json"].name,
        "sidecar_csv": paths["csv"].name,
        "pose_json": pose_path.name,
        "frames_dir": "frames" if pngs else None,
        "n_png": len(pngs),
        "still_mesh": still.name if still else None,
        "mujoco_preview_mp4": preview_mp4.name if preview_mp4 else None,
        "bpy_script": "confluence_blender_hud.py",
        "renders_dir": "renders",
        "viz_complete": True,
        "mesh": MESH_NAME if live_mesh else None,
        "mesh_live": live_mesh,
        "n_frames": len(rows),
        "solver": DEFAULT_SOLVER,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "out_dir": str(out_dir),
        "n_frames": len(rows),
        "n_png": len(pngs),
        "mesh_live": live_mesh,
        "sidecar": str(paths["json"]),
        "manifest": str(out_dir / "manifest.json"),
        "label": SIMULATION_LABEL,
        "non_claim": NON_CLAIM,
    }
