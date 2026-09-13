"""Render a short closed-loop clip from the real flybody MuJoCo mesh.

    export MUJOCO_GL=osmesa
    python -m confluence.demo_cinematic --out docs/demo/cinematic.mp4

Refuses to write stub / CPG footage. If TuragaLab/flybody cannot render
fruitfly.xml, this job exits nonzero.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")

from confluence.cinematic import overlay_hud
from confluence.controllers import make_controller
from confluence.embodiment.flybody_bridge import flybody_available, flybody_status
from confluence.loop import ClosedLoopSimulator


def _ffmpeg_write_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    h, w = frames.shape[1], frames.shape[2]
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-preset", "medium",
        str(path),
    ]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True, capture_output=True)


def _ffmpeg_png(path: Path, rgb: np.ndarray) -> None:
    h, w = rgb.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-i", "-",
        "-frames:v", "1",
        str(path),
    ]
    subprocess.run(cmd, input=np.ascontiguousarray(rgb).tobytes(), check=True, capture_output=True)


def render_clip(
    seconds: float = 12.0,
    fps: int = 24,
    width: int = 1280,
    height: int = 720,
) -> tuple:
    if not flybody_available():
        raise SystemExit(
            "demo_cinematic refuses to ship stub footage. "
            "Install the anatomical flybody mesh:\n"
            f"  {flybody_status()['install']}"
        )
    n = int(round(seconds * fps))
    ctrl = make_controller("F", n_neurons=512, seed=3)
    sim = ClosedLoopSimulator(
        archetype="glioblastoma",
        controller=ctrl,
        dt=0.15,
        seed=3,
        embodiment_enabled=True,
    )
    sim.embodiment.camera = "walker/hero"
    sim.embodiment.render_size = (width, height)
    # Skip per-step JPEG; this job writes high-res RGB frames itself.
    sim.embodiment.render_enabled = False
    if sim.embodiment.backend != "flybody" or sim.embodiment._env is None:
        raise SystemExit(
            "flybody imported but the MuJoCo env is not live. "
            "Check MUJOCO_GL=osmesa (or egl) and fruitfly.xml assets."
        )
    frames = np.zeros((n, height, width, 3), dtype=np.uint8)
    last = sim.history[-1]
    for i in range(n):
        last = sim.step(run_cancer=True, run_embodiment=True)
        rgb = sim.embodiment.render_rgb(width=width, height=height)
        if rgb is None or rgb.shape != (height, width, 3):
            raise SystemExit(f"real fruitfly.xml render failed at frame {i}")
        proteins = [
            k
            for k, v in (last.action.infusion or {}).items()
            if k.startswith("protein_") and float(v) > 0.05
        ]
        frames[i] = overlay_hud(
            rgb.copy(),
            last.latent.tumor_burden,
            last.latent.resistance_frequency,
            float((last.connectome or {}).get("da") or 0.0),
            proteins,
            last.t,
            immune=last.latent.I_act,
            fusion_af=last.latent.fusion_allele_fraction,
        )
    return frames, last, sim


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Confluence cinematic clip (real flybody mesh only)")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--out", default="docs/demo/cinematic.mp4")
    parser.add_argument(
        "--prefer-real",
        action="store_true",
        help="Accepted for compatibility; real mesh is required either way.",
    )
    parser.add_argument(
        "--allow-stub",
        action="store_true",
        help="Disabled. Stub footage is never written.",
    )
    args = parser.parse_args(argv)
    if args.allow_stub:
        raise SystemExit("--allow-stub is rejected: share clips must be fruitfly.xml")

    print("Rendering flybody fruitfly.xml clip — research visualization, not a clinical demo.")
    print("Simulated therapy metrics are research scores, not a clinical outcome.")
    frames, last, sim = render_clip(
        seconds=args.seconds,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    out = Path(args.out)
    _ffmpeg_write_mp4(out, frames, args.fps)
    still_a = out.with_name("still_hero.png")
    still_b = out.with_name("still_mid.png")
    still_mesh = out.with_name("still_mesh.png")
    still_track = out.with_name("still_track.png")
    _ffmpeg_png(still_a, frames[0])
    _ffmpeg_png(still_b, frames[len(frames) // 2])
    # Clean anatomical stills (no HUD) for flybody / Nature / Menagerie comparison.
    clean = sim.embodiment.render_rgb(width=args.width, height=args.height)
    if clean is None:
        raise SystemExit("clean fruitfly.xml still failed after the clip")
    _ffmpeg_png(still_mesh, clean)
    sim.embodiment.camera = "walker/track1"
    track = sim.embodiment.render_rgb(width=args.width, height=args.height)
    if track is None:
        raise SystemExit("walker/track1 fruitfly.xml still failed")
    _ffmpeg_png(still_track, track)
    print(f"mesh=TuragaLab/flybody fruitfly.xml  backend={last.embodiment.get('backend') if last.embodiment else '?'}")
    print(f"wrote {out}  {frames.shape[0]} frames  t={last.t:.2f}d  burden={last.latent.tumor_burden:.2f}")
    print(f"stills {still_a} {still_b} {still_mesh} {still_track}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
