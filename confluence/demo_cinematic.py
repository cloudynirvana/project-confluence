"""Render a short dark-lab closed-loop clip for README / sharing.

    python -m confluence.demo_cinematic --seconds 12 --out docs/demo/cinematic.mp4

Uses the cinematic stub fly when MuJoCo is absent. With flybody extras:

    export MUJOCO_GL=osmesa
    pip install -e ".[flybody]"
    python -m confluence.demo_cinematic --seconds 12 --prefer-real
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from confluence.cinematic import render_frame
from confluence.controllers import make_controller
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
    proc = subprocess.run(cmd, input=frames.tobytes(), check=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode()[:800])


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
    prefer_real: bool = False,
) -> tuple:
    n = int(round(seconds * fps))
    ctrl = make_controller("F", n_neurons=512, seed=3)
    sim = ClosedLoopSimulator(
        archetype="glioblastoma",
        controller=ctrl,
        dt=0.15,
        seed=3,
        embodiment_enabled=True,
    )
    if not prefer_real:
        sim.embodiment.backend = "kinematic_stub"
        sim.embodiment._env = None
        sim.embodiment.render_enabled = False
    frames = np.zeros((n, height, width, 3), dtype=np.uint8)
    last = sim.history[-1]
    for i in range(n):
        last = sim.step(run_cancer=True, run_embodiment=True)
        proteins = (last.action.infusion and [
            k for k, v in last.action.infusion.items()
            if k.startswith("protein_") and float(v) > 0.05
        ]) or []
        frames[i] = render_frame(
            width=width,
            height=height,
            embodiment=last.embodiment or {},
            connectome=last.connectome or {},
            latent={
                "tumor_burden": last.latent.tumor_burden,
                "resistance_frequency": last.latent.resistance_frequency,
            },
            proteins=proteins,
            t_days=last.t,
            seed=i,
        )
    return frames, last


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Confluence cinematic demo clip")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--out", default="docs/demo/cinematic.mp4")
    parser.add_argument("--prefer-real", action="store_true")
    args = parser.parse_args(argv)

    print("Rendering cinematic clip (research simulation, not a clinical demo).")
    frames, last = render_clip(
        seconds=args.seconds,
        fps=args.fps,
        width=args.width,
        height=args.height,
        prefer_real=args.prefer_real,
    )
    out = Path(args.out)
    _ffmpeg_write_mp4(out, frames, args.fps)
    still_a = out.with_name("still_hero.png")
    still_b = out.with_name("still_mid.png")
    _ffmpeg_png(still_a, frames[0])
    _ffmpeg_png(still_b, frames[len(frames) // 2])
    print(f"wrote {out}  {frames.shape[0]} frames  t={last.t:.2f}d  burden={last.latent.tumor_burden:.2f}")
    print(f"stills {still_a} {still_b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
