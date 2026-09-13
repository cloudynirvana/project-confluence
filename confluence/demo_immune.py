"""Fly-brain immune + chimeric-protein demo on the real flybody mesh.

    export MUJOCO_GL=osmesa
    python -m confluence.demo_immune --out docs/demo/immune_chimeric.mp4

Shows controller F (sparse secretory net) driving IFN-γ / IL-2 / chimeric
T-cell engager / checkpoint + fusion TKIs while I_act rises. Research
visualization only — not a clinical immune-therapy demo and not a cure.

Refuses stub / CPG footage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

from confluence.cinematic import overlay_hud
from confluence.controllers import make_controller
from confluence.demo_cinematic import _ffmpeg_png, _ffmpeg_write_mp4
from confluence.embodiment.flybody_bridge import flybody_available, flybody_status
from confluence.loop import ClosedLoopSimulator


def render_immune_clip(
    seconds: float = 8.0,
    fps: int = 20,
    width: int = 1280,
    height: int = 720,
):
    if not flybody_available():
        raise SystemExit(
            "demo_immune refuses to ship stub footage. "
            f"Install the anatomical mesh:\n  {flybody_status()['install']}"
        )
    n = int(round(seconds * fps))
    ctrl = make_controller("F", n_neurons=512, seed=11)
    sim = ClosedLoopSimulator(
        archetype="melanoma_persister",
        controller=ctrl,
        # Short cancer clock so an 8 s / 20 fps clip stays in the
        # immune-rise window (~8 sim days), not a 50-day burden wipe.
        dt=0.05,
        seed=11,
        embodiment_enabled=True,
    )
    sim.embodiment.camera = "walker/hero"
    sim.embodiment.render_size = (width, height)
    sim.embodiment.render_enabled = False
    if sim.embodiment.backend != "flybody" or sim.embodiment._env is None:
        raise SystemExit("flybody imported but MuJoCo env is not live (MUJOCO_GL=osmesa).")

    import numpy as np

    frames = np.zeros((n, height, width, 3), dtype=np.uint8)
    start = sim.history[-1]
    last = start
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
    metrics = {
        "research_only": True,
        "not_a_cure": True,
        "mesh": "TuragaLab/flybody fruitfly.xml",
        "controller": "F",
        "archetype": "melanoma_persister",
        "fusion_class": last.latent.fusion_display,
        "t_days": float(last.t),
        "I_act_start": float(start.latent.I_act),
        "I_act_end": float(last.latent.I_act),
        "fusion_af_start": float(start.latent.fusion_allele_fraction),
        "fusion_af_end": float(last.latent.fusion_allele_fraction),
        "burden_start": float(start.latent.tumor_burden),
        "burden_end": float(last.latent.tumor_burden),
        "proteins_end": [
            k
            for k, v in (last.action.infusion or {}).items()
            if k.startswith("protein_") and float(v) > 0.05
        ],
        "fusion_tki_end": [
            k
            for k, v in (last.action.infusion or {}).items()
            if k.startswith("tki_") and float(v) > 0.05
        ],
        "engager": float((last.action.infusion or {}).get("protein_chimeric_engager") or 0.0),
    }
    return frames, last, sim, metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fly-brain immune + chimeric-protein demo (real mesh)")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--out", default="docs/demo/immune_chimeric.mp4")
    parser.add_argument("--allow-stub", action="store_true")
    args = parser.parse_args(argv)
    if args.allow_stub:
        raise SystemExit("--allow-stub is rejected: this demo must be fruitfly.xml")

    print("Fly-brain immune + chimeric-protein demo — research visualization, not a cure.")
    frames, last, _sim, metrics = render_immune_clip(
        seconds=args.seconds, fps=args.fps, width=args.width, height=args.height
    )
    out = Path(args.out)
    _ffmpeg_write_mp4(out, frames, args.fps)
    still = out.with_name("still_immune_hero.png")
    mid = out.with_name("still_immune_mid.png")
    _ffmpeg_png(still, frames[0])
    _ffmpeg_png(mid, frames[len(frames) // 2])
    metrics_path = out.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out} {still} {mid} {metrics_path}")
    if metrics["I_act_end"] <= metrics["I_act_start"]:
        print("note: I_act did not rise in this short horizon (still a valid research clip).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
