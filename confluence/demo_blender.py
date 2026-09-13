"""One-command scientific visualization export for Blender.

    export MUJOCO_GL=osmesa
    python3 -m confluence.demo_blender --out docs/demo/blender

Dumps MuJoCo fruitfly.xml PNG frames + a JSON/CSV HUD sidecar from the
real closed loop. A bpy script in the same folder is for local Blender 4.x.

Simulation / research only. Not a cure clip. Not generative biology.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from confluence.viz.blender_export import NON_CLAIM, SIMULATION_LABEL, export_closed_loop_blender


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Export logged Confluence sims for Blender (simulation / research)"
    )
    parser.add_argument("--out", default="docs/demo/blender")
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--controller", default="E", help="A/B/E/F (F is F-256 proxy)")
    parser.add_argument(
        "--require-mesh",
        action="store_true",
        help="Fail if fruitfly.xml cannot render (default: sidecar-only fallback)",
    )
    parser.add_argument("--allow-stub", action="store_true", help="Rejected.")
    args = parser.parse_args(argv)
    if args.allow_stub:
        raise SystemExit("--allow-stub is rejected: mesh frames must be fruitfly.xml")
    print(SIMULATION_LABEL)
    print(NON_CLAIM)
    report = export_closed_loop_blender(
        Path(args.out),
        n_frames=args.frames,
        dt=args.dt,
        fps=args.fps,
        width=args.width,
        height=args.height,
        seed=args.seed,
        controller=args.controller,
        require_mesh=args.require_mesh,
    )
    print("wrote", report["out_dir"], "frames", report["n_frames"], "png", report["n_png"], "mesh_live", report["mesh_live"])
    print("sidecar", report["sidecar"])
    if not report["mesh_live"]:
        print("No fruitfly.xml dump. Sidecar is still the logged ODE series. Open bpy script locally after installing .[flybody].")
    return 0


if __name__ == "__main__":
    sys.exit(main())
