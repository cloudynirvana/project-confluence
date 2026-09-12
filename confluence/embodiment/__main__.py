"""Headless embodiment demo.

    python -m confluence.embodiment
    python -m confluence.embodiment --task template
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from confluence.embodiment.flybody_bridge import FlybodyBridge, flybody_status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Confluence flybody embodiment demo")
    parser.add_argument("--task", default="walk_imitation",
                        help="walk_imitation | template | flight_imitation")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--stub", action="store_true", help="force kinematic stub")
    args = parser.parse_args(argv)

    print(json.dumps(flybody_status(), indent=2))
    bridge = FlybodyBridge(task=args.task, prefer_real=not args.stub)
    print("backend", bridge.backend, "action_dim", bridge.action_dim)
    print(bridge.notes)
    u = np.array([0.4, 0.2, 0.3, 0.1, 0.25])
    mbon = np.linspace(0.1, 0.8, 8)
    for i in range(args.steps):
        tel = bridge.step(u, mbon)
        if i == 0 or i == args.steps - 1:
            print(
                f"step {i:03d} t_fly={tel.t_fly:.3f} rms={tel.action_rms:.3f} "
                f"reward={tel.reward:.3f} xpos={tel.xpos}"
            )
            if not np.all(np.isfinite(tel.action)):
                raise SystemExit("NaN/Inf in fly action")
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
