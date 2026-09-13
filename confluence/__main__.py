"""Launch the Confluence v2 interactive session.

    python -m confluence
    python -m confluence --port 8765
    python -m confluence --benchmark --trials 2
    python -m confluence.embodiment --task template
    python -m confluence.train_full_brain --neurons 2048 --episodes 3
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Confluence v2 interactive / benchmark")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--horizon", type=float, default=40.0)
    args = parser.parse_args(argv)

    if args.benchmark:
        from confluence.benchmarks.runner import main as bench_main

        bench_main(["--trials", str(args.trials), "--horizon", str(args.horizon)])
        return 0

    import uvicorn

    print()
    print("  Confluence v2 — computational closed-loop session")
    print("  Research simulation only. Not a medical device.")
    print(f"  Open  http://{args.host}:{args.port}")
    print("  Stop  Ctrl+C")
    print()
    uvicorn.run(
        "confluence.telemetry.websocket_server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
