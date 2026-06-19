"""Run the PDAC rogue-closure model and write reproducible outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.pdac_rogue_closure import SCENARIOS, simulate, summarize, write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PDAC rogue-closure model.")
    parser.add_argument("--scenario", choices=SCENARIOS, default="adaptive_closure")
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--days", type=float, default=180.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--output-dir", default="results/pdac_rogue_closure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = SCENARIOS if args.all_scenarios else [args.scenario]
    all_points = []

    for scenario in scenarios:
        points = simulate(scenario=scenario, days=args.days, dt=args.dt)
        all_points.extend(points)
        summary = summarize(points)
        print(
            f"{scenario}: final_tumor={summary['final_tumor']:.3f}, "
            f"final_closure={summary['final_rogue_closure']:.3f}, "
            f"final_access={summary['final_host_access']:.3f}, "
            f"resistance={summary['final_resistance']:.3f}"
        )

    csv_path, json_path, svg_path = write_outputs(all_points, Path(args.output_dir))
    print(f"CSV: {csv_path}")
    print(f"Summary: {json_path}")
    print(f"SVG: {svg_path}")


if __name__ == "__main__":
    main()

