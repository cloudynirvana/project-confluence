#!/usr/bin/env python3
"""Emit gated Disease Profile JSON + SUMMARY.md from a case table.

Research / in-silico only. Refuses rows that try to write ODE Θ.
Never edits CancerODE.

  python3 scripts/build_disease_profile_pack.py
  python3 scripts/build_disease_profile_pack.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from confluence.profiles.pack import (  # noqa: E402
    DEFAULT_OUT,
    DEFAULT_TABLE,
    build_pack,
    load_table,
    write_pack,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build research-only Disease Profile JSON from a case table."
    )
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print counts; still exercises refuse_examples.",
    )
    args = parser.parse_args(argv)

    accepted, refused_cases = build_pack(args.table, include_refuse_examples=False)
    _, refused_all = build_pack(args.table, include_refuse_examples=True)
    example_slugs = {row.get("slug") for row in (load_table(args.table).get("refuse_examples") or [])}
    refuse_examples = [reason for row, reason in refused_all if row.get("slug") in example_slugs]

    summary = {
        "accepted": len(accepted),
        "refused_cases": len(refused_cases),
        "refused_examples": refuse_examples,
        "disclaimer": "research / in-silico only; not clinical CDS",
        "files": [name for name, _ in accepted],
    }
    if not accepted or refused_cases:
        print(json.dumps(summary, indent=2))
        return 1
    if example_slugs and not refuse_examples:
        print(json.dumps(summary, indent=2))
        print("expected refuse_examples to be rejected", file=sys.stderr)
        return 1
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    write_pack(accepted, args.out, refused=list(refused_all))
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
