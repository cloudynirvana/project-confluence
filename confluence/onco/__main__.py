"""CLI: python -m confluence.onco bind --id ldha"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from confluence.onco.attribution import ATTRIBUTION
from confluence.onco.client import OncoClient, OncoClientError
from confluence.onco.mapper import bind, wired_claims


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="OnCo adapter (read-only). Does not modify CancerODE."
    )
    parser.add_argument("--api", default=None, help="ONCO_API root or local fixture dir")
    parser.add_argument("--cache", default=None, help="cache directory")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_meta = sub.add_parser("meta", help="print OnCo meta.json")
    p_meta.set_defaults(cmd="meta")

    p_bind = sub.add_parser("bind", help="map a name/id onto Confluence slots")
    p_bind.add_argument("--id", required=True)

    p_get = sub.add_parser("get", help="fetch one OnCo entity")
    p_get.add_argument("--id", required=True)

    p_wired = sub.add_parser("wired", help="list seed EvidenceObjects already in the RHS")

    args = parser.parse_args(argv)
    print(ATTRIBUTION, file=sys.stderr)

    if args.cmd == "bind":
        hits = bind(args.id)
        if not hits:
            print(json.dumps({"query": args.id, "bindings": [], "attribution": ATTRIBUTION}, indent=2))
            return 1
        payload = {
            "query": args.id,
            "bindings": [h.__dict__ for h in hits],
            "attribution": ATTRIBUTION,
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.cmd == "wired":
        print(json.dumps({"wired_claims": wired_claims(), "attribution": ATTRIBUTION}, indent=2))
        return 0

    cache = Path(args.cache) if args.cache else None
    client = OncoClient(api_root=args.api, cache_dir=cache)
    try:
        if args.cmd == "meta":
            meta = client.get_meta()
            meta["attribution"] = ATTRIBUTION
            print(json.dumps(meta, indent=2))
            return 0
        if args.cmd == "get":
            ent = client.get_entity(args.id)
            ref = client.as_ref(ent, client.get_meta() if _can_meta(client) else None)
            print(json.dumps({"ref": ref.model_dump(), "entity_keys": sorted(ent.keys()), "attribution": ATTRIBUTION}, indent=2))
            return 0
    except OncoClientError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _can_meta(client: OncoClient) -> bool:
    try:
        client.get_meta()
        return True
    except OncoClientError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
