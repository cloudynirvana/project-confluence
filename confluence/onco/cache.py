"""On-disk cache for OnCo JSON. Never commit the payload tree.

Cached records remain OnCo data (CC BY-NC 4.0), not MIT Confluence code.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from confluence.onco.attribution import ATTRIBUTION, LICENCE, SOURCE_URL


def content_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def envelope(payload: Any, *, rel: str, api_root: str, build: Optional[str] = None) -> Dict[str, Any]:
    return {
        "source": "OnCo",
        "source_uri": SOURCE_URL,
        "api_root": api_root,
        "rel": rel,
        "license": LICENCE,
        "attribution": ATTRIBUTION,
        "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "api_version": build,
        "schema_version": "onco-adapter-0.3",
        "content_hash": content_hash(payload),
        "payload": payload,
    }


class OncoCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        safe = key.strip("/").replace("..", "_")
        return self.root / f"{safe}.json"

    def get(self, key: str) -> Optional[Any]:
        path = self.path_for(key)
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "payload" in raw and "attribution" in raw:
            return raw["payload"]
        return raw

    def get_envelope(self, key: str) -> Optional[Dict[str, Any]]:
        path = self.path_for(key)
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None

    def put(
        self,
        key: str,
        payload: Any,
        api_root: str = "",
        build: Optional[str] = None,
    ) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        wrapped = envelope(payload, rel=key, api_root=api_root, build=build)
        path.write_text(json.dumps(wrapped, indent=2, sort_keys=True), encoding="utf-8")
        return path
