"""Read-only HTTP client for the OnCo static API."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from confluence.onco.attribution import ATTRIBUTION
from confluence.onco.cache import OncoCache
from confluence.onco.schemas import OncoRef

DEFAULT_API = "https://onco.cc/api/v1"
USER_AGENT = "project-confluence-onco-adapter/0.1 (research; CC-BY-NC-4.0 consumer)"


class OncoClientError(RuntimeError):
    pass


class OncoClient:
    """Fetch OnCo records. Never writes ODE terms."""

    def __init__(
        self,
        api_root: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        timeout: float = 30.0,
    ):
        self.api_root = (api_root or os.environ.get("ONCO_API") or DEFAULT_API).rstrip("/")
        self.timeout = timeout
        self.cache = OncoCache(cache_dir) if cache_dir else None

    def _get_json(self, rel: str) -> Any:
        rel = rel.lstrip("/")
        if self.cache:
            hit = self.cache.get(rel)
            if hit is not None:
                return hit
        if self.api_root.startswith("file://"):
            path = Path(self.api_root[7:]) / rel
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif Path(self.api_root).exists():
            path = Path(self.api_root) / rel
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            url = f"{self.api_root}/{rel}"
            req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            try:
                with urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except HTTPError as exc:
                raise OncoClientError(f"HTTP {exc.code} for {url}") from exc
            except URLError as exc:
                raise OncoClientError(f"network error for {url}: {exc}") from exc
        if self.cache:
            self.cache.put(rel, payload, api_root=self.api_root)
        return payload

    def get_meta(self) -> Dict[str, Any]:
        return self._get_json("meta.json")

    def get_entity(self, entity_id: str) -> Dict[str, Any]:
        return self._get_json(f"entities/{entity_id}.json")

    def list_kind(self, kind: str) -> Any:
        plural = kind if kind.endswith("s") else f"{kind}s"
        return self._get_json(f"{plural}.json")

    def search_index(self) -> Any:
        return self._get_json("search.json")

    def iter_search(self, query: str, kind: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        q = query.lower()
        blob = self.search_index()
        docs = blob if isinstance(blob, list) else blob.get("documents") or blob.get("items") or []
        for doc in docs:
            if kind and str(doc.get("kind", "")).lower() != kind.lower():
                continue
            hay = " ".join(
                str(doc.get(k, "")) for k in ("id", "name", "aka", "tldr", "route")
            ).lower()
            if q in hay:
                yield doc

    def as_ref(self, entity: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> OncoRef:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        build = None
        if meta:
            build = str(meta.get("buildDate") or meta.get("version") or "")
        return OncoRef(
            onco_id=str(entity.get("id") or entity.get("onco_id")),
            kind=str(entity.get("kind") or "unknown"),
            name=str(entity.get("name") or entity.get("id")),
            route=entity.get("route"),
            tldr=entity.get("tldr"),
            onco_as_of=entity.get("asOf") or entity.get("as_of"),
            retrieved_at=now,
            onco_build=build,
            attribution=ATTRIBUTION,
        )
