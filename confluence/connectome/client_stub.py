"""Credential-free FlyWire client.

Real CAVEclient / fafbseg ingestion needs a CAVE token and network access
to FlyWire materializations. This adapter exposes the same surface so a
later token-backed implementation can drop in without changing controllers.
"""

from __future__ import annotations

import os
from typing import Optional

from confluence.connectome.fafb_loader import FAFBLoader
from confluence.connectome.schemas import ConnectomeGraph


class FlyWireClient:
    """Load an MB subcircuit. Falls back to the biological stub graph."""

    def __init__(
        self,
        token: Optional[str] = None,
        datastack: str = "flywire_fafb_production",
    ):
        self.token = token or os.environ.get("CAVE_TOKEN") or os.environ.get("FLYWIRE_TOKEN")
        self.datastack = datastack
        self.loader = FAFBLoader(token=self.token, datastack=datastack)

    @property
    def using_stub(self) -> bool:
        return not self.loader.has_credentials

    def fetch_mushroom_body(self, n_kc: int = 256) -> ConnectomeGraph:
        """Return a Kenyon-cell-centered MB graph.

        Without credentials this always returns the structured stub.
        With credentials, `FAFBLoader.load_v783` is the ingestion hook
        (currently still stubbed — see that class for the intended CAVE path).
        """
        return self.loader.load_v783(n_kc=n_kc)
