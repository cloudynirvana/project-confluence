"""Optional flybody embodiment layer (DeepMind / HHMI Janelia)."""

from confluence.embodiment.flybody_bridge import (
    FLYBODY_COMMIT,
    FLYBODY_REPO,
    WALK_ACTION_DIM,
    EmbodimentTelemetry,
    FlybodyBridge,
    flybody_available,
    flybody_status,
)

__all__ = [
    "FLYBODY_COMMIT",
    "FLYBODY_REPO",
    "WALK_ACTION_DIM",
    "EmbodimentTelemetry",
    "FlybodyBridge",
    "flybody_available",
    "flybody_status",
]
