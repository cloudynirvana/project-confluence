"""Weight checkpoints for the sparse full-brain network."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from confluence.neural_engine.full_brain import FullBrainNetwork


def save_checkpoint(
    network: FullBrainNetwork,
    path: str | Path,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    path = network.save(path)
    if extra:
        data = dict(np.load(path, allow_pickle=True))
        for key, value in extra.items():
            data[key] = np.asarray(value)
        np.savez_compressed(path, **data)
    return Path(path)


def load_checkpoint(network: FullBrainNetwork, path: str | Path) -> Dict[str, Any]:
    network.load(path)
    raw = np.load(path, allow_pickle=True)
    meta = {}
    for key in raw.files:
        if key in {"episode", "reward", "n_neurons", "seed"}:
            meta[key] = raw[key].item() if raw[key].shape == () else raw[key]
    return meta
