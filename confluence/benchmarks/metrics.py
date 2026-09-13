"""Trial-level oncology simulation metrics (computational, not clinical)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TrialMetrics:
    archetype: str
    controller: str
    seed: int
    pfs_days: float
    resistance_emergence_day: Optional[float]
    cumulative_toxicity: float
    pharmacological_burden: float
    final_burden: float
    final_resistance: float
    terminal_toxicity: bool
    steps: int

    def as_dict(self) -> Dict:
        return {
            "archetype": self.archetype,
            "controller": self.controller,
            "seed": self.seed,
            "pfs_days": self.pfs_days,
            "resistance_emergence_day": self.resistance_emergence_day,
            "cumulative_toxicity": self.cumulative_toxicity,
            "pharmacological_burden": self.pharmacological_burden,
            "final_burden": self.final_burden,
            "final_resistance": self.final_resistance,
            "terminal_toxicity": self.terminal_toxicity,
            "steps": self.steps,
        }


def compute_metrics(
    times: np.ndarray,
    burdens: np.ndarray,
    resists: np.ndarray,
    health: np.ndarray,
    conc_sum: np.ndarray,
    infusions: np.ndarray,
    archetype: str,
    controller: str,
    seed: int,
    baseline_burden: float,
    resist_threshold: float = 0.45,
    progression_factor: float = 1.20,
) -> TrialMetrics:
    """PFS: first time burden exceeds 120% of baseline (or host terminal)."""
    terminal_idx = np.where(health <= 0.2)[0]
    progress_idx = np.where(burdens >= progression_factor * max(baseline_burden, 1e-6))[0]

    pfs = float(times[-1])
    if terminal_idx.size:
        pfs = min(pfs, float(times[terminal_idx[0]]))
    if progress_idx.size:
        pfs = min(pfs, float(times[progress_idx[0]]))

    emerge = np.where(resists >= resist_threshold)[0]
    emerge_day = float(times[emerge[0]]) if emerge.size else None

    dt = float(np.mean(np.diff(times))) if times.size > 1 else 1.0
    tox = float(np.sum(np.maximum(0.2 - health, 0.0)) + np.sum(np.maximum(0.45 - health, 0.0)) * 0.25)
    pharm = float(np.sum(conc_sum) * dt + np.sum(infusions) * dt)

    return TrialMetrics(
        archetype=archetype,
        controller=controller,
        seed=seed,
        pfs_days=pfs,
        resistance_emergence_day=emerge_day,
        cumulative_toxicity=tox,
        pharmacological_burden=pharm,
        final_burden=float(burdens[-1]),
        final_resistance=float(resists[-1]),
        terminal_toxicity=bool(health[-1] <= 0.2),
        steps=int(times.size),
    )


def summarize(rows: List[TrialMetrics]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[TrialMetrics]] = {}
    for row in rows:
        key = f"{row.archetype}:{row.controller}"
        grouped.setdefault(key, []).append(row)
    out = {}
    for key, items in grouped.items():
        pfs = np.array([r.pfs_days for r in items], dtype=float)
        tox = np.array([r.cumulative_toxicity for r in items], dtype=float)
        pharm = np.array([r.pharmacological_burden for r in items], dtype=float)
        out[key] = {
            "n": float(len(items)),
            "pfs_mean": float(np.mean(pfs)),
            "pfs_std": float(np.std(pfs)),
            "toxicity_mean": float(np.mean(tox)),
            "pharm_burden_mean": float(np.mean(pharm)),
            "terminal_rate": float(np.mean([r.terminal_toxicity for r in items])),
        }
    return out
