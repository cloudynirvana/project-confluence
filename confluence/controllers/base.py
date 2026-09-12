"""Controller interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from confluence.contracts import CONTROL_DRUG_IDS, InterventionAction, ObservationRecord


@dataclass
class ControllerContext:
    concentrations: Dict[str, float] = field(default_factory=dict)
    archetype: str = ""
    dt: float = 0.1
    step_index: int = 0


class BaseController:
    letter: str = "?"
    name: str = "base"
    uses_connectome: bool = False

    def __init__(self, drug_ids: Sequence[str] = CONTROL_DRUG_IDS):
        self.drug_ids = tuple(drug_ids)

    def reset(self) -> None:
        return None

    def decide(self, observation: ObservationRecord, context: ControllerContext) -> InterventionAction:
        raise NotImplementedError

    def connectome_telemetry(self) -> Optional[Dict]:
        return None

    def _action(self, t: float, u: np.ndarray, source: str, notes: str = "") -> InterventionAction:
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        clipped = bool(np.any(u < 0) or np.any(u > 1))
        infusion = {d: float(u[i]) for i, d in enumerate(self.drug_ids)}
        return InterventionAction(t=t, infusion=infusion, source=source, clipped=clipped, notes=notes)
