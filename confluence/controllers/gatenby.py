"""Controller B — Gatenby adaptive therapy.

Treat at MTD until tumor burden drops 50% from the current reference,
then halt. Resume when burden recovers to the reference.

Reference: Gatenby et al., Cancer Research 2009 (adaptive therapy).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from confluence.contracts import InterventionAction, ObservationRecord
from confluence.controllers.base import BaseController, ControllerContext
from confluence.controllers.mtd import SOC_SCHEDULE


class GatenbyAdaptiveController(BaseController):
    letter = "B"
    name = "Gatenby adaptive"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reference: Optional[float] = None
        self.treating = True

    def reset(self) -> None:
        self.reference = None
        self.treating = True

    def decide(self, observation: ObservationRecord, context: ControllerContext) -> InterventionAction:
        burden = max(observation.tumor_burden, 1e-8)
        if self.reference is None:
            self.reference = burden
            self.treating = True

        if self.treating and burden <= 0.50 * self.reference:
            self.treating = False
        elif (not self.treating) and burden >= self.reference:
            self.treating = True

        mix = SOC_SCHEDULE.get(context.archetype, SOC_SCHEDULE["glioblastoma"])
        if self.treating:
            u = np.array([mix.get(d, 0.6) for d in self.drug_ids], dtype=float)
            notes = f"treat; ref={self.reference:.3f}"
        else:
            u = np.zeros(len(self.drug_ids), dtype=float)
            notes = f"holiday; ref={self.reference:.3f}"
        return self._action(observation.t, u, source="B", notes=notes)
