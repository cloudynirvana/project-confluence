"""Controller A — standard-of-care maximum tolerated dose schedule."""

from __future__ import annotations

import numpy as np

from confluence.contracts import InterventionAction, ObservationRecord
from confluence.controllers.base import BaseController, ControllerContext

# Archetype-specific SOC mixes (simulation units, fraction of MTD).
SOC_SCHEDULE = {
    "glioblastoma": {
        "anti_pd1": 0.35,
        "tgfb_inhibitor": 0.15,
        "mct1": 0.20,
        "hdac": 0.10,
        "targeted_kinase": 0.85,
    },
    "pancreatic_pdac": {
        "anti_pd1": 0.20,
        "tgfb_inhibitor": 0.80,
        "mct1": 0.25,
        "hdac": 0.15,
        "targeted_kinase": 0.55,
    },
    "melanoma_persister": {
        "anti_pd1": 0.85,
        "tgfb_inhibitor": 0.10,
        "mct1": 0.15,
        "hdac": 0.20,
        "targeted_kinase": 0.75,
    },
}


class MTDController(BaseController):
    letter = "A"
    name = "MTD / standard-of-care"

    def decide(self, observation: ObservationRecord, context: ControllerContext) -> InterventionAction:
        mix = SOC_SCHEDULE.get(context.archetype, SOC_SCHEDULE["glioblastoma"])
        u = np.array([mix.get(d, 0.6) for d in self.drug_ids], dtype=float)
        # Continuous MTD: no holidays. Toxicity is the intended weakness.
        return self._action(observation.t, u, source="A", notes="continuous MTD mix")
