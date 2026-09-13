"""Controller F — sparse full-brain net with protein/biologic effectors."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from confluence.contracts import ALL_EFFECTOR_IDS, INTERACTIVE_BRAIN_NEURONS, InterventionAction, ObservationRecord
from confluence.controllers.base import BaseController, ControllerContext
from confluence.controllers.immune_prior import apply_immune_secretory_prior
from confluence.neural_engine.full_brain import FullBrainConfig, FullBrainNetwork


class FullBrainController(BaseController):
    letter = "F"
    name = "Full-brain secretory"
    uses_connectome = True

    def __init__(
        self,
        n_neurons: int = INTERACTIVE_BRAIN_NEURONS,
        seed: int = 7,
        checkpoint: Optional[str | Path] = None,
        drug_ids: Sequence[str] = ALL_EFFECTOR_IDS,
        **kwargs,
    ):
        kwargs.pop("n_kc", None)
        super().__init__(drug_ids=drug_ids)
        self.network = FullBrainNetwork(
            FullBrainConfig(n_neurons=int(n_neurons), seed=int(seed), plastic=True),
            drug_ids=self.drug_ids,
        )
        if checkpoint:
            self.network.load(checkpoint)

    def reset(self) -> None:
        self.network.reset_rates()

    def decide(self, observation: ObservationRecord, context: ControllerContext) -> InterventionAction:
        conc_sum = sum(context.concentrations.values())
        u = self.network.step(observation, conc_sum, context.dt)
        u = apply_immune_secretory_prior(u, self.drug_ids, observation)
        if observation.host_toxicity_warning:
            u = u * 0.45
        return self._action(
            observation.t,
            u,
            source="F",
            notes="sparse full-brain → immune + chimeric-protein effectors (simulated)",
        )

    def connectome_telemetry(self):
        return self.network.telemetry()
