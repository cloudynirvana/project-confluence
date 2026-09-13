"""Controller E — plastic mushroom-body with dopamine-modulated weights."""

from __future__ import annotations

from confluence.contracts import CONTROL_DRUG_IDS, FUSION_CHANNEL_IDS, InterventionAction, ObservationRecord
from confluence.controllers.base import BaseController, ControllerContext
from confluence.neural_engine.network import MushroomBodyNetwork, NetworkConfig


class PlasticMushroomBodyController(BaseController):
    letter = "E"
    name = "Plastic mushroom body"
    uses_connectome = True

    def __init__(self, n_kc: int = 256, seed: int = 7, **kwargs):
        kwargs.setdefault("drug_ids", CONTROL_DRUG_IDS + FUSION_CHANNEL_IDS)
        super().__init__(**kwargs)
        self.network = MushroomBodyNetwork(
            NetworkConfig(n_kc=n_kc, seed=seed, plastic=True, n_obs=7, n_out=len(self.drug_ids)),
            drug_ids=self.drug_ids,
        )

    def reset(self) -> None:
        self.network.mbon_rate[:] = 0.0
        self.network.prev_burden = None
        self.network.prev_resist = None
        self.network.prev_fusion = None

    def decide(self, observation: ObservationRecord, context: ControllerContext) -> InterventionAction:
        conc_sum = sum(context.concentrations.values())
        u = self.network.step(observation, conc_sum, context.dt)
        if observation.host_toxicity_warning:
            u = u * 0.45
        return self._action(observation.t, u, source="E", notes="plastic MB + DA")

    def connectome_telemetry(self):
        return self.network.telemetry()
