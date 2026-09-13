"""Controller E — plastic mushroom-body with dopamine-modulated weights."""

from __future__ import annotations

from confluence.contracts import ALL_EFFECTOR_IDS, OBS_VECTOR_NAMES, InterventionAction, ObservationRecord
from confluence.controllers.base import BaseController, ControllerContext
from confluence.controllers.immune_prior import apply_immune_secretory_prior
from confluence.neural_engine.network import MushroomBodyNetwork, NetworkConfig


class PlasticMushroomBodyController(BaseController):
    letter = "E"
    name = "Plastic mushroom body"
    uses_connectome = True

    def __init__(self, n_kc: int = 256, seed: int = 7, plastic: bool = True, **kwargs):
        kwargs.setdefault("drug_ids", ALL_EFFECTOR_IDS)
        super().__init__(**kwargs)
        self.network = MushroomBodyNetwork(
            NetworkConfig(
                n_kc=n_kc,
                seed=seed,
                plastic=bool(plastic),
                n_obs=len(OBS_VECTOR_NAMES),
                n_out=len(self.drug_ids),
            ),
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
        u = apply_immune_secretory_prior(u, self.drug_ids, observation)
        if observation.host_toxicity_warning:
            u = u * 0.45
        return self._action(observation.t, u, source="E", notes="plastic MB + immune/chimeric prior")

    def connectome_telemetry(self):
        return self.network.telemetry()
