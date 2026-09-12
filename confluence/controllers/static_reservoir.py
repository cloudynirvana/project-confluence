"""Controller D — frozen mushroom-body reservoir + linear readout."""

from __future__ import annotations

from typing import Optional

import numpy as np

from confluence.contracts import InterventionAction, ObservationRecord
from confluence.controllers.base import BaseController, ControllerContext
from confluence.neural_engine.network import MushroomBodyNetwork, NetworkConfig


class StaticReservoirController(BaseController):
    letter = "D"
    name = "Static MB reservoir"
    uses_connectome = True

    def __init__(self, n_kc: int = 256, seed: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.network = MushroomBodyNetwork(
            NetworkConfig(n_kc=n_kc, seed=seed, plastic=False)
        )
        self._calibrated = False

    def reset(self) -> None:
        self.network.mbon_rate[:] = 0.0
        self.network.prev_burden = None
        self.network.prev_resist = None

    def _calibrate_readout(self) -> None:
        """One-shot ridge readout: map random MBON patterns to SOC-like mixes."""
        rng = np.random.default_rng(11)
        n_mbon = self.network.n_mbon
        n_out = len(self.drug_ids)
        X = rng.random((48, n_mbon))
        Y = 0.3 + 0.5 * rng.random((48, n_out))
        # Ridge
        xtx = X.T @ X + 0.25 * np.eye(n_mbon)
        self.network.w_out = np.linalg.solve(xtx, X.T @ Y).T
        self._calibrated = True

    def decide(self, observation: ObservationRecord, context: ControllerContext) -> InterventionAction:
        if not self._calibrated:
            self._calibrate_readout()
        self.network.encode(observation)
        u = self.network.decode()
        if observation.host_toxicity_warning:
            u = u * 0.4
        return self._action(observation.t, u, source="D", notes="frozen reservoir")

    def connectome_telemetry(self):
        return self.network.telemetry()
