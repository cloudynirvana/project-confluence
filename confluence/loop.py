"""Closed-loop simulator: observations → controller → drugs → ODE → Y'."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from confluence.cancer_env.archetypes import get_archetype
from confluence.cancer_env.ode_system import CancerODE
from confluence.cancer_env.observation_layer import ObservationLayer
from confluence.contracts import (
    CONTROL_DRUG_IDS,
    InterventionAction,
    LatentCancerState,
    ObservationRecord,
)
from confluence.controllers.base import BaseController, ControllerContext
from confluence.controllers.plastic_mb import PlasticMushroomBodyController
from confluence.pharmacology.pk_pd_model import PKPDModel


@dataclass
class SimFrame:
    t: float
    latent: LatentCancerState
    observation: ObservationRecord
    action: InterventionAction
    concentrations: Dict[str, float]
    occupancies: Dict[str, float]
    connectome: Optional[Dict] = None
    terminal: bool = False


@dataclass
class ClosedLoopSimulator:
    archetype: str = "glioblastoma"
    controller: Optional[BaseController] = None
    dt: float = 0.25
    seed: int = 0
    history: List[SimFrame] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.params = get_archetype(self.archetype)
        self.pk = PKPDModel()
        self.ode = CancerODE(self.params, self.pk)
        self.observer = ObservationLayer(seed=self.seed)
        if self.controller is None:
            self.controller = PlasticMushroomBodyController()
        self.manual_override = False
        self.manual_u = {d: 0.0 for d in CONTROL_DRUG_IDS}
        self.reset()

    def reset(self, archetype: Optional[str] = None, seed: Optional[int] = None) -> SimFrame:
        if archetype is not None:
            self.archetype = archetype
            self.params = get_archetype(archetype)
            self.ode = CancerODE(self.params, self.pk)
        if seed is not None:
            self.seed = seed
            self.observer = ObservationLayer(seed=seed)
        self.x, self.c = self.ode.initial_state()
        self.t = 0.0
        self.history.clear()
        if hasattr(self.controller, "reset"):
            self.controller.reset()
        return self._frame(self._observe(), self._idle_action())

    def _observe(self) -> ObservationRecord:
        latent = self.ode.to_latent(self.x, t=self.t)
        return self.observer.observe(latent)

    def _idle_action(self) -> InterventionAction:
        return InterventionAction(
            t=self.t,
            infusion={d: 0.0 for d in CONTROL_DRUG_IDS},
            source="init",
        )

    def _frame(self, obs: ObservationRecord, action: InterventionAction) -> SimFrame:
        latent = self.ode.to_latent(self.x, t=self.t)
        conc = self.pk.as_dict(self.c)
        tel = None
        if hasattr(self.controller, "connectome_telemetry"):
            tel = self.controller.connectome_telemetry()
        frame = SimFrame(
            t=self.t,
            latent=latent,
            observation=obs,
            action=action,
            concentrations=conc,
            occupancies=self.pk.occupancies(self.c),
            connectome=tel,
            terminal=latent.terminal_toxicity,
        )
        self.history.append(frame)
        return frame

    def set_manual(self, enabled: bool, infusion: Optional[Dict[str, float]] = None) -> None:
        self.manual_override = enabled
        if infusion:
            self.manual_u.update({k: float(np.clip(v, 0.0, 1.0)) for k, v in infusion.items()})

    def set_controller(self, controller: BaseController) -> None:
        self.controller = controller
        if hasattr(controller, "reset"):
            controller.reset()

    def step(self) -> SimFrame:
        obs = self._observe()
        if self.manual_override:
            action = InterventionAction(
                t=self.t,
                infusion=dict(self.manual_u),
                source="manual",
                notes="human override",
            )
        else:
            ctx = ControllerContext(
                concentrations=self.pk.as_dict(self.c),
                archetype=self.params.name,
                dt=self.dt,
                step_index=len(self.history),
            )
            action = self.controller.decide(obs, ctx)
        u = self.pk.clip_infusion(action.infusion)
        self.x, self.c = self.ode.step(self.x, self.c, u, self.dt)
        self.t += self.dt
        return self._frame(obs, action)
