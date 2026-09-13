"""Closed-loop simulator: observations → controller → drugs → ODE → Y'."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from confluence.cancer_env.archetypes import get_archetype
from confluence.cancer_env.ode_system import CancerODE
from confluence.cancer_env.observation_layer import ObservationLayer
from confluence.contracts import (
    ALL_EFFECTOR_IDS,
    InterventionAction,
    LatentCancerState,
    ObservationRecord,
)
from confluence.controllers.base import BaseController, ControllerContext
from confluence.controllers.plastic_mb import PlasticMushroomBodyController
from confluence.embodiment.flybody_bridge import FlybodyBridge
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
    embodiment: Optional[Dict] = None
    terminal: bool = False


@dataclass
class ClosedLoopSimulator:
    archetype: str = "glioblastoma"
    controller: Optional[BaseController] = None
    dt: float = 0.25
    seed: int = 0
    embodiment_enabled: bool = True
    solver: str = "LSODA"
    history: List[SimFrame] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.params = get_archetype(self.archetype)
        # Always allocate protein + small-molecule PK states. Controllers A–E
        # still emit 5-D U; missing keys pad to 0 so existing policies are unchanged.
        self.pk = PKPDModel(drug_ids=ALL_EFFECTOR_IDS)
        self.ode = CancerODE(self.params, self.pk, seed=self.seed)
        self.observer = ObservationLayer(seed=self.seed, params=self.params)
        if self.controller is None:
            self.controller = PlasticMushroomBodyController()
        self.manual_override = False
        self.manual_u = {d: 0.0 for d in ALL_EFFECTOR_IDS}
        self.embodiment = FlybodyBridge(task="template", prefer_real=True, seed=self.seed)
        self.embodiment_alpha = 0.25
        self.reset()

    def reset(self, archetype: Optional[str] = None, seed: Optional[int] = None) -> SimFrame:
        if archetype is not None:
            self.archetype = archetype
            self.params = get_archetype(archetype)
            self.ode = CancerODE(self.params, self.pk, seed=self.seed if seed is None else seed)
        if seed is not None:
            self.seed = seed
            self.observer = ObservationLayer(seed=seed, params=self.params)
        else:
            self.observer.set_params(self.params)
        self.x, self.c = self.ode.initial_state()
        self.t = 0.0
        self.history.clear()
        if hasattr(self.controller, "reset"):
            self.controller.reset()
        if self.embodiment_enabled:
            self.embodiment.reset()
        return self._frame(self._observe(), self._idle_action())

    def _observe(self) -> ObservationRecord:
        latent = self.ode.to_latent(self.x, t=self.t)
        return self.observer.observe(latent)

    def _idle_action(self) -> InterventionAction:
        return InterventionAction(
            t=self.t,
            infusion={d: 0.0 for d in self.pk.drug_ids},
            source="init",
        )

    def _frame(self, obs: ObservationRecord, action: InterventionAction) -> SimFrame:
        latent = self.ode.to_latent(self.x, t=self.t)
        conc = self.pk.as_dict(self.c)
        tel = None
        if hasattr(self.controller, "connectome_telemetry"):
            tel = self.controller.connectome_telemetry()
        emb = None
        if self.embodiment_enabled and self.embodiment.last is not None:
            emb = self.embodiment.last.as_dict()
        frame = SimFrame(
            t=self.t,
            latent=latent,
            observation=obs,
            action=action,
            concentrations=conc,
            occupancies=self.pk.occupancies(self.c),
            connectome=tel,
            embodiment=emb,
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

    def _maybe_mix(self, obs: ObservationRecord) -> ObservationRecord:
        if not self.embodiment_enabled or self.embodiment.last is None:
            return obs
        mixed = self.embodiment.mix_observation(obs.as_vector(), alpha=self.embodiment_alpha)
        update = {
            "tumor_burden": float(max(0.0, mixed[0])),
            "resistance_frequency": float(np.clip(mixed[1], 0.0, 1.0)),
            "lactate": float(max(0.0, mixed[2])),
            "tgfb": float(max(0.0, mixed[3])),
            "immune_competence_ratio": float(max(0.0, mixed[4])),
        }
        if mixed.size >= 7:
            update["fusion_allele_fraction"] = float(np.clip(mixed[5], 0.0, 1.0))
            update["junction_neoantigen"] = float(max(0.0, mixed[6]))
        if mixed.size >= 11:
            update["occult_allele_fraction"] = float(np.clip(mixed[7], 0.0, 1.0))
            update["dormancy_exit"] = float(np.clip(mixed[8], 0.0, 1.5))
            update["immune_surveillance"] = float(max(0.0, mixed[9]))
            update["antibody_readiness"] = float(max(0.0, mixed[10]))
        return obs.model_copy(update=update)

    def step(self, run_cancer: bool = True, run_embodiment: bool = True) -> SimFrame:
        obs_true = self._observe()
        obs_ctrl = self._maybe_mix(obs_true) if run_embodiment else obs_true
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
            action = self.controller.decide(obs_ctrl, ctx)
        u = self.pk.clip_infusion(action.infusion)
        if run_embodiment and self.embodiment_enabled:
            mbon = []
            if hasattr(self.controller, "connectome_telemetry"):
                tel = self.controller.connectome_telemetry() or {}
                mbon = tel.get("mbon_rates") or []
            self.embodiment.step(u, mbon)
        if run_cancer:
            self.x, self.c = self.ode.step(self.x, self.c, u, self.dt, method=self.solver)
            self.t += self.dt
        return self._frame(obs_true, action)
