"""Competing therapy controllers A–F."""

from confluence.controllers.base import BaseController, ControllerContext
from confluence.controllers.full_brain import FullBrainController
from confluence.controllers.gatenby import GatenbyAdaptiveController
from confluence.controllers.mtd import MTDController
from confluence.controllers.plastic_mb import PlasticMushroomBodyController
from confluence.controllers.ppo import PPOController
from confluence.controllers.static_reservoir import StaticReservoirController

CONTROLLER_REGISTRY = {
    "A": MTDController,
    "B": GatenbyAdaptiveController,
    "C": PPOController,
    "D": StaticReservoirController,
    "E": PlasticMushroomBodyController,
    "F": FullBrainController,
    "mtd": MTDController,
    "gatenby": GatenbyAdaptiveController,
    "ppo": PPOController,
    "reservoir": StaticReservoirController,
    "plastic_mb": PlasticMushroomBodyController,
    "full_brain": FullBrainController,
}


def make_controller(name: str, **kwargs) -> BaseController:
    key = name.strip()
    if key not in CONTROLLER_REGISTRY:
        # allow "A: MTD" style labels
        key = key.split(":")[0].strip()
    cls = CONTROLLER_REGISTRY.get(key) or CONTROLLER_REGISTRY.get(key.upper())
    if cls is None:
        raise KeyError(f"Unknown controller '{name}'")
    return cls(**kwargs)


__all__ = [
    "CONTROLLER_REGISTRY",
    "BaseController",
    "ControllerContext",
    "FullBrainController",
    "GatenbyAdaptiveController",
    "MTDController",
    "PPOController",
    "PlasticMushroomBodyController",
    "StaticReservoirController",
    "make_controller",
]
