"""Rate-based mushroom-body network and dopamine plasticity."""

from confluence.neural_engine.full_brain import (
    FULL_BRAIN_NEURONS,
    FullBrainConfig,
    FullBrainNetwork,
    INTERACTIVE_BRAIN_NEURONS,
    population_sizes,
)
from confluence.neural_engine.network import MushroomBodyNetwork
from confluence.neural_engine.plasticity import DopaminePlasticity, dopamine_signal

__all__ = [
    "DopaminePlasticity",
    "FULL_BRAIN_NEURONS",
    "FullBrainConfig",
    "FullBrainNetwork",
    "INTERACTIVE_BRAIN_NEURONS",
    "MushroomBodyNetwork",
    "dopamine_signal",
    "population_sizes",
]
