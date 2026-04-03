"""
Project Confluence — Space Medicine Module
============================================

Adaptive complexity-restoring countermeasure optimization
for human spaceflight.

This module extends the Confluence framework from oncology to
space physiology, treating astronaut deconditioning as a
multi-system loss of adaptive complexity.

Core components:
    - AstronautResilienceProfile: 7D state vector (Ψ_space)
    - SpaceCountermeasureController: Closed-loop adaptive controller
    - SpacePhysiologyODE: 7D coupled ODE system
    - MissionPhase: Mission phase enum with gravity model

Framing: Research & decision-support tool for flight surgeons.
Not an autonomous treatment system.
"""

from .state_vector import (
    AstronautResilienceProfile,
    PsiDimension,
    SpaceBiomarkerMap,
    HEALTHY_GROUND_REFERENCE,
    SAFE_CORRIDOR,
)
from .countermeasures import (
    CountermeasureAction,
    CountermeasureVector,
    CountermeasureConstraints,
    COUNTERMEASURE_SCHEMA,
)

__all__ = [
    "AstronautResilienceProfile",
    "PsiDimension",
    "SpaceBiomarkerMap",
    "HEALTHY_GROUND_REFERENCE",
    "SAFE_CORRIDOR",
    "CountermeasureAction",
    "CountermeasureVector",
    "CountermeasureConstraints",
    "COUNTERMEASURE_SCHEMA",
]
