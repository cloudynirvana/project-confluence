"""
Confluence v2 — computational closed-loop oncology research.

A Drosophila melanogaster mushroom-body-style connectome steers a
multi-drug infusion vector over a mechanistic cancer microenvironment.

This package is a simulation / research instrument. It is not a medical
device and does not claim clinical efficacy or a path to cure.
"""

from confluence.contracts import (
    ConnectomeSubcircuit,
    DrugSpecification,
    InterventionAction,
    LatentCancerState,
    ObservationRecord,
)

__version__ = "2.0.0"

__all__ = [
    "ConnectomeSubcircuit",
    "DrugSpecification",
    "InterventionAction",
    "LatentCancerState",
    "ObservationRecord",
    "__version__",
]
