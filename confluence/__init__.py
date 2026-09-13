"""
Confluence v2 — computational closed-loop oncology research.

A Drosophila melanogaster mushroom-body-style connectome steers a
multi-drug infusion vector over a mechanistic cancer microenvironment.

This package is a simulation / research instrument. It is not a medical
device and does not claim clinical efficacy, disease eradication,
or a treatment path.
"""

from confluence.contracts import (
    ALL_EFFECTOR_IDS,
    ConnectomeSubcircuit,
    DrugSpecification,
    FULL_BRAIN_NEURONS,
    FUSION_CHANNEL_IDS,
    InterventionAction,
    LatentCancerState,
    ObservationRecord,
    PROTEIN_CHANNEL_IDS,
)

__version__ = "2.0.0"

__all__ = [
    "ALL_EFFECTOR_IDS",
    "ConnectomeSubcircuit",
    "DrugSpecification",
    "FULL_BRAIN_NEURONS",
    "InterventionAction",
    "LatentCancerState",
    "ObservationRecord",
    "PROTEIN_CHANNEL_IDS",
    "FUSION_CHANNEL_IDS",
    "__version__",
]
