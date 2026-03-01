"""
SAEM Cancer POC — Stochastic Attractor Escape Model
=====================================================

A computational framework for cancer therapy optimization using
geometric attractor dynamics and the Kramers escape theory.

Modules:
    generator         — Extract generator matrix A from metabolomics data
    coherence         — Coherence analysis (spectral, stability, coupling)
    intervention      — Drug library, PK engine, and protocol generation
    geometric_optimization — Basin curvature, flattening, entropic resonance
    immune_dynamics   — Lymphocyte force field and exhaustion physics
    restoration       — Corrective δA computation
    calibration       — Grid search + local refinement calibration
    tnbc_ode          — Literature-grounded ODE systems (TNBC + pan-cancer)
"""

from .generator import GeneratorExtractor, simulate_dynamics, generate_synthetic_system
from .coherence import CoherenceAnalyzer, compute_phase_coherence
from .intervention import (
    TherapeuticIntervention,
    InterventionMapper,
    TNBCMetabolicModel,
    DrugEfficiencyEngine,
    PathologyScalingTemplate,
)
from .geometric_optimization import (
    GeometricState,
    GeometricOptimizer,
    ProtocolPhase,
    TherapeuticProtocolOptimizer,
)
from .immune_dynamics import ImmuneParams, LymphocyteForceField
from .restoration import RestorationComputer
from .calibration import CalibrationResult, coarse_grid_search, refine_around_best, calibrate_full
from .tnbc_ode import TNBCODESystem, ODEParams, METABOLITES, simulate_trajectory, simulate_treatment_protocol

__all__ = [
    # Generator
    "GeneratorExtractor", "simulate_dynamics", "generate_synthetic_system",
    # Coherence
    "CoherenceAnalyzer", "compute_phase_coherence",
    # Intervention
    "TherapeuticIntervention", "InterventionMapper", "TNBCMetabolicModel",
    "DrugEfficiencyEngine", "PathologyScalingTemplate",
    # Geometric Optimization
    "GeometricState", "GeometricOptimizer", "ProtocolPhase", "TherapeuticProtocolOptimizer",
    # Immune
    "ImmuneParams", "LymphocyteForceField",
    # Restoration
    "RestorationComputer",
    # Calibration
    "CalibrationResult", "coarse_grid_search", "refine_around_best", "calibrate_full",
    # ODE Systems
    "TNBCODESystem", "ODEParams", "METABOLITES", "simulate_trajectory", "simulate_treatment_protocol",
]
