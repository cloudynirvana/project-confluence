"""
Mission Phase Model — Space Medicine Module
=============================================

Defines mission phases, gravity models, and phase-dependent
stressor profiles for spaceflight simulation.

Each mission phase has distinct:
    - Gravity environment
    - Dominant stressors
    - Countermeasure priorities
    - Ψ decay rate profiles

References:
    NASA Human Spaceflight Risks: https://www.nasa.gov/hrp/risks/
    SPRINT protocol for exercise countermeasures
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto


class MissionPhase(Enum):
    """Mission phases with distinct physiological profiles."""
    PRE_FLIGHT = auto()
    LAUNCH = auto()
    EARLY_ADAPTATION = auto()       # Days 1–14 in microgravity
    CRUISE_LEO = auto()             # Months in LEO (ISS)
    DEEP_SPACE_TRANSIT = auto()     # Beyond LEO (Artemis, Mars)
    PLANETARY_SURFACE = auto()      # Moon (0.16 G) or Mars (0.38 G)
    REENTRY = auto()
    POST_FLIGHT_REHAB = auto()


@dataclass
class PhaseProfile:
    """Physiological profile for a mission phase."""
    phase: MissionPhase
    gravity_g: float                # Gravity level (Earth = 1.0)
    duration_days: float            # Typical phase duration
    radiation_factor: float         # Radiation relative to LEO (LEO = 1.0)

    # Per-dimension Ψ decay rates (per day, in microgravity without countermeasures)
    # Negative = degradation, zero = stable
    baseline_decay: np.ndarray = None  # Shape (7,)

    # Countermeasure priority ranking (0 = highest priority)
    countermeasure_priority: np.ndarray = None  # Shape (7,)

    def __post_init__(self):
        if self.baseline_decay is None:
            self.baseline_decay = np.zeros(7)
        if self.countermeasure_priority is None:
            self.countermeasure_priority = np.arange(7, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE PROFILES
# ═══════════════════════════════════════════════════════════════════════════

PHASE_PROFILES = {
    MissionPhase.PRE_FLIGHT: PhaseProfile(
        phase=MissionPhase.PRE_FLIGHT,
        gravity_g=1.0,
        duration_days=180,  # ~6 months pre-flight training
        radiation_factor=0.0,
        baseline_decay=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        countermeasure_priority=np.array([6, 5, 4, 3, 2, 1, 0]),
    ),
    MissionPhase.LAUNCH: PhaseProfile(
        phase=MissionPhase.LAUNCH,
        gravity_g=3.5,  # Average sustained G during ascent
        duration_days=0.01,  # ~15 minutes
        radiation_factor=0.5,
        # Acute stress, vestibular shock
        baseline_decay=np.array([
            -0.10,  # Circadian: disrupted by launch timing
            -0.15,  # Autonomic: stress response
            -0.05,  # Immune: acute stress hormones
            0.00,   # Microbiome: minimal acute effect
            0.00,   # Musculoskeletal: no acute effect
            -0.05,  # Neuro-ocular: G-force fluid shift
            -0.10,  # Cognitive: stress, task load
        ]),
        countermeasure_priority=np.array([4, 3, 5, 6, 6, 2, 1]),
    ),
    MissionPhase.EARLY_ADAPTATION: PhaseProfile(
        phase=MissionPhase.EARLY_ADAPTATION,
        gravity_g=0.0,
        duration_days=14,
        radiation_factor=1.0,
        # Space Adaptation Syndrome, fluid shift, circadian disruption
        baseline_decay=np.array([
            -0.04,  # Circadian: 16 sunrises/day on ISS
            -0.03,  # Autonomic: baroreceptor recalibration
            -0.02,  # Immune: stress-induced suppression
            -0.01,  # Microbiome: dietary change onset
            -0.02,  # Musculoskeletal: unloading begins
            -0.04,  # Neuro-ocular: cephalad fluid shift
            -0.03,  # Cognitive: SAS, disorientation
        ]),
        countermeasure_priority=np.array([1, 3, 4, 5, 6, 0, 2]),
    ),
    MissionPhase.CRUISE_LEO: PhaseProfile(
        phase=MissionPhase.CRUISE_LEO,
        gravity_g=0.0,
        duration_days=180,  # Typical ISS increment
        radiation_factor=1.0,
        # Chronic deconditioning — slower but persistent
        baseline_decay=np.array([
            -0.015,  # Circadian: persistent disruption without good lighting
            -0.010,  # Autonomic: gradual deconditioning
            -0.008,  # Immune: latent virus reactivation risk
            -0.005,  # Microbiome: gradual diversity loss
            -0.012,  # Musculoskeletal: ~1% BMD loss/month without exercise
            -0.010,  # Neuro-ocular: SANS progression
            -0.008,  # Cognitive: fatigue accumulation
        ]),
        countermeasure_priority=np.array([2, 3, 4, 5, 0, 1, 6]),
    ),
    MissionPhase.DEEP_SPACE_TRANSIT: PhaseProfile(
        phase=MissionPhase.DEEP_SPACE_TRANSIT,
        gravity_g=0.0,
        duration_days=270,  # Mars transit
        radiation_factor=3.0,  # GCR beyond LEO
        # All stressors amplified by isolation + radiation
        baseline_decay=np.array([
            -0.018,  # Circadian: no Earth zeitgebers
            -0.012,  # Autonomic: deeper deconditioning
            -0.012,  # Immune: radiation + stress synergy
            -0.008,  # Microbiome: limited food diversity
            -0.015,  # Musculoskeletal: prolonged unloading
            -0.012,  # Neuro-ocular: prolonged fluid shift
            -0.015,  # Cognitive: isolation, comm delays
        ]),
        countermeasure_priority=np.array([0, 2, 1, 4, 3, 5, 6]),
    ),
    MissionPhase.PLANETARY_SURFACE: PhaseProfile(
        phase=MissionPhase.PLANETARY_SURFACE,
        gravity_g=0.38,  # Mars (or 0.16 for Moon)
        duration_days=500,  # Mars surface stay
        radiation_factor=2.0,  # Surface radiation (some shielding)
        # Partial gravity provides some loading
        baseline_decay=np.array([
            -0.008,  # Circadian: Mars sol ≈ 24.6h — manageable
            -0.005,  # Autonomic: partial gravity helps
            -0.006,  # Immune: reduced but persistent
            -0.004,  # Microbiome: dust exposure new factor
            -0.005,  # Musculoskeletal: partial loading helps
            -0.005,  # Neuro-ocular: partial gravity alleviates
            -0.008,  # Cognitive: new environment stress
        ]),
        countermeasure_priority=np.array([3, 5, 1, 2, 4, 6, 0]),
    ),
    MissionPhase.REENTRY: PhaseProfile(
        phase=MissionPhase.REENTRY,
        gravity_g=1.5,  # Sustained G during reentry
        duration_days=0.02,  # ~30 minutes
        radiation_factor=0.2,
        # Orthostatic intolerance risk
        baseline_decay=np.array([
            -0.05,  # Circadian: disrupted by reentry timing
            -0.20,  # Autonomic: orthostatic challenge
            -0.02,  # Immune: stress hormones
            0.00,   # Microbiome: no acute change
            -0.05,  # Musculoskeletal: G-loading on deconditioned body
            -0.10,  # Neuro-ocular: gravity return → fluid redistribution
            -0.10,  # Cognitive: high task load
        ]),
        countermeasure_priority=np.array([5, 0, 4, 6, 2, 1, 3]),
    ),
    MissionPhase.POST_FLIGHT_REHAB: PhaseProfile(
        phase=MissionPhase.POST_FLIGHT_REHAB,
        gravity_g=1.0,
        duration_days=45,  # ~6 weeks structured rehab
        radiation_factor=0.0,
        # Recovery in 1 G — most systems trending positive
        baseline_decay=np.array([
            0.03,   # Circadian: recovering with Earth zeitgebers
            0.02,   # Autonomic: readapting to gravity
            0.02,   # Immune: normalizing
            0.01,   # Microbiome: diet normalization
            0.02,   # Musculoskeletal: reloading
            0.02,   # Neuro-ocular: gravity restoring fluid balance
            0.03,   # Cognitive: normalized schedule + rest
        ]),
        countermeasure_priority=np.array([4, 1, 3, 5, 0, 2, 6]),
    ),
}


def get_phase_profile(phase: MissionPhase) -> PhaseProfile:
    """Get the physiological profile for a mission phase."""
    return PHASE_PROFILES[phase]


def gravity_factor(gravity_g: float) -> float:
    """Convert gravity level to a stressor factor.

    Returns:
        factor: 1.0 in microgravity (maximum stressor),
                0.0 at 1 G (no gravity-related stressor),
                negative above 1 G (hypergravity stress).
    """
    if gravity_g <= 0.01:
        return 1.0
    elif gravity_g >= 1.0:
        return max(0.0, -(gravity_g - 1.0) * 0.5)  # Hypergravity penalty
    else:
        return 1.0 - gravity_g  # Linear interpolation


def get_phase_at_day(mission_day: float,
                     mission_type: str = "iss_6month") -> MissionPhase:
    """Determine mission phase from elapsed mission day.

    Args:
        mission_day: Days since launch.
        mission_type: One of 'iss_6month', 'lunar_30day', 'mars_30month'.

    Returns:
        Current MissionPhase.
    """
    if mission_type == "iss_6month":
        if mission_day < 0:
            return MissionPhase.PRE_FLIGHT
        elif mission_day < 0.01:
            return MissionPhase.LAUNCH
        elif mission_day < 14:
            return MissionPhase.EARLY_ADAPTATION
        elif mission_day < 180:
            return MissionPhase.CRUISE_LEO
        elif mission_day < 180.02:
            return MissionPhase.REENTRY
        else:
            return MissionPhase.POST_FLIGHT_REHAB

    elif mission_type == "lunar_30day":
        if mission_day < 0:
            return MissionPhase.PRE_FLIGHT
        elif mission_day < 0.01:
            return MissionPhase.LAUNCH
        elif mission_day < 4:
            return MissionPhase.DEEP_SPACE_TRANSIT
        elif mission_day < 4 + 21:
            return MissionPhase.PLANETARY_SURFACE
        elif mission_day < 4 + 21 + 4:
            return MissionPhase.DEEP_SPACE_TRANSIT
        elif mission_day < 30:
            return MissionPhase.REENTRY
        else:
            return MissionPhase.POST_FLIGHT_REHAB

    elif mission_type == "mars_30month":
        if mission_day < 0:
            return MissionPhase.PRE_FLIGHT
        elif mission_day < 0.01:
            return MissionPhase.LAUNCH
        elif mission_day < 14:
            return MissionPhase.EARLY_ADAPTATION
        elif mission_day < 270:
            return MissionPhase.DEEP_SPACE_TRANSIT
        elif mission_day < 270 + 500:
            return MissionPhase.PLANETARY_SURFACE
        elif mission_day < 270 + 500 + 270:
            return MissionPhase.DEEP_SPACE_TRANSIT
        else:
            return MissionPhase.POST_FLIGHT_REHAB

    return MissionPhase.CRUISE_LEO  # Default
