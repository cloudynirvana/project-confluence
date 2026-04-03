"""
Astronaut Resilience Profile (ARP) — 7D State Vector
=====================================================

Defines the 7-dimensional state vector Ψ_space that tracks astronaut
physiological complexity across coupled systems. Each dimension maps
to specific NASA Standard Measures data streams.

The ARP is the space-medicine analogue of the oncology Φ (Phi) vector.
Both measure adaptive complexity; the domain mapping differs.

Dimensions:
    Ψ₁ Circadian Coherence          ← actigraphy, light exposure
    Ψ₂ Autonomic Variability        ← HRV (SDNN, RMSSD, DFA α₁)
    Ψ₃ Immune-Inflammatory Balance  ← cytokines, WBC differential
    Ψ₄ Microbiome Stability         ← 16S/metagenomics diversity
    Ψ₅ Bone-Muscle Loading Balance  ← DEXA, exercise telemetry
    Ψ₆ Fluid-Shift/Neuro-Ocular    ← IOP, OCT, ultrasound
    Ψ₇ Cognitive-Behavioral         ← PVT, Cognition battery, sleep

References:
    NASA Standard Measures: https://www.nasa.gov/reference/standard-measures/
    Garrett-Bakelman et al. (2019) — Twins Study, Science
    Baevsky et al. (2017) — HRV in long-duration spaceflight
    Flynn-Evans et al. (2016) — Circadian disruption on ISS
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto


# ═══════════════════════════════════════════════════════════════════════════
# 1. DIMENSION ENUMERATION
# ═══════════════════════════════════════════════════════════════════════════

class PsiDimension(Enum):
    """The 7 dimensions of the Astronaut Resilience Profile."""
    CIRCADIAN = 0
    AUTONOMIC = 1
    IMMUNE = 2
    MICROBIOME = 3
    MUSCULOSKELETAL = 4
    NEURO_OCULAR = 5
    COGNITIVE = 6


PSI_DIMENSION_NAMES = [
    "Psi_circadian",
    "Psi_autonomic",
    "Psi_immune",
    "Psi_microbiome",
    "Psi_musculoskeletal",
    "Psi_neuro_ocular",
    "Psi_cognitive",
]

PSI_LABELS = [
    "Circadian Coherence",
    "Autonomic Variability",
    "Immune-Inflammatory Balance",
    "Microbiome Stability",
    "Bone-Muscle Loading Balance",
    "Fluid-Shift / Neuro-Ocular Stability",
    "Cognitive-Behavioral Resilience",
]

N_DIMENSIONS = 7


# ═══════════════════════════════════════════════════════════════════════════
# 2. BIOMARKER MAPPING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BiomarkerSpec:
    """Specification for a single biomarker within a Ψ dimension."""
    name: str
    unit: str
    healthy_low: float
    healthy_high: float
    critical_low: Optional[float] = None
    critical_high: Optional[float] = None
    nasa_measure: str = ""
    measurement_modality: str = ""
    sampling_frequency: str = ""


# Primary biomarkers for each Ψ dimension
BIOMARKER_DEFINITIONS: Dict[PsiDimension, List[BiomarkerSpec]] = {
    PsiDimension.CIRCADIAN: [
        BiomarkerSpec(
            name="interdaily_stability",
            unit="IS (0-1)",
            healthy_low=0.50, healthy_high=0.85,
            critical_low=0.25, critical_high=None,
            nasa_measure="Actigraphy",
            measurement_modality="Wrist accelerometer",
            sampling_frequency="Continuous",
        ),
        BiomarkerSpec(
            name="intradaily_variability",
            unit="IV (0-2)",
            healthy_low=0.30, healthy_high=0.70,
            critical_low=None, critical_high=1.20,
            nasa_measure="Actigraphy",
            measurement_modality="Wrist accelerometer",
            sampling_frequency="Continuous",
        ),
        BiomarkerSpec(
            name="melatonin_phase_angle",
            unit="hours",
            healthy_low=-3.0, healthy_high=-1.0,
            critical_low=-6.0, critical_high=1.0,
            nasa_measure="Salivary melatonin",
            measurement_modality="Saliva sample",
            sampling_frequency="Weekly",
        ),
        BiomarkerSpec(
            name="light_exposure_timing_score",
            unit="score (0-1)",
            healthy_low=0.60, healthy_high=1.00,
            critical_low=0.30, critical_high=None,
            nasa_measure="Light exposure logs",
            measurement_modality="Light sensor",
            sampling_frequency="Continuous",
        ),
    ],
    PsiDimension.AUTONOMIC: [
        BiomarkerSpec(
            name="SDNN",
            unit="ms",
            healthy_low=100.0, healthy_high=200.0,
            critical_low=50.0, critical_high=None,
            nasa_measure="ECG/Holter",
            measurement_modality="Chest-worn ECG",
            sampling_frequency="Daily 5-min recordings",
        ),
        BiomarkerSpec(
            name="RMSSD",
            unit="ms",
            healthy_low=25.0, healthy_high=80.0,
            critical_low=12.0, critical_high=None,
            nasa_measure="ECG/Holter",
            measurement_modality="Chest-worn ECG",
            sampling_frequency="Daily 5-min recordings",
        ),
        BiomarkerSpec(
            name="DFA_alpha1",
            unit="dimensionless",
            healthy_low=0.75, healthy_high=1.15,
            critical_low=0.50, critical_high=1.50,
            nasa_measure="ECG/Holter",
            measurement_modality="Derived from RR intervals",
            sampling_frequency="Daily",
        ),
        BiomarkerSpec(
            name="LF_HF_ratio",
            unit="dimensionless",
            healthy_low=0.5, healthy_high=3.0,
            critical_low=0.2, critical_high=6.0,
            nasa_measure="ECG/Holter",
            measurement_modality="Spectral analysis of HRV",
            sampling_frequency="Daily",
        ),
    ],
    PsiDimension.IMMUNE: [
        BiomarkerSpec(
            name="IL6_IL10_ratio",
            unit="ratio",
            healthy_low=0.3, healthy_high=2.0,
            critical_low=None, critical_high=5.0,
            nasa_measure="Blood draw",
            measurement_modality="Venipuncture + ELISA",
            sampling_frequency="Monthly",
        ),
        BiomarkerSpec(
            name="CD4_CD8_ratio",
            unit="ratio",
            healthy_low=1.0, healthy_high=3.0,
            critical_low=0.5, critical_high=5.0,
            nasa_measure="Blood draw",
            measurement_modality="Flow cytometry",
            sampling_frequency="Monthly",
        ),
        BiomarkerSpec(
            name="NK_cell_activity",
            unit="% cytotoxicity",
            healthy_low=15.0, healthy_high=45.0,
            critical_low=8.0, critical_high=None,
            nasa_measure="Blood draw",
            measurement_modality="Cr-release assay",
            sampling_frequency="Monthly",
        ),
        BiomarkerSpec(
            name="salivary_cortisol",
            unit="μg/dL",
            healthy_low=0.10, healthy_high=0.40,
            critical_low=None, critical_high=0.80,
            nasa_measure="Saliva sample",
            measurement_modality="Immunoassay",
            sampling_frequency="Weekly",
        ),
    ],
    PsiDimension.MICROBIOME: [
        BiomarkerSpec(
            name="shannon_diversity",
            unit="H' (bits)",
            healthy_low=3.0, healthy_high=5.5,
            critical_low=2.0, critical_high=None,
            nasa_measure="Fecal/oral swab",
            measurement_modality="16S rRNA sequencing",
            sampling_frequency="Monthly",
        ),
        BiomarkerSpec(
            name="firmicutes_bacteroidetes_ratio",
            unit="ratio",
            healthy_low=1.0, healthy_high=3.0,
            critical_low=0.3, critical_high=8.0,
            nasa_measure="Fecal swab",
            measurement_modality="Shotgun metagenomics",
            sampling_frequency="Monthly",
        ),
        BiomarkerSpec(
            name="functional_gene_stability",
            unit="Bray-Curtis similarity to baseline",
            healthy_low=0.60, healthy_high=1.00,
            critical_low=0.35, critical_high=None,
            nasa_measure="Fecal swab",
            measurement_modality="Shotgun metagenomics",
            sampling_frequency="Monthly",
        ),
    ],
    PsiDimension.MUSCULOSKELETAL: [
        BiomarkerSpec(
            name="bmd_loss_rate",
            unit="% per month",
            healthy_low=-0.50, healthy_high=0.10,
            critical_low=None, critical_high=None,
            nasa_measure="DEXA (pre/post)",
            measurement_modality="Dual-energy X-ray absorptiometry",
            sampling_frequency="Pre-flight / post-flight",
        ),
        BiomarkerSpec(
            name="lean_mass_change",
            unit="% from baseline",
            healthy_low=-2.0, healthy_high=2.0,
            critical_low=-5.0, critical_high=None,
            nasa_measure="Body composition",
            measurement_modality="DEXA / bioimpedance",
            sampling_frequency="Monthly",
        ),
        BiomarkerSpec(
            name="exercise_compliance",
            unit="ratio (actual/prescribed)",
            healthy_low=0.80, healthy_high=1.20,
            critical_low=0.50, critical_high=None,
            nasa_measure="ARED/CEVIS/T2 logs",
            measurement_modality="Exercise telemetry",
            sampling_frequency="Daily",
        ),
    ],
    PsiDimension.NEURO_OCULAR: [
        BiomarkerSpec(
            name="intraocular_pressure",
            unit="mmHg",
            healthy_low=10.0, healthy_high=20.0,
            critical_low=None, critical_high=25.0,
            nasa_measure="Tonometry",
            measurement_modality="Portable tonometer",
            sampling_frequency="Weekly",
        ),
        BiomarkerSpec(
            name="rnfl_thickness",
            unit="μm",
            healthy_low=80.0, healthy_high=110.0,
            critical_low=70.0, critical_high=130.0,
            nasa_measure="OCT",
            measurement_modality="Portable OCT",
            sampling_frequency="Monthly",
        ),
        BiomarkerSpec(
            name="choroidal_thickness_change",
            unit="μm from baseline",
            healthy_low=-20.0, healthy_high=40.0,
            critical_low=None, critical_high=80.0,
            nasa_measure="OCT / ultrasound",
            measurement_modality="Portable OCT + B-scan US",
            sampling_frequency="Monthly",
        ),
    ],
    PsiDimension.COGNITIVE: [
        BiomarkerSpec(
            name="pvt_lapses",
            unit="count per 10-min test",
            healthy_low=0.0, healthy_high=5.0,
            critical_low=None, critical_high=12.0,
            nasa_measure="Cognition battery",
            measurement_modality="Tablet-based PVT",
            sampling_frequency="Daily",
        ),
        BiomarkerSpec(
            name="cognition_composite_score",
            unit="z-score from baseline",
            healthy_low=-0.5, healthy_high=0.5,
            critical_low=-1.5, critical_high=None,
            nasa_measure="Cognition battery",
            measurement_modality="Tablet cognitive battery",
            sampling_frequency="Weekly",
        ),
        BiomarkerSpec(
            name="sleep_efficiency",
            unit="%",
            healthy_low=85.0, healthy_high=98.0,
            critical_low=70.0, critical_high=None,
            nasa_measure="Actigraphy + sleep logs",
            measurement_modality="Wrist accelerometer",
            sampling_frequency="Daily",
        ),
    ],
}


class SpaceBiomarkerMap:
    """Registry of all biomarkers across Ψ dimensions.

    Provides normalization, lookup, and export utilities.
    """

    def __init__(self):
        self._map = BIOMARKER_DEFINITIONS

    def get_biomarkers(self, dim: PsiDimension) -> List[BiomarkerSpec]:
        """Get all biomarkers for a Ψ dimension."""
        return self._map.get(dim, [])

    def normalize_value(self, dim: PsiDimension, biomarker_name: str,
                        raw_value: float) -> float:
        """Normalize a raw biomarker value to [0, 1] within its healthy range.

        Returns:
            1.0 at the healthy midpoint, tapering to 0.0 at critical bounds.
        """
        specs = self._map.get(dim, [])
        spec = next((s for s in specs if s.name == biomarker_name), None)
        if spec is None:
            return 0.5  # Unknown biomarker → neutral

        midpoint = (spec.healthy_low + spec.healthy_high) / 2.0
        half_range = (spec.healthy_high - spec.healthy_low) / 2.0
        if half_range < 1e-10:
            return 1.0 if abs(raw_value - midpoint) < 1e-10 else 0.0

        # Gaussian-like normalization centered on healthy midpoint
        deviation = abs(raw_value - midpoint)
        sigma = half_range  # 1σ = half the healthy range
        score = float(np.exp(-0.5 * (deviation / sigma) ** 2))
        return np.clip(score, 0.0, 1.0)

    def compute_dimension_score(self, dim: PsiDimension,
                                measurements: Dict[str, float]) -> float:
        """Compute a single Ψ dimension score from raw biomarker values.

        Averages the normalized scores of all available biomarkers for
        that dimension.
        """
        specs = self._map.get(dim, [])
        scores = []
        for spec in specs:
            if spec.name in measurements:
                score = self.normalize_value(dim, spec.name,
                                             measurements[spec.name])
                scores.append(score)
        return float(np.mean(scores)) if scores else 0.5

    def is_critical(self, dim: PsiDimension, biomarker_name: str,
                    raw_value: float) -> bool:
        """Check if a biomarker value is outside critical bounds."""
        specs = self._map.get(dim, [])
        spec = next((s for s in specs if s.name == biomarker_name), None)
        if spec is None:
            return False
        if spec.critical_low is not None and raw_value < spec.critical_low:
            return True
        if spec.critical_high is not None and raw_value > spec.critical_high:
            return True
        return False

    def to_json(self, path: Optional[str] = None) -> str:
        """Export the full biomarker map as JSON."""
        data = {}
        for dim, specs in self._map.items():
            data[dim.name] = [asdict(s) for s in specs]
        json_str = json.dumps(data, indent=2)
        if path:
            with open(path, 'w') as f:
                f.write(json_str)
        return json_str


# ═══════════════════════════════════════════════════════════════════════════
# 3. HEALTHY REFERENCE & SAFE CORRIDOR
# ═══════════════════════════════════════════════════════════════════════════

# Ground-based healthy reference values (pre-flight baseline)
HEALTHY_GROUND_REFERENCE = np.array([
    0.75,   # Ψ₁ Circadian: well-entrained, regular sleep-wake
    0.80,   # Ψ₂ Autonomic: good HRV, balanced sympathovagal
    0.70,   # Ψ₃ Immune: balanced pro/anti-inflammatory
    0.75,   # Ψ₄ Microbiome: diverse, stable
    0.85,   # Ψ₅ Musculoskeletal: full loading, trained
    0.80,   # Ψ₆ Neuro-ocular: normal IOP, stable RNFL
    0.80,   # Ψ₇ Cognitive: alert, well-rested
])

# Safe corridor: min/max acceptable values for each Ψ dimension
SAFE_CORRIDOR = {
    PsiDimension.CIRCADIAN:      (0.35, 1.00),
    PsiDimension.AUTONOMIC:      (0.35, 1.00),
    PsiDimension.IMMUNE:         (0.30, 1.00),
    PsiDimension.MICROBIOME:     (0.30, 1.00),
    PsiDimension.MUSCULOSKELETAL: (0.40, 1.00),
    PsiDimension.NEURO_OCULAR:   (0.35, 1.00),
    PsiDimension.COGNITIVE:      (0.40, 1.00),
}


# ═══════════════════════════════════════════════════════════════════════════
# 4. ASTRONAUT RESILIENCE PROFILE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AstronautResilienceProfile:
    """
    7-Dimensional Astronaut Resilience Profile (Ψ_space).

    The space-medicine analogue of the oncology PhiProfile.
    Each dimension ∈ [0, 1], where 1 = maximum adaptive complexity
    and 0 = complete loss of regulation.

    Healthy astronauts maintain Ψ ≈ HEALTHY_GROUND_REFERENCE.
    Spaceflight causes gradual drift toward lower values across
    multiple coupled dimensions.
    """
    psi_circadian: float = 0.75
    psi_autonomic: float = 0.80
    psi_immune: float = 0.70
    psi_microbiome: float = 0.75
    psi_musculoskeletal: float = 0.85
    psi_neuro_ocular: float = 0.80
    psi_cognitive: float = 0.80

    # Crew member metadata
    crew_id: str = ""
    mission_day: float = 0.0
    mission_phase: str = "pre-flight"
    timestamp: str = ""

    # Classification
    archetype: str = "Healthy Adapted"
    archetype_confidence: float = 1.0

    # Raw measurement values used to compute this profile
    raw_measurements: Dict[str, float] = field(default_factory=dict)

    # Alert flags
    alerts: List[str] = field(default_factory=list)

    @property
    def psi_vector(self) -> np.ndarray:
        """Return the 7D Ψ vector as numpy array."""
        return np.array([
            self.psi_circadian, self.psi_autonomic, self.psi_immune,
            self.psi_microbiome, self.psi_musculoskeletal,
            self.psi_neuro_ocular, self.psi_cognitive,
        ])

    @property
    def psi_magnitude(self) -> float:
        """L2 norm — overall resilience magnitude."""
        return float(np.linalg.norm(self.psi_vector))

    @property
    def psi_mean(self) -> float:
        """Mean across all dimensions — simple overall score."""
        return float(np.mean(self.psi_vector))

    @property
    def distance_from_healthy(self) -> float:
        """Euclidean distance from healthy ground reference."""
        return float(np.linalg.norm(self.psi_vector - HEALTHY_GROUND_REFERENCE))

    @property
    def min_dimension(self) -> Tuple[str, float]:
        """Identify the weakest dimension (highest priority for intervention)."""
        v = self.psi_vector
        idx = int(np.argmin(v))
        return PSI_LABELS[idx], float(v[idx])

    def check_safe_corridor(self) -> List[str]:
        """Check if any dimension is outside the safe corridor.

        Returns list of alert strings for dimensions outside bounds.
        """
        alerts = []
        v = self.psi_vector
        for dim in PsiDimension:
            low, high = SAFE_CORRIDOR[dim]
            val = v[dim.value]
            if val < low:
                alerts.append(
                    f"ALERT: {PSI_LABELS[dim.value]} = {val:.3f} "
                    f"below safe minimum ({low:.2f})")
            elif val > high:
                alerts.append(
                    f"WARNING: {PSI_LABELS[dim.value]} = {val:.3f} "
                    f"above safe maximum ({high:.2f})")
        return alerts

    def classify_archetype(self) -> Tuple[str, float]:
        """Classify the astronaut's current physiological archetype.

        Archetypes:
            Healthy Adapted     — All dimensions in safe corridor
            Circadian Drifter   — Ψ₁ low, others moderate
            Deconditioning      — Ψ₅ low, Ψ₂ moderate-low
            Immune Drift        — Ψ₃ and/or Ψ₄ low
            SANS Risk           — Ψ₆ low, fluid-shift markers abnormal
            Cognitive Fatigue   — Ψ₇ low, often coupled with Ψ₁
            Multi-System Decline — ≥3 dimensions below safe corridor
            Critical             — Any dimension at critical bounds
        """
        v = self.psi_vector
        below_safe = []
        for dim in PsiDimension:
            low, _ = SAFE_CORRIDOR[dim]
            if v[dim.value] < low:
                below_safe.append(dim)

        # Multi-system decline
        if len(below_safe) >= 3:
            return "Multi-System Decline", 0.90

        # Critical
        critical_threshold = 0.20
        if np.any(v < critical_threshold):
            return "Critical", 0.95

        # Single-system archetypes
        if PsiDimension.CIRCADIAN in below_safe:
            conf = 1.0 - v[PsiDimension.CIRCADIAN.value]
            return "Circadian Drifter", round(conf, 3)

        if PsiDimension.MUSCULOSKELETAL in below_safe:
            conf = 1.0 - v[PsiDimension.MUSCULOSKELETAL.value]
            return "Deconditioning", round(conf, 3)

        if PsiDimension.IMMUNE in below_safe or PsiDimension.MICROBIOME in below_safe:
            vals = [v[PsiDimension.IMMUNE.value], v[PsiDimension.MICROBIOME.value]]
            conf = 1.0 - min(vals)
            return "Immune Drift", round(conf, 3)

        if PsiDimension.NEURO_OCULAR in below_safe:
            conf = 1.0 - v[PsiDimension.NEURO_OCULAR.value]
            return "SANS Risk", round(conf, 3)

        if PsiDimension.COGNITIVE in below_safe:
            conf = 1.0 - v[PsiDimension.COGNITIVE.value]
            return "Cognitive Fatigue", round(conf, 3)

        # Healthy
        return "Healthy Adapted", round(float(np.min(v)), 3)

    @classmethod
    def from_measurements(cls, measurements: Dict[str, float],
                          crew_id: str = "",
                          mission_day: float = 0.0,
                          mission_phase: str = "in-flight") -> "AstronautResilienceProfile":
        """Construct an ARP from raw biomarker measurements.

        Args:
            measurements: Dict of biomarker_name → raw_value.
                         Names must match BiomarkerSpec.name values.
            crew_id: Crew member identifier.
            mission_day: Mission elapsed time in days.
            mission_phase: Current mission phase string.

        Returns:
            AstronautResilienceProfile with computed Ψ values.
        """
        bmap = SpaceBiomarkerMap()
        profile = cls(
            crew_id=crew_id,
            mission_day=mission_day,
            mission_phase=mission_phase,
            raw_measurements=measurements.copy(),
        )

        # Compute each dimension from available biomarkers
        profile.psi_circadian = bmap.compute_dimension_score(
            PsiDimension.CIRCADIAN, measurements)
        profile.psi_autonomic = bmap.compute_dimension_score(
            PsiDimension.AUTONOMIC, measurements)
        profile.psi_immune = bmap.compute_dimension_score(
            PsiDimension.IMMUNE, measurements)
        profile.psi_microbiome = bmap.compute_dimension_score(
            PsiDimension.MICROBIOME, measurements)
        profile.psi_musculoskeletal = bmap.compute_dimension_score(
            PsiDimension.MUSCULOSKELETAL, measurements)
        profile.psi_neuro_ocular = bmap.compute_dimension_score(
            PsiDimension.NEURO_OCULAR, measurements)
        profile.psi_cognitive = bmap.compute_dimension_score(
            PsiDimension.COGNITIVE, measurements)

        # Classify and check alerts
        profile.archetype, profile.archetype_confidence = \
            profile.classify_archetype()
        profile.alerts = profile.check_safe_corridor()

        return profile

    @classmethod
    def healthy_baseline(cls, crew_id: str = "baseline") -> "AstronautResilienceProfile":
        """Return a healthy ground-based baseline profile."""
        ref = HEALTHY_GROUND_REFERENCE
        return cls(
            psi_circadian=ref[0],
            psi_autonomic=ref[1],
            psi_immune=ref[2],
            psi_microbiome=ref[3],
            psi_musculoskeletal=ref[4],
            psi_neuro_ocular=ref[5],
            psi_cognitive=ref[6],
            crew_id=crew_id,
            mission_day=0.0,
            mission_phase="pre-flight",
            archetype="Healthy Adapted",
            archetype_confidence=1.0,
        )

    def to_json(self, path: Optional[str] = None) -> str:
        """Export as JSON."""
        data = {
            "psi_vector": {
                name: round(self.psi_vector[i], 4)
                for i, name in enumerate(PSI_DIMENSION_NAMES)
            },
            "summary": {
                "psi_magnitude": round(self.psi_magnitude, 4),
                "psi_mean": round(self.psi_mean, 4),
                "distance_from_healthy": round(self.distance_from_healthy, 4),
                "archetype": self.archetype,
                "archetype_confidence": round(self.archetype_confidence, 4),
                "min_dimension": self.min_dimension[0],
                "min_value": round(self.min_dimension[1], 4),
            },
            "metadata": {
                "crew_id": self.crew_id,
                "mission_day": self.mission_day,
                "mission_phase": self.mission_phase,
            },
            "alerts": self.alerts,
        }
        json_str = json.dumps(data, indent=2)
        if path:
            with open(path, 'w') as f:
                f.write(json_str)
        return json_str

    def __repr__(self) -> str:
        v = self.psi_vector
        dims = " ".join(f"{v[i]:.2f}" for i in range(N_DIMENSIONS))
        return (f"ARP(Ψ=[{dims}] "
                f"|Ψ|={self.psi_magnitude:.3f} "
                f"arch={self.archetype} "
                f"day={self.mission_day:.0f})")


# ═══════════════════════════════════════════════════════════════════════════
# 5. CROSS-SYSTEM COUPLING MATRIX
# ═══════════════════════════════════════════════════════════════════════════

# How each dimension's deficit propagates to others.
# C[i, j] = effect of Ψ_j deficit on Ψ_i drift rate.
# Derived from spaceflight physiology literature.
COUPLING_MATRIX = np.array([
    # Circ  Auto  Immun Micro Musc  Neuro Cogn
    [0.00, 0.35, 0.20, 0.10, 0.05, 0.10, 0.40],  # Circadian
    [0.30, 0.00, 0.25, 0.05, 0.15, 0.20, 0.20],  # Autonomic
    [0.15, 0.20, 0.00, 0.35, 0.10, 0.05, 0.10],  # Immune
    [0.10, 0.05, 0.30, 0.00, 0.05, 0.00, 0.05],  # Microbiome
    [0.05, 0.15, 0.10, 0.05, 0.00, 0.10, 0.15],  # Musculoskeletal
    [0.10, 0.25, 0.05, 0.00, 0.05, 0.00, 0.15],  # Neuro-ocular
    [0.35, 0.20, 0.10, 0.05, 0.10, 0.10, 0.00],  # Cognitive
])


def compute_coupling_drift(psi_current: np.ndarray,
                           psi_healthy: np.ndarray = HEALTHY_GROUND_REFERENCE,
                           coupling_strength: float = 1.0) -> np.ndarray:
    """Compute the cross-system coupling drift vector.

    When one system degrades, it pulls coupled systems down.

    Args:
        psi_current: Current 7D Ψ vector.
        psi_healthy: Healthy reference vector.
        coupling_strength: Global scaling factor for coupling.

    Returns:
        drift: 7D vector of coupling-induced drift rates (negative = degradation).
    """
    deficit = psi_healthy - psi_current  # Positive where degraded
    deficit = np.maximum(deficit, 0.0)   # Only deficits drive coupling
    drift = -coupling_strength * COUPLING_MATRIX @ deficit
    return drift
