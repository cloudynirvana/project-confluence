"""
Spatial Dynamics Module — Project Confluence
=============================================

Models tumor spatial heterogeneity using a simplified 1D radial
compartment model: Core (hypoxic), Rim (normoxic), Stroma.

Each compartment has its own metabolic state and drug penetration
profile. The tumor's overall escape distance is a weighted average
across compartments.

Key features:
  - 3-compartment radial model (core/rim/stroma)
  - Drug penetration: exponential decay from vasculature
  - Oxygen gradient: affects ROS dynamics in each compartment
  - Stromal barrier: PDAC desmoplasia modeled explicitly
  - Weighted escape: compartment-weighted final distance

References:
  - Minchinton & Tannock 2006, Nat Rev Cancer: Drug penetration
  - Helmlinger et al. 1997, Nature Medicine: Oxygen gradients
  - Olive et al. 2009, Science: PDAC stromal depletion
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CompartmentParams:
    """Parameters for a single spatial compartment."""
    name: str = "core"
    volume_fraction: float = 0.33       # Fraction of total tumor volume
    oxygen_level: float = 0.2           # 0 = anoxic, 1 = normoxic
    vascular_density: float = 0.1       # Vessel density (0-1)
    drug_penetration: float = 0.3       # Fraction of systemic dose that reaches here
    ph: float = 6.5                     # Local pH (acid = lower drug uptake for weak bases)
    cell_density: float = 0.8           # Normalized cell packing density

    # Metabolite modifiers (multiplicative on generator diagonal)
    glucose_supply: float = 0.5         # Glucose availability (low in core)
    lactate_accumulation: float = 1.5   # Lactate buildup (high in core)
    ros_baseline: float = 1.0           # ROS level modifier


# Pre-configured spatial profiles for cancer types
SPATIAL_PROFILES: Dict[str, List[CompartmentParams]] = {
    "TNBC": [
        CompartmentParams("core",   0.30, 0.15, 0.05, 0.20, 6.4, 0.9, 0.3, 2.0, 1.5),
        CompartmentParams("rim",    0.50, 0.70, 0.60, 0.80, 7.0, 0.7, 0.9, 0.8, 0.8),
        CompartmentParams("stroma", 0.20, 0.85, 0.70, 0.90, 7.2, 0.3, 1.0, 0.5, 0.5),
    ],
    "PDAC": [
        CompartmentParams("core",   0.25, 0.10, 0.03, 0.10, 6.3, 0.95, 0.2, 2.5, 2.0),
        CompartmentParams("rim",    0.30, 0.50, 0.30, 0.50, 6.8, 0.8,  0.7, 1.2, 1.0),
        CompartmentParams("stroma", 0.45, 0.80, 0.50, 0.40, 7.1, 0.4,  0.8, 0.6, 0.6),
    ],
    "GBM": [
        CompartmentParams("core",   0.35, 0.05, 0.02, 0.15, 6.2, 0.95, 0.1, 3.0, 2.5),
        CompartmentParams("rim",    0.45, 0.60, 0.40, 0.60, 6.9, 0.75, 0.8, 1.0, 0.9),
        CompartmentParams("stroma", 0.20, 0.90, 0.80, 0.85, 7.3, 0.2,  1.0, 0.4, 0.4),
    ],
    "NSCLC": [
        CompartmentParams("core",   0.25, 0.20, 0.10, 0.30, 6.5, 0.85, 0.4, 1.8, 1.3),
        CompartmentParams("rim",    0.55, 0.75, 0.65, 0.85, 7.1, 0.65, 0.95, 0.7, 0.7),
        CompartmentParams("stroma", 0.20, 0.90, 0.75, 0.90, 7.3, 0.25, 1.0, 0.5, 0.5),
    ],
    "Melanoma": [
        CompartmentParams("core",   0.20, 0.25, 0.15, 0.35, 6.6, 0.80, 0.5, 1.5, 1.2),
        CompartmentParams("rim",    0.60, 0.80, 0.70, 0.90, 7.2, 0.60, 1.0, 0.6, 0.6),
        CompartmentParams("stroma", 0.20, 0.90, 0.80, 0.95, 7.3, 0.20, 1.0, 0.4, 0.4),
    ],
    "CRC": [
        CompartmentParams("core",   0.30, 0.20, 0.08, 0.25, 6.4, 0.85, 0.3, 2.0, 1.4),
        CompartmentParams("rim",    0.50, 0.70, 0.55, 0.80, 7.0, 0.70, 0.9, 0.8, 0.8),
        CompartmentParams("stroma", 0.20, 0.85, 0.65, 0.85, 7.2, 0.30, 1.0, 0.5, 0.5),
    ],
}

# Default profile for unlisted cancer types
DEFAULT_SPATIAL = [
    CompartmentParams("core",   0.30, 0.20, 0.10, 0.25, 6.5, 0.85, 0.4, 1.8, 1.3),
    CompartmentParams("rim",    0.50, 0.70, 0.60, 0.80, 7.0, 0.65, 0.9, 0.8, 0.8),
    CompartmentParams("stroma", 0.20, 0.85, 0.70, 0.90, 7.3, 0.25, 1.0, 0.5, 0.5),
]


class SpatialTumorModel:
    """
    1D radial tumor model with 3 spatial compartments.

    Each compartment modifies the base cancer generator to reflect
    local conditions (oxygen, pH, drug penetration). The overall
    treatment response is the volume-weighted average.
    """

    def __init__(self, cancer_type: str = "TNBC",
                 tumor_radius_mm: float = 15.0):
        self.cancer_type = cancer_type
        self.tumor_radius_mm = tumor_radius_mm
        self.compartments = SPATIAL_PROFILES.get(cancer_type, DEFAULT_SPATIAL)

        # Validate volume fractions sum to 1
        total_vol = sum(c.volume_fraction for c in self.compartments)
        if abs(total_vol - 1.0) > 0.01:
            for c in self.compartments:
                c.volume_fraction /= total_vol

    def modify_generator(self, A_cancer: np.ndarray,
                         compartment: CompartmentParams) -> np.ndarray:
        """
        Apply spatial modifications to the cancer generator matrix.

        Adjustments based on local conditions:
          - Glucose supply affects glycolysis rates (row 0)
          - Lactate accumulation (row 1)
          - ROS dynamics (row 9)
          - Oxygen-dependent OXPHOS coupling (row 3: ATP, row 4: NADH)
        """
        A_local = A_cancer.copy()
        n = A_local.shape[0]

        # Glucose (index 0): supply modifies uptake rate
        if n > 0:
            A_local[0, 0] *= compartment.glucose_supply

        # Lactate (index 1): accumulation in hypoxic core
        if n > 1:
            A_local[1, 1] *= (1.0 / compartment.lactate_accumulation)
            # Higher lactate production from anaerobic glycolysis
            if n > 2:
                A_local[1, 2] += 0.05 * (1.0 - compartment.oxygen_level)

        # ATP (index 3): oxygen-dependent OXPHOS
        if n > 3:
            oxphos_efficiency = compartment.oxygen_level * 0.8
            A_local[3, 3] *= (0.4 + 0.6 * oxphos_efficiency)

        # NADH (index 4): electron transport depends on O2
        if n > 4:
            A_local[4, 4] *= (0.5 + 0.5 * compartment.oxygen_level)

        # ROS (index 9): baseline ROS modification
        if n > 9:
            A_local[9, 9] *= compartment.ros_baseline
            # Hypoxia increases ROS leak from complex III
            if n > 4:
                A_local[9, 4] += 0.05 * (1.0 - compartment.oxygen_level)

        return A_local

    def apply_drug_penetration(self, delta_A: np.ndarray,
                               compartment: CompartmentParams) -> np.ndarray:
        """
        Scale drug effect by local penetration.

        Drug penetration depends on:
          - Vascular density (distance from vessels)
          - pH (affects weak acid/base drug uptake)
          - Cell density (diffusion barrier)
        """
        # Base penetration
        penetration = compartment.drug_penetration

        # pH adjustment: weak acids penetrate better in acidic tumors,
        # weak bases penetrate worse. Net effect for most chemo is negative.
        ph_factor = 1.0 - 0.1 * (7.4 - compartment.ph)

        # Cell density barrier
        density_factor = 1.0 - 0.2 * (compartment.cell_density - 0.5)

        effective_penetration = penetration * ph_factor * density_factor
        effective_penetration = np.clip(effective_penetration, 0.05, 1.0)

        return delta_A * effective_penetration

    def compute_spatial_escape(self, escape_distances: Dict[str, float]) -> float:
        """
        Compute volume-weighted escape distance across compartments.

        Args:
            escape_distances: {compartment_name: escape_distance}

        Returns:
            Weighted average escape distance
        """
        total = 0.0
        for comp in self.compartments:
            dist = escape_distances.get(comp.name, 0.0)
            total += comp.volume_fraction * dist
        return total

    def get_compartment_summary(self) -> List[Dict]:
        """Return summary of all compartments for reporting."""
        return [
            {
                "name": c.name,
                "volume_fraction": round(c.volume_fraction, 2),
                "oxygen": round(c.oxygen_level, 2),
                "drug_penetration": round(c.drug_penetration, 2),
                "vascular_density": round(c.vascular_density, 2),
                "ph": round(c.ph, 1),
                "glucose_supply": round(c.glucose_supply, 2),
                "ros_baseline": round(c.ros_baseline, 2),
            }
            for c in self.compartments
        ]


def compute_diffusion_profile(radius_mm: float, vascular_density: float,
                               drug_diffusion_coeff: float = 0.1,
                               n_points: int = 50) -> np.ndarray:
    """
    Compute drug concentration profile along tumor radius.

    Models drug diffusion from vasculature with exponential decay:
      C(r) = C_0 * exp(-r / lambda)
    where lambda = diffusion_length = sqrt(D / k_elim)

    Args:
        radius_mm: Tumor radius in mm
        vascular_density: Vessel density (0-1), scales diffusion length
        drug_diffusion_coeff: Drug-specific diffusion coefficient
        n_points: Resolution of the radial profile

    Returns:
        concentration_profile: Normalized drug concentration (0-1) at each
                              point from r=0 (center) to r=radius
    """
    r = np.linspace(0, radius_mm, n_points)

    # Diffusion length depends on vessel density and drug properties
    lambda_mm = drug_diffusion_coeff * (0.5 + 2.0 * vascular_density)

    # Vessels are at the rim; drug diffuses inward
    # Distance from nearest vessel ≈ (radius - r) for simplified 1D
    distance_from_vessel = radius_mm - r
    concentration = np.exp(-distance_from_vessel / max(lambda_mm, 0.1))

    return concentration / max(concentration.max(), 1e-10)
