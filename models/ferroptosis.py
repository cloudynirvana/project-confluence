"""
Ferroptosis Module — Project Confluence
=======================================

Iron-dependent, lipid peroxidation-driven cell death pathway.
Distinct from apoptosis — critical for drug-resistant tumors.

Models 3 key biochemical variables:
  1. GPX4 activity    — antioxidant enzyme (prevents lipid peroxidation)
  2. Labile iron pool — free iron catalyzes Fenton reaction
  3. Lipid peroxides  — accumulation beyond threshold = cell death

Wires into the existing 10D phase space via ROS (index 9).

References:
  - Dixon et al. 2012, Cell: Ferroptosis defined
  - Jiang et al. 2021, Nature: GPX4 as master regulator
  - Hassannia et al. 2019, Cancer Cell: Ferroptosis in cancer therapy

Usage:
    from ferroptosis import FerroptosisEngine
    fe = FerroptosisEngine()
    fe.step(metabolites, active_drugs, dt=0.5)
    if fe.cell_death_fraction > 0.5:
        # Significant ferroptotic death occurring
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List


# ═══════════════════════════════════════════════════════════════════════
# FERROPTOSIS PARAMETERS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FerroptosisParams:
    """Biologically grounded ferroptosis parameters."""

    # GPX4 dynamics
    gpx4_basal: float = 1.0          # Normalized basal GPX4 activity
    gpx4_synthesis_rate: float = 0.05 # Recovery rate when not inhibited
    gpx4_half_life: float = 12.0     # Hours (rapid turnover)

    # Iron dynamics
    iron_basal: float = 0.3          # µM labile iron (normal: 0.2-0.5 µM)
    iron_import_rate: float = 0.01   # Transferrin receptor uptake
    iron_export_rate: float = 0.02   # Ferroportin export
    iron_storage_rate: float = 0.03  # Ferritin sequestration

    # Lipid peroxidation  
    lpo_generation_rate: float = 0.1  # Base ROS → lipid peroxide rate
    lpo_clearance_rate: float = 0.08  # GPX4-dependent clearance
    lpo_death_threshold: float = 2.0  # Above this = ferroptotic death

    # Fenton reaction kinetics
    fenton_rate: float = 0.15        # Fe²⁺ + H₂O₂ → •OH rate constant

    # Cancer-type specific sensitivity
    sensitivity_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "TNBC": 1.3,     # High: glycolytic, ROS-stressed
        "PDAC": 0.7,     # Low: stromal protection
        "GBM": 1.1,      # Moderate: lipid-rich membranes
        "NSCLC": 1.2,    # High: KEAP1/NRF2 mutations common
        "CRC": 1.0,      # Moderate
        "Melanoma": 1.4,  # High: iron metabolism dysregulated
        "AML": 0.9,      # Moderate: liquid tumor
        "HCC": 1.5,      # Highest: hepatic iron overload
        "HGSOC": 1.1,    # Moderate
        "mCRPC": 0.8,    # Low: androgen-regulated iron
    })


# ═══════════════════════════════════════════════════════════════════════
# DRUG DEFINITIONS (for intervention.py integration)
# ═══════════════════════════════════════════════════════════════════════

FERROPTOSIS_DRUGS = {
    "RSL3": {
        "mechanism": "GPX4 covalent inhibitor",
        "gpx4_inhibition_rate": 0.15,   # Per day
        "iron_effect": 0.0,
        "category": "ferroptosis_inducer",
        "half_life_days": 0.5,          # Short-acting
        "evidence_level": "Preclinical (Phase I pending)",
    },
    "Erastin": {
        "mechanism": "System Xc⁻ inhibitor → glutathione depletion → GPX4 starved",
        "gpx4_inhibition_rate": 0.08,   # Indirect, slower
        "iron_effect": 0.03,            # Mild iron accumulation
        "category": "ferroptosis_inducer",
        "half_life_days": 1.0,
        "evidence_level": "Preclinical",
    },
    "FerricAmmoniumCitrate": {
        "mechanism": "Exogenous iron loading → Fenton reaction amplification",
        "gpx4_inhibition_rate": 0.0,
        "iron_effect": 0.10,            # Direct iron loading
        "category": "iron_modulator",
        "half_life_days": 2.0,
        "evidence_level": "Clinical (iron supplementation)",
    },
    "Sorafenib": {
        "mechanism": "Multi-kinase inhibitor with ferroptosis-inducing activity",
        "gpx4_inhibition_rate": 0.04,
        "iron_effect": 0.02,
        "category": "ferroptosis_inducer",
        "half_life_days": 1.1,          # ~25-48h clinical half-life
        "evidence_level": "FDA approved (HCC, RCC)",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# FERROPTOSIS ENGINE
# ═══════════════════════════════════════════════════════════════════════

class FerroptosisEngine:
    """
    Models ferroptotic cell death dynamics.
    
    Integrates with the SAEM 10D phase space by reading ROS (index 9)
    and feeding back cell death fraction as an additional force on
    the metabolic state.
    """

    ROS_INDEX = 9  # ROS position in the 10D metabolic vector

    def __init__(
        self,
        params: Optional[FerroptosisParams] = None,
        cancer_type: str = "TNBC",
        seed: int = 42,
    ):
        self.params = params or FerroptosisParams()
        self.cancer_type = cancer_type
        self.rng = np.random.RandomState(seed)

        # Sensitivity multiplier for this cancer type
        self.sensitivity = self.params.sensitivity_multipliers.get(cancer_type, 1.0)

        # State variables
        self.gpx4_activity = self.params.gpx4_basal
        self.labile_iron = self.params.iron_basal
        self.lipid_peroxides = 0.0
        self.cell_death_fraction = 0.0  # 0-1: fraction of tumor undergoing ferroptosis

        # History tracking
        self.history: List[Dict] = []
        self.total_ferroptotic_death = 0.0

    def reset(self):
        """Reset to initial state."""
        self.gpx4_activity = self.params.gpx4_basal
        self.labile_iron = self.params.iron_basal
        self.lipid_peroxides = 0.0
        self.cell_death_fraction = 0.0
        self.total_ferroptotic_death = 0.0
        self.history.clear()

    def step(
        self,
        metabolites: np.ndarray,
        active_drugs: Dict[str, float],
        dt: float = 0.5,
    ) -> float:
        """
        Advance ferroptosis state by dt days.
        
        Args:
            metabolites: 10D metabolic state vector
            active_drugs: {drug_name: dose_fraction} dict
            dt: time step in days
            
        Returns:
            cell_death_fraction: 0-1 ferroptotic death rate
        """
        p = self.params

        # ── 1. GPX4 dynamics ──
        # Natural synthesis (recovery)
        gpx4_synthesis = p.gpx4_synthesis_rate * dt
        
        # Drug-induced inhibition
        gpx4_inhibition = 0.0
        for drug_name, dose in active_drugs.items():
            drug_def = FERROPTOSIS_DRUGS.get(drug_name, {})
            inhibit_rate = drug_def.get("gpx4_inhibition_rate", 0)
            gpx4_inhibition += inhibit_rate * dose * dt

        # Natural degradation
        gpx4_decay = (np.log(2) / (p.gpx4_half_life / 24.0)) * self.gpx4_activity * dt

        self.gpx4_activity += gpx4_synthesis - gpx4_inhibition - gpx4_decay
        self.gpx4_activity = np.clip(self.gpx4_activity, 0.01, 2.0)

        # ── 2. Iron dynamics ──
        iron_import = p.iron_import_rate * dt
        iron_export = p.iron_export_rate * self.labile_iron * dt
        iron_storage = p.iron_storage_rate * max(0, self.labile_iron - p.iron_basal) * dt

        # Drug-induced iron loading
        iron_drugs = 0.0
        for drug_name, dose in active_drugs.items():
            drug_def = FERROPTOSIS_DRUGS.get(drug_name, {})
            iron_drugs += drug_def.get("iron_effect", 0) * dose * dt

        self.labile_iron += iron_import + iron_drugs - iron_export - iron_storage
        self.labile_iron = np.clip(self.labile_iron, 0.05, 5.0)

        # ── 3. Lipid peroxidation (the killing mechanism) ──
        # ROS from metabolic state drives peroxidation
        ros_level = metabolites[self.ROS_INDEX] if len(metabolites) > self.ROS_INDEX else 0.5
        ros_normalized = np.clip(float(ros_level), 0.1, 3.0)  # Clamp to physiological range

        # Fenton reaction: Fe²⁺ + H₂O₂ → •OH (hydroxyl radical)
        # Use log-linear approximation when GPX4 is critically low to prevent overflow
        if self.gpx4_activity < 0.1 and self.labile_iron > 1.0:
            # Near-zero GPX4 + high iron: log-linear regime
            fenton_flux = p.fenton_rate * np.log1p(self.labile_iron) * ros_normalized * self.sensitivity * 0.5
        else:
            fenton_flux = p.fenton_rate * self.labile_iron * ros_normalized * self.sensitivity

        # Lipid peroxide generation = Fenton + basal ROS (capped)
        lpo_generation = min((p.lpo_generation_rate * ros_normalized + fenton_flux) * dt, 0.5)

        # GPX4-dependent clearance
        lpo_clearance = p.lpo_clearance_rate * self.gpx4_activity * self.lipid_peroxides * dt

        # Stochastic noise (biological variability)
        noise = self.rng.randn() * 0.02 * np.sqrt(dt)

        self.lipid_peroxides += lpo_generation - lpo_clearance + noise
        self.lipid_peroxides = np.clip(self.lipid_peroxides, 0.0, 5.0)  # Hard cap prevents overflow

        # ── 4. Cell death calculation ──
        # Sigmoid death curve: sharp transition at threshold
        if self.lipid_peroxides > p.lpo_death_threshold * 0.5:
            # Hill function: death = LPO^n / (K^n + LPO^n)
            n_hill = 3.0  # Steep cooperativity
            K = p.lpo_death_threshold
            self.cell_death_fraction = (
                self.lipid_peroxides ** n_hill /
                (K ** n_hill + self.lipid_peroxides ** n_hill)
            )
        else:
            self.cell_death_fraction = 0.0

        self.total_ferroptotic_death += self.cell_death_fraction * dt

        # ── 5. Record history ──
        self.history.append({
            "gpx4": round(self.gpx4_activity, 4),
            "iron": round(self.labile_iron, 4),
            "lpo": round(self.lipid_peroxides, 4),
            "death_frac": round(self.cell_death_fraction, 4),
            "ros": round(ros_normalized, 4),
        })

        return self.cell_death_fraction

    def compute_metabolic_force(self, n_metabolites: int = 10) -> np.ndarray:
        """
        Convert ferroptotic death into a force on the metabolic state.
        
        Ferroptosis disrupts cellular metabolism:
        - Depletes glutathione (→ affects ROS, glutamine/glutamate)
        - Releases iron (→ amplifies further peroxidation)
        - Disrupts mitochondrial membrane (→ ATP/NADH collapse)
        """
        force = np.zeros(n_metabolites)

        if self.cell_death_fraction < 0.01:
            return force

        death = self.cell_death_fraction

        # Metabolite indices: 0=Glucose, 1=Lactate, 2=Pyruvate, 3=ATP,
        # 4=NADH, 5=Glutamine, 6=Glutamate, 7=αKG, 8=Citrate, 9=ROS

        # Ferroptosis effects on metabolism:
        force[3] -= death * 0.3   # ATP depletion (mitochondrial damage)
        force[4] -= death * 0.2   # NADH depletion (ETC disruption)
        force[5] -= death * 0.15  # Glutamine consumption (glutathione synthesis attempt)
        force[6] += death * 0.1   # Glutamate release (from dying cells)
        force[9] += death * 0.08  # ROS amplification (CAPPED to prevent runaway feedback)

        return force

    def get_summary(self) -> dict:
        """Return current ferroptosis state summary."""
        return {
            "gpx4_activity": round(self.gpx4_activity, 4),
            "labile_iron_uM": round(self.labile_iron, 4),
            "lipid_peroxides": round(self.lipid_peroxides, 4),
            "cell_death_fraction": round(self.cell_death_fraction, 4),
            "total_ferroptotic_death": round(self.total_ferroptotic_death, 4),
            "cancer_type": self.cancer_type,
            "sensitivity": self.sensitivity,
        }


# ═══════════════════════════════════════════════════════════════════════
# QUICK VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def validate_ferroptosis():
    """Demonstrate the ferroptosis engine dynamics."""
    print("=" * 60)
    print("FERROPTOSIS ENGINE — Dynamics Validation")
    print("=" * 60)

    cancers = ["TNBC", "HCC", "Melanoma", "PDAC"]
    dt = 0.5
    days = 30

    for cancer in cancers:
        fe = FerroptosisEngine(cancer_type=cancer, seed=42)
        metabolites = np.ones(10) * 0.5
        metabolites[9] = 0.6  # Elevated ROS

        # Simulate with RSL3 + iron loading
        for day in range(int(days / dt)):
            if day < 10 / dt:
                # No drugs yet
                fe.step(metabolites, {}, dt)
            else:
                # Apply RSL3 + ferric iron
                fe.step(metabolites, {"RSL3": 0.8, "FerricAmmoniumCitrate": 0.5}, dt)

        summary = fe.get_summary()
        death = summary["total_ferroptotic_death"]
        status = "💀" if death > 2.0 else "⚗️" if death > 0.5 else "🛡️"

        print(
            f"  {cancer:<10} | GPX4={summary['gpx4_activity']:.3f} "
            f"| Fe={summary['labile_iron_uM']:.3f}µM "
            f"| LPO={summary['lipid_peroxides']:.3f} "
            f"| Death={death:.2f} {status}"
        )

    print()
    print("Legend: 💀 = significant death | ⚗️ = moderate | 🛡️ = resistant")
    print()


if __name__ == "__main__":
    validate_ferroptosis()
