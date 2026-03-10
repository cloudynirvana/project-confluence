"""
Realistic Failure Model — Project Confluence
=============================================

Injects biologically grounded stochastic failure mechanisms into the SAEM
simulation to bring escape rates from 100% down to clinically plausible
ranges (30-70% depending on cancer type).

Three failure modes:
  1. PRIMARY RESISTANCE: Some patients never respond (genomic/epigenomic)
  2. IMMUNE EVASION: MHC-I downregulation, β2m loss mid-treatment
  3. METABOLIC PLASTICITY: Warburg ↔ OXPHOS switching under drug pressure

Clinical calibration sources:
  - KEYNOTE-522 (TNBC), IMbrave150 (HCC), CheckMate-067 (Melanoma)
  - KEYNOTE-177 (CRC MSI-H), Phase III data for all 10 cancer types

Usage:
    from realistic_failure import RealisticFailureModel
    fm = RealisticFailureModel("TNBC", seed=42)
    if not fm.responds_to_treatment():
        return "PRIMARY_RESISTANCE"
    # ... run simulation ...
    if fm.immune_evasion_occurs(day=15, immune_force=0.3):
        immune_force *= fm.evasion_factor
    if fm.metabolic_switch_occurs(day=25, drug_pressure=0.8):
        A_cancer = fm.apply_metabolic_switch(A_cancer)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# CLINICAL RESPONSE RATES (calibration targets)
# ═══════════════════════════════════════════════════════════════════════
# These represent the BEST available combination therapy response rates.
# The model should NOT exceed these + 10% margin.

CLINICAL_RESPONSE_RATES: Dict[str, dict] = {
    "TNBC": {
        "response_rate": 0.40,   # KEYNOTE-522: pCR 64% but durable response ~40%
        "source": "KEYNOTE-522 (pembrolizumab + chemo)",
        "primary_resistance_frac": 0.25,  # Never respond at all
        "immune_evasion_prob": 0.20,      # Lose response mid-treatment
        "metabolic_switch_prob": 0.15,    # Warburg↔OXPHOS switch
    },
    "PDAC": {
        "response_rate": 0.12,
        "source": "FOLFIRINOX + gemcitabine combinations",
        "primary_resistance_frac": 0.55,
        "immune_evasion_prob": 0.15,
        "metabolic_switch_prob": 0.18,
    },
    "GBM": {
        "response_rate": 0.10,
        "source": "Temozolomide + radiation (Stupp protocol)",
        "primary_resistance_frac": 0.50,
        "immune_evasion_prob": 0.25,  # Immune privilege behind BBB
        "metabolic_switch_prob": 0.15,
    },
    "NSCLC": {
        "response_rate": 0.45,
        "source": "Pembrolizumab monotherapy (PD-L1 ≥50%)",
        "primary_resistance_frac": 0.20,
        "immune_evasion_prob": 0.18,
        "metabolic_switch_prob": 0.17,
    },
    "CRC": {
        "response_rate": 0.50,
        "source": "Pembrolizumab (KEYNOTE-177, MSI-H)",
        "primary_resistance_frac": 0.20,
        "immune_evasion_prob": 0.15,
        "metabolic_switch_prob": 0.15,
    },
    "Melanoma": {
        "response_rate": 0.60,
        "source": "Nivolumab + ipilimumab (CheckMate-067)",
        "primary_resistance_frac": 0.15,
        "immune_evasion_prob": 0.12,
        "metabolic_switch_prob": 0.13,
    },
    "AML": {
        "response_rate": 0.30,
        "source": "Venetoclax + azacitidine",
        "primary_resistance_frac": 0.35,
        "immune_evasion_prob": 0.18,
        "metabolic_switch_prob": 0.17,
    },
    "HCC": {
        "response_rate": 0.30,
        "source": "Atezolizumab + bevacizumab (IMbrave150)",
        "primary_resistance_frac": 0.35,
        "immune_evasion_prob": 0.20,
        "metabolic_switch_prob": 0.15,
    },
    "HGSOC": {
        "response_rate": 0.35,
        "source": "PARP inhibitors + platinum (SOLO-1)",
        "primary_resistance_frac": 0.30,
        "immune_evasion_prob": 0.18,
        "metabolic_switch_prob": 0.17,
    },
    "mCRPC": {
        "response_rate": 0.25,
        "source": "Enzalutamide + pembrolizumab",
        "primary_resistance_frac": 0.40,
        "immune_evasion_prob": 0.18,
        "metabolic_switch_prob": 0.17,
    },
}


# ═══════════════════════════════════════════════════════════════════════
# FAILURE MODEL
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FailureEvent:
    """Records a failure event during simulation."""
    day: float
    mechanism: str  # "primary_resistance", "immune_evasion", "metabolic_switch"
    severity: float  # 0-1, how much it impacts treatment
    details: str


class RealisticFailureModel:
    """
    Stochastic failure model calibrated to clinical response rates.

    Creates a per-patient failure profile that determines:
    - Whether the patient responds at all (primary resistance)
    - Whether immune evasion occurs mid-treatment
    - Whether metabolic plasticity undermines drug efficacy
    """

    def __init__(self, cancer_type: str, seed: int = 42):
        self.cancer_type = cancer_type
        self.rng = np.random.RandomState(seed)
        self.events: list = []

        clinical = CLINICAL_RESPONSE_RATES.get(cancer_type, {
            "response_rate": 0.40,
            "primary_resistance_frac": 0.30,
            "immune_evasion_prob": 0.15,
            "metabolic_switch_prob": 0.15,
        })

        self.base_response_rate = clinical["response_rate"]
        self.primary_resistance_frac = clinical["primary_resistance_frac"]
        self.immune_evasion_prob = clinical["immune_evasion_prob"]
        self.metabolic_switch_prob = clinical["metabolic_switch_prob"]

        # Pre-determine patient-level characteristics
        self._is_primary_resistant = self.rng.random() < self.primary_resistance_frac
        self._evasion_day = None
        self._switch_day = None
        self._evasion_severity = 0.0
        self._switch_magnitude = 0.0

        if not self._is_primary_resistant:
            # Determine if and when immune evasion occurs
            if self.rng.random() < self.immune_evasion_prob:
                self._evasion_day = self.rng.uniform(10, 40)  # Days 10-40
                self._evasion_severity = self.rng.uniform(0.6, 0.9)  # 60-90% force reduction

            # Determine if and when metabolic switch occurs
            if self.rng.random() < self.metabolic_switch_prob:
                self._switch_day = self.rng.uniform(15, 45)
                self._switch_magnitude = self.rng.uniform(0.3, 0.6)

    # ── Check 1: Primary Resistance ──
    def responds_to_treatment(self) -> bool:
        """Does this patient respond to treatment at all?"""
        if self._is_primary_resistant:
            self.events.append(FailureEvent(
                day=0,
                mechanism="primary_resistance",
                severity=1.0,
                details=(
                    f"Patient has intrinsic resistance to therapy. "
                    f"This occurs in {self.primary_resistance_frac:.0%} of "
                    f"{self.cancer_type} patients."
                ),
            ))
            return False
        return True

    # ── Check 2: Immune Evasion ──
    def immune_evasion_occurs(self, day: float) -> bool:
        """Check if immune evasion activates at this timepoint."""
        if self._evasion_day is not None and day >= self._evasion_day:
            if not any(e.mechanism == "immune_evasion" for e in self.events):
                self.events.append(FailureEvent(
                    day=day,
                    mechanism="immune_evasion",
                    severity=self._evasion_severity,
                    details=(
                        f"Immune evasion at day {day:.0f}: MHC-I downregulation "
                        f"reduces immune force by {self._evasion_severity:.0%}."
                    ),
                ))
            return True
        return False

    @property
    def evasion_factor(self) -> float:
        """Multiplicative factor on immune force (1.0 = no evasion)."""
        if self._evasion_day is not None:
            return 1.0 - self._evasion_severity
        return 1.0

    # ── Check 3: Metabolic Plasticity ──
    def metabolic_switch_occurs(self, day: float, drug_pressure: float = 0.5) -> bool:
        """
        Check if metabolic switching (Warburg ↔ OXPHOS) occurs.
        Higher drug pressure increases switch probability.
        """
        if self._switch_day is not None and day >= self._switch_day:
            # Drug pressure modulates switch: more pressure = more likely
            effective_prob = min(1.0, drug_pressure * 1.5)
            if self.rng.random() < effective_prob:
                if not any(e.mechanism == "metabolic_switch" for e in self.events):
                    self.events.append(FailureEvent(
                        day=day,
                        mechanism="metabolic_switch",
                        severity=self._switch_magnitude,
                        details=(
                            f"Metabolic plasticity at day {day:.0f}: "
                            f"Warburg↔OXPHOS switch with magnitude "
                            f"{self._switch_magnitude:.2f}."
                        ),
                    ))
                return True
        return False

    def apply_metabolic_switch(self, A_cancer: np.ndarray) -> np.ndarray:
        """
        Apply metabolic switching to the cancer generator.

        Models Warburg ↔ OXPHOS transition: perturbs the generator's
        eigenstructure, partially undoing the Flatten phase.

        The switch rotates the cancer attractor in metabolic space,
        creating a "moving target" that drug combinations struggle to track.
        """
        n = A_cancer.shape[0]
        magnitude = self._switch_magnitude

        # Create perturbation that rotates the attractor
        # This models the metabolic reprogramming under drug pressure
        perturbation = np.zeros((n, n))

        # Glycolysis ↔ OXPHOS axis (indices 0=Glucose, 1=Lactate, 3=ATP, 4=NADH)
        glycolysis_idx = [0, 1]    # Glucose, Lactate
        oxphos_idx = [3, 4]        # ATP, NADH

        # Weaken glycolysis dependencies (tumor shifts to OXPHOS)
        for i in glycolysis_idx:
            for j in glycolysis_idx:
                perturbation[i, j] += magnitude * 0.3 * self.rng.randn()

        # Strengthen OXPHOS coupling (compensatory)
        for i in oxphos_idx:
            for j in oxphos_idx:
                perturbation[i, j] -= magnitude * 0.2 * self.rng.randn()

        # Cross-coupling shift (metabolic rewiring)
        for i in glycolysis_idx:
            for j in oxphos_idx:
                perturbation[i, j] += magnitude * 0.15 * self.rng.randn()
                perturbation[j, i] -= magnitude * 0.15 * self.rng.randn()

        # Apply: new cancer attractor = old + perturbation
        A_switched = A_cancer + perturbation

        return A_switched

    # ── Summary ──
    def get_failure_summary(self) -> dict:
        """Return summary of all failure events for this patient."""
        return {
            "cancer_type": self.cancer_type,
            "responded": not self._is_primary_resistant,
            "num_events": len(self.events),
            "events": [
                {
                    "day": e.day,
                    "mechanism": e.mechanism,
                    "severity": e.severity,
                    "details": e.details,
                }
                for e in self.events
            ],
            "clinical_response_rate": self.base_response_rate,
        }


# ═══════════════════════════════════════════════════════════════════════
# QUICK VALIDATION
# ═══════════════════════════════════════════════════════════════════════
def validate_failure_model(n_patients: int = 1000):
    """
    Validate that the failure model produces response rates
    close to clinical calibration targets.
    """
    print("=" * 65)
    print("REALISTIC FAILURE MODEL — Calibration Validation")
    print("=" * 65)
    print(f"{'Cancer':<10} {'Clinical':>10} {'Simulated':>10} {'Δ':>8} {'Status':>8}")
    print("-" * 65)

    all_ok = True
    for cancer_type, clinical in CLINICAL_RESPONSE_RATES.items():
        responds = 0
        evasions = 0
        switches = 0

        for i in range(n_patients):
            fm = RealisticFailureModel(cancer_type, seed=i)
            if fm.responds_to_treatment():
                responds += 1
                # Check mid-treatment failures
                for day in range(0, 60):
                    if fm.immune_evasion_occurs(day):
                        evasions += 1
                        break
                for day in range(0, 60):
                    if fm.metabolic_switch_occurs(day, drug_pressure=0.7):
                        switches += 1
                        break

        response_rate = responds / n_patients
        # Effective cure rate considers that evasion/switch may prevent cure
        # even in responders
        effective_rate = (responds - evasions * 0.7 - switches * 0.5) / n_patients
        effective_rate = max(0, effective_rate)

        target = clinical["response_rate"]
        delta = effective_rate - target
        ok = abs(delta) < 0.15  # Within 15% of target
        status = "✅" if ok else "⚠️"
        if not ok:
            all_ok = False

        print(
            f"{cancer_type:<10} "
            f"{target:>9.0%} "
            f"{effective_rate:>9.0%} "
            f"{delta:>+7.0%} "
            f"{status:>8}"
        )

    print("-" * 65)
    print(f"Overall: {'ALL CALIBRATED' if all_ok else 'NEEDS TUNING'}")
    print()

    return all_ok


if __name__ == "__main__":
    validate_failure_model()
