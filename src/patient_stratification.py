"""
Patient Stratification Module — Project Confluence
====================================================

Framework for patient-specific protocol adaptation using individual
metabolomics data to calibrate generators and tailor treatment.

Key features:
  - Patient generator extraction from metabolomics time-series
  - Risk stratification based on attractor depth + immune status
  - Protocol adaptation: phase durations and drug selection per-patient
  - Cohort simulation with inter-patient variance

References:
  - Ghaffari Laleh et al. 2022, Nature Cancer: ML for patient stratification
  - Barker et al. 2020, Nature Medicine: Adaptive therapy clinical trials
  - TCGA/CCLE metabolomics for population-level generators
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PatientProfile:
    """Individual patient metabolic and immune profile."""
    patient_id: str
    cancer_type: str

    # Metabolomics data (10-metabolite panel measured at multiple timepoints)
    metabolite_timeseries: Optional[np.ndarray] = None  # Shape: (n_timepoints, 10)
    sampling_interval_days: float = 1.0

    # Extracted patient-specific generator
    generator: Optional[np.ndarray] = None  # 10×10 matrix

    # Immune status
    cd8_count_per_ul: float = 500.0         # CD8+ T-cells/µL blood
    pd_l1_expression: float = 0.3           # Tumor PD-L1 TPS (0-1)
    tmb_mutations_per_mb: float = 5.0       # Tumor mutational burden
    msi_status: str = "MSS"                 # MSS or MSI-H

    # Tumor characteristics
    tumor_size_mm: float = 20.0
    ki67_percent: float = 30.0              # Proliferation index
    grade: int = 2                          # Histological grade (1-3)

    # Pharmacogenomics
    cyp3a4_activity: str = "normal"         # "poor", "intermediate", "normal", "rapid"
    dpd_deficiency: bool = False            # Dihydropyrimidine dehydrogenase

    # Derived scores (computed by stratification)
    seriousness_score: float = 0.0
    risk_tier: str = ""                     # "low", "medium", "high", "extreme"
    predicted_cure_rate: float = 0.0
    recommended_protocol_adjustments: Dict = field(default_factory=dict)


class PatientStratifier:
    """
    Stratify patients into risk tiers and adapt treatment protocols.

    Uses patient metabolomics to extract individual generators,
    then maps attractor depth + immune status → risk tier →
    protocol modifications.
    """

    # Risk tier definitions
    RISK_TIERS = {
        "low":     {"seriousness_range": (0.0, 0.30), "expected_cure": 0.95},
        "medium":  {"seriousness_range": (0.30, 0.45), "expected_cure": 0.85},
        "high":    {"seriousness_range": (0.45, 0.55), "expected_cure": 0.70},
        "extreme": {"seriousness_range": (0.55, 1.0), "expected_cure": 0.50},
    }

    # Protocol adjustment templates by risk tier
    PROTOCOL_ADJUSTMENTS = {
        "low": {
            "phase1_duration_mult": 0.8,    # Shorter flattening (easier basin)
            "phase2_duration_mult": 1.0,
            "phase3_duration_mult": 0.9,
            "drug_dose_mult": 0.9,          # Slightly lower doses
            "add_drugs": [],
            "notes": "Standard protocol with mild dose reduction",
        },
        "medium": {
            "phase1_duration_mult": 1.0,
            "phase2_duration_mult": 1.0,
            "phase3_duration_mult": 1.0,
            "drug_dose_mult": 1.0,
            "add_drugs": [],
            "notes": "Standard protocol",
        },
        "high": {
            "phase1_duration_mult": 1.2,    # Longer flattening
            "phase2_duration_mult": 1.1,
            "phase3_duration_mult": 1.2,
            "drug_dose_mult": 1.1,          # Higher doses
            "add_drugs": ["Hydroxychloroquine (HCQ)"],
            "notes": "Extended protocol + autophagy blockade",
        },
        "extreme": {
            "phase1_duration_mult": 1.5,    # Much longer flattening
            "phase2_duration_mult": 1.2,
            "phase3_duration_mult": 1.4,
            "drug_dose_mult": 1.2,
            "add_drugs": [
                "Hydroxychloroquine (HCQ)",
                "Bevacizumab (Anti-VEGF)",
            ],
            "notes": "Aggressive protocol + TME remodeling + autophagy block",
        },
    }

    def __init__(self, healthy_generator: np.ndarray):
        """
        Args:
            healthy_generator: 10×10 healthy reference generator matrix
        """
        self.healthy_generator = healthy_generator
        self.n = healthy_generator.shape[0]

    def extract_generator(self, profile: PatientProfile,
                          alpha: float = 0.1) -> np.ndarray:
        """
        Extract patient-specific generator from metabolomics time-series.

        Uses Ridge regression to estimate the generator matrix A such that
        dx/dt ≈ A * x fits the observed metabolite trajectories.

        Args:
            profile: Patient profile with metabolite_timeseries
            alpha: Ridge regression regularization strength

        Returns:
            A: Patient-specific 10×10 generator matrix
        """
        if profile.metabolite_timeseries is None:
            return None

        X = profile.metabolite_timeseries
        n_t, n_m = X.shape
        assert n_m == self.n, f"Expected {self.n} metabolites, got {n_m}"

        if n_t < 3:
            return None

        # Compute finite differences: dx/dt ≈ (x_{t+1} - x_t) / dt
        dt = profile.sampling_interval_days
        dXdt = np.diff(X, axis=0) / dt  # Shape: (n_t-1, n_m)
        X_mid = (X[:-1] + X[1:]) / 2    # Midpoint states

        # Ridge regression: dXdt = X_mid @ A.T → A.T = (X.T X + αI)⁻¹ X.T dXdt
        XtX = X_mid.T @ X_mid + alpha * np.eye(n_m)
        XtY = X_mid.T @ dXdt
        A = np.linalg.solve(XtX, XtY).T

        profile.generator = A
        return A

    def compute_seriousness(self, profile: PatientProfile) -> float:
        """
        Compute patient-specific seriousness score.

        Combines:
          - Attractor depth (from generator eigenvalues)
          - Immune status (CD8+ count, PD-L1, TMB)
          - Tumor burden (size, grade, Ki67)
          - Cancer type severity
        """
        score_components = []

        # 1. Generator-based depth (if available)
        A = profile.generator
        if A is not None:
            delta = A - self.healthy_generator
            frob_dist = np.linalg.norm(delta, 'fro')
            # Normalize: typical range is 1-5
            depth_score = min(1.0, frob_dist / 5.0)
            score_components.append(("attractor_depth", depth_score, 0.35))
        else:
            # Fallback: use cancer type average
            depth_score = 0.4
            score_components.append(("attractor_depth_estimated", depth_score, 0.35))

        # 2. Immune status score
        # High CD8+ and TMB = better prognosis; high PD-L1 = better CPI response
        cd8_norm = min(1.0, profile.cd8_count_per_ul / 1000)
        tmb_norm = min(1.0, profile.tmb_mutations_per_mb / 20)
        msi_bonus = 0.15 if profile.msi_status == "MSI-H" else 0.0
        immune_favorability = (cd8_norm * 0.4 + tmb_norm * 0.3
                               + profile.pd_l1_expression * 0.2 + msi_bonus)
        immune_score = 1.0 - immune_favorability  # Invert: higher = worse
        score_components.append(("immune_unfavorability", immune_score, 0.25))

        # 3. Tumor burden
        size_norm = min(1.0, profile.tumor_size_mm / 50)
        ki67_norm = min(1.0, profile.ki67_percent / 80)
        grade_norm = (profile.grade - 1) / 2.0
        burden = size_norm * 0.4 + ki67_norm * 0.3 + grade_norm * 0.3
        score_components.append(("tumor_burden", burden, 0.20))

        # 4. PK risk (CYP activity affects drug efficacy)
        pk_risk = {"poor": 0.7, "intermediate": 0.4, "normal": 0.2, "rapid": 0.1}
        pk_score = pk_risk.get(profile.cyp3a4_activity, 0.2)
        if profile.dpd_deficiency:
            pk_score += 0.3
        score_components.append(("pk_risk", min(1.0, pk_score), 0.10))

        # 5. Pharmacogenomics penalty
        dpd_penalty = 0.15 if profile.dpd_deficiency else 0.0
        score_components.append(("pharmacogenomics", dpd_penalty, 0.10))

        # Weighted sum
        total = sum(score * weight for _, score, weight in score_components)
        profile.seriousness_score = round(total, 3)

        # Assign risk tier
        for tier, info in self.RISK_TIERS.items():
            low, high = info["seriousness_range"]
            if low <= total < high:
                profile.risk_tier = tier
                profile.predicted_cure_rate = info["expected_cure"]
                break
        else:
            profile.risk_tier = "extreme"
            profile.predicted_cure_rate = 0.50

        return profile.seriousness_score

    def adapt_protocol(self, profile: PatientProfile) -> Dict:
        """
        Generate patient-specific protocol adjustments.

        Returns:
            Dict with phase duration multipliers, dose adjustments,
            and additional drug recommendations
        """
        if not profile.risk_tier:
            self.compute_seriousness(profile)

        adjustments = self.PROTOCOL_ADJUSTMENTS.get(
            profile.risk_tier,
            self.PROTOCOL_ADJUSTMENTS["medium"]
        ).copy()

        # Immune-specific adjustments
        if profile.pd_l1_expression > 0.5:
            adjustments["prioritize_checkpoint"] = True
            adjustments["notes"] += "; High PD-L1 → prioritize Anti-PD-1"

        if profile.tmb_mutations_per_mb > 10:
            adjustments["prioritize_checkpoint"] = True
            adjustments["notes"] += "; High TMB → checkpoint responsive"

        if profile.msi_status == "MSI-H":
            adjustments["add_drugs"].append("Anti-PD-1 (Pembrolizumab)")
            adjustments["notes"] += "; MSI-H → FDA-approved for pembrolizumab"

        # CYP-based dose adjustments
        if profile.cyp3a4_activity == "poor":
            adjustments["drug_dose_mult"] *= 0.7
            adjustments["notes"] += "; CYP3A4 poor metabolizer → 30% dose reduction"
        elif profile.cyp3a4_activity == "rapid":
            adjustments["drug_dose_mult"] *= 1.15
            adjustments["notes"] += "; CYP3A4 rapid metabolizer → 15% dose increase"

        profile.recommended_protocol_adjustments = adjustments
        return adjustments

    def stratify_cohort(self, profiles: List[PatientProfile]) -> Dict:
        """
        Stratify a cohort of patients and return tier distribution.

        Returns:
            Dict with tier counts, average seriousness, and per-patient summaries
        """
        tier_counts = {"low": 0, "medium": 0, "high": 0, "extreme": 0}
        scores = []
        summaries = []

        for p in profiles:
            score = self.compute_seriousness(p)
            self.adapt_protocol(p)
            scores.append(score)
            tier_counts[p.risk_tier] += 1
            summaries.append({
                "id": p.patient_id,
                "cancer": p.cancer_type,
                "seriousness": round(score, 3),
                "tier": p.risk_tier,
                "cure_rate": p.predicted_cure_rate,
                "notes": p.recommended_protocol_adjustments.get("notes", ""),
            })

        return {
            "n_patients": len(profiles),
            "tier_distribution": tier_counts,
            "mean_seriousness": round(np.mean(scores), 3) if scores else 0,
            "std_seriousness": round(np.std(scores), 3) if scores else 0,
            "patients": summaries,
        }
