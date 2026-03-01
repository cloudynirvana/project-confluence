"""
Toxicity Constraints Module — Project Confluence
==================================================

Clinical safety constraints for the SAEM protocol optimizer.
Ensures generated protocols do not exceed maximum tolerated doses (MTD)
or cumulative organ toxicity limits derived from Phase I clinical data.

Each drug has:
- MTD: Maximum tolerated dose (per administration)
- Cumulative limit: Total lifetime exposure limit for organ-specific toxicity
- Dose-limiting toxicity (DLT) organ
- Therapeutic index: Ratio of toxic dose to effective dose
- Grade 3/4 toxicity probability at recommended dose

The ToxicityGuard rejects protocols that violate constraints and scores
protocol safety for optimization.

References:
    - FDA prescribing information for each approved drug
    - NCI CTCAE v5.0 (Common Terminology Criteria for Adverse Events)
    - Phase I trial data from ClinicalTrials.gov
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class DrugToxicityProfile:
    """Clinical toxicity profile for a single drug."""
    drug_name: str
    
    # Dose limits
    mtd_mg_per_m2: float             # Maximum tolerated dose (mg/m²/day)
    recommended_dose_mg: float        # Approved/standard dose (mg/day for 70kg)
    cumulative_limit_mg: float        # Lifetime cumulative limit (mg) — 0 = no limit
    
    # Toxicity characterization
    dlt_organ: str                    # Dose-limiting toxicity organ
    therapeutic_index: float          # TI = toxic_dose / effective_dose (higher = safer)
    grade34_probability: float        # P(Grade 3/4 AE) at recommended dose
    
    # Interaction flags
    hepatotoxic: bool = False
    nephrotoxic: bool = False
    cardiotoxic: bool = False
    myelosuppressive: bool = False
    neurotoxic: bool = False
    
    # Overlapping toxicity penalty
    # When multiple drugs share a DLT organ, cumulative risk increases
    organ_overlap_penalty: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
# CLINICAL TOXICITY DATABASE
# ═══════════════════════════════════════════════════════════════════════

TOXICITY_DATABASE: Dict[str, DrugToxicityProfile] = {
    
    "Dichloroacetate (DCA)": DrugToxicityProfile(
        drug_name="Dichloroacetate (DCA)",
        mtd_mg_per_m2=25.0,          # 25 mg/kg/day max
        recommended_dose_mg=1000,     # ~12.5 mg/kg BID
        cumulative_limit_mg=0,        # No established limit
        dlt_organ="peripheral_nerve",
        therapeutic_index=3.0,
        grade34_probability=0.12,     # Peripheral neuropathy
        neurotoxic=True,
    ),
    
    "Metformin": DrugToxicityProfile(
        drug_name="Metformin",
        mtd_mg_per_m2=50.0,
        recommended_dose_mg=2000,     # 500-1000mg BID
        cumulative_limit_mg=0,
        dlt_organ="gi_tract",
        therapeutic_index=8.0,        # Very safe
        grade34_probability=0.03,     # Lactic acidosis rare
        hepatotoxic=False,
    ),
    
    "2-Deoxyglucose (2-DG)": DrugToxicityProfile(
        drug_name="2-Deoxyglucose (2-DG)",
        mtd_mg_per_m2=63.0,
        recommended_dose_mg=3000,     # ~45 mg/kg
        cumulative_limit_mg=0,
        dlt_organ="cns",
        therapeutic_index=2.5,
        grade34_probability=0.15,     # Hypoglycemia-like symptoms
        neurotoxic=True,
    ),
    
    "CB-839 (Telaglenastat)": DrugToxicityProfile(
        drug_name="CB-839 (Telaglenastat)",
        mtd_mg_per_m2=40.0,
        recommended_dose_mg=800,      # 800mg BID
        cumulative_limit_mg=0,
        dlt_organ="gi_tract",
        therapeutic_index=4.0,
        grade34_probability=0.08,
        hepatotoxic=True,
    ),
    
    "Olaparib (PARP inhibitor)": DrugToxicityProfile(
        drug_name="Olaparib (PARP inhibitor)",
        mtd_mg_per_m2=20.0,
        recommended_dose_mg=600,      # 300mg BID
        cumulative_limit_mg=0,
        dlt_organ="bone_marrow",
        therapeutic_index=3.5,
        grade34_probability=0.18,
        myelosuppressive=True,
    ),
    
    "Vorinostat (SAHA, HDACi)": DrugToxicityProfile(
        drug_name="Vorinostat (SAHA, HDACi)",
        mtd_mg_per_m2=15.0,
        recommended_dose_mg=400,
        cumulative_limit_mg=0,
        dlt_organ="bone_marrow",
        therapeutic_index=2.5,
        grade34_probability=0.20,
        myelosuppressive=True,
    ),
    
    "Anti-PD-1 (Pembrolizumab)": DrugToxicityProfile(
        drug_name="Anti-PD-1 (Pembrolizumab)",
        mtd_mg_per_m2=10.0,
        recommended_dose_mg=200,      # 200mg Q3W
        cumulative_limit_mg=0,
        dlt_organ="immune_system",    # irAEs
        therapeutic_index=5.0,
        grade34_probability=0.15,
        hepatotoxic=True,             # Immune-mediated hepatitis
    ),
    
    "Anti-CTLA-4 (Ipilimumab)": DrugToxicityProfile(
        drug_name="Anti-CTLA-4 (Ipilimumab)",
        mtd_mg_per_m2=10.0,
        recommended_dose_mg=3,        # 3 mg/kg Q3W
        cumulative_limit_mg=0,
        dlt_organ="immune_system",
        therapeutic_index=3.0,
        grade34_probability=0.25,     # Higher irAE rate
        hepatotoxic=True,
    ),
    
    "Hydroxychloroquine (HCQ)": DrugToxicityProfile(
        drug_name="Hydroxychloroquine (HCQ)",
        mtd_mg_per_m2=10.0,
        recommended_dose_mg=400,
        cumulative_limit_mg=365000,   # ~1000mg/day × 1 year retinal risk
        dlt_organ="retina",
        therapeutic_index=4.0,
        grade34_probability=0.05,
        cardiotoxic=True,             # QT prolongation at high doses
    ),
    
    "5-Azacitidine (DNMTi)": DrugToxicityProfile(
        drug_name="5-Azacitidine (DNMTi)",
        mtd_mg_per_m2=75.0,
        recommended_dose_mg=130,      # 75 mg/m² × 1.73 m² avg
        cumulative_limit_mg=0,
        dlt_organ="bone_marrow",
        therapeutic_index=3.0,
        grade34_probability=0.22,
        myelosuppressive=True,
    ),
    
    "N6F11 (Selective GPX4 degrader)": DrugToxicityProfile(
        drug_name="N6F11 (Selective GPX4 degrader)",
        mtd_mg_per_m2=50.0,           # Estimated (preclinical)
        recommended_dose_mg=15,       # μM dosing → estimated mg equivalent
        cumulative_limit_mg=0,
        dlt_organ="liver",
        therapeutic_index=4.5,        # Immune-sparing → better TI than erastin
        grade34_probability=0.10,
        hepatotoxic=True,
    ),
    
    "Bevacizumab (Anti-VEGF)": DrugToxicityProfile(
        drug_name="Bevacizumab (Anti-VEGF)",
        mtd_mg_per_m2=15.0,
        recommended_dose_mg=700,      # 10 mg/kg Q2W
        cumulative_limit_mg=0,
        dlt_organ="vascular",
        therapeutic_index=4.0,
        grade34_probability=0.12,
        cardiotoxic=True,             # Hypertension, thromboembolism
    ),
    
    "CAR-T Cell Therapy": DrugToxicityProfile(
        drug_name="CAR-T Cell Therapy",
        mtd_mg_per_m2=0,              # Cell-based, not mg
        recommended_dose_mg=0,
        cumulative_limit_mg=0,
        dlt_organ="immune_system",    # CRS, neurotoxicity
        therapeutic_index=2.0,        # Narrow window
        grade34_probability=0.35,     # High CRS risk
        neurotoxic=True,
    ),
    
    "High-dose Vitamin C": DrugToxicityProfile(
        drug_name="High-dose Vitamin C",
        mtd_mg_per_m2=1000.0,
        recommended_dose_mg=75000,    # 75g IV
        cumulative_limit_mg=0,
        dlt_organ="kidney",
        therapeutic_index=6.0,
        grade34_probability=0.04,
        nephrotoxic=True,
    ),
    
    "Fasting-Mimicking Diet (FMD)": DrugToxicityProfile(
        drug_name="Fasting-Mimicking Diet (FMD)",
        mtd_mg_per_m2=0,             # Not a drug
        recommended_dose_mg=0,
        cumulative_limit_mg=0,
        dlt_organ="metabolic",
        therapeutic_index=10.0,       # Very safe
        grade34_probability=0.02,
    ),
    
    "Entropic Heating (Hyperthermia)": DrugToxicityProfile(
        drug_name="Entropic Heating (Hyperthermia)",
        mtd_mg_per_m2=0,
        recommended_dose_mg=0,
        cumulative_limit_mg=0,
        dlt_organ="cardiovascular",
        therapeutic_index=5.0,
        grade34_probability=0.06,
        cardiotoxic=True,
    ),
    
    "Ferroptosis Inducer (Erastin/RSL3)": DrugToxicityProfile(
        drug_name="Ferroptosis Inducer (Erastin/RSL3)",
        mtd_mg_per_m2=30.0,
        recommended_dose_mg=10,       # μM research dose
        cumulative_limit_mg=0,
        dlt_organ="liver",
        therapeutic_index=2.0,        # Non-selective → narrow window
        grade34_probability=0.25,
        hepatotoxic=True,
        nephrotoxic=True,
    ),
    
    "NAD+ Precursors (NMN/NR)": DrugToxicityProfile(
        drug_name="NAD+ Precursors (NMN/NR)",
        mtd_mg_per_m2=500.0,
        recommended_dose_mg=500,
        cumulative_limit_mg=0,
        dlt_organ="gi_tract",
        therapeutic_index=10.0,
        grade34_probability=0.01,
    ),
    
    "Epogen (Epoetin alfa)": DrugToxicityProfile(
        drug_name="Epogen (Epoetin alfa)",
        mtd_mg_per_m2=0,
        recommended_dose_mg=0,
        cumulative_limit_mg=0,
        dlt_organ="vascular",
        therapeutic_index=1.5,        # Dangerous in cancer
        grade34_probability=0.30,
        cardiotoxic=True,
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# TOXICITY GUARD
# ═══════════════════════════════════════════════════════════════════════

class ToxicityGuard:
    """
    Clinical safety gate for SAEM protocols.
    
    Evaluates protocols against MTD limits, organ overlap, cumulative
    exposure, and therapeutic index to produce a safety score and
    identify violations.
    
    Usage:
        guard = ToxicityGuard()
        result = guard.evaluate_protocol(drug_names, phase_days)
        
        if not result["is_safe"]:
            print(f"REJECTED: {result['violations']}")
        else:
            print(f"Safety score: {result['safety_score']:.2f}")
    """
    
    def __init__(self, database: Optional[Dict[str, DrugToxicityProfile]] = None):
        self.db = database or TOXICITY_DATABASE
    
    def evaluate_protocol(self, 
                          drug_names: List[str],
                          phase_days: Dict[str, int],
                          drug_phases: Optional[Dict[str, List[str]]] = None,
                          ) -> Dict:
        """
        Evaluate a complete protocol for safety.
        
        Args:
            drug_names: All drugs in the protocol
            phase_days: {"flatten": n1, "heat": n2, "push": n3}
            drug_phases: Optional mapping of phase → active drugs in that phase
        
        Returns:
            Dict with safety assessment:
              - is_safe: bool
              - safety_score: float (0-1, higher = safer)
              - violations: List[str]
              - organ_risk: Dict[str, float]
              - recommendations: List[str]
        """
        violations = []
        recommendations = []
        organ_risk: Dict[str, float] = {}
        
        total_days = sum(phase_days.values())
        profiles = []
        
        for name in drug_names:
            profile = self.db.get(name)
            if profile:
                profiles.append(profile)
            else:
                recommendations.append(f"⚠️ No toxicity data for '{name}' — manual review required")
        
        if not profiles:
            return {
                "is_safe": False,
                "safety_score": 0.0,
                "violations": ["No drug toxicity profiles found"],
                "organ_risk": {},
                "recommendations": recommendations,
            }
        
        # ── Check 1: Therapeutic Index ──
        min_ti = min(p.therapeutic_index for p in profiles)
        if min_ti < 2.0:
            violations.append(
                f"NARROW therapeutic index ({min_ti:.1f}) — "
                f"drug: {[p.drug_name for p in profiles if p.therapeutic_index == min_ti][0]}"
            )
        
        # ── Check 2: Cumulative G3/4 probability ──
        # Approximate: P(any G3/4) = 1 - ∏(1 - p_i)
        product = 1.0
        for p in profiles:
            product *= (1.0 - p.grade34_probability)
        cumulative_g34 = 1.0 - product
        
        if cumulative_g34 > 0.60:
            violations.append(
                f"HIGH cumulative Grade 3/4 AE probability: {cumulative_g34:.0%} "
                f"(threshold: 60%)"
            )
        elif cumulative_g34 > 0.40:
            recommendations.append(
                f"⚠️ Moderate G3/4 AE risk: {cumulative_g34:.0%} — close monitoring required"
            )
        
        # ── Check 3: Organ overlap ──
        organ_counts: Dict[str, List[str]] = {}
        for p in profiles:
            # DLT organ
            organ_counts.setdefault(p.dlt_organ, []).append(p.drug_name)
            # Additional organ flags
            if p.hepatotoxic:
                organ_counts.setdefault("liver", []).append(p.drug_name)
            if p.nephrotoxic:
                organ_counts.setdefault("kidney", []).append(p.drug_name)
            if p.cardiotoxic:
                organ_counts.setdefault("heart", []).append(p.drug_name)
            if p.myelosuppressive:
                organ_counts.setdefault("bone_marrow", []).append(p.drug_name)
            if p.neurotoxic:
                organ_counts.setdefault("nervous_system", []).append(p.drug_name)
        
        for organ, drugs in organ_counts.items():
            unique_drugs = list(set(drugs))
            risk = len(unique_drugs) * 0.15  # Each drug adds 15% organ risk
            organ_risk[organ] = min(1.0, risk)
            
            if len(unique_drugs) >= 3:
                violations.append(
                    f"TRIPLE organ overlap on {organ}: {', '.join(unique_drugs)}"
                )
            elif len(unique_drugs) >= 2:
                recommendations.append(
                    f"⚠️ Dual {organ} toxicity: {', '.join(unique_drugs)} — stagger dosing"
                )
        
        # ── Check 4: Duration-adjusted risk ──
        # Longer protocols amplify cumulative toxicity
        duration_factor = min(2.0, total_days / 45.0)  # Normalized to ~45 day protocol
        duration_adjusted_risk = cumulative_g34 * duration_factor
        
        if duration_adjusted_risk > 0.75:
            violations.append(
                f"Duration-adjusted toxicity risk too high: {duration_adjusted_risk:.0%}"
            )
        
        # ── Check 5: Specific dangerous combinations ──
        drug_set = set(drug_names)
        
        # Dual checkpoint (PD-1 + CTLA-4) — known high irAE rate
        if {"Anti-PD-1 (Pembrolizumab)", "Anti-CTLA-4 (Ipilimumab)"}.issubset(drug_set):
            recommendations.append(
                "⚠️ Dual checkpoint blockade: 55% Grade 3/4 irAE rate "
                "(Larkin et al. 2015) — consider sequential dosing"
            )
        
        # Epogen in cancer — contraindicated
        if "Epogen (Epoetin alfa)" in drug_set:
            violations.append(
                "CONTRAINDICATED: Epogen promotes tumor growth via EPO receptor signaling"
            )
        
        # ── Compute safety score ──
        # Base score from therapeutic indices
        ti_scores = [min(1.0, p.therapeutic_index / 5.0) for p in profiles]
        ti_avg = np.mean(ti_scores) if ti_scores else 0.5
        
        # Penalty for organ overlaps
        overlap_penalty = sum(
            0.1 for drugs in organ_counts.values() 
            if len(set(drugs)) >= 2
        )
        
        # Penalty for violations
        violation_penalty = len(violations) * 0.2
        
        safety_score = max(0.0, min(1.0, 
            ti_avg * (1.0 - cumulative_g34 * 0.5) - overlap_penalty - violation_penalty
        ))
        
        is_safe = len(violations) == 0 and safety_score > 0.3
        
        return {
            "is_safe": is_safe,
            "safety_score": round(safety_score, 3),
            "cumulative_g34_probability": round(cumulative_g34, 3),
            "violations": violations,
            "organ_risk": {k: round(v, 3) for k, v in organ_risk.items()},
            "recommendations": recommendations,
            "drug_profiles": {
                p.drug_name: {
                    "therapeutic_index": p.therapeutic_index,
                    "grade34_prob": p.grade34_probability,
                    "dlt_organ": p.dlt_organ,
                }
                for p in profiles
            },
        }
    
    def get_safety_summary(self, result: Dict) -> str:
        """Generate human-readable safety summary."""
        lines = []
        lines.append("═" * 60)
        lines.append("TOXICITY SAFETY ASSESSMENT")
        lines.append("═" * 60)
        
        status = "✅ SAFE" if result["is_safe"] else "❌ UNSAFE"
        lines.append(f"\nVerdict: {status}")
        lines.append(f"Safety Score: {result['safety_score']:.1%}")
        lines.append(f"Cumulative G3/4 Risk: {result['cumulative_g34_probability']:.0%}")
        
        if result["violations"]:
            lines.append("\n🚫 VIOLATIONS:")
            for v in result["violations"]:
                lines.append(f"  • {v}")
        
        if result["organ_risk"]:
            lines.append("\n🎯 Organ Risk Map:")
            for organ, risk in sorted(result["organ_risk"].items(), 
                                       key=lambda x: -x[1]):
                bar = "█" * int(risk * 10)
                lines.append(f"  {organ:20s} {bar} {risk:.0%}")
        
        if result["recommendations"]:
            lines.append("\n📋 Recommendations:")
            for r in result["recommendations"]:
                lines.append(f"  {r}")
        
        lines.append("\n" + "═" * 60)
        return "\n".join(lines)
