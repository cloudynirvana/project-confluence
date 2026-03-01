"""
Diabetes ODE System — Project Confluence Extension
=====================================================

10-dimensional metabolic/hormonal generator system for diabetes modeling.
Applies SAEM geometric alignment framework to insulin resistance,
beta-cell failure, and diabetic complications.

Key Axes (10D State Space):
  0. Glucose       — Plasma glucose (mmol/L normalized)
  1. Insulin       — Circulating insulin (reduced = T1D, elevated early = T2D)
  2. FFA           — Free Fatty Acids (lipotoxicity driver)
  3. Glucagon      — Counter-regulatory (elevated in diabetes)
  4. ATP_beta      — Beta-cell ATP (drives insulin secretion coupling)
  5. ROS_beta      — Beta-cell ROS (oxidative stress → apoptosis)
  6. Adiponectin   — Insulin sensitizer (reduced in T2D)
  7. TNF_alpha     — Inflammatory cytokine (insulin resistance)
  8. HbA1c_proxy   — Glycation proxy (cumulative damage marker)
  9. Cortisol      — Stress hormone (counter-regulatory, insulin antagonist)

Key References:
  - Stumvoll et al. 2005, Lancet (T2D pathophysiology)
  - DeFronzo 2009, Diabetes (ominous octet)
  - Poitout & Robertson 2008, Endocrine Reviews (glucolipotoxicity)
  - Bergman 2005, Diabetes (minimal model)
  - ADA Standards of Care 2024
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field


DIABETES_AXES = [
    "Glucose", "Insulin", "FFA", "Glucagon", "ATP_beta",
    "ROS_beta", "Adiponectin", "TNF_alpha", "HbA1c_proxy", "Cortisol",
]


@dataclass
class DiabetesParams:
    """Tunable parameters for diabetes ODE system."""
    # Glucose dynamics
    glucose_clearance: float = -0.60     # Insulin-mediated uptake (healthy)
    hepatic_output: float = 0.20         # Liver glucose production
    glucagon_glucose: float = 0.25       # Glucagon stimulates glucose
    
    # Insulin dynamics
    insulin_clearance: float = -0.50     # Hepatic insulin degradation
    glucose_stimulated: float = 0.40     # Glucose → insulin secretion
    atp_coupling: float = 0.30           # ATP drives insulin exocytosis
    
    # Lipid axis
    ffa_clearance: float = -0.45         # FFA oxidation/storage
    insulin_suppresses_ffa: float = -0.30 # Insulin → antilipolysis
    
    # Counter-regulation
    glucagon_clearance: float = -0.40
    cortisol_clearance: float = -0.35
    
    # Beta-cell homeostasis
    ros_clearance: float = -0.70         # Antioxidant capacity (healthy)
    adiponectin_turnover: float = -0.30
    tnf_clearance: float = -0.50
    hba1c_turnover: float = -0.10        # Slow (RBC lifespan ~120 days)


@dataclass
class DiabetesMetadata:
    """Metadata for a diabetes generator."""
    subtype: str
    description: str
    risk_factors: List[str]
    complications: List[str]
    mortality_rate: str        # Annual mortality description
    prevalence: str
    confidence: str = "high"


# ═══════════════════════════════════════════════════════════════
# PER-SUBTYPE METADATA
# ═══════════════════════════════════════════════════════════════

DIABETES_METADATA: Dict[str, DiabetesMetadata] = {
    "Healthy": DiabetesMetadata(
        subtype="Healthy",
        description="Normal glucose homeostasis with intact beta-cell function",
        risk_factors=[],
        complications=[],
        mortality_rate="Baseline",
        prevalence="~90% of population",
    ),
    "PreDiabetes": DiabetesMetadata(
        subtype="PreDiabetes",
        description="Impaired fasting glucose / impaired glucose tolerance. Reversible.",
        risk_factors=["Obesity", "Sedentary", "Family history", "PCOS"],
        complications=["Elevated cardiovascular risk (1.5x)"],
        mortality_rate="~5-10% progress to T2D per year",
        prevalence="~38% of US adults (96 million)",
    ),
    "T2D_Early": DiabetesMetadata(
        subtype="T2D_Early",
        description="Early T2D: insulin resistance with compensatory hyperinsulinemia",
        risk_factors=["Obesity", "Insulin resistance", "Metabolic syndrome"],
        complications=["Microvascular onset (retinopathy, nephropathy)"],
        mortality_rate="1.8x all-cause mortality vs healthy",
        prevalence="~37 million in US",
    ),
    "T2D_Advanced": DiabetesMetadata(
        subtype="T2D_Advanced",
        description="Advanced T2D: beta-cell failure, insulin dependency, complications",
        risk_factors=["Duration >10y", "Poor glycemic control", "Comorbidities"],
        complications=["CVD (2-4x)", "CKD", "Neuropathy", "Amputation", "Blindness"],
        mortality_rate="3x all-cause mortality; #1 cause: cardiovascular",
        prevalence="~40% of T2D patients progress here",
    ),
    "T1D": DiabetesMetadata(
        subtype="T1D",
        description="Autoimmune beta-cell destruction. Absolute insulin deficiency.",
        risk_factors=["HLA-DR3/DR4", "Autoantibodies (GAD65, IA-2, ZnT8)"],
        complications=["DKA (acute)", "Hypoglycemia (acute)", "All microvascular"],
        mortality_rate="3-4x all-cause mortality; life expectancy -12 years",
        prevalence="~1.6 million in US",
    ),
    "GDM": DiabetesMetadata(
        subtype="GDM",
        description="Gestational diabetes: insulin resistance from placental hormones",
        risk_factors=["Obesity", "Age >35", "Prior GDM", "PCOS"],
        complications=["Macrosomia", "Preeclampsia", "50% risk of T2D within 10y"],
        mortality_rate="Low acute; high long-term T2D conversion",
        prevalence="~10% of pregnancies",
    ),
}


class DiabetesODESystem:
    """
    10-dimensional diabetes metabolic/hormonal generator system.
    
    Applies the same SAEM geometric framework used for cancer:
    - Generator matrices define attractor basins
    - Disease = trapped in pathological attractor
    - Cure = flatten basin → add noise → push toward healthy attractor
    """
    
    N = 10  # Same dimensionality as cancer framework
    
    @classmethod
    def healthy_generator(cls, params: Optional[DiabetesParams] = None) -> np.ndarray:
        """
        Healthy glucose homeostasis generator.
        
        Balanced insulin-glucose feedback, intact beta-cells,
        low inflammation, normal lipid handling.
        
        Ref: Bergman minimal model extended; Stumvoll et al. 2005 Lancet
        """
        p = params or DiabetesParams()
        n = cls.N
        A = np.zeros((n, n))
        
        # ── Diagonal: turnover/clearance rates ──
        A[0, 0] = p.glucose_clearance     # Glucose disposal (muscle + liver)
        A[1, 1] = p.insulin_clearance      # Hepatic insulin degradation
        A[2, 2] = p.ffa_clearance          # FFA oxidation/storage
        A[3, 3] = p.glucagon_clearance     # Glucagon clearance
        A[4, 4] = -0.30                    # Beta-cell ATP homeostasis
        A[5, 5] = p.ros_clearance          # Beta-cell antioxidant capacity
        A[6, 6] = p.adiponectin_turnover   # Adiponectin turnover
        A[7, 7] = p.tnf_clearance          # TNF-α clearance
        A[8, 8] = p.hba1c_turnover         # HbA1c slow turnover (120d RBC)
        A[9, 9] = p.cortisol_clearance     # Cortisol clearance
        
        # ── Key couplings: insulin-glucose axis ──
        A[0, 1] = p.glucose_stimulated     # Glucose stimulates insulin
        A[1, 0] = -0.35                    # Insulin promotes glucose uptake
        
        # ── Beta-cell coupling ──
        A[0, 4] = 0.20                     # Glucose → beta-cell ATP (GSIS)
        A[4, 1] = p.atp_coupling           # ATP drives insulin exocytosis
        
        # ── Counter-regulation ──
        A[3, 0] = p.glucagon_glucose       # Glucagon → hepatic glucose output
        A[0, 3] = 0.15                     # Low glucose stimulates glucagon
        
        # ── Lipid axis ──
        A[1, 2] = p.insulin_suppresses_ffa # Insulin suppresses lipolysis
        A[2, 0] = 0.10                     # Lipids contribute to glucose (gluco)
        
        # ── Inflammatory axis ──
        A[7, 6] = -0.20                    # Adiponectin suppresses TNF-α
        A[6, 1] = 0.15                     # Insulin promotes adiponectin
        
        # ── Damage/glycation ──
        A[0, 8] = 0.05                     # Glucose → glycation (HbA1c)
        
        # ── ROS from metabolism ──
        A[2, 5] = 0.10                     # FFA → beta-cell ROS (lipotoxicity)
        A[0, 5] = 0.08                     # Glucose → beta-cell ROS (glucotoxicity)
        
        # ── Cortisol counter-regulation ──
        A[9, 0] = 0.12                     # Cortisol → glucose (gluconeogenesis)
        A[9, 1] = -0.10                    # Cortisol → insulin resistance
        
        return A

    @classmethod
    def prediabetes_generator(cls, params: Optional[DiabetesParams] = None) -> np.ndarray:
        """
        Pre-diabetes: early insulin resistance with compensatory secretion.
        
        The basin is shallow — this is the REVERSIBLE stage.
        Lifestyle intervention (weight loss, exercise) can flatten
        this attractor back to healthy.
        
        Ref: ADA 2024 Standards of Care; DeFronzo 2009 (ominous octet)
        """
        A = cls.healthy_generator(params)
        
        # Mild insulin resistance (muscle + liver)
        A[1, 0] = -0.22          # Reduced insulin→glucose disposal
        A[0, 0] = -0.48          # Slower glucose clearance
        
        # Compensatory hyperinsulinemia (beta-cells working harder)
        A[0, 1] = 0.55           # More glucose-stimulated insulin
        A[4, 4] = -0.35          # Beta-cell ATP demand increases
        
        # Early FFA elevation (visceral adiposity)
        A[2, 2] = -0.35          # Slower FFA clearance
        A[1, 2] = -0.22          # Reduced insulin suppression of lipolysis
        
        # Early inflammation
        A[7, 7] = -0.40          # Slower TNF-α clearance
        A[6, 6] = -0.35          # Adiponectin starting to decline
        
        # Mild beta-cell stress
        A[5, 5] = -0.60          # Slightly reduced ROS clearance
        A[2, 5] = 0.15           # More FFA → ROS
        
        # Cortisol: stress axis mildly activated
        A[9, 9] = -0.30          # Slightly slower clearance
        
        return A
    
    @classmethod
    def t2d_early_generator(cls, params: Optional[DiabetesParams] = None) -> np.ndarray:
        """
        Early T2D: frank insulin resistance + compensatory hyperinsulinemia.
        
        Deeper attractor than pre-diabetes. Beta-cells still functional
        but under significant glucolipotoxic stress.
        
        Ref: DeFronzo 2009, Diabetes (ominous octet)
        """
        A = cls.healthy_generator(params)
        
        # Significant insulin resistance
        A[1, 0] = -0.15          # Significant resistance to glucose disposal
        A[0, 0] = -0.38          # Poor glucose clearance
        
        # Hyperinsulinemia (compensation)
        A[0, 1] = 0.65           # Ramped-up GSIS
        A[1, 1] = -0.40          # Insulin clearance slightly reduced (fatty liver)
        
        # Hepatic glucose overproduction
        A[3, 0] = 0.35           # Glucagon driving more hepatic output
        A[3, 3] = -0.30          # Slower glucagon suppression
        
        # FFA elevation (lipolysis uninhibited)
        A[2, 2] = -0.28          # Poor FFA clearance
        A[1, 2] = -0.15          # Weak antilipolytic effect of insulin
        A[2, 0] = 0.18           # More lipid → glucose (gluconeogenesis)
        
        # Beta-cell stress
        A[5, 5] = -0.45          # Reduced antioxidant capacity
        A[2, 5] = 0.22           # More lipotoxicity
        A[0, 5] = 0.15           # More glucotoxicity
        A[4, 4] = -0.42          # ATP demand exhausting
        
        # Inflammation
        A[7, 7] = -0.32          # TNF-α accumulating
        A[6, 6] = -0.40          # Adiponectin declining
        A[7, 6] = -0.12          # Weakened adiponectin→TNF suppression
        
        # Glycation damage
        A[0, 8] = 0.12           # Elevated glucose → HbA1c
        
        # Cortisol dysregulation
        A[9, 0] = 0.18           # More cortisol-driven gluconeogenesis
        
        return A
    
    @classmethod
    def t2d_advanced_generator(cls, params: Optional[DiabetesParams] = None) -> np.ndarray:
        """
        Advanced T2D: beta-cell failure + multi-organ complications.
        
        DEEP attractor — analogous to PDAC in cancer. 
        Beta-cells destroyed (>50%), insulin-dependent,
        complications driving mortality.
        
        Ref: UK Prospective Diabetes Study (UKPDS); 
             Poitout & Robertson 2008 (glucolipotoxicity)
        """
        A = cls.healthy_generator(params)
        
        # Severe insulin resistance
        A[1, 0] = -0.08          # Near-total resistance
        A[0, 0] = -0.25          # Very poor glucose clearance
        
        # Beta-cell failure (declining insulin output)
        A[0, 1] = 0.25           # Insufficient GSIS (beta-cell mass <50%)
        A[4, 4] = -0.55          # ATP production failing
        A[4, 1] = 0.12           # Weak ATP→insulin coupling
        
        # Severe FFA/lipotoxicity
        A[2, 2] = -0.18          # FFA accumulating (fatty liver + muscle)
        A[1, 2] = -0.05          # Almost no antilipolytic effect
        A[2, 5] = 0.35           # Massive lipotoxic ROS
        
        # Glucagon excess (alpha-cell dysfunction)
        A[3, 3] = -0.22          # Inappropriate glucagon
        A[3, 0] = 0.45           # Excessive hepatic glucose output
        
        # Beta-cell ROS destruction
        A[5, 5] = -0.25          # Overwhelmed antioxidants
        A[0, 5] = 0.25           # Severe glucotoxicity
        
        # Chronic inflammation
        A[7, 7] = -0.22          # TNF-α accumulation
        A[6, 6] = -0.50          # Adiponectin very low
        A[7, 6] = -0.05          # Adiponectin→TNF pathway broken
        
        # Glycation damage (complications driver)
        A[0, 8] = 0.20           # Chronic hyperglycemia → HbA1c → AGEs
        A[8, 8] = -0.08          # Slow clearance of glycated products
        
        # Cortisol excess (stress axis overactive)
        A[9, 9] = -0.25          # Chronic hypercortisolism
        A[9, 0] = 0.25           # More gluconeogenesis
        A[9, 1] = -0.18          # More insulin resistance
        
        return A
    
    @classmethod
    def t1d_generator(cls, params: Optional[DiabetesParams] = None) -> np.ndarray:
        """
        Type 1 Diabetes: autoimmune beta-cell destruction.
        
        Unique attractor — immune-mediated, not metabolic.
        Insulin secretion near-zero. Survival depends on exogenous insulin.
        
        Ref: Atkinson et al. 2014, Lancet
        """
        A = cls.healthy_generator(params)
        
        # Complete beta-cell failure
        A[0, 1] = 0.05           # Minimal residual insulin secretion
        A[4, 4] = -0.60          # Beta-cell ATP depleted (apoptosis)
        A[4, 1] = 0.05           # Near-zero coupling
        
        # Total insulin deficiency consequences
        A[1, 0] = -0.10          # Poor glucose disposal
        A[0, 0] = -0.20          # Very poor clearance
        A[1, 1] = -0.60          # Rapid insulin degradation (exogenous)
        
        # FFA surge (no insulin → uncontrolled lipolysis)
        A[2, 2] = -0.15          # FFA accumulating
        A[1, 2] = -0.02          # No antilipolytic effect
        
        # Glucagon excess (no paracrine insulin suppression)
        A[3, 3] = -0.20          # Inappropriate glucagon
        A[3, 0] = 0.50           # Massive hepatic output
        
        # Beta-cell ROS (autoimmune + metabolic)
        A[5, 5] = -0.20          # Destroyed antioxidant capacity
        A[0, 5] = 0.30           # Severe glucotoxicity
        
        # DKA risk axis (ketogenesis from FFA)
        A[2, 0] = 0.25           # FFA → glucose (ketogenic substrate)
        
        # Inflammation (autoimmune ongoing)
        A[7, 7] = -0.25          # Chronic inflammation
        
        return A
    
    @classmethod
    def all_generators(cls) -> Dict[str, np.ndarray]:
        """Return all diabetes generator matrices."""
        return {
            "Healthy": cls.healthy_generator(),
            "PreDiabetes": cls.prediabetes_generator(),
            "T2D_Early": cls.t2d_early_generator(),
            "T2D_Advanced": cls.t2d_advanced_generator(),
            "T1D": cls.t1d_generator(),
        }


# ═══════════════════════════════════════════════════════════════
# DIABETES THERAPEUTIC INTERVENTIONS
# ═══════════════════════════════════════════════════════════════

@dataclass
class DiabetesIntervention:
    """A diabetes therapeutic intervention mapped to generator correction."""
    name: str
    mechanism: str
    expected_effect: np.ndarray   # 10×10 δA correction
    category: str                 # lifestyle, oral, injectable, surgical
    evidence_level: str
    references: List[str]
    curvature_multiplier: float = 1.0
    fatality_reduction: str = ""  # Proven mortality benefit


def build_diabetes_drug_library(n: int = 10) -> List[DiabetesIntervention]:
    """Build library of diabetes interventions mapped to generator corrections."""
    interventions = []
    
    # ────────────────────────────────────────────────────────
    # 1. METFORMIN (First-line T2D)
    # Ref: UKPDS 1998 (34% CVD mortality reduction)
    # ────────────────────────────────────────────────────────
    met = np.zeros((n, n))
    met[0, 0] = -0.15       # Improves glucose clearance
    met[3, 0] = -0.12       # Reduces hepatic glucose output
    met[2, 2] = -0.08       # Mild FFA reduction
    met[7, 7] = -0.05       # Anti-inflammatory
    interventions.append(DiabetesIntervention(
        name="Metformin",
        mechanism="AMPK activation → hepatic glucose suppression + insulin sensitization",
        expected_effect=met,
        category="oral",
        evidence_level="Gold Standard",
        references=["UKPDS 1998 Lancet (34% CVD mortality reduction)", "DeFronzo 2009"],
        curvature_multiplier=0.80,
        fatality_reduction="34% reduction in diabetes-related mortality (UKPDS)",
    ))
    
    # ────────────────────────────────────────────────────────
    # 2. GLP-1 RECEPTOR AGONISTS (Semaglutide/Liraglutide)
    # Ref: SUSTAIN-6 (26% MACE reduction), LEADER trial
    # ────────────────────────────────────────────────────────
    glp1 = np.zeros((n, n))
    glp1[0, 1] = 0.20       # Enhances glucose-stimulated insulin
    glp1[3, 3] = -0.15      # Suppresses glucagon
    glp1[0, 0] = -0.12      # Improves glucose clearance
    glp1[2, 2] = -0.10      # FFA reduction (weight loss)
    glp1[4, 4] = 0.08       # Beta-cell protection
    glp1[5, 5] = -0.05      # Mild antioxidant
    interventions.append(DiabetesIntervention(
        name="GLP-1 RA (Semaglutide)",
        mechanism="GLP-1R agonism → insulin secretion + glucagon suppression + weight loss",
        expected_effect=glp1,
        category="injectable",
        evidence_level="Clinical Standard",
        references=[
            "SUSTAIN-6 2016 NEJM (26% MACE reduction)",
            "LEADER 2016 NEJM (liraglutide CVD benefit)",
            "SELECT 2023 NEJM (semaglutide 20% MACE in obesity)",
        ],
        curvature_multiplier=0.65,
        fatality_reduction="26% MACE reduction; 15% all-cause mortality reduction",
    ))
    
    # ────────────────────────────────────────────────────────
    # 3. SGLT2 INHIBITORS (Empagliflozin/Dapagliflozin)
    # Ref: EMPA-REG (38% CVD death reduction!)
    # ────────────────────────────────────────────────────────
    sglt2 = np.zeros((n, n))
    sglt2[0, 0] = -0.18     # Glucose excretion (renal)
    sglt2[2, 2] = -0.10     # FFA shift to ketone metabolism
    sglt2[5, 5] = -0.08     # Reduced oxidative stress
    sglt2[7, 7] = -0.06     # Anti-inflammatory
    interventions.append(DiabetesIntervention(
        name="SGLT2i (Empagliflozin)",
        mechanism="Renal glucose excretion + hemodynamic benefit + ketone metabolism",
        expected_effect=sglt2,
        category="oral",
        evidence_level="Clinical Standard",
        references=[
            "EMPA-REG 2015 NEJM (38% CV death reduction!)",
            "DAPA-HF 2019 NEJM (heart failure benefit)",
            "CREDENCE 2019 NEJM (renal protection)",
        ],
        curvature_multiplier=0.70,
        fatality_reduction="38% CV death reduction (EMPA-REG); strongest mortality signal",
    ))
    
    # ────────────────────────────────────────────────────────
    # 4. INSULIN (Exogenous — T1D essential, T2D advanced)
    # ────────────────────────────────────────────────────────
    ins = np.zeros((n, n))
    ins[1, 0] = -0.25       # Restores glucose disposal
    ins[0, 0] = -0.20       # Improves clearance
    ins[1, 2] = -0.20       # Suppresses lipolysis
    ins[3, 3] = -0.10       # Suppresses glucagon
    interventions.append(DiabetesIntervention(
        name="Insulin (Exogenous)",
        mechanism="Direct glucose disposal + lipolysis suppression + glucagon suppression",
        expected_effect=ins,
        category="injectable",
        evidence_level="Essential",
        references=["Banting & Best 1922", "DCCT 1993 NEJM"],
        curvature_multiplier=0.60,
        fatality_reduction="Life-saving in T1D; 76% microvascular reduction (DCCT)",
    ))
    
    # ────────────────────────────────────────────────────────
    # 5. LIFESTYLE (Weight loss, exercise, diet)
    # Ref: DPP 2002 (58% T2D prevention)
    # ────────────────────────────────────────────────────────
    life = np.zeros((n, n))
    life[1, 0] = -0.18      # Restored insulin sensitivity
    life[0, 0] = -0.15      # Better glucose clearance
    life[2, 2] = -0.15      # FFA reduction
    life[6, 6] = 0.10       # Adiponectin increase
    life[7, 7] = -0.12      # TNF-α reduction
    life[5, 5] = -0.10      # ROS reduction
    life[9, 9] = -0.08      # Cortisol normalization
    interventions.append(DiabetesIntervention(
        name="Lifestyle (7% weight loss + exercise)",
        mechanism="Insulin sensitization + adipokine normalization + inflammation reduction",
        expected_effect=life,
        category="lifestyle",
        evidence_level="Gold Standard",
        references=[
            "DPP 2002 NEJM (58% T2D prevention)",
            "Look AHEAD 2013",
            "Finnish DPS 2001 NEJM",
        ],
        curvature_multiplier=0.55,
        fatality_reduction="58% reduction in T2D incidence; reverses pre-diabetes",
    ))
    
    # ────────────────────────────────────────────────────────
    # 6. BARIATRIC SURGERY (Metabolic surgery)
    # Ref: STAMPEDE 2012 NEJM (T2D remission)
    # ────────────────────────────────────────────────────────
    bari = np.zeros((n, n))
    bari[1, 0] = -0.30      # Major insulin sensitization
    bari[0, 0] = -0.25      # Strong glucose clearance
    bari[2, 2] = -0.25      # Major FFA reduction
    bari[0, 1] = 0.30       # Restored GSIS (incretin effect)
    bari[6, 6] = 0.15       # Adiponectin restoration
    bari[7, 7] = -0.20      # Inflammation resolution
    bari[4, 4] = 0.15       # Beta-cell recovery
    bari[5, 5] = -0.15      # ROS reduction
    interventions.append(DiabetesIntervention(
        name="Bariatric Surgery (RYGB/Sleeve)",
        mechanism="Gut hormone remodeling + caloric restriction + adipose reduction",
        expected_effect=bari,
        category="surgical",
        evidence_level="Clinical Standard",
        references=[
            "STAMPEDE 2012 NEJM (T2D remission 37-42%)",
            "Swedish Obese Subjects Study 2012 NEJM (29% mortality reduction)",
            "ADA/ASMBS 2022 Guidelines",
        ],
        curvature_multiplier=0.40,
        fatality_reduction="29% all-cause mortality reduction (SOS); 56% CV mortality reduction",
    ))
    
    # ────────────────────────────────────────────────────────
    # 7. TIRZEPATIDE (Dual GIP/GLP-1 agonist — latest)
    # Ref: SURPASS-4, SURMOUNT-1
    # ────────────────────────────────────────────────────────
    tirz = np.zeros((n, n))
    tirz[0, 1] = 0.25       # Enhanced insulin secretion
    tirz[3, 3] = -0.18      # Glucagon suppression
    tirz[0, 0] = -0.18      # Strong glucose clearance
    tirz[2, 2] = -0.15      # Major FFA reduction (22% weight loss)
    tirz[4, 4] = 0.10       # Beta-cell protection
    tirz[6, 6] = 0.08       # Adiponectin improvement
    tirz[7, 7] = -0.08      # Anti-inflammatory
    interventions.append(DiabetesIntervention(
        name="Tirzepatide (Dual GIP/GLP-1)",
        mechanism="Dual incretin agonism → superior GSIS + glucagon suppression + 22% weight loss",
        expected_effect=tirz,
        category="injectable",
        evidence_level="Clinical Standard",
        references=[
            "SURPASS-4 2021 Lancet (HbA1c reduction to 6.4%)",
            "SURMOUNT-1 2022 NEJM (22.5% weight loss)",
        ],
        curvature_multiplier=0.55,
        fatality_reduction="HbA1c to near-normal; CVOT pending (SURPASS-CVOT)",
    ))
    
    return interventions
