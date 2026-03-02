"""
Cardiovascular ODE System — Project Confluence Extension
==========================================================

10-dimensional hemodynamic/vascular generator system for cardiovascular disease modeling.
Applies SAEM geometric alignment framework to atherosclerosis,
heart failure, and thrombotic events.

Key Axes (10D State Space):
  0. LDL_ox       — Oxidized LDL (atherogenesis driver)
  1. HDL          — Protective lipoprotein (reverse cholesterol transport)
  2. Endothelial  — Endothelial function (NO bioavailability)
  3. VSMC         — Vascular smooth muscle cell proliferation (plaque growth)
  4. Platelet     — Platelet activation/aggregation (thrombosis risk)
  5. CRP          — C-reactive protein (systemic inflammation marker)
  6. RAAS         — Renin-angiotensin-aldosterone system activity
  7. BNP          — B-type natriuretic peptide (myocardial stress)
  8. EF           — Ejection fraction proxy (cardiac output, higher = healthier)
  9. Fibrosis     — Myocardial/vascular fibrosis (remodeling)

Key References:
  - Libby et al. 2019, Nature Reviews Disease Primers (atherosclerosis)
  - Ridker et al. 2017, NEJM (CANTOS — inflammation in CVD)
  - McMurray et al. 2014, NEJM (PARADIGM-HF — sacubitril/valsartan)
  - Cannon et al. 2015, NEJM (IMPROVE-IT — ezetimibe)
  - Yusuf et al. 2000, NEJM (HOPE — ramipril)
  - ACC/AHA 2022 Guidelines
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field


CVD_AXES = [
    "LDL_ox", "HDL", "Endothelial", "VSMC", "Platelet",
    "CRP", "RAAS", "BNP", "EF", "Fibrosis",
]


@dataclass
class CVDParams:
    """Tunable parameters for cardiovascular ODE system."""
    # Lipid dynamics
    ldl_clearance: float = -0.55      # Hepatic LDL receptor uptake
    hdl_turnover: float = -0.30       # HDL catabolism
    hdl_efflux: float = 0.25          # HDL-mediated cholesterol efflux

    # Endothelial dynamics
    endo_recovery: float = -0.40      # Endothelial repair rate
    no_production: float = 0.30       # Nitric oxide synthesis

    # Vascular remodeling
    vsmc_clearance: float = -0.35     # VSMC apoptosis/quiescence
    fibrosis_turnover: float = -0.15  # MMP-mediated ECM degradation (slow)

    # Hemostasis
    platelet_clearance: float = -0.50 # Platelet consumption/clearance
    raas_clearance: float = -0.40     # RAAS feedback suppression

    # Cardiac
    bnp_clearance: float = -0.45      # Neprilysin clearance
    ef_homeostasis: float = -0.20     # Cardiac output regulation

    # Inflammation
    crp_clearance: float = -0.50      # CRP hepatic clearance


@dataclass
class CVDMetadata:
    """Metadata for a cardiovascular generator."""
    subtype: str
    description: str
    risk_factors: List[str]
    complications: List[str]
    mortality_rate: str
    prevalence: str
    confidence: str = "high"


# ═══════════════════════════════════════════════════════════════
# PER-SUBTYPE METADATA
# ═══════════════════════════════════════════════════════════════

CVD_METADATA: Dict[str, CVDMetadata] = {
    "Healthy": CVDMetadata(
        subtype="Healthy",
        description="Normal cardiovascular function with intact endothelium",
        risk_factors=[],
        complications=[],
        mortality_rate="Baseline",
        prevalence="~60% of adults under 40",
    ),
    "Dyslipidemia": CVDMetadata(
        subtype="Dyslipidemia",
        description="Elevated LDL, reduced HDL. Early atherogenic substrate.",
        risk_factors=["Diet", "Obesity", "Familial hypercholesterolemia", "Sedentary"],
        complications=["Subclinical atherosclerosis", "Fatty streaks"],
        mortality_rate="2x CVD risk per 1mmol/L LDL increase",
        prevalence="~38% of US adults (high LDL)",
    ),
    "Atherosclerosis": CVDMetadata(
        subtype="Atherosclerosis",
        description="Established plaque with inflammatory remodeling. Moderate basin depth.",
        risk_factors=["Smoking", "Hypertension", "Diabetes", "High LDL", "Family history"],
        complications=["Stable angina", "ACS risk", "PAD", "Carotid stenosis"],
        mortality_rate="3.7x all-cause mortality (multi-vessel CAD)",
        prevalence="~50% of adults over 45 have subclinical atherosclerosis",
    ),
    "ACS": CVDMetadata(
        subtype="ACS",
        description="Acute coronary syndrome — plaque rupture + thrombosis. Deep acute attractor.",
        risk_factors=["Unstable plaque", "High platelet activity", "Inflammation"],
        complications=["MI", "Sudden cardiac death", "Arrhythmia", "Cardiogenic shock"],
        mortality_rate="6-12% in-hospital mortality (STEMI); 30-day 5-8%",
        prevalence="~805,000 MIs per year in US",
    ),
    "HeartFailure": CVDMetadata(
        subtype="HeartFailure",
        description="HFrEF — reduced ejection fraction with neurohormonal activation. Deepest basin.",
        risk_factors=["Prior MI", "Hypertension", "Valvular disease", "Cardiomyopathy"],
        complications=["Pulmonary edema", "Arrhythmia", "Renal failure", "Cachexia"],
        mortality_rate="50% 5-year mortality (NYHA III-IV); #1 cause of hospitalization >65",
        prevalence="~6.7 million in US; rising globally",
    ),
    "Hypertension": CVDMetadata(
        subtype="Hypertension",
        description="Chronic RAAS overactivation with endothelial dysfunction. Moderate-deep basin.",
        risk_factors=["Salt intake", "Obesity", "Stress", "Age", "Genetics"],
        complications=["LVH", "CKD", "Stroke", "Retinopathy", "Accelerated atherosclerosis"],
        mortality_rate="2x stroke risk; 1.5x MI risk per 20/10 mmHg increase",
        prevalence="~47% of US adults (116 million)",
    ),
}


class CardiovascularODESystem:
    """
    10-dimensional cardiovascular hemodynamic/vascular generator system.

    Applies the same SAEM geometric framework used for cancer and diabetes:
    - Generator matrices define attractor basins
    - Disease = trapped in pathological attractor
    - Cure = flatten basin → add noise → push toward healthy attractor
    """

    N = 10  # Same dimensionality as cancer/diabetes frameworks

    @classmethod
    def healthy_generator(cls, params: Optional[CVDParams] = None) -> np.ndarray:
        """
        Healthy cardiovascular homeostasis generator.

        Balanced lipids, intact endothelium, quiescent RAAS,
        normal platelet function, preserved ejection fraction.

        Ref: ACC/AHA 2022 Guidelines; Libby et al. 2019 NRDP
        """
        p = params or CVDParams()
        n = cls.N
        A = np.zeros((n, n))

        # ── Diagonal: turnover/clearance rates ──
        A[0, 0] = p.ldl_clearance       # LDL-ox clearance (liver + macrophage)
        A[1, 1] = p.hdl_turnover        # HDL catabolism
        A[2, 2] = p.endo_recovery       # Endothelial repair
        A[3, 3] = p.vsmc_clearance      # VSMC quiescence
        A[4, 4] = p.platelet_clearance  # Platelet consumption
        A[5, 5] = p.crp_clearance       # CRP clearance
        A[6, 6] = p.raas_clearance      # RAAS feedback
        A[7, 7] = p.bnp_clearance       # BNP clearance
        A[8, 8] = p.ef_homeostasis      # EF homeostasis
        A[9, 9] = p.fibrosis_turnover   # Fibrosis resolution

        # ── Key couplings: lipid-endothelial axis ──
        A[0, 2] = -0.25     # LDL-ox damages endothelium
        A[2, 1] = 0.20      # HDL protects endothelium (reverse transport)
        A[1, 0] = -0.15     # HDL counteracts LDL-ox accumulation

        # ── Inflammation axis ──
        A[0, 5] = 0.15      # LDL-ox drives CRP/inflammation
        A[5, 2] = -0.18     # Inflammation damages endothelium
        A[5, 3] = 0.12      # Inflammation promotes VSMC proliferation

        # ── Plaque dynamics ──
        A[0, 3] = 0.10      # LDL-ox feeds plaque growth (foam cells)
        A[3, 2] = -0.08     # VSMC proliferation narrows lumen

        # ── Thrombosis axis ──
        A[2, 4] = -0.10     # Endothelial damage activates platelets
        A[5, 4] = 0.08      # Inflammation activates platelets

        # ── RAAS-vascular coupling ──
        A[6, 2] = -0.12     # RAAS overactivation damages endothelium
        A[6, 9] = 0.10      # RAAS drives fibrosis (aldosterone)
        A[6, 7] = 0.08      # RAAS overload → BNP elevation

        # ── Cardiac axis ──
        A[9, 8] = -0.15     # Fibrosis reduces EF (stiff ventricle)
        A[7, 8] = -0.10     # BNP elevation signals reduced EF

        # ── Protective couplings ──
        A[2, 8] = 0.12      # Healthy endothelium supports cardiac output
        A[1, 5] = -0.10     # HDL is anti-inflammatory

        return A

    @classmethod
    def dyslipidemia_generator(cls, params: Optional[CVDParams] = None) -> np.ndarray:
        """
        Dyslipidemia: elevated LDL, reduced HDL. Shallow atherogenic basin.

        The basin is shallow — statin therapy can flatten it back to healthy.

        Ref: Baigent et al. 2010, Lancet (CTT meta-analysis — statins)
        """
        A = cls.healthy_generator(params)

        # Elevated LDL-ox
        A[0, 0] = -0.38     # Slower LDL clearance (fewer receptors)
        A[0, 2] = -0.32     # More LDL-mediated endothelial damage

        # Reduced HDL
        A[1, 1] = -0.38     # Faster HDL catabolism
        A[2, 1] = 0.12      # Weaker HDL protection
        A[1, 0] = -0.08     # Weaker reverse transport

        # Early inflammation
        A[5, 5] = -0.42     # Slightly slower CRP clearance
        A[0, 5] = 0.20      # More LDL-driven inflammation

        # Mild endothelial dysfunction
        A[2, 2] = -0.35     # Slightly impaired repair

        return A

    @classmethod
    def atherosclerosis_generator(cls, params: Optional[CVDParams] = None) -> np.ndarray:
        """
        Established atherosclerosis: inflammatory plaque with remodeling.

        Moderate-depth basin. Multi-drug therapy needed to flatten.

        Ref: Libby et al. 2019, NRDP; Ridker 2017 NEJM (CANTOS)
        """
        A = cls.healthy_generator(params)

        # Advanced lipid dysfunction
        A[0, 0] = -0.30     # Poor LDL clearance
        A[1, 1] = -0.42     # HDL dysfunction (pro-inflammatory HDL)
        A[0, 2] = -0.40     # Major endothelial damage from ox-LDL

        # Chronic inflammation (IL-1β / IL-6 / CRP loop)
        A[5, 5] = -0.30     # CRP accumulating
        A[0, 5] = 0.28      # ox-LDL → macrophage → cytokines → CRP
        A[5, 3] = 0.22      # Inflammation → VSMC proliferation

        # Active plaque growth
        A[3, 3] = -0.22     # VSMC proliferating (fibrous cap thickening)
        A[0, 3] = 0.20      # Foam cell accumulation in intima

        # Endothelial dysfunction
        A[2, 2] = -0.28     # Impaired NO production
        A[6, 2] = -0.20     # RAAS worsening endothelium

        # RAAS activation
        A[6, 6] = -0.30     # RAAS overactive
        A[6, 9] = 0.18      # RAAS driving vascular fibrosis

        # Platelet priming
        A[4, 4] = -0.38     # Platelets more reactive
        A[2, 4] = -0.18     # Dysfunctional endothelium = more activation

        # Early fibrosis
        A[9, 9] = -0.12     # Fibrosis accumulating

        return A

    @classmethod
    def acs_generator(cls, params: Optional[CVDParams] = None) -> np.ndarray:
        """
        Acute Coronary Syndrome: plaque rupture + thrombosis.

        ACUTE deep attractor — analogous to DKA in diabetes.
        Immediate intervention required to prevent death.

        Ref: Amsterdam et al. 2014, JACC (ACS guidelines);
             Antman et al. 2004, NEJM (TIMI risk score)
        """
        A = cls.healthy_generator(params)

        # Plaque rupture → massive platelet activation
        A[4, 4] = -0.20     # Massive platelet aggregation (thrombus forming)
        A[2, 4] = -0.35     # Exposed subendothelial collagen → platelet
        A[5, 4] = 0.25      # Inflammatory milieu amplifies thrombosis

        # Acute inflammation surge
        A[5, 5] = -0.22     # CRP spike (troponin correlates)
        A[0, 5] = 0.35      # ox-LDL from ruptured plaque → inflammation storm

        # Severe endothelial disruption
        A[2, 2] = -0.18     # Endothelium breached
        A[0, 2] = -0.45     # Plaque contents toxic to vessel wall

        # Cardiac damage
        A[8, 8] = -0.35     # EF dropping (ischemic myocardium)
        A[7, 7] = -0.25     # BNP surging (myocardial stress)
        A[9, 8] = -0.25     # Acute fibrosis from infarction

        # RAAS reflex activation
        A[6, 6] = -0.25     # RAAS overactivation (compensatory)
        A[6, 7] = 0.20      # RAAS drive → BNP

        # Fibrosis acceleration (scar formation)
        A[9, 9] = -0.10     # Rapid scar tissue deposition
        A[6, 9] = 0.25      # RAAS → aldosterone → fibrosis

        return A

    @classmethod
    def heart_failure_generator(cls, params: Optional[CVDParams] = None) -> np.ndarray:
        """
        Heart Failure (HFrEF): neurohormonal overactivation + ventricular remodeling.

        DEEPEST cardiovascular attractor — analogous to PDAC in cancer.
        Chronic, progressive, high mortality without intervention.

        Ref: McMurray et al. 2014, NEJM (PARADIGM-HF);
             Zannad et al. 2011, NEJM (EMPHASIS-HF);
             DAPA-HF 2019 NEJM
        """
        A = cls.healthy_generator(params)

        # Severely reduced cardiac output
        A[8, 8] = -0.45     # EF critically low (<35%)
        A[7, 7] = -0.20     # BNP chronically elevated (stretched myocardium)

        # RAAS overactivation (neurohormonal spiral)
        A[6, 6] = -0.22     # Chronic RAAS — aldosterone escape
        A[6, 2] = -0.25     # RAAS destroying endothelium
        A[6, 9] = 0.30      # RAAS → massive fibrosis
        A[6, 7] = 0.22      # RAAS loop → BNP

        # Severe fibrosis (ventricular remodeling)
        A[9, 9] = -0.08     # Fibrosis near-irreversible
        A[9, 8] = -0.30     # Fibrosis → diastolic dysfunction → lower EF

        # Endothelial destruction
        A[2, 2] = -0.20     # Near-total NO depletion
        A[2, 8] = 0.05      # Minimal endothelial contribution to output

        # Chronic inflammation
        A[5, 5] = -0.28     # Chronic low-grade inflammation
        A[5, 3] = 0.15      # Ongoing vascular remodeling

        # Lipid axis secondary
        A[0, 0] = -0.40     # Lipid handling impaired (hepatic congestion)

        # Platelet dysfunction
        A[4, 4] = -0.35     # Hypercoagulable state (stasis)

        return A

    @classmethod
    def hypertension_generator(cls, params: Optional[CVDParams] = None) -> np.ndarray:
        """
        Chronic hypertension: RAAS overactivation with vascular remodeling.

        Moderate-deep basin. Primary driver of stroke, CKD, HF.

        Ref: SPRINT 2015, NEJM (intensive BP lowering);
             Yusuf et al. 2000, NEJM (HOPE — ramipril)
        """
        A = cls.healthy_generator(params)

        # RAAS overactivation (primary driver)
        A[6, 6] = -0.25     # Chronically overactive RAAS
        A[6, 2] = -0.22     # RAAS damages endothelium (angiotensin II)
        A[6, 9] = 0.22      # RAAS → aldosterone → cardiac/vascular fibrosis
        A[6, 7] = 0.15      # RAAS → ventricular hypertrophy → BNP

        # Endothelial dysfunction
        A[2, 2] = -0.28     # Shear stress → reduced NO
        A[2, 4] = -0.15     # Dysfunctional endothelium primes platelets

        # Vascular remodeling
        A[3, 3] = -0.25     # VSMC hypertrophy (arteriolar thickening)
        A[9, 9] = -0.10     # Progressive fibrosis (arterial stiffness)

        # Cardiac adaptation
        A[8, 8] = -0.25     # EF initially preserved, then declining (LVH → HF)
        A[7, 7] = -0.35     # BNP mildly elevated (wall stress)

        # Mild inflammation
        A[5, 5] = -0.40     # Modest CRP elevation
        A[0, 5] = 0.18      # Accelerated atherogenesis

        return A

    @classmethod
    def all_generators(cls) -> Dict[str, np.ndarray]:
        """Return all cardiovascular generator matrices."""
        return {
            "Healthy": cls.healthy_generator(),
            "Dyslipidemia": cls.dyslipidemia_generator(),
            "Atherosclerosis": cls.atherosclerosis_generator(),
            "ACS": cls.acs_generator(),
            "HeartFailure": cls.heart_failure_generator(),
            "Hypertension": cls.hypertension_generator(),
        }


# ═══════════════════════════════════════════════════════════════
# CARDIOVASCULAR THERAPEUTIC INTERVENTIONS
# ═══════════════════════════════════════════════════════════════

@dataclass
class CVDIntervention:
    """A cardiovascular therapeutic intervention mapped to generator correction."""
    name: str
    mechanism: str
    expected_effect: np.ndarray   # 10×10 δA correction
    category: str                 # statin, antiplatelet, ACEi, ARB, betablocker, etc.
    evidence_level: str
    references: List[str]
    curvature_multiplier: float = 1.0
    fatality_reduction: str = ""


def build_cvd_drug_library(n: int = 10) -> List[CVDIntervention]:
    """Build library of cardiovascular interventions mapped to generator corrections."""
    interventions = []

    # ────────────────────────────────────────────────────────
    # 1. HIGH-INTENSITY STATIN (Atorvastatin 80mg / Rosuvastatin 40mg)
    # Ref: CTT 2010 Lancet (22% RRR per mmol/L LDL reduction)
    #      PROVE-IT 2004 NEJM (atorvastatin 80 vs pravastatin 40)
    # ────────────────────────────────────────────────────────
    statin = np.zeros((n, n))
    statin[0, 0] = -0.25    # Major LDL-ox reduction (upregulate LDL-R)
    statin[1, 1] = 0.08     # Modest HDL increase (~5-10%)
    statin[5, 5] = -0.12    # Anti-inflammatory (pleiotropic — CRP reduction)
    statin[2, 2] = -0.08    # Endothelial improvement (NO restoration)
    statin[3, 3] = -0.05    # Mild plaque stabilization
    interventions.append(CVDIntervention(
        name="High-Intensity Statin (Atorvastatin 80mg)",
        mechanism="HMG-CoA reductase inhibition → LDL-R upregulation + pleiotropic anti-inflammatory",
        expected_effect=statin,
        category="statin",
        evidence_level="Gold Standard",
        references=[
            "CTT meta-analysis 2010 Lancet (22% RRR per mmol/L LDL reduction)",
            "PROVE-IT 2004 NEJM (intensive vs moderate statin)",
            "JUPITER 2008 NEJM (rosuvastatin in elevated CRP)",
        ],
        curvature_multiplier=0.70,
        fatality_reduction="22% reduction in major vascular events per mmol/L LDL decrease; NNT=28 over 5 years",
    ))

    # ────────────────────────────────────────────────────────
    # 2. ACE INHIBITOR (Ramipril)
    # Ref: HOPE 2000 NEJM (22% CVD death reduction)
    # ────────────────────────────────────────────────────────
    acei = np.zeros((n, n))
    acei[6, 6] = -0.20      # RAAS suppression (primary mechanism)
    acei[6, 2] = 0.12       # Reduced RAAS → endothelial recovery
    acei[6, 9] = -0.15      # Reduced aldosterone → less fibrosis
    acei[2, 2] = -0.08      # Bradykinin-mediated NO increase
    acei[7, 7] = -0.06      # Reduced ventricular remodeling → lower BNP
    interventions.append(CVDIntervention(
        name="ACE Inhibitor (Ramipril)",
        mechanism="ACE inhibition → reduced angiotensin II + increased bradykinin → vasodilation + reduced fibrosis",
        expected_effect=acei,
        category="ACEi",
        evidence_level="Gold Standard",
        references=[
            "HOPE 2000 NEJM (22% reduction in CVD death/MI/stroke)",
            "EUROPA 2003 Lancet (perindopril — 20% RRR)",
            "SOLVD 1991 NEJM (enalapril in HF)",
        ],
        curvature_multiplier=0.72,
        fatality_reduction="22% reduction in CVD death + MI + stroke (HOPE); cornerstone of HF therapy",
    ))

    # ────────────────────────────────────────────────────────
    # 3. DUAL ANTIPLATELET (Aspirin + Clopidogrel/Ticagrelor)
    # Ref: PLATO 2009 NEJM (ticagrelor — 16% reduction in CV death)
    # ────────────────────────────────────────────────────────
    dapt = np.zeros((n, n))
    dapt[4, 4] = -0.30      # Major platelet inhibition (dual pathway)
    dapt[2, 4] = 0.10       # Reduced platelet-mediated endothelial damage
    dapt[5, 5] = -0.05      # Mild anti-inflammatory (aspirin COX-1)
    interventions.append(CVDIntervention(
        name="DAPT (Aspirin + Ticagrelor)",
        mechanism="COX-1 + P2Y12 inhibition → dual platelet aggregation block → reduced thrombosis",
        expected_effect=dapt,
        category="antiplatelet",
        evidence_level="Clinical Standard",
        references=[
            "PLATO 2009 NEJM (ticagrelor: 16% CV death reduction vs clopidogrel)",
            "CURE 2001 NEJM (clopidogrel in ACS)",
            "ISIS-2 1988 Lancet (aspirin in MI — 23% mortality reduction)",
        ],
        curvature_multiplier=0.65,
        fatality_reduction="16% CV death reduction (PLATO); 23% mortality reduction with aspirin alone (ISIS-2)",
    ))

    # ────────────────────────────────────────────────────────
    # 4. SACUBITRIL/VALSARTAN (ARNI — Heart Failure standard)
    # Ref: PARADIGM-HF 2014 NEJM (20% CV death/HF hospitalization reduction)
    # ────────────────────────────────────────────────────────
    arni = np.zeros((n, n))
    arni[6, 6] = -0.22      # ARB component → RAAS blockade
    arni[7, 7] = -0.20      # Neprilysin inhibition → BNP preserved (beneficial)
    arni[6, 9] = -0.18      # Reduced RAAS-driven fibrosis
    arni[8, 8] = 0.10       # EF improvement (reverse remodeling)
    arni[2, 2] = -0.08      # Natriuretic peptide → vasodilation
    interventions.append(CVDIntervention(
        name="Sacubitril/Valsartan (ARNI)",
        mechanism="Neprilysin inhibition + AT1 blockade → enhanced natriuretic peptides + RAAS suppression",
        expected_effect=arni,
        category="ARNI",
        evidence_level="Clinical Standard",
        references=[
            "PARADIGM-HF 2014 NEJM (20% reduction in CV death + HF hospitalization)",
            "PARAGON-HF 2019 NEJM (HFpEF — marginal benefit)",
        ],
        curvature_multiplier=0.60,
        fatality_reduction="20% reduction in CV death + HF hospitalization (PARADIGM-HF); now first-line for HFrEF",
    ))

    # ────────────────────────────────────────────────────────
    # 5. SGLT2 INHIBITOR (Empagliflozin/Dapagliflozin — dual CVD+diabetes benefit)
    # Ref: DAPA-HF 2019 NEJM (26% CV death/HF hospitalization reduction)
    #      EMPEROR-Reduced 2020 NEJM
    # ────────────────────────────────────────────────────────
    sglt2 = np.zeros((n, n))
    sglt2[7, 7] = -0.15     # BNP reduction (hemodynamic unloading)
    sglt2[8, 8] = 0.08      # EF improvement (volume reduction)
    sglt2[6, 6] = -0.08     # Mild RAAS modulation
    sglt2[9, 9] = -0.08     # Anti-fibrotic (emerging data)
    sglt2[5, 5] = -0.06     # Anti-inflammatory (ketone metabolism)
    interventions.append(CVDIntervention(
        name="SGLT2i (Dapagliflozin)",
        mechanism="Renal glucose/sodium excretion → hemodynamic unloading + ketone metabolism + anti-fibrotic",
        expected_effect=sglt2,
        category="SGLT2i",
        evidence_level="Clinical Standard",
        references=[
            "DAPA-HF 2019 NEJM (26% CV death/HF hospitalization — regardless of diabetes!)",
            "EMPEROR-Reduced 2020 NEJM (empagliflozin in HF)",
            "EMPA-REG 2015 NEJM (38% CV death in T2D)",
        ],
        curvature_multiplier=0.65,
        fatality_reduction="26% CV death + HF hospitalization reduction (DAPA-HF); works even without diabetes",
    ))

    # ────────────────────────────────────────────────────────
    # 6. BETA-BLOCKER (Carvedilol/Bisoprolol/Metoprolol succinate)
    # Ref: MERIT-HF 1999 Lancet (34% all-cause mortality reduction in HF)
    # ────────────────────────────────────────────────────────
    bb = np.zeros((n, n))
    bb[8, 8] = 0.10         # EF improvement (reverse remodeling over months)
    bb[6, 6] = -0.10        # Reduced RAAS activation (sympatholytic)
    bb[4, 4] = -0.05        # Mild antiplatelet (reduced shear stress)
    bb[7, 7] = -0.08        # BNP reduction (reduced wall stress)
    interventions.append(CVDIntervention(
        name="Beta-Blocker (Carvedilol)",
        mechanism="β1/β2/α1 blockade → reduced heart rate + SVR + neurohormonal suppression",
        expected_effect=bb,
        category="betablocker",
        evidence_level="Gold Standard",
        references=[
            "MERIT-HF 1999 Lancet (34% all-cause mortality reduction)",
            "COPERNICUS 2001 NEJM (carvedilol in severe HF — 35% mortality reduction)",
            "CIBIS-II 1999 Lancet (bisoprolol — 34% mortality)",
        ],
        curvature_multiplier=0.68,
        fatality_reduction="34% all-cause mortality reduction in HFrEF (MERIT-HF, COPERNICUS, CIBIS-II)",
    ))

    # ────────────────────────────────────────────────────────
    # 7. PCSK9 INHIBITOR (Evolocumab/Alirocumab)
    # Ref: FOURIER 2017 NEJM (15% MACE reduction)
    # ────────────────────────────────────────────────────────
    pcsk9 = np.zeros((n, n))
    pcsk9[0, 0] = -0.35     # Dramatic LDL reduction (60% on top of statin)
    pcsk9[1, 1] = 0.05      # Modest HDL increase
    pcsk9[5, 5] = -0.08     # CRP reduction (less atherogenic burden)
    pcsk9[0, 2] = 0.10      # Reduced ox-LDL → endothelial relief
    interventions.append(CVDIntervention(
        name="PCSK9 Inhibitor (Evolocumab)",
        mechanism="PCSK9 monoclonal antibody → massive LDL-R upregulation → LDL to <20 mg/dL",
        expected_effect=pcsk9,
        category="PCSK9i",
        evidence_level="Clinical Standard",
        references=[
            "FOURIER 2017 NEJM (15% MACE reduction with evolocumab)",
            "ODYSSEY 2018 NEJM (alirocumab — 15% all-cause mortality reduction in ACS)",
        ],
        curvature_multiplier=0.62,
        fatality_reduction="15% MACE reduction (FOURIER); 15% all-cause mortality in ACS (ODYSSEY post-hoc)",
    ))

    # ────────────────────────────────────────────────────────
    # 8. LIFESTYLE (Exercise + Mediterranean diet + smoking cessation)
    # Ref: PREDIMED 2018 NEJM-retracted/rerandomized (30% CVD reduction)
    #      Ekelund et al. 2015, Lancet (PA and mortality)
    # ────────────────────────────────────────────────────────
    life = np.zeros((n, n))
    life[0, 0] = -0.12      # LDL reduction (diet)
    life[1, 1] = 0.10       # HDL increase (exercise)
    life[2, 2] = -0.15      # Endothelial improvement (exercise → shear stress → NO)
    life[5, 5] = -0.15      # Anti-inflammatory (weight loss + exercise)
    life[6, 6] = -0.10      # RAAS normalization (weight loss)
    life[8, 8] = 0.08       # EF improvement (cardiac conditioning)
    life[9, 9] = -0.05      # Mild anti-fibrotic
    interventions.append(CVDIntervention(
        name="Lifestyle (Exercise + Mediterranean Diet + Smoking Cessation)",
        mechanism="Multi-pathway: lipid improvement + endothelial restoration + anti-inflammatory + RAAS normalization",
        expected_effect=life,
        category="lifestyle",
        evidence_level="Gold Standard",
        references=[
            "PREDIMED 2018 NEJM (30% CVD reduction with Mediterranean diet)",
            "Ekelund et al. 2015, Lancet (20% mortality reduction per 150 min/week exercise)",
            "Critchley & Capewell 2003 BMJ (36% mortality reduction with smoking cessation)",
        ],
        curvature_multiplier=0.55,
        fatality_reduction="30% CVD reduction (diet); 36% mortality reduction (smoking cessation); strongest combined NNT",
    ))

    return interventions
