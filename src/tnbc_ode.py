"""
TNBC ODE System Module
=======================

Deterministic, literature-grounded 10-metabolite ODE system for
Triple-Negative Breast Cancer metabolic modeling.

Replaces the ad hoc random-noise generators with reproducible,
parametrized matrices derived from published TNBC metabolomics.

Key References:
  - Warburg effect parameters: Vander Heiden et al. 2009, Science
  - Glutaminolysis flux: DeBerardinis et al. 2007, PNAS
  - ROS dynamics: Gorrini et al. 2013, Nat Rev Drug Discov
  - TNBC-specific: Lanning et al. 2017, Cell Reports
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field


METABOLITES = [
    "Glucose", "Lactate", "Pyruvate", "ATP", "NADH",
    "Glutamine", "Glutamate", "aKG", "Citrate", "ROS",
]


@dataclass
class ODEParams:
    """Tunable ODE parameters for sensitivity analysis."""
    # Glycolysis
    glucose_uptake: float = -0.50     # Healthy glucose turnover
    glycolysis_flux: float = 0.40     # Glucose → Pyruvate
    lactate_clearance: float = -0.80  # Lactate washout
    pyruvate_to_lactate: float = 0.10 # Low in healthy tissue
    pyruvate_to_atp: float = 0.30     # TCA/OXPHOS coupling

    # Mitochondrial
    nadh_to_atp: float = 0.20        # Electron transport
    atp_turnover: float = -0.20      # ATP consumption
    nadh_cycling: float = -0.30      # NAD+/NADH balance

    # Glutamine axis
    glutamine_utilization: float = -0.40
    glutaminolysis: float = 0.30     # Gln → Glu
    glutamate_to_akg: float = 0.30   # Glu → α-KG
    akg_to_citrate: float = 0.20     # TCA cycle

    # Redox
    ros_clearance: float = -0.90     # Antioxidant capacity
    nadh_ros_leak: float = 0.10      # ETC side-reaction
    ros_atp_damage: float = -0.10    # ROS inhibits ATP

    # Feedback
    atp_inhibits_glucose: float = -0.10

    # Citrate
    citrate_turnover: float = -0.30


@dataclass
class GeneratorMetadata:
    """Metadata for a cancer generator: traceability, confidence, and tags."""
    cancer_type: str
    evidence_notes: str        # Short citations / justification text
    confidence: str            # 'high', 'medium', 'low'
    tags: List[str] = field(default_factory=list)  # e.g. ['glycolytic', 'immune-cold']
    immune_suppression: float = 0.3  # 0-1 scale, higher = more immunosuppressive
    stromal_coupling: float = 0.2    # 0-1 scale, higher = more desmoplastic stroma


# ═══════════════════════════════════════════════════════════════════════
# PER-CANCER METADATA — evidence, confidence, tags for every generator
# ═══════════════════════════════════════════════════════════════════════
GENERATOR_METADATA: Dict[str, GeneratorMetadata] = {
    'TNBC': GeneratorMetadata(
        cancer_type='TNBC',
        evidence_notes='Warburg glycolysis (Vander Heiden 2009), glutamine addiction (Lanning 2017 Cell Reports)',
        confidence='high',
        tags=['glycolytic', 'glutamine-addicted', 'immune-excluded', 'redox-stressed'],
        immune_suppression=0.55,
        stromal_coupling=0.30,
    ),
    'PDAC': GeneratorMetadata(
        cancer_type='PDAC',
        evidence_notes='Halbrook & Lyssiotis 2017 Cell Metab; macropinocytosis; dense desmoplastic stroma',
        confidence='high',
        tags=['glycolytic', 'glutamine-rewired', 'desmoplastic', 'immune-cold'],
        immune_suppression=0.80,
        stromal_coupling=0.85,
    ),
    'NSCLC': GeneratorMetadata(
        cancer_type='NSCLC',
        evidence_notes='Hensley et al. 2016 Cell; metabolically flexible, MCT1 lactate recycling',
        confidence='high',
        tags=['flexible', 'oxphos-active', 'kras-driven'],
        immune_suppression=0.35,
        stromal_coupling=0.25,
    ),
    'Melanoma': GeneratorMetadata(
        cancer_type='Melanoma',
        evidence_notes='Fischer et al. 2018 Mol Cell; PGC1a+ OXPHOS; BRAF-adaptive; MITF/ROS signaling',
        confidence='high',
        tags=['oxphos', 'braf', 'ros-adaptive', 'immune-responsive'],
        immune_suppression=0.20,
        stromal_coupling=0.15,
    ),
    'GBM': GeneratorMetadata(
        cancer_type='GBM',
        evidence_notes='Marin-Valencia 2012 Cell Metab; SREBP-1 lipogenesis; Vlashi 2011 JNCI GSC OXPHOS',
        confidence='high',
        tags=['glycolytic', 'lipid-dependent', 'bbb-protected', 'immune-cold'],
        immune_suppression=0.70,
        stromal_coupling=0.40,
    ),
    'CRC': GeneratorMetadata(
        cancer_type='CRC',
        evidence_notes='Pate et al. 2014 PNAS; PKM2 glycolysis; butyrate-HDAC vulnerability; MYC/Wnt',
        confidence='high',
        tags=['wnt-driven', 'glycolytic', 'butyrate-sensitive', 'microbiome-coupled'],
        immune_suppression=0.40,
        stromal_coupling=0.30,
    ),
    'HGSOC': GeneratorMetadata(
        cancer_type='HGSOC',
        evidence_notes='Nieman et al. 2011 Nat Med (omental lipid transfer); Konstantinopoulos 2015 (BRCA/HRD)',
        confidence='medium',
        tags=['lipid-dependent', 'folate-1c', 'brca-hrdef', 'immune-variable'],
        immune_suppression=0.50,
        stromal_coupling=0.55,
    ),
    'mCRPC': GeneratorMetadata(
        cancer_type='mCRPC',
        evidence_notes='Zadra et al. 2019 Nat Rev Cancer (lipogenesis); Bader & McGuire 2020 (AR→OXPHOS)',
        confidence='medium',
        tags=['lipogenic', 'ar-driven', 'oxphos-dependent', 'immune-cold'],
        immune_suppression=0.65,
        stromal_coupling=0.35,
    ),
    'AML': GeneratorMetadata(
        cancer_type='AML',
        evidence_notes='Ward et al. 2010 Cancer Cell (IDH/2-HG); Lagadinou et al. 2013 (LSC OXPHOS/BCL-2)',
        confidence='medium',
        tags=['idh-mutant', '2hg', 'bh3-dependent', 'oxphos', 'immune-variable'],
        immune_suppression=0.45,
        stromal_coupling=0.20,
    ),
    'HCC': GeneratorMetadata(
        cancer_type='HCC',
        evidence_notes='Ally et al. 2017 Cell (TCGA-LIHC); urea cycle defects; Wnt/β-catenin; CTNNB1 mutations',
        confidence='medium',
        tags=['lipogenic', 'urea-cycle-defect', 'wnt-driven', 'immune-excluded'],
        immune_suppression=0.60,
        stromal_coupling=0.50,
    ),
}


def validate_generator(name: str, A: np.ndarray, all_generators: Optional[dict] = None) -> List[str]:
    """
    Validate a cancer generator matrix. Returns list of issues (empty = valid).

    Checks:
      1. Shape is 10×10
      2. All entries are numerically bounded (|a_ij| < 5.0)
      3. Not degenerate (not identical to another generator within tolerance)
    """
    issues: List[str] = []

    # 1. Shape
    if A.shape != (10, 10):
        issues.append(f"{name}: expected shape (10, 10), got {A.shape}")
        return issues  # Can't do further checks

    # 2. Bounded
    max_abs = float(np.max(np.abs(A)))
    if max_abs > 5.0:
        issues.append(f"{name}: max |entry| = {max_abs:.3f} exceeds bound 5.0")

    # 3. Non-degenerate
    if all_generators is not None:
        for other_name, A_other in all_generators.items():
            if other_name == name:
                continue
            diff = float(np.linalg.norm(A - A_other))
            if diff < 0.01:
                issues.append(f"{name}: nearly identical to {other_name} (diff={diff:.6f})")

    return issues


def validate_all_generators() -> Dict[str, List[str]]:
    """Run validation on all 10 pan-cancer generators. Returns {name: [issues]}."""
    generators = TNBCODESystem.pan_cancer_generators()
    results = {}
    for name, A in generators.items():
        results[name] = validate_generator(name, A, generators)
    return results


class TNBCODESystem:
    """
    Deterministic TNBC metabolic generator system.

    Unlike TNBCMetabolicModel, this produces identical matrices every time
    and supports continuous alpha-interpolation between healthy and cancer.
    """

    N = 10  # Number of metabolites

    @classmethod
    def healthy_generator(cls, params: Optional[ODEParams] = None) -> np.ndarray:
        """
        Build a deterministic healthy-tissue metabolic generator.

        All rates from ODEParams; fully reproducible.
        """
        p = params or ODEParams()
        n = cls.N
        A = np.zeros((n, n))

        # Diagonal: decay/turnover rates
        A[0, 0] = p.glucose_uptake
        A[1, 1] = p.lactate_clearance
        A[2, 2] = -0.30  # Pyruvate processing
        A[3, 3] = p.atp_turnover
        A[4, 4] = p.nadh_cycling
        A[5, 5] = p.glutamine_utilization
        A[6, 6] = -0.50  # Glutamate processing
        A[7, 7] = -0.40  # α-KG in TCA
        A[8, 8] = p.citrate_turnover
        A[9, 9] = p.ros_clearance

        # Metabolic fluxes
        A[0, 2] = p.glycolysis_flux        # Glucose → Pyruvate
        A[2, 3] = p.pyruvate_to_atp        # Pyruvate → ATP
        A[2, 1] = p.pyruvate_to_lactate    # Pyruvate → Lactate
        A[4, 3] = p.nadh_to_atp            # NADH → ATP
        A[5, 6] = p.glutaminolysis         # Glutamine → Glutamate
        A[6, 7] = p.glutamate_to_akg       # Glutamate → α-KG
        A[7, 8] = p.akg_to_citrate         # α-KG → Citrate

        # Regulatory feedbacks
        A[3, 0] = p.atp_inhibits_glucose   # ATP inhibits glucose uptake
        A[9, 4] = p.nadh_ros_leak          # NADH generates some ROS
        A[9, 3] = p.ros_atp_damage         # ROS damages ATP production

        return A

    @classmethod
    def tnbc_generator(cls, params: Optional[ODEParams] = None) -> np.ndarray:
        """
        Build a deterministic TNBC cancer metabolic generator.

        Applies published TNBC metabolic shifts to healthy baseline:
        - Enhanced glycolysis (Warburg effect)
        - Increased lactate production
        - Glutamine addiction
        - Compromised ROS clearance
        - Citrate diversion to lipogenesis
        """
        A = cls.healthy_generator(params)

        # 1. Warburg effect: enhanced glycolysis
        A[0, 0] = -0.30       # Increased glucose uptake
        A[0, 2] = 0.60        # Enhanced glycolysis flux
        A[2, 1] = 0.40        # Much more pyruvate → lactate
        A[2, 3] = 0.15        # Less pyruvate to OXPHOS
        A[1, 1] = -0.30       # Lactate accumulates

        # 2. Glutamine addiction
        A[5, 5] = -0.20       # Increased dependence
        A[5, 6] = 0.50        # Enhanced glutaminolysis
        A[6, 7] = 0.50        # More glutamate → α-KG

        # 3. Compromised mitochondria
        A[4, 3] = 0.10        # Less efficient ETC
        A[3, 3] = -0.15       # Slower ATP turnover

        # 4. Redox imbalance
        A[9, 9] = -0.40       # Reduced ROS clearance
        A[9, 4] = 0.20        # More ROS from NADH

        # 5. Lipogenesis
        A[8, 8] = -0.15       # Citrate diverted

        return A

    @classmethod
    def interpolated_generator(
        cls,
        alpha: float,
        params: Optional[ODEParams] = None,
    ) -> np.ndarray:
        """
        Continuous interpolation: alpha=0 → healthy, alpha=1 → TNBC.

        Enables bifurcation analysis: find the critical alpha where
        the attractor geometry shifts from escapable to trapping.
        """
        A_h = cls.healthy_generator(params)
        A_c = cls.tnbc_generator(params)
        return (1.0 - alpha) * A_h + alpha * A_c

    @classmethod
    def bifurcation_scan(
        cls,
        n_points: int = 50,
        params: Optional[ODEParams] = None,
    ) -> dict:
        """
        Scan alpha from 0→1 and return eigenvalue trajectories.

        Returns:
            dict with 'alpha', 'max_real_eigenvalue', 'curvature' arrays
        """
        from geometric_optimization import GeometricOptimizer
        optimizer = GeometricOptimizer(cls.N)

        alphas = np.linspace(0.0, 1.0, n_points)
        max_reals = []
        curvatures = []

        for a in alphas:
            A = cls.interpolated_generator(a, params)
            evals = np.linalg.eigvals(A)
            max_reals.append(float(np.max(evals.real)))
            curvatures.append(float(optimizer.compute_basin_curvature(A)))

        return {
            "alpha": alphas.tolist(),
            "max_real_eigenvalue": max_reals,
            "curvature": curvatures,
        }

    # ══════════════════════════════════════════════════════════════
    # PAN-CANCER GENERATOR TEMPLATES
    # Each models the metabolic signature of a distinct cancer type
    # on the same 10D metabolite axis for direct comparison.
    # ══════════════════════════════════════════════════════════════

    @classmethod
    def pdac_generator(cls, params: Optional[ODEParams] = None) -> np.ndarray:
        """
        Pancreatic Ductal Adenocarcinoma (PDAC) generator.
        
        Key metabolic features (Ref: Halbrook & Lyssiotis 2017, Cell Metab):
        - Extreme glucose dependence + branching to hexosamine/PPP
        - Non-canonical glutamine usage (transamination)
        - Macropinocytosis-driven amino acid scavenging
        - Dense desmoplastic stroma limits drug delivery -> resistance
        - Very deep attractor (hardest to escape)
        """
        A = cls.healthy_generator(params)
        
        # Deep glycolytic dependence
        A[0, 0] = -0.25
        A[0, 2] = 0.70        # Even more glycolysis than TNBC
        A[2, 1] = 0.50        # Heavy lactate production
        A[2, 3] = 0.10        # Very little OXPHOS
        A[1, 1] = -0.25       # Lactate accumulates heavily
        
        # Non-canonical glutamine (transamination pathway)
        A[5, 5] = -0.15
        A[5, 6] = 0.60
        A[6, 7] = 0.55
        A[7, 8] = 0.25        # aKG heavily feeds TCA
        
        # Mitochondrial dysfunction (severe)
        A[4, 3] = 0.05
        A[3, 3] = -0.10       # Very low ATP turnover
        
        # ROS balance (moderate, stroma is protective)
        A[9, 9] = -0.50       # Strong ROS clearance (GPX4 upregulated)
        A[9, 4] = 0.15
        
        return A
    
    @classmethod
    def nsclc_generator(cls, params: Optional[ODEParams] = None) -> np.ndarray:
        """
        Non-Small Cell Lung Cancer (NSCLC) generator.
        
        Key metabolic features (Ref: Hensley et al. 2016, Cell):
        - In vivo: uses BOTH glycolysis and OXPHOS (metabolically flexible)
        - Glucose and lactate can both fuel TCA
        - High glutamine usage in KRAS-mutant subtype
        - Moderate attractor depth (more amenable to therapy)
        """
        A = cls.healthy_generator(params)
        
        # Metabolic flexibility: glycolysis AND OXPHOS active
        A[0, 0] = -0.30
        A[0, 2] = 0.45        # Moderate glycolysis
        A[2, 1] = 0.30        # Some lactate
        A[2, 3] = 0.30        # Also feeds OXPHOS
        A[1, 3] = 0.15        # Lactate can re-enter as fuel (MCT1)
        A[1, 1] = -0.35
        
        # Glutamine
        A[5, 5] = -0.25
        A[5, 6] = 0.45
        A[6, 7] = 0.40
        
        # Better mitochondrial function than TNBC
        A[4, 3] = 0.20
        A[3, 3] = -0.25
        
        # ROS
        A[9, 9] = -0.45
        A[9, 4] = 0.18
        
        return A
    
    @classmethod
    def melanoma_generator(cls, params: Optional[ODEParams] = None) -> np.ndarray:
        """
        Melanoma generator.
        
        Key metabolic features (Ref: Fischer et al. 2018, Mol Cell):
        - OXPHOS-dependent (PGC1a+ subtype)
        - High OXPHOS = immunotherapy responsive (favorable geometry)
        - BRAF-mutant: glycolytic shift (Haq et al. 2013)
        - ROS-adaptive: uses ROS for signaling (MITF)
        - Shallowest attractor of the 4 (most amenable to immune push)
        """
        A = cls.healthy_generator(params)
        
        # OXPHOS-dependent metabolism
        A[0, 0] = -0.35
        A[0, 2] = 0.35        # Lower glycolysis
        A[2, 1] = 0.20        # Less lactate
        A[2, 3] = 0.40        # More OXPHOS
        A[1, 1] = -0.40
        
        # Moderate glutamine
        A[5, 5] = -0.30
        A[5, 6] = 0.35
        A[6, 7] = 0.35
        
        # Good mitochondria (PGC1a+)
        A[4, 3] = 0.25
        A[3, 3] = -0.30
        
        # ROS-adaptive (uses ROS for MITF signaling)
        A[9, 9] = -0.35       # Tolerates ROS
        A[9, 4] = 0.25
        
        return A
    
    @classmethod
    def gbm_generator(cls, params: Optional[ODEParams] = None) -> np.ndarray:
        """
        Glioblastoma Multiforme (GBM) generator — IDH wild-type profile.
        
        Key metabolic features:
        - High glycolytic rate (up to 200x normal) but OXPHOS also active
          Ref: Marin-Valencia et al. 2012, Cell Metabolism
        - Lipid metabolism dependency (SREBP-1 driven de novo lipogenesis)
          Ref: Guo et al. 2011, PNAS; Vlashi et al. 2011, JNCI
        - Impaired ADP phosphorylation despite active ETC
          Ref: Feichtinger et al. 2014, J Neurooncol
        - Glioma stem cells use OXPHOS in hypoxic niches for ROS production
          Ref: Vlashi et al. 2011, JNCI
        - Moderate-deep attractor: glycolysis + lipid deps create resilience
        """
        A = cls.healthy_generator(params)
        
        # High glycolytic rate (Warburg, but less extreme than PDAC)
        A[0, 0] = -0.30       # Increased glucose uptake
        A[0, 2] = 0.55        # Strong glycolysis flux
        A[2, 1] = 0.35        # Moderate lactate production
        A[2, 3] = 0.20        # Some pyruvate to OXPHOS (impaired)
        A[1, 1] = -0.35       # Moderate lactate accumulation
        
        # Glutamine: moderate addiction (varies by subtype)
        A[5, 5] = -0.25
        A[5, 6] = 0.40
        A[6, 7] = 0.35
        
        # Mitochondrial: impaired ADP phosphorylation
        A[4, 3] = 0.12        # Reduced NADH → ATP efficiency
        A[3, 3] = -0.18       # Low ATP turnover (impaired coupling)
        
        # Lipid axis: citrate diverted to de novo lipogenesis (SREBP-1)
        A[8, 8] = -0.10       # Strong citrate diversion to lipid synthesis
        A[7, 8] = 0.25        # aKG feeds TCA → citrate for lipids
        
        # ROS: tolerated, used for GSC signaling
        A[9, 9] = -0.42       # Moderate ROS clearance
        A[9, 4] = 0.22        # ROS leak from impaired ETC
        
        return A
    
    @classmethod
    def crc_generator(cls, params: Optional[ODEParams] = None) -> np.ndarray:
        """
        Colorectal Cancer (CRC) generator — Wnt/β-catenin driven profile.
        
        Key metabolic features:
        - Wnt/β-catenin activates PDK1 → blocks pyruvate→mito flux
          Ref: Pate et al. 2014, PNAS
        - PKM2 overexpression drives efficient glycolysis
          Ref: Amin et al. 2019, Cells
        - Butyrate-sensitive: HDAC inhibition reverses Warburg in CRC
          Ref: Donohoe et al. 2012, Cancer Cell
        - MYC-dependent glycolytic program via Wnt target genes
          Ref: Dang 2012, Cell
        - Intermediate attractor depth (butyrate vulnerability = flattening target)
        """
        A = cls.healthy_generator(params)
        
        # Wnt-driven glycolysis (PDK1 activation blocks pyruvate→mito)
        A[0, 0] = -0.30       # Increased glucose uptake
        A[0, 2] = 0.55        # Enhanced glycolysis (PKM2 high)
        A[2, 1] = 0.45        # Heavy lactate production (PDK1 → LDH)
        A[2, 3] = 0.12        # PDK1 blocks pyruvate entry to mito
        A[1, 1] = -0.30       # Lactate accumulation
        
        # Moderate glutamine usage
        A[5, 5] = -0.30
        A[5, 6] = 0.40
        A[6, 7] = 0.40
        
        # Mitochondrial: partially functional but underused
        A[4, 3] = 0.15        # Reduced OXPHOS coupling
        A[3, 3] = -0.18       # Lower ATP turnover
        
        # Citrate: moderate diversion
        A[8, 8] = -0.20
        
        # ROS: moderately elevated (microsatellite instability adds variance)
        A[9, 9] = -0.45       # Moderate ROS clearance
        A[9, 4] = 0.18
        
        return A

    @classmethod
    def hgsoc_generator(cls, params: Optional[ODEParams] = None) -> np.ndarray:
        """
        High-Grade Serous Ovarian Carcinoma (HGSOC) generator.

        Key metabolic features:
        - Omental fat pad co-option for lipid transfer
          Ref: Nieman et al. 2011, Nature Medicine
        - BRCA1/2 deficiency → impaired homologous recombination → PARP vulnerability
          Ref: Konstantinopoulos et al. 2015, Cancer Discovery
        - Folate/1-carbon metabolism upregulation (MTHFD2)
          Ref: Nilsson et al. 2014, Nature Communications
        - Moderate-deep attractor (lipid dependencies + peritoneal spread)
        """
        A = cls.healthy_generator(params)

        # Glycolysis: moderate Warburg shift
        A[0, 0] = -0.32
        A[0, 2] = 0.48        # Moderate glycolysis
        A[2, 1] = 0.32        # Some lactate production
        A[2, 3] = 0.22        # Partial OXPHOS retained
        A[1, 1] = -0.35

        # Glutamine: moderate (peritoneal milieu provides amino acids)
        A[5, 5] = -0.28
        A[5, 6] = 0.42
        A[6, 7] = 0.38

        # Mitochondria: functional but lipid-dependent
        A[4, 3] = 0.18
        A[3, 3] = -0.22

        # Lipid dependency: omental adipocytes feed citrate/fatty acid synthesis
        A[8, 8] = -0.12       # Strong citrate diversion to lipogenesis
        A[7, 8] = 0.28        # Enhanced aKG→citrate for lipid synthesis

        # ROS: elevated due to DNA repair defects (BRCA-related)
        A[9, 9] = -0.38       # Moderate ROS clearance
        A[9, 4] = 0.22        # ROS leakage

        return A

    @classmethod
    def mcrpc_generator(cls, params: Optional[ODEParams] = None) -> np.ndarray:
        """
        Metastatic Castration-Resistant Prostate Cancer (mCRPC) generator.

        Key metabolic features:
        - AR-driven de novo lipogenesis (FASN, ACC1 upregulation)
          Ref: Zadra et al. 2019, Nature Reviews Cancer
        - OXPHOS-dependent (unlike most solid tumors)
          Ref: Bader & McGuire 2020, Molecular Cancer Research
        - Low glycolytic rate (atypical Warburg)
          Ref: Eidelman et al. 2017, Oncotarget
        - Moderate attractor with OXPHOS dependency as vulnerability
        """
        A = cls.healthy_generator(params)

        # Low glycolysis (anti-Warburg)
        A[0, 0] = -0.38       # Near-normal glucose uptake
        A[0, 2] = 0.35        # Low glycolytic flux
        A[2, 1] = 0.18        # Low lactate production
        A[2, 3] = 0.38        # Strong pyruvate→OXPHOS
        A[1, 1] = -0.50       # Lactate cleared efficiently

        # Glutamine: moderate
        A[5, 5] = -0.32
        A[5, 6] = 0.38
        A[6, 7] = 0.35

        # Mitochondria: functional, OXPHOS-dependent
        A[4, 3] = 0.28        # Strong NADH→ATP
        A[3, 3] = -0.28       # Active ATP turnover

        # AR-driven lipogenesis: strong citrate diversion
        A[8, 8] = -0.10       # Heavy citrate diversion to fatty acids
        A[7, 8] = 0.30        # Enhanced TCA feeding for lipogenesis

        # ROS: relatively controlled
        A[9, 9] = -0.55       # Good ROS clearance (NRF2 pathway)
        A[9, 4] = 0.15        # Low ROS leak

        return A

    @classmethod
    def aml_generator(cls, params: Optional[ODEParams] = None) -> np.ndarray:
        """
        Acute Myeloid Leukemia (AML) generator — IDH-mutant profile.

        Key metabolic features:
        - IDH1/2 mutations produce 2-hydroxyglutarate (2-HG oncometabolite)
          Ref: Ward et al. 2010, Cancer Cell
        - Leukemia stem cells (LSCs) are OXPHOS-dependent, BCL-2 reliant
          Ref: Lagadinou et al. 2013, Cell Stem Cell
        - Venetoclax (BH3-mimetic) targets BCL-2 + OXPHOS dependency
          Ref: DiNardo et al. 2018, Blood
        - Variable attractor depth (IDH-mutant vs wildtype)
        """
        A = cls.healthy_generator(params)

        # Glycolysis: moderate (bulk blasts are glycolytic)
        A[0, 0] = -0.30
        A[0, 2] = 0.50        # Moderate glycolysis
        A[2, 1] = 0.35        # Lactate production
        A[2, 3] = 0.25        # Some OXPHOS (LSCs)
        A[1, 1] = -0.30

        # Glutamine: moderate-high (anaplerosis for TCA)
        A[5, 5] = -0.22
        A[5, 6] = 0.48        # Enhanced glutaminolysis
        A[6, 7] = 0.45        # Strong glutamate→aKG

        # IDH mutation: aKG accumulation / 2-HG production
        # Model as altered aKG clearance (some aKG diverted to 2-HG)
        A[7, 7] = -0.30       # Slower aKG processing (diverted to 2-HG)
        A[7, 8] = 0.15        # Reduced TCA flux from aKG

        # Mitochondria: LSCs depend on OXPHOS (BCL-2 maintains mito integrity)
        A[4, 3] = 0.22
        A[3, 3] = -0.20

        # Citrate: moderate
        A[8, 8] = -0.22

        # ROS: elevated (IDH mutations increase oxidative stress)
        A[9, 9] = -0.35       # Compromised clearance
        A[9, 4] = 0.25        # High ROS from mitochondrial dysfunction

        return A

    @classmethod
    def hcc_generator(cls, params: Optional[ODEParams] = None) -> np.ndarray:
        """
        Hepatocellular Carcinoma (HCC) generator.

        Key metabolic features:
        - De novo lipogenesis driven by SREBP/ACLY/FASN
          Ref: Calvisi et al. 2011, Hepatology
        - Urea cycle defects → arginine auxotrophy in subset
          Ref: Yau et al. 2019, J Hepatol
        - Wnt/β-catenin (CTNNB1) mutations drive glycolytic program
          Ref: Ally et al. 2017, Cell (TCGA-LIHC)
        - Deep attractor (liver regeneration capacity + lipid rewiring)
        """
        A = cls.healthy_generator(params)

        # Glycolysis: Wnt/CTNNB1-driven, strong Warburg
        A[0, 0] = -0.28
        A[0, 2] = 0.62        # High glycolytic flux
        A[2, 1] = 0.42        # Significant lactate production
        A[2, 3] = 0.15        # Reduced OXPHOS
        A[1, 1] = -0.28       # Lactate accumulates

        # Glutamine: variable (some HCC is glutamine-dependent)
        A[5, 5] = -0.25
        A[5, 6] = 0.45
        A[6, 7] = 0.42

        # Mitochondria: partially functional
        A[4, 3] = 0.12
        A[3, 3] = -0.15       # Low ATP turnover

        # Lipogenesis: very strong (SREBP/ACLY/FASN axis)
        A[8, 8] = -0.08       # Extreme citrate diversion to lipids
        A[7, 8] = 0.32        # Heavy TCA→citrate for fatty acid synthesis

        # ROS: elevated (fibrosis/cirrhosis background + lipid peroxidation)
        A[9, 9] = -0.32       # Weak ROS clearance
        A[9, 4] = 0.28        # High ROS production

        return A

    @classmethod
    def pan_cancer_generators(cls) -> dict:
        """Return all 10 cancer generator matrices for pan-cancer analysis."""
        return {
            'TNBC': cls.tnbc_generator(),
            'PDAC': cls.pdac_generator(),
            'NSCLC': cls.nsclc_generator(),
            'Melanoma': cls.melanoma_generator(),
            'GBM': cls.gbm_generator(),
            'CRC': cls.crc_generator(),
            'HGSOC': cls.hgsoc_generator(),
            'mCRPC': cls.mcrpc_generator(),
            'AML': cls.aml_generator(),
            'HCC': cls.hcc_generator(),
        }


def simulate_trajectory(
    A: np.ndarray,
    x0: np.ndarray,
    days: int = 60,
    dt: float = 0.1,
    noise_std: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Euler-Maruyama integrator for dx = Ax dt + sigma dW.

    Args:
        A:         Generator matrix (n x n)
        x0:        Initial state (n,)
        days:      Simulation duration
        dt:        Time step
        noise_std: SDE noise level (0 = deterministic)
        seed:      RNG seed for reproducibility

    Returns:
        (times, trajectory): shape (steps,) and (steps, n)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    steps = int(days / dt)
    n = len(x0)
    trajectory = np.zeros((steps, n))
    times = np.linspace(0.0, days, steps)
    x = x0.copy().astype(float)

    sqrt_dt = np.sqrt(dt)

    for i in range(steps):
        trajectory[i] = x
        dx = A @ x * dt
        if noise_std > 0:
            dx += rng.standard_normal(n) * noise_std * sqrt_dt
        x = x + dx

    return times, trajectory

def simulate_treatment_protocol(
    A_base: np.ndarray,
    x0: np.ndarray,
    protocol: list, # List of ProtocolPhase objects from geometric_optimization
    metabolite_effects: dict, # Map name -> Intervention effect matrix
    immune_effects: dict,     # Map name -> immune parameter dictionary
    intervention_objects: Optional[dict] = None, # Map name -> TherapeuticIntervention (for PK)
    dt: float = 0.1,
    base_noise: float = 0.08,
    resistance_rate: float = 0.05, # Models attractor re-deepening over time
    adaptive_scheduling: bool = True, # Use biomarker thresholds for phase triggers
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Dynamic Euler-Maruyama integrating a multi-phase protocol over time.
    
    Now supports pharmacokinetic drug efficiency via DrugEfficiencyEngine:
    each drug's effect ramps up (onset), sustains (peak window), and
    washes out (half-life decay) — modeling shorter, realistic drug windows.
    
    Returns (times, trajectory, metrics_dict).
    """
    from immune_dynamics import LymphocyteForceField, ImmuneParams
    from geometric_optimization import GeometricOptimizer
    from intervention import DrugEfficiencyEngine

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    total_days = sum(phase.duration for phase in protocol)
    if total_days == 0:
        total_days = 60 # Default if empty
        
    steps = int(total_days / dt)
    n = len(x0)
    
    trajectory = np.zeros((steps, n))
    times = np.linspace(0.0, total_days, steps)
    
    # Store dynamic metrics for analysis
    metrics = {
        'curvature': np.zeros(steps),
        'escape_rate': np.zeros(steps),
        'phase_active': np.zeros(steps, dtype=int),
        'drug_efficacy': np.zeros(steps),  # Aggregate drug efficacy at each step
    }
    
    x = x0.copy().astype(float)
    sqrt_dt = np.sqrt(dt)
    
    geom_opt = GeometricOptimizer(n)
    
    # Build a flat dosing schedule for the DrugEfficiencyEngine
    # Each entry: (dose_day, TherapeuticIntervention, dose_amount)
    pk_schedule = []
    use_pk = intervention_objects is not None and len(intervention_objects) > 0
    
    if use_pk:
        for phase in protocol:
            for drug_name, dose in phase.interventions:
                if drug_name in intervention_objects:
                    pk_schedule.append((float(phase.day_start), intervention_objects[drug_name], dose))
    
    current_day = 0.0
    phase_idx = 0
    current_noise = base_noise
    
    # Track base immune state
    immune_params = ImmuneParams(base_force=0.85, exhaustion_rate=0.12, treg_load=0.24)
    immune_field = LymphocyteForceField(n, immune_params)
    
    # For the non-PK fallback, we still accumulate deltas
    target_A_delta_static = np.zeros_like(A_base)
    
    def apply_phase_static(phase_index):
        """Fallback: constant-dose phase accumulation (no PK)."""
        nonlocal target_A_delta_static, current_noise
        target_A_delta_static = np.zeros_like(A_base)
        current_noise = base_noise
        
        for p in protocol[:phase_index+1]:
            for drug_name, dose in p.interventions:
                if drug_name in metabolite_effects:
                    target_A_delta_static += metabolite_effects[drug_name] * dose
                elif drug_name in immune_effects:
                    imm_effect = immune_effects[drug_name]
                    if 'entropic_driver' in imm_effect:
                        current_noise += imm_effect['entropic_driver'] * dose
                    if 'pd1_blockade' in imm_effect:
                        immune_field.params.pd1_blockade = min(1.0, immune_field.params.pd1_blockade + imm_effect['pd1_blockade'] * dose)
                    if 'ctla4_blockade' in imm_effect:
                        immune_field.params.ctla4_blockade = min(1.0, immune_field.params.ctla4_blockade + imm_effect['ctla4_blockade'] * dose)
                        
    if protocol and not use_pk:
        apply_phase_static(0)

    for i in range(steps):
        current_day = i * dt
        
        # --- 1. Compute effective generator ---
        if use_pk:
            # PK-aware: drug effects are time-weighted via DrugEfficiencyEngine
            pk_delta = DrugEfficiencyEngine.compute_effective_delta(current_day, pk_schedule)
            
            # Also apply immune effects based on PK efficacy
            current_noise = base_noise
            for dose_day, inv_obj, dose in pk_schedule:
                eff = DrugEfficiencyEngine.efficacy_at_time(current_day, dose_day, inv_obj)
                if inv_obj.name in immune_effects:
                    imm = immune_effects[inv_obj.name]
                    if 'entropic_driver' in imm:
                        current_noise += imm['entropic_driver'] * dose * eff
                    if 'pd1_blockade' in imm:
                        immune_field.params.pd1_blockade = min(1.0, imm['pd1_blockade'] * dose * eff)
                    if 'ctla4_blockade' in imm:
                        immune_field.params.ctla4_blockade = min(1.0, imm['ctla4_blockade'] * dose * eff)
            
            target_A_delta = pk_delta
            # Track aggregate efficacy
            if pk_schedule:
                agg_eff = np.mean([DrugEfficiencyEngine.efficacy_at_time(current_day, d, inv, ) for d, inv, _ in pk_schedule])
            else:
                agg_eff = 0.0
            metrics['drug_efficacy'][i] = agg_eff
        else:
            target_A_delta = target_A_delta_static
            metrics['drug_efficacy'][i] = 1.0  # Constant dose assumed
        
        # --- 2. Resistance Modeling ---
        resistance_reversion = np.exp(-resistance_rate * current_day)
        realized_A = A_base + (target_A_delta * resistance_reversion)
        
        current_curvature = geom_opt.compute_basin_curvature(realized_A)
        
        # --- 3. Adaptive Scheduling (only for non-PK mode) ---
        if not use_pk and phase_idx < len(protocol) - 1:
            next_phase = protocol[phase_idx + 1]
            transition = False
            
            if adaptive_scheduling:
                if phase_idx == 0:
                    target_mu = protocol[phase_idx].expected_curvature * 1.5 
                    if current_curvature <= target_mu or current_day >= next_phase.day_start:
                        transition = True
                elif phase_idx == 1:
                    if current_noise > base_noise * 1.5 and current_day >= next_phase.day_start:
                        transition = True
            else:
                if current_day >= next_phase.day_start:
                    transition = True
                    
            if transition:
                phase_idx += 1
                apply_phase_static(phase_idx)
        
        # Track which phase we're in (for PK mode, derive from day)
        if use_pk:
            for pi, p in enumerate(protocol):
                if current_day >= p.day_start:
                    phase_idx = pi
                
        # --- 4. Integrate Step ---
        trajectory[i] = x
        metrics['curvature'][i] = current_curvature
        metrics['phase_active'][i] = phase_idx
        
        # Force computation
        force_vector = immune_field.compute_net_force(x, current_curvature, dt)
        force_mag = np.linalg.norm(force_vector)
        metrics['escape_rate'][i] = geom_opt.compute_kramers_escape_rate(realized_A, current_noise, force_mag)
        
        # Deterministic drift
        dx_det = realized_A @ x * dt
        
        # Stochastic noise
        dx_stoch = np.zeros(n)
        if current_noise > 0:
            dx_stoch = rng.standard_normal(n) * current_noise * sqrt_dt
            
        dx = dx_det + dx_stoch + force_vector * dt
        x = x + dx

    return times, trajectory, metrics
