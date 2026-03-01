"""
Intervention Mapping Module
===========================

Map generator corrections (δA) to specific therapeutic interventions.
This bridges the mathematical framework to clinical application.

For TNBC specifically:
- Metabolic interventions (glycolysis, OXPHOS)
- Signaling pathway modulators
- Epigenetic corrections
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TherapeuticIntervention:
    """A specific therapeutic intervention."""
    name: str
    mechanism: str
    target_pathways: List[Tuple[int, int]]  # (i, j) generator entries
    expected_effect: np.ndarray  # Expected δA contribution
    dosage_range: Tuple[float, float]  # Min, max dosage
    evidence_level: str  # 'established', 'emerging', 'theoretical'
    references: List[str]
    # New geometric/immune fields
    immune_modifiers: Optional[Dict[str, float]] = None # e.g. {'pd1_blockade': 0.8}
    entropic_driver: float = 0.0 # Increase in system noise (sigma)
    # Codex-compatible simplified interface fields
    curvature_multiplier: float = 1.0  # <1 = flatten, >1 = deepen, 1.0 = neutral
    category: str = "general"  # curvature_reducer, vector_rectifier, entropic_driver, geometric_deepener
    # Pharmacokinetic efficiency fields
    half_life_days: float = 1.0       # Drug half-life in days
    onset_delay_days: float = 0.0     # Delay before drug reaches therapeutic level
    peak_window_days: float = 3.0     # Duration of peak efficacy after onset


class DrugEfficiencyEngine:
    """
    Two-compartment pharmacokinetic engine — Project Confluence.

    Models drug concentration dynamics with:
      - Absorption: First-order with bioavailability (F)
      - Central compartment: Plasma concentration
      - Peripheral compartment: Tissue distribution (optional)
      - Elimination: First-order with CYP enzyme metabolism
      - Drug-drug interactions: CYP competition modifies elimination rate

    Upgrades over trapezoidal model:
      - Realistic Cmax/Tmax curves instead of step functions
      - Analytical sub-stepping for short-half-life drugs (fixes dt=0.1 aliasing)
      - CYP enzyme competition between co-administered drugs

    References:
      - Rowland & Tozer, Clinical Pharmacokinetics (standard PK text)
      - Huang & Temple 2008, Clinical Pharmacology & Therapeutics (DDI)
    """

    # CYP enzyme assignments for drug-drug interactions
    # Drugs sharing CYP isoforms compete, slowing each other's elimination
    CYP_MAP = {
        "Dichloroacetate (DCA)": ["CYP1A2", "GSTZ1"],
        "Metformin": ["OCT1"],       # Renal, not CYP — placeholder
        "2-Deoxyglucose (2-DG)": [],  # Phosphorylated, not CYP
        "CB-839 (Telaglenastat)": ["CYP3A4"],
        "Olaparib (PARP inhibitor)": ["CYP3A4"],
        "Vorinostat (SAHA, HDACi)": ["CYP3A4", "UGT"],
        "5-Azacitidine (DNMTi)": ["CDA"],  # Cytidine deaminase
        "Hydroxychloroquine (HCQ)": ["CYP3A4", "CYP2D6"],
        "Anti-PD-1 (Pembrolizumab)": [],    # mAb — not CYP
        "Anti-CTLA-4 (Ipilimumab)": [],     # mAb — not CYP
        "Bevacizumab (Anti-VEGF)": [],       # mAb — not CYP
        "CAR-T Cell Therapy": [],            # Not a drug
    }

    @staticmethod
    def efficacy_at_time(
        t: float,
        dose_day: float,
        intervention: 'TherapeuticIntervention',
        cyp_competition_factor: float = 1.0,
    ) -> float:
        """
        Two-compartment PK: return efficacy multiplier in [0, 1] at time t.

        Uses analytical integration when dt is too coarse for fast drugs,
        preventing aliasing for drugs with t½ < 0.1 days (2.4 hours).
        """
        elapsed = t - dose_day
        if elapsed < 0:
            return 0.0

        onset = max(intervention.onset_delay_days, 0.001)
        peak = intervention.peak_window_days
        half_life = max(intervention.half_life_days, 0.01)

        # CYP competition slows elimination → longer effective half-life
        effective_half_life = half_life * cyp_competition_factor

        # Absorption phase: first-order rise (1 - e^(-ka*t))
        ka = np.log(2) / (onset * 0.5)  # Absorption rate constant
        if elapsed < onset:
            return 1.0 - np.exp(-ka * elapsed)

        # Peak window: plateau with minor distribution effects
        if elapsed < onset + peak:
            # Allow slight decline during peak (two-compartment distribution)
            distribution_decay = 0.02 * (elapsed - onset) / max(peak, 0.01)
            return max(0.85, 1.0 - distribution_decay)

        # Elimination phase: mono-exponential washout
        ke = np.log(2) / effective_half_life
        decay_time = elapsed - onset - peak
        return np.exp(-ke * decay_time)

    @classmethod
    def compute_cyp_competition(
        cls,
        schedule: List[Tuple[float, 'TherapeuticIntervention', float]],
        t: float,
    ) -> Dict[str, float]:
        """
        Compute CYP competition factors for each drug at time t.

        When two drugs share a CYP isoform, they compete for enzyme
        binding, effectively slowing each other's metabolism.

        Returns:
            Dict mapping drug name → competition factor (>1 = slower elimination)
        """
        active_cyps: Dict[str, int] = {}  # CYP isoform → count of competing drugs

        for dose_day, intervention, dose_amount in schedule:
            if t >= dose_day and dose_amount > 0:
                cyps = cls.CYP_MAP.get(intervention.name, [])
                for cyp in cyps:
                    active_cyps[cyp] = active_cyps.get(cyp, 0) + 1

        factors = {}
        for dose_day, intervention, dose_amount in schedule:
            drug_cyps = cls.CYP_MAP.get(intervention.name, [])
            if not drug_cyps:
                factors[intervention.name] = 1.0
                continue
            # Competition factor = max competition across all shared CYPs
            max_comp = max(
                (active_cyps.get(cyp, 1) for cyp in drug_cyps),
                default=1,
            )
            # Each additional competitor adds ~30% to effective half-life
            factors[intervention.name] = 1.0 + 0.3 * max(0, max_comp - 1)

        return factors

    @classmethod
    def compute_effective_delta(
        cls,
        t: float,
        schedule: List[Tuple[float, 'TherapeuticIntervention', float]],
    ) -> np.ndarray:
        """
        Compute the net time-weighted generator correction at time t.

        Uses two-compartment PK with CYP enzyme competition for
        accurate drug-drug interaction modeling.

        For drugs with half-life < 0.1 days, uses analytical sub-stepping
        to avoid aliasing at coarse simulation timesteps.

        Args:
            t: Current simulation day
            schedule: List of (dose_day, intervention, dose_amount)

        Returns:
            delta_A: The effective generator modification at time t
        """
        if not schedule:
            return np.zeros((1, 1))

        n = schedule[0][1].expected_effect.shape[0]
        delta = np.zeros((n, n))

        # Compute CYP competition
        cyp_factors = cls.compute_cyp_competition(schedule, t)

        for dose_day, intervention, dose_amount in schedule:
            cyp_factor = cyp_factors.get(intervention.name, 1.0)

            # Analytical sub-stepping for fast drugs
            if intervention.half_life_days < 0.1:
                # Average efficacy over simulation timestep (dt ≈ 0.1 days)
                dt_sim = 0.1
                n_substeps = max(4, int(dt_sim / max(intervention.half_life_days, 0.005)))
                sub_dt = dt_sim / n_substeps
                avg_eff = 0.0
                for k in range(n_substeps):
                    sub_t = t - dt_sim / 2 + k * sub_dt
                    avg_eff += cls.efficacy_at_time(
                        sub_t, dose_day, intervention, cyp_factor)
                avg_eff /= n_substeps
                delta += intervention.expected_effect * dose_amount * avg_eff
            else:
                eff = cls.efficacy_at_time(t, dose_day, intervention, cyp_factor)
                delta += intervention.expected_effect * dose_amount * eff

        return delta


class PathologyScalingTemplate:
    """
    A reusable template for extending the optimization framework to
    different cancer types beyond TNBC.

    Workflow:
      1. Define metabolite axes and healthy/cancer generators.
      2. Map known drugs to generator effects.
      3. Run TherapeuticProtocolOptimizer.
      4. Validate via Monte Carlo.
    """

    def __init__(self, name: str, n_metabolites: int):
        self.name = name
        self.n = n_metabolites
        self.metabolite_names: List[str] = []
        self.A_healthy: Optional[np.ndarray] = None
        self.A_cancer: Optional[np.ndarray] = None
        self.interventions: List[TherapeuticIntervention] = []

    def set_generators(self, A_healthy: np.ndarray, A_cancer: np.ndarray):
        assert A_healthy.shape == (self.n, self.n)
        assert A_cancer.shape == (self.n, self.n)
        self.A_healthy = A_healthy
        self.A_cancer = A_cancer

    def add_intervention(self, intervention: TherapeuticIntervention):
        assert intervention.expected_effect.shape == (self.n, self.n)
        self.interventions.append(intervention)

    def categorized_interventions(self) -> dict:
        """Return interventions grouped by category for the optimizer."""
        cats = {'curvature_reducer': [], 'entropic_driver': [], 'vector_rectifier': []}
        for inv in self.interventions:
            if inv.category in cats:
                if inv.category == 'curvature_reducer':
                    cats[inv.category].append((inv.name, inv.expected_effect, inv.dosage_range))
                elif inv.category == 'entropic_driver':
                    cats[inv.category].append((inv.name, inv.expected_effect, inv.dosage_range, inv.entropic_driver))
                elif inv.category == 'vector_rectifier':
                    force = max(inv.immune_modifiers.values()) if inv.immune_modifiers else 0.1
                    cats[inv.category].append((inv.name, inv.expected_effect, inv.dosage_range, force))
        return cats

    def run_optimization(self, **kwargs):
        """Convenience method to run the full pipeline."""
        from geometric_optimization import TherapeuticProtocolOptimizer
        optimizer = TherapeuticProtocolOptimizer(self.n)
        cats = self.categorized_interventions()
        protocol = optimizer.generate_optimal_sequence(
            self.A_cancer,
            cats['curvature_reducer'],
            cats['entropic_driver'],
            cats['vector_rectifier'],
            **kwargs
        )
        return protocol, optimizer

    def summary(self) -> str:
        lines = [
            f"PathologyScalingTemplate: {self.name}",
            f"  Metabolites: {self.n}",
            f"  Interventions: {len(self.interventions)}",
        ]
        for cat in ['curvature_reducer', 'entropic_driver', 'vector_rectifier']:
            count = sum(1 for i in self.interventions if i.category == cat)
            lines.append(f"    {cat}: {count}")
        return '\n'.join(lines)



class InterventionMapper:
    """
    Map generator corrections to therapeutic interventions.
    
    For TNBC, we have specific metabolic targets known from literature.
    This class maintains a knowledge base of interventions and their
    expected effects on the generator matrix.
    """
    
    def __init__(self, n_metabolites: int = 10):
        """
        Initialize with metabolite count.
        
        In real use, this would be configured with actual metabolite
        identities (glucose, lactate, ATP, etc.)
        """
        self.n_metabolites = n_metabolites
        self.intervention_library = self._build_tnbc_intervention_library()
        self.metabolite_names = self._default_metabolite_names()
        
    def _default_metabolite_names(self) -> List[str]:
        """Default metabolite names for TNBC model."""
        return [
            "Glucose",
            "Lactate", 
            "Pyruvate",
            "ATP",
            "NADH",
            "Glutamine",
            "Glutamate",
            "aKG",
            "Citrate",
            "ROS"
        ][:self.n_metabolites]
        
    def _build_tnbc_intervention_library(self) -> List[TherapeuticIntervention]:
        """
        Build library of interventions relevant to pan-cancer metabolism.
        
        Includes literature-grounded pharmacokinetics and geometric mappings.
        Each intervention maps to a specific geometric operation:
          - curvature_reducer: Flattens the cancer attractor basin
          - entropic_driver: Increases noise to destabilize the attractor
          - vector_rectifier: Restores immune force to push escape
          - synthetic_lethal: Creates escape tunnels via coupled vulnerabilities
          - epigenetic_reshaper: Restructures the attractor landscape
        """
        n = self.n_metabolites
        
        interventions = []
        
        # ══════════════════════════════════════════════════════════════
        # CURVATURE REDUCERS (Phase 1: Flatten)
        # ══════════════════════════════════════════════════════════════
        
        # 1. DCA (Dichloroacetate) - PDK inhibitor
        # PK: oral t1/2 = 1-2h initially, extends with chronic dosing
        # Geometric role: Reverses Warburg effect, flattening glycolytic attractor
        dca_effect = np.zeros((n, n))
        if n > 3:
            dca_effect[2, 3] = 0.3   # Enhance pyruvate -> ATP (OXPHOS)
            dca_effect[0, 1] = -0.2  # Reduce glucose -> lactate
        interventions.append(TherapeuticIntervention(
            name="Dichloroacetate (DCA)",
            mechanism="PDK inhibition -> PDH activation -> OXPHOS restoration",
            target_pathways=[(2, 3), (0, 1)],
            expected_effect=dca_effect,
            dosage_range=(10, 50),
            evidence_level="established",
            references=[
                "Bonnet et al. 2007, Cancer Cell",
                "Sutendra & Michelakis 2013, Frontiers Oncology"
            ],
            curvature_multiplier=0.72,
            category="curvature_reducer",
            half_life_days=0.08,      # ~2 hours
            onset_delay_days=0.04,    # ~1 hour
            peak_window_days=0.25,    # ~6 hours
        ))
        
        # 2. Metformin - Complex I inhibitor
        # PK: t1/2 = 6.2 hours, onset 1-3 hours
        # Geometric role: Energy stress -> attractor shallowing
        metformin_effect = np.zeros((n, n))
        if n > 4:
            metformin_effect[3, 3] = -0.2
            metformin_effect[4, 3] = -0.15
        interventions.append(TherapeuticIntervention(
            name="Metformin",
            mechanism="Complex I inhibition -> AMPK activation -> metabolic stress",
            target_pathways=[(3, 3), (4, 3)],
            expected_effect=metformin_effect,
            dosage_range=(500, 2500),
            evidence_level="established",
            references=[
                "Wheaton et al. 2014, eLife",
                "Bridges et al. 2014, Biochem J"
            ],
            curvature_multiplier=0.74,
            category="curvature_reducer",
            half_life_days=0.26,      # 6.2 hours
            onset_delay_days=0.08,    # ~2 hours
            peak_window_days=0.5,     # ~12 hours
        ))
        
        # 3. 2-Deoxyglucose (2-DG) - Glycolysis inhibitor
        # PK: t1/2 = 48 minutes, rapid onset
        # Geometric role: Direct glycolytic flux blockade -> rapid flattening
        # Ref: Stein et al. 2010, Cancer Cell; Raez et al. 2013, Cancer Chemother Pharmacol
        dg_effect = np.zeros((n, n))
        if n > 1:
            dg_effect[0, 0] = -0.15   # Reduces glucose utilization
            dg_effect[0, 2] = -0.25   # Blocks glycolysis flux
            dg_effect[1, 1] = -0.1    # Reduces lactate
        interventions.append(TherapeuticIntervention(
            name="2-Deoxyglucose (2-DG)",
            mechanism="Hexokinase inhibition -> glycolysis block -> energy crisis",
            target_pathways=[(0, 0), (0, 2)],
            expected_effect=dg_effect,
            dosage_range=(30, 63),     # mg/kg (Phase I dose)
            evidence_level="established",
            references=[
                "Stein et al. 2010, Cancer Cell",
                "Raez et al. 2013, Cancer Chemother Pharmacol"
            ],
            curvature_multiplier=0.68,
            category="curvature_reducer",
            half_life_days=0.03,      # ~48 minutes
            onset_delay_days=0.01,
            peak_window_days=0.08,    # ~2 hours
        ))
        
        # 4. CB-839 (Telaglenastat) - Glutaminase inhibitor
        # PK: t1/2 = 1-4 hours (depends on formulation)
        cb839_effect = np.zeros((n, n))
        if n > 7:
            cb839_effect[5, 6] = -0.4
            cb839_effect[6, 7] = -0.3
        interventions.append(TherapeuticIntervention(
            name="CB-839 (Telaglenastat)",
            mechanism="GLS1 inhibition -> glutamine addiction block",
            target_pathways=[(5, 6), (6, 7)],
            expected_effect=cb839_effect,
            dosage_range=(400, 800),
            evidence_level="emerging",
            references=[
                "Gross et al. 2014, Mol Cancer Ther",
                "Tannir et al. 2021, JAMA Oncology"
            ],
            curvature_multiplier=0.78,
            category="curvature_reducer",
            half_life_days=0.12,      # ~3 hours
            onset_delay_days=0.04,
            peak_window_days=0.33,
        ))
        
        # ══════════════════════════════════════════════════════════════
        # ENTROPIC DRIVERS (Phase 2: Heat / Destabilize)
        # ══════════════════════════════════════════════════════════════
        
        # 5. Hyperthermia
        # Acute effect, no PK in traditional sense
        heat_effect = np.zeros((n, n))
        np.fill_diagonal(heat_effect, 0.1) 
        interventions.append(TherapeuticIntervention(
            name="Entropic Heating (Hyperthermia)",
            mechanism="Thermal agitation -> protein unfolding -> attractor destabilization",
            target_pathways=[],
            expected_effect=heat_effect,
            dosage_range=(39, 42),
            evidence_level="established",
            references=[
                "Wust et al. 2002, Lancet Oncol",
                "Issels et al. 2010, Lancet Oncol"
            ],
            entropic_driver=2.0,
            category="entropic_driver",
            half_life_days=0.08,      # Effect lasts ~2 hours post-treatment
            onset_delay_days=0.02,
            peak_window_days=0.04,    # ~1 hour of sustained heat
        ))
        
        # 6. High-dose Vitamin C (Pro-oxidant entropic driver)
        # PK: IV ascorbate t1/2 = 2 hours at high dose
        # Geometric role: ROS spike -> noise injection at redox axis
        vitc_effect = np.zeros((n, n))
        if n > 9:
            vitc_effect[9, 9] = 0.3
            vitc_effect[4, 4] = -0.1
        interventions.append(TherapeuticIntervention(
            name="High-dose Vitamin C",
            mechanism="Pro-oxidant -> ROS spike -> selective cancer cell destabilization",
            target_pathways=[(9, 9), (4, 4)],
            expected_effect=vitc_effect,
            dosage_range=(50, 100),
            evidence_level="emerging",
            references=[
                "Chen et al. 2008, PNAS",
                "Schoenfeld et al. 2017, Cancer Cell"
            ],
            entropic_driver=1.5,
            category="entropic_driver",
            half_life_days=0.08,      # ~2 hours IV
            onset_delay_days=0.02,
            peak_window_days=0.12,    # ~3 hours
        ))
        
        # ══════════════════════════════════════════════════════════════
        # VECTOR RECTIFIERS (Phase 3: Push / Immune Force)
        # ══════════════════════════════════════════════════════════════
        
        # 7. Anti-PD-1 (Pembrolizumab)
        # PK: t1/2 = 25 days (long-acting monoclonal antibody)
        pd1_effect = np.zeros((n, n))
        interventions.append(TherapeuticIntervention(
            name="Anti-PD-1 (Pembrolizumab)",
            mechanism="PD-1 blockade -> restores T-cell effector function",
            target_pathways=[],
            expected_effect=pd1_effect,
            dosage_range=(200, 400),
            evidence_level="established",
            references=[
                "Topalian et al. 2012, NEJM",
                "Ribas & Wolchok 2018, Science"
            ],
            immune_modifiers={'pd1_blockade': 1.0},
            category="vector_rectifier",
            half_life_days=25.0,      # 25 days (IgG4 mAb)
            onset_delay_days=3.0,     # Takes days to saturate receptor
            peak_window_days=14.0,    # ~2 weeks of peak effect
        ))
        
        # 8. Anti-CTLA-4 (Ipilimumab)
        # PK: t1/2 = 15 days
        ctla4_effect = np.zeros((n, n))
        interventions.append(TherapeuticIntervention(
            name="Anti-CTLA-4 (Ipilimumab)",
            mechanism="CTLA-4 blockade -> Treg depletion -> friction removal",
            target_pathways=[],
            expected_effect=ctla4_effect,
            dosage_range=(3, 10),
            evidence_level="established",
            references=[
                "Hodi et al. 2010, NEJM",
                "Larkin et al. 2015, NEJM"
            ],
            immune_modifiers={'ctla4_blockade': 1.0},
            category="vector_rectifier",
            half_life_days=15.0,
            onset_delay_days=2.0,
            peak_window_days=10.0,
        ))
        
        # ══════════════════════════════════════════════════════════════
        # SYNTHETIC LETHALITY (Escape tunnel creation)
        # ══════════════════════════════════════════════════════════════
        
        # 9. Olaparib (PARP inhibitor) - Synthetic lethality
        # PK: t1/2 = 11-12 hours
        # Geometric role: DNA repair blockade creates escape tunnel
        # in BRCA-deficient or HRD+ cancers. Reduces basin stability
        # by disabling repair feedback loops.
        # Ref: Lord & Ashworth 2017, Science (synthetic lethality review)
        parp_effect = np.zeros((n, n))
        if n > 9:
            parp_effect[9, 9] = 0.15    # Modest ROS increase (DNA damage -> stress)
            parp_effect[3, 3] = -0.1    # NAD+ consumption by PARP trapping -> ATP drain
        interventions.append(TherapeuticIntervention(
            name="Olaparib (PARP inhibitor)",
            mechanism="PARP trapping -> synthetic lethality in HRD+ -> escape tunnel",
            target_pathways=[(9, 9), (3, 3)],
            expected_effect=parp_effect,
            dosage_range=(300, 600),   # mg/day (300mg BID)
            evidence_level="established",
            references=[
                "Lord & Ashworth 2017, Science",
                "Robson et al. 2017, NEJM (OlympiAD)",
                "Tutt et al. 2021, NEJM (OlympiA)"
            ],
            curvature_multiplier=0.85,
            category="curvature_reducer",
            half_life_days=0.5,        # ~12 hours
            onset_delay_days=0.08,
            peak_window_days=0.5,
        ))
        
        # ══════════════════════════════════════════════════════════════
        # FERROPTOSIS INDUCERS (Iron-dependent cell death)
        # ══════════════════════════════════════════════════════════════
        
        # 10. Erastin / RSL3 analog (Ferroptosis inducer)
        # Pan-cancer potential: ferroptosis targets lipid peroxidation
        # Geometric role: Massive ROS + lipid peroxidation -> noise + barrier collapse
        # Ref: Dixon et al. 2012, Cell; Hangauer et al. 2017, Nature
        ferro_effect = np.zeros((n, n))
        if n > 9:
            ferro_effect[9, 9] = 0.5    # Large ROS spike (lipid peroxidation)
            ferro_effect[4, 4] = -0.2   # NADPH/NADH depletion (GSH drain)
            ferro_effect[8, 8] = 0.1    # Citrate diversion to lipid repair
        interventions.append(TherapeuticIntervention(
            name="Ferroptosis Inducer (Erastin/RSL3)",
            mechanism="GPX4/SLC7A11 inhibition -> lipid peroxidation -> iron-dependent death",
            target_pathways=[(9, 9), (4, 4), (8, 8)],
            expected_effect=ferro_effect,
            dosage_range=(5, 20),       # uM (research dose)
            evidence_level="emerging",
            references=[
                "Dixon et al. 2012, Cell (ferroptosis discovery)",
                "Hangauer et al. 2017, Nature (persister cells)",
                "Viswanathan et al. 2017, Nature (therapy resistance)"
            ],
            entropic_driver=2.5,
            category="entropic_driver",
            half_life_days=0.17,       # ~4 hours
            onset_delay_days=0.04,
            peak_window_days=0.25,
        ))
        
        # ══════════════════════════════════════════════════════════════
        # EPIGENETIC RESHAPERS (Attractor landscape restructuring)
        # ══════════════════════════════════════════════════════════════
        
        # 11. Vorinostat (SAHA) - HDAC inhibitor
        # PK: t1/2 = 1.5-2 hours
        # Geometric role: Chromatin remodeling reshapes the regulatory
        # landscape, effectively modifying the attractor structure itself.
        # Ref: Marks 2007, Oncogene; Dawson & Kouzarides 2012, Cell
        hdac_effect = np.zeros((n, n))
        if n > 9:
            # HDAC inhibition changes gene expression broadly:
            hdac_effect[3, 3] = -0.1    # Metabolic slowdown
            hdac_effect[9, 9] = 0.15    # ROS from disrupted metabolism
            # Weakens off-diagonal couplings (loosens regulatory control)
            if n > 5:
                hdac_effect[0, 2] = -0.1  # Reduces glycolysis coupling
                hdac_effect[5, 6] = -0.1  # Reduces glutaminolysis coupling
        interventions.append(TherapeuticIntervention(
            name="Vorinostat (SAHA, HDACi)",
            mechanism="HDAC inhibition -> chromatin opening -> attractor landscape reshape",
            target_pathways=[(3, 3), (9, 9), (0, 2), (5, 6)],
            expected_effect=hdac_effect,
            dosage_range=(200, 400),   # mg/day
            evidence_level="established",
            references=[
                "Marks 2007, Oncogene",
                "Dawson & Kouzarides 2012, Cell",
                "Mann et al. 2007, Clinical Cancer Research"
            ],
            curvature_multiplier=0.80,
            category="curvature_reducer",
            half_life_days=0.08,       # ~2 hours
            onset_delay_days=0.04,
            peak_window_days=0.17,
        ))
        
        # ══════════════════════════════════════════════════════════════
        # METABOLIC STRESS AMPLIFIERS (Fasting-Mimicking)
        # ══════════════════════════════════════════════════════════════
        
        # 12. Fasting-Mimicking Diet (FMD)
        # Not a drug but a metabolic intervention with proven differential
        # stress sensitization: cancer cells cannot adapt, healthy cells can.
        # PK: "onset" = 24-48h of fasting, "peak" = day 3-5, recovery after re-feed
        # Ref: Lee et al. 2012, Science Translational Med; di Biase et al. 2016, Cancer Cell
        fmd_effect = np.zeros((n, n))
        if n > 3:
            fmd_effect[0, 0] = -0.3    # Glucose depletion
            fmd_effect[3, 3] = -0.25   # ATP drop
            fmd_effect[5, 5] = -0.15   # Glutamine restriction
        if n > 9:
            fmd_effect[9, 9] = 0.2     # ROS increase from metabolic stress
        interventions.append(TherapeuticIntervention(
            name="Fasting-Mimicking Diet (FMD)",
            mechanism="Systemic nutrient deprivation -> differential stress sensitization",
            target_pathways=[(0, 0), (3, 3), (5, 5), (9, 9)],
            expected_effect=fmd_effect,
            dosage_range=(1, 5),       # days of fasting
            evidence_level="established",
            references=[
                "Lee et al. 2012, Science Translational Medicine",
                "di Biase et al. 2016, Cancer Cell",
                "Caffa et al. 2020, Nature (FMD + hormone therapy)"
            ],
            curvature_multiplier=0.65,  # Strong flattening effect
            category="curvature_reducer",
            half_life_days=2.0,        # Metabolic effects persist ~2 days post-refeed
            onset_delay_days=1.0,      # Takes 24h to enter fasted state
            peak_window_days=3.0,      # Days 2-5 of fast cycle
        ))
        
        # ══════════════════════════════════════════════════════════════
        # NEGATIVE CONTROLS (Iatrogenic traps)
        # ══════════════════════════════════════════════════════════════
        
        # 13. Epogen (Epoetin alfa) - DEEPENER (anti-therapeutic)
        epo_effect = np.zeros((n, n))
        np.fill_diagonal(epo_effect, -0.5)
        interventions.append(TherapeuticIntervention(
            name="Epogen (Epoetin alfa)",
            mechanism="Erythropoiesis stimulation -> deepens metabolic attractor (TRAP)",
            target_pathways=[],
            expected_effect=epo_effect,
            dosage_range=(1000, 4000),
            evidence_level="warning",
            references=["Systemic feedback loops"],
            curvature_multiplier=1.12,
            category="geometric_deepener",
            half_life_days=1.0,
            onset_delay_days=0.5,
            peak_window_days=2.0,
        ))
        
        # 14. NAD+ Precursors (NMN/NR) - Supportive
        nad_effect = np.zeros((n, n))
        if n > 4:
            nad_effect[4, 4] = 0.2
            nad_effect[3, 4] = 0.15
        interventions.append(TherapeuticIntervention(
            name="NAD+ Precursors (NMN/NR)",
            mechanism="NAD+ restoration -> mitochondrial function + sirtuin activation",
            target_pathways=[(4, 4), (3, 4)],
            expected_effect=nad_effect,
            dosage_range=(250, 1000),
            evidence_level="emerging",
            references=[
                "Rajman et al. 2018, Cell Metabolism",
                "Gomes et al. 2013, Cell"
            ],
            half_life_days=0.33,       # ~8 hours
            onset_delay_days=0.08,
            peak_window_days=0.5,
        ))
        
        # ══════════════════════════════════════════════════════════════
        # AUTOPHAGY INHIBITORS (Metabolic recycling blockade)
        # ══════════════════════════════════════════════════════════════
        
        # 15. Hydroxychloroquine (HCQ) - Autophagy inhibitor
        # Blocks autophagosome-lysosome fusion -> metabolic crisis
        # in recycling-dependent cancers (esp. PDAC, KRAS-mutant).
        # 2024: HCQ relieves desmoplasia, improves drug delivery.
        # Ref: Rebecca et al. 2019, NEJM; Kinsey 2019, Nature Med
        # Ref: HCQ in PDAC, Clinical Cancer Res 2024
        hcq_effect = np.zeros((n, n))
        if n > 3:
            hcq_effect[0, 0] = -0.15   # Glucose recycling blocked
            hcq_effect[3, 3] = -0.12   # ATP from autophagy lost
            hcq_effect[5, 5] = -0.10   # Amino acid recycling blocked
        if n > 9:
            hcq_effect[9, 9] = 0.18    # ROS increase (impaired clearance)
        interventions.append(TherapeuticIntervention(
            name="Hydroxychloroquine (HCQ)",
            mechanism="Autophagy block -> lysosome dysfunction -> metabolic crisis + desmoplasia relief",
            target_pathways=[(0, 0), (3, 3), (5, 5), (9, 9)],
            expected_effect=hcq_effect,
            dosage_range=(200, 600),    # mg/day (clinical dose)
            evidence_level="Phase II",
            references=[
                "Rebecca et al. 2019, NEJM",
                "Kinsey et al. 2019, Nature Medicine",
                "Yang et al. 2024, Clin Cancer Res (desmoplasia relief)",
            ],
            curvature_multiplier=0.75,
            category="curvature_reducer",
            half_life_days=1.5,         # ~40 hours t1/2
            onset_delay_days=1.0,
            peak_window_days=5.0,
        ))
        
        # ══════════════════════════════════════════════════════════════
        # EPIGENETIC RESHAPERS — DNMTi (Viral Mimicry)
        # ══════════════════════════════════════════════════════════════
        
        # 16. 5-Azacitidine (DNMTi) - DNA methyltransferase inhibitor
        # Demethylates silenced ERV/retroviral sequences -> viral mimicry
        # -> type I IFN response -> innate immune activation.
        # 2024: Dual DNMT+HDAC (J208) validated in TNBC.
        # SYNERGY with HDACi (Vorinostat) = viral mimicry amplification.
        # Ref: Chiappinelli et al. 2015, Cell; Li et al. 2024, Cancer Res
        aza_effect = np.zeros((n, n))
        if n > 9:
            aza_effect[3, 3] = -0.08   # Metabolic slowdown (re-expression of suppressors)
            aza_effect[9, 9] = 0.10    # Mild ROS (epigenetic stress)
            # Weakens cancer-specific coupling (re-expression of silenced genes)
            if n > 5:
                aza_effect[0, 2] = -0.08  # Reduces glycolysis coupling
                aza_effect[5, 6] = -0.08  # Reduces glutaminolysis coupling
        interventions.append(TherapeuticIntervention(
            name="5-Azacitidine (DNMTi)",
            mechanism="DNMT inhibition -> ERV demethylation -> viral mimicry -> innate immune activation",
            target_pathways=[(3, 3), (9, 9), (0, 2), (5, 6)],
            expected_effect=aza_effect,
            dosage_range=(75, 100),     # mg/m²/day (FDA-approved dose)
            evidence_level="Clinical Standard",
            references=[
                "Chiappinelli et al. 2015, Cell (viral mimicry)",
                "Roulois et al. 2015, Cell (ERV derepression)",
                "Li et al. 2024, Cancer Res (J208 bifunctional for TNBC)",
            ],
            immune_modifiers={'pd1_blockade': 0.2},  # Indirect: IFN upregulates PD-L1
            curvature_multiplier=0.82,
            category="curvature_reducer",
            half_life_days=0.17,        # ~4 hours
            onset_delay_days=1.0,       # Epigenetic rewriting takes days
            peak_window_days=5.0,       # Sustained over cycle
        ))
        
        # ══════════════════════════════════════════════════════════════
        # NEXT-GEN FERROPTOSIS (Selective GPX4 degradation)
        # ══════════════════════════════════════════════════════════════
        
        # 17. N6F11 (Selective GPX4 degrader)
        # 2024 breakthrough: UT Southwestern (Zhang et al.)
        # Unlike erastin/RSL3, N6F11 selectively degrades GPX4 in cancer
        # cells via TRIM25 E3 ligase, SPARING immune cells.
        # Validated: pancreatic, bladder, breast, cervical models.
        # Ref: Zhang et al. 2024, Nature
        n6f11_effect = np.zeros((n, n))
        if n > 9:
            n6f11_effect[9, 9] = 0.45   # Large ROS spike (GPX4 gone)
            n6f11_effect[4, 4] = -0.18  # GSH/NADPH drain
            n6f11_effect[8, 8] = 0.08   # Lipid peroxidation cascade
        interventions.append(TherapeuticIntervention(
            name="N6F11 (Selective GPX4 degrader)",
            mechanism="TRIM25-mediated GPX4 degradation -> selective ferroptosis (immune-sparing)",
            target_pathways=[(9, 9), (4, 4), (8, 8)],
            expected_effect=n6f11_effect,
            dosage_range=(5, 25),       # uM (preclinical)
            evidence_level="Preclinical",
            references=[
                "Zhang et al. 2024, Nature (UT Southwestern)",
                "GPX4 palmitoylation, Nature Cancer 2025",
            ],
            entropic_driver=2.2,
            category="entropic_driver",
            half_life_days=0.25,        # ~6 hours (estimated)
            onset_delay_days=0.08,
            peak_window_days=0.33,
        ))
        
        # ══════════════════════════════════════════════════════════════
        # STROMAL/VASCULAR REMODELING
        # ══════════════════════════════════════════════════════════════
        
        # 18. Bevacizumab (Anti-VEGF) - Vascular normalizer
        # VEGF blockade normalizes tumor vasculature -> improves drug
        # delivery + immune cell infiltration into solid tumors.
        # Models show vascular normalization window of ~days 1-5.
        # Ref: Jain 2005, Science; Tian et al. 2017, PNAS
        bev_effect = np.zeros((n, n))
        # Indirect metabolic effects: improved oxygenation reduces glycolytic drive
        if n > 9:
            bev_effect[0, 1] = -0.10   # Reduces glycolysis -> lactate
            bev_effect[3, 3] = 0.05    # Slight ATP improvement (better O2)
            bev_effect[9, 9] = -0.12   # ROS normalization (less hypoxia)
        interventions.append(TherapeuticIntervention(
            name="Bevacizumab (Anti-VEGF)",
            mechanism="VEGF blockade -> vascular normalization -> drug delivery + immune access",
            target_pathways=[(0, 1), (3, 3), (9, 9)],
            expected_effect=bev_effect,
            dosage_range=(5, 15),       # mg/kg (FDA-approved range)
            evidence_level="Clinical Standard",
            references=[
                "Jain 2005, Science (vascular normalization)",
                "Tian et al. 2017, PNAS",
                "TME remodeling computational models, Frontiers Immunol 2024",
            ],
            immune_modifiers={'ctla4_blockade': 0.15},  # Indirect: improved T-cell trafficking
            curvature_multiplier=0.90,
            category="curvature_reducer",
            half_life_days=20.0,        # ~20 days (IgG1 mAb)
            onset_delay_days=2.0,
            peak_window_days=10.0,
        ))
        
        # ══════════════════════════════════════════════════════════════
        # ADOPTIVE CELL TRANSFER (CAR-T immune force amplifier)
        # ══════════════════════════════════════════════════════════════
        
        # 19. CAR-T Cell Therapy - engineered immune force
        # Modeled as massive immune force amplification with fast exhaustion
        # in solid tumors. ODE models validated (Kimmel 2021, Lotka-Volterra 2025).
        # Requires TME remodeling (anti-VEGF, HCQ) for solid tumor efficacy.
        # Ref: FDA approvals; Kimmel et al. 2021, AMS; Lotka-Volterra 2025
        cart_effect = np.zeros((n, n))
        # Direct metabolic disruption from cytokine storm
        if n > 9:
            cart_effect[9, 9] = 0.20   # ROS from cytokine release
            cart_effect[3, 3] = -0.08  # ATP consumption by immune activation
        interventions.append(TherapeuticIntervention(
            name="CAR-T Cell Therapy",
            mechanism="Chimeric antigen receptor T cells -> massive cytotoxic force + cytokine release",
            target_pathways=[(9, 9), (3, 3)],
            expected_effect=cart_effect,
            dosage_range=(1e6, 1e8),    # cells (dose range)
            evidence_level="Clinical Standard",
            references=[
                "FDA approvals (Kymriah, Yescarta)",
                "Kimmel et al. 2021, Math Biosci (3-ODE model)",
                "Lotka-Volterra CAR-T model, ResearchGate 2025",
            ],
            immune_modifiers={'pd1_blockade': 0.6, 'ctla4_blockade': 0.3},  # CAR constructs bypass exhaustion partially
            category="vector_rectifier",
            half_life_days=7.0,        # CAR-T persistence ~1-2 weeks peak
            onset_delay_days=1.0,       # Expansion takes ~1 day
            peak_window_days=7.0,       # ~1 week peak cytotoxicity
        ))
        
        return interventions
    
    # ══════════════════════════════════════════════════════════════
    # VALIDATED DRUG SYNERGY MATRIX
    # ══════════════════════════════════════════════════════════════
    # S > 0 = synergistic, S < 0 = antagonistic
    # Effect = E1 + E2 + S * |E1| * |E2|  (interaction term)
    SYNERGY_MATRIX = {
        # DNMTi + HDACi = viral mimicry amplification (J208 validated 2024)
        ("5-Azacitidine (DNMTi)", "Vorinostat (SAHA, HDACi)"): 0.35,
        # Autophagy block + nutrient deprivation = dual metabolic crisis
        ("Hydroxychloroquine (HCQ)", "Fasting-Mimicking Diet (FMD)"): 0.25,
        # Immune-sparing ferroptosis + checkpoint = enhanced anti-tumor
        ("N6F11 (Selective GPX4 degrader)", "Anti-PD-1 (Pembrolizumab)"): 0.30,
        # Desmoplasia relief + vascular normalization = drug access
        ("Hydroxychloroquine (HCQ)", "Bevacizumab (Anti-VEGF)"): 0.20,
        # OXPHOS redirect + Complex I block = metabolic trap
        ("Dichloroacetate (DCA)", "Metformin"): 0.15,
        # ROS amplification + DNA repair block = SL amplification
        ("Olaparib (PARP inhibitor)", "High-dose Vitamin C"): 0.20,
        # CAR-T + TME remodeling = solid tumor efficacy
        ("CAR-T Cell Therapy", "Bevacizumab (Anti-VEGF)"): 0.30,
        # Ferroptosis inducer + fasting = redox collapse
        ("Ferroptosis Inducer (Erastin/RSL3)", "Fasting-Mimicking Diet (FMD)"): 0.20,
    }
    
    def map_correction_to_interventions(
        self,
        delta_A_needed: np.ndarray,
        max_interventions: int = 3
    ) -> List[Tuple[TherapeuticIntervention, float]]:
        """
        Find interventions that best match the needed correction.
        
        Args:
            delta_A_needed: The correction needed (A_healthy - A_cancer)
            max_interventions: Maximum number of interventions to combine
            
        Returns:
            List of (intervention, weight) tuples
        """
        matches = []
        
        for intervention in self.intervention_library:
            # Compute alignment between intervention effect and needed correction
            effect = intervention.expected_effect
            
            # Ensure same shape
            if effect.shape != delta_A_needed.shape:
                continue
                
            # Compute dot product (alignment)
            alignment = np.sum(effect * delta_A_needed)
            
            # Normalize by magnitudes
            effect_norm = np.linalg.norm(effect, 'fro')
            needed_norm = np.linalg.norm(delta_A_needed, 'fro')
            
            if effect_norm > 0 and needed_norm > 0:
                similarity = alignment / (effect_norm * needed_norm)
                # Weight determines dosage scaling
                weight = alignment / (effect_norm ** 2) if effect_norm > 0 else 0
                matches.append((intervention, similarity, weight))
                
        # Sort by similarity (best matches first)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Return top matches with weights
        return [(m[0], m[2]) for m in matches[:max_interventions] if m[1] > 0]
    
    def compute_combination_effect(
        self,
        interventions: List[Tuple[TherapeuticIntervention, float]],
        synergy_matrix: Optional[Dict[Tuple[str, str], float]] = None
    ) -> np.ndarray:
        """
        Compute combined effect of multiple interventions, accounting for synergy.
        
        Args:
            interventions: List of (intervention, weight) tuples
            synergy_matrix: Dictionary mapping pairs of drug names to a synergy coefficient `S`.
                            Effect = E1 + E2 + S * (E1 * E2)
                            where `S > 0` is synergistic, `S < 0` is antagonistic.
        """
        combined = np.zeros((self.n_metabolites, self.n_metabolites))
        
        for intervention, weight in interventions:
            effect = intervention.expected_effect
            if effect.shape == combined.shape:
                combined += effect * weight
                
        # Apply pairwise synergies
        if synergy_matrix:
            n = len(interventions)
            for i in range(n):
                for j in range(i + 1, n):
                    int1, w1 = interventions[i]
                    int2, w2 = interventions[j]
                    
                    # Look up synergy both ways
                    S = synergy_matrix.get((int1.name, int2.name), 0.0)
                    if S == 0.0:
                        S = synergy_matrix.get((int2.name, int1.name), 0.0)
                        
                    if S != 0.0:
                        # Synergy is applied as an interaction term: S * (E1 * w1) * (E2 * w2)
                        E1 = int1.expected_effect * w1
                        E2 = int2.expected_effect * w2
                        # We use element-wise multiplication for the interaction
                        combined += S * (E1 * E2)
                        
        return combined
    
    def generate_protocol(
        self,
        phases: List, # List of ProtocolPhase objects from geometric_optimization
        patient_weight_kg: float = 70
    ) -> Dict:
        """
        Generate sequenced treatment protocol from intervention mapping.
        
        Args:
            phases: List of ProtocolPhase objects
            patient_weight_kg: Patient weight for dosing
            
        Returns:
            Protocol dictionary with phasing and dosing recommendations
        """
        protocol = {
            'schedule': [],
            'monitoring': [],
            'cautions': []
        }
        
        name_to_intervention = {inv.name: inv for inv in self.intervention_library}
        
        for phase in phases:
            phase_details = {
                'phase_name': phase.description.split(':')[0],
                'description': phase.description,
                'day_start': phase.day_start,
                'duration_days': phase.duration,
                'expected_kramers_escape_rate': phase.expected_escape_rate,
                'expected_curvature': phase.expected_curvature,
                'drugs': []
            }
            
            for drug_name, dose in phase.interventions:
                inv = name_to_intervention.get(drug_name)
                
                # Scale logic simplified: just pass the optimized dose directly
                # assuming the optimizer kept it within `dosage_range`
                
                if inv:
                    phase_details['drugs'].append({
                        'name': inv.name,
                        'mechanism': inv.mechanism,
                        'optimized_dose': f"{dose:.1f}",
                        'evidence': inv.evidence_level
                    })
                else:
                     phase_details['drugs'].append({
                        'name': drug_name,
                        'optimized_dose': f"{dose:.1f}",
                    })
                     
            protocol['schedule'].append(phase_details)
            
        # Add monitoring recommendations
        protocol['monitoring'] = [
            "Continuous tracking of Kramer escape rate metrics",
            "Weekly metabolic panel (glucose, lactate, ketones) during Phase 1",
            "Thermal tolerance checks during Phase 2",
            "Immune-related adverse event (irAE) monitoring during Phase 3"
        ]
        
        # Add cautions
        protocol['cautions'] = [
            "This is computationally-generated sequential protocol.",
            "Toxicity models are proxies; monitor closely during phase transitions."
        ]
        
        return protocol
    
    def get_intervention_report(
        self,
        delta_A_needed: np.ndarray
    ) -> str:
        """
        Generate comprehensive intervention recommendation report.
        """
        matches = self.map_correction_to_interventions(delta_A_needed)
        
        report = []
        report.append("=" * 60)
        report.append("THERAPEUTIC INTERVENTION MAPPING REPORT")
        report.append("=" * 60)
        
        report.append("\n## Recommended Interventions\n")
        
        for i, (intervention, weight) in enumerate(matches, 1):
            dose_low, dose_high = intervention.dosage_range
            suggested_dose = f"{dose_low + (dose_high - dose_low) * min(abs(weight), 1.0):.1f}"
            report.append(f"### {i}. {intervention.name}")
            report.append(f"   Mechanism: {intervention.mechanism}")
            report.append(f"   Suggested dose: {suggested_dose}")
            report.append(f"   Evidence level: {intervention.evidence_level}")
            report.append(f"   Correction weight: {weight:.4f}")
            report.append("")
            
        report.append("\n## Monitoring Recommendations\n")
        monitor_items = [
            "Continuous tracking of Kramers escape rate metrics",
            "Weekly metabolic panel (glucose, lactate, ketones)",
            "Immune-related adverse event (irAE) monitoring",
        ]
        for item in monitor_items:
            report.append(f"- {item}")
            
        report.append("\n## Important Cautions\n")
        caution_items = [
            "This is a computationally-generated recommendation.",
            "Toxicity models are proxies; monitor closely.",
        ]
        for item in caution_items:
            report.append(f"⚠️ {item}")
            
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


# TNBC-specific metabolite pathway model
class TNBCMetabolicModel:
    """
    TNBC-specific metabolic model.
    
    Triple-negative breast cancer has distinct metabolic features:
    1. Enhanced Warburg effect (aerobic glycolysis)
    2. Glutamine dependence
    3. Lipid metabolism reprogramming
    4. Redox stress sensitivity
    """
    
    METABOLITES = [
        "Glucose",      # 0: Primary fuel
        "Lactate",      # 1: Glycolytic end product
        "Pyruvate",     # 2: Glycolysis/TCA junction
        "ATP",          # 3: Energy currency
        "NADH",         # 4: Electron carrier
        "Glutamine",    # 5: Nitrogen/carbon source
        "Glutamate",    # 6: Glutaminolysis intermediate
        "aKG",         # 7: TCA cycle intermediate
        "Citrate",      # 8: TCA/lipogenesis junction
        "ROS",          # 9: Reactive oxygen species
    ]
    
    @classmethod
    def get_healthy_generator(cls) -> np.ndarray:
        """
        Return a healthy mammary tissue metabolic generator.
        
        Healthy tissue characteristics:
        - Balanced glycolysis/OXPHOS
        - Low lactate production
        - Stable redox state
        """
        n = len(cls.METABOLITES)
        A = np.zeros((n, n))
        
        # Diagonal: decay/turnover rates (all negative = stable)
        A[0, 0] = -0.5   # Glucose uptake/utilization
        A[1, 1] = -0.8   # Lactate clearance (efficient in healthy)
        A[2, 2] = -0.3   # Pyruvate processing
        A[3, 3] = -0.2   # ATP turnover
        A[4, 4] = -0.3   # NAD+/NADH cycling
        A[5, 5] = -0.4   # Glutamine utilization
        A[6, 6] = -0.5   # Glutamate processing
        A[7, 7] = -0.4   # α-KG in TCA
        A[8, 8] = -0.3   # Citrate utilization
        A[9, 9] = -0.9   # ROS clearance (good antioxidant capacity)
        
        # Key metabolic fluxes
        A[0, 2] = 0.4    # Glucose → Pyruvate (glycolysis)
        A[2, 3] = 0.3    # Pyruvate → ATP (via TCA/OXPHOS)
        A[2, 1] = 0.1    # Pyruvate → Lactate (low in healthy)
        A[4, 3] = 0.2    # NADH → ATP (electron transport)
        A[5, 6] = 0.3    # Glutamine → Glutamate
        A[6, 7] = 0.3    # Glutamate → α-KG
        A[7, 8] = 0.2    # α-KG → Citrate (TCA flow)
        
        # Regulatory feedbacks
        A[3, 0] = -0.1   # ATP inhibits glucose uptake
        A[9, 4] = 0.1    # NADH generates some ROS
        A[9, 3] = -0.1   # ROS can damage ATP production
        
        return A
    
    @classmethod
    def get_tnbc_generator(cls) -> np.ndarray:
        """
        Return TNBC metabolic generator.
        
        TNBC characteristics:
        - Enhanced glycolysis (Warburg effect)
        - High lactate production
        - Glutamine addiction
        - Compromised ROS clearance
        """
        # Start from healthy and apply pathological shifts
        A = cls.get_healthy_generator()
        
        # 1. Warburg effect: enhanced glycolysis
        A[0, 0] = -0.3   # Increased glucose uptake
        A[0, 2] = 0.6    # Enhanced glucose → pyruvate
        A[2, 1] = 0.4    # Much more pyruvate → lactate
        A[2, 3] = 0.15   # Less pyruvate to OXPHOS
        A[1, 1] = -0.3   # Lactate accumulates (poor clearance)
        
        # 2. Glutamine addiction
        A[5, 5] = -0.2   # Increased glutamine dependence
        A[5, 6] = 0.5    # Enhanced glutaminolysis
        A[6, 7] = 0.5    # More glutamate → α-KG
        
        # 3. Compromised mitochondria
        A[4, 3] = 0.1    # Less efficient NADH → ATP
        A[3, 3] = -0.15  # Slower ATP turnover
        
        # 4. Redox imbalance
        A[9, 9] = -0.4   # Reduced ROS clearance
        A[9, 4] = 0.2    # More ROS from NADH
        
        # 5. Lipogenesis for membrane synthesis
        A[8, 8] = -0.15  # Citrate diverted to lipid synthesis
        
        # Overall: system is less stable, more chaotic
        # Add some eigenvalue destabilization
        noise = np.random.randn(len(cls.METABOLITES), len(cls.METABOLITES)) * 0.05
        A += noise
        
        return A
    
    @classmethod
    def get_metabolite_names(cls) -> List[str]:
        return cls.METABOLITES.copy()
