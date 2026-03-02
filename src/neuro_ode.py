"""
Neurodegeneration ODE System — Project Confluence Extension
=============================================================

10-dimensional neuro-proteomic generator system for neurodegenerative diseases.
Applies SAEM geometric alignment framework to Alzheimer's, Parkinson's, and ALS.

Key Axes (10D State Space):
  0. Amyloid      — Amyloid-β42 aggregation (AD primary)
  1. Tau          — Phospho-tau / neurofibrillary tangles
  2. Alpha_syn    — α-synuclein aggregation (PD primary)
  3. Dopamine     — Dopaminergic neurotransmission (PD)
  4. Glutamate    — Excitotoxicity (ALS primary, all neurodegeneration)
  5. Mito         — Mitochondrial function (bioenergetic failure)
  6. Neuroinflam  — Microglial activation / neuroinflammation
  7. BDNF         — Brain-derived neurotrophic factor (neuroprotection)
  8. Autophagy    — Autophagy/proteasome flux (protein clearance)
  9. Synaptic     — Synaptic density / connectivity (functional output)

Key References:
  - Jack et al. 2018, Alzheimer's & Dementia (AT(N) framework)
  - Bloem et al. 2021, Lancet (Parkinson's disease)
  - Brown & Al-Chalabi 2017, NEJM (ALS)
  - van Dyck et al. 2023, NEJM (lecanemab — CLARITY AD)
  - Pagano et al. 2024, NEJM (prasinezumab — PASADENA)
"""

import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass


NEURO_AXES = [
    "Amyloid", "Tau", "Alpha_syn", "Dopamine", "Glutamate",
    "Mito", "Neuroinflam", "BDNF", "Autophagy", "Synaptic",
]


@dataclass
class NeuroParams:
    """Tunable parameters for neurodegeneration ODE system."""
    # Protein clearance
    amyloid_clearance: float = -0.50     # Glymphatic + enzymatic clearance
    tau_clearance: float = -0.45         # Proteasomal degradation
    alphasyn_clearance: float = -0.45    # Lysosomal/autophagy clearance

    # Neurotransmission
    dopamine_turnover: float = -0.40     # MAO-B + COMT degradation
    glutamate_clearance: float = -0.55   # Astrocyte uptake (EAAT2)

    # Cellular health
    mito_homeostasis: float = -0.35      # Mitophagy + biogenesis balance
    neuroinflam_resolution: float = -0.50 # Microglial resolution
    bdnf_turnover: float = -0.30         # BDNF production/degradation
    autophagy_rate: float = -0.40        # Basal autophagy flux
    synaptic_maintenance: float = -0.25  # Synaptic turnover/pruning


@dataclass
class NeuroMetadata:
    """Metadata for a neurodegeneration generator."""
    subtype: str
    description: str
    risk_factors: List[str]
    complications: List[str]
    mortality_rate: str
    prevalence: str
    confidence: str = "high"


NEURO_METADATA: Dict[str, NeuroMetadata] = {
    "Healthy_Aging": NeuroMetadata(
        subtype="Healthy_Aging",
        description="Normal aging with preserved cognitive and motor function",
        risk_factors=[],
        complications=["Mild cognitive slowing (normal)"],
        mortality_rate="Baseline",
        prevalence="~70% of adults over 65",
    ),
    "MCI_Amyloid": NeuroMetadata(
        subtype="MCI_Amyloid",
        description="Mild cognitive impairment with amyloid positivity. Prodromal AD.",
        risk_factors=["APOE4", "Age >65", "Family history", "Cardiovascular risk"],
        complications=["~15% per year convert to dementia"],
        mortality_rate="2x conversion to dementia within 5 years",
        prevalence="~15-20% of adults over 65",
    ),
    "Alzheimers": NeuroMetadata(
        subtype="Alzheimers",
        description="Alzheimer's disease: amyloid plaques + tau tangles + synaptic loss",
        risk_factors=["APOE4 homozygosity", "Age", "Female sex", "Head trauma", "Low education"],
        complications=["Progressive dementia", "Behavioral changes", "Loss of independence", "Aspiration pneumonia"],
        mortality_rate="3-4x all-cause mortality; median survival 4-8 years from diagnosis",
        prevalence="~6.7 million in US; #5 cause of death >65",
    ),
    "Parkinsons": NeuroMetadata(
        subtype="Parkinsons",
        description="Parkinson's disease: α-synuclein aggregation + dopaminergic neuron loss",
        risk_factors=["Age >60", "Male sex", "Pesticide exposure", "LRRK2/GBA mutations"],
        complications=["Motor: tremor, bradykinesia, rigidity", "Non-motor: dementia, depression, autonomic failure"],
        mortality_rate="2x all-cause mortality; life expectancy reduced 5-10 years",
        prevalence="~1 million in US; second most common neurodegeneration",
    ),
    "ALS": NeuroMetadata(
        subtype="ALS",
        description="Amyotrophic lateral sclerosis: motor neuron degeneration via excitotoxicity + proteostasis failure",
        risk_factors=["Age 40-70", "Male sex", "SOD1/C9orf72 mutations", "Military service"],
        complications=["Progressive paralysis", "Respiratory failure", "Dysphagia", "Frontotemporal dementia (15%)"],
        mortality_rate="Median survival 2-5 years from symptom onset; respiratory failure #1 cause",
        prevalence="~30,000 in US; incidence 2/100,000",
    ),
}


class NeurodegenerationODESystem:
    """
    10-dimensional neuro-proteomic generator system.

    Applies the same SAEM geometric framework used for cancer, diabetes, and CVD:
    - Generator matrices define attractor basins
    - Disease = trapped in pathological attractor (protein aggregation + neuronal loss)
    - Cure = flatten basin → add noise → push toward healthy attractor
    """

    N = 10

    @classmethod
    def healthy_aging_generator(cls, params: Optional[NeuroParams] = None) -> np.ndarray:
        """
        Healthy aging: balanced protein clearance, intact synapses, active neuroprotection.

        Ref: Jack et al. 2018 (AT(N) framework — amyloid/tau/neurodegeneration)
        """
        p = params or NeuroParams()
        n = cls.N
        A = np.zeros((n, n))

        # ── Diagonal: clearance/turnover ──
        A[0, 0] = p.amyloid_clearance        # Amyloid clearance (glymphatic)
        A[1, 1] = p.tau_clearance             # Tau degradation
        A[2, 2] = p.alphasyn_clearance        # α-syn clearance
        A[3, 3] = p.dopamine_turnover         # DA turnover (MAO-B)
        A[4, 4] = p.glutamate_clearance       # Glutamate reuptake (EAAT2)
        A[5, 5] = p.mito_homeostasis          # Mitochondrial balance
        A[6, 6] = p.neuroinflam_resolution    # Microglial homeostasis
        A[7, 7] = p.bdnf_turnover             # BDNF turnover
        A[8, 8] = p.autophagy_rate            # Autophagy flux
        A[9, 9] = p.synaptic_maintenance      # Synaptic maintenance

        # ── Protein clearance couplings ──
        A[8, 0] = -0.20     # Autophagy clears amyloid
        A[8, 1] = -0.18     # Autophagy clears tau
        A[8, 2] = -0.20     # Autophagy clears α-synuclein

        # ── Neuroprotection axis ──
        A[7, 9] = 0.25      # BDNF maintains synaptic density
        A[7, 5] = 0.15      # BDNF supports mitochondrial biogenesis
        A[5, 9] = 0.18      # Healthy mitochondria support synaptic function

        # ── Toxicity couplings ──
        A[0, 6] = 0.10      # Amyloid activates microglia (low level = phagocytosis)
        A[4, 9] = -0.12     # Excess glutamate damages synapses (excitotoxicity)
        A[6, 9] = -0.08     # Inflammation damages synapses

        # ── Neurotransmission ──
        A[3, 9] = 0.15      # Dopamine required for motor circuits
        A[5, 3] = 0.12      # Mitochondria power dopamine synthesis

        # ── Homeostatic feedback ──
        A[0, 8] = 0.10      # Amyloid accumulation boosts autophagy response
        A[6, 7] = -0.08     # Mild inflammation reduces BDNF

        return A

    @classmethod
    def mci_amyloid_generator(cls, params: Optional[NeuroParams] = None) -> np.ndarray:
        """
        MCI with amyloid positivity: prodromal Alzheimer's. Shallow basin — potentially modifiable.

        Ref: Hansson et al. 2006 Lancet (CSF biomarkers predict AD)
        """
        A = cls.healthy_aging_generator(params)

        # Early amyloid accumulation
        A[0, 0] = -0.32     # Reduced amyloid clearance
        A[0, 6] = 0.18      # More amyloid → microglial activation
        A[0, 1] = 0.08      # Early amyloid → tau phosphorylation

        # Autophagy beginning to fail
        A[8, 8] = -0.32     # Slightly impaired autophagy
        A[8, 0] = -0.14     # Reduced amyloid clearance via autophagy

        # Mild synaptic weakening
        A[9, 9] = -0.28     # Early synaptic loss
        A[7, 9] = 0.18      # BDNF still partially protective

        # Mild neuroinflammation
        A[6, 6] = -0.42     # Microglia mildly overactivated

        return A

    @classmethod
    def alzheimers_generator(cls, params: Optional[NeuroParams] = None) -> np.ndarray:
        """
        Alzheimer's disease: amyloid cascade + tau propagation + synaptic devastation.

        Deep attractor. Current therapies (lecanemab) slow but don't reverse.

        Ref: van Dyck et al. 2023, NEJM (CLARITY AD — lecanemab);
             Sevigny et al. 2016, Nature (aducanumab);
             Jack et al. 2018 (AT(N) biomarker framework)
        """
        A = cls.healthy_aging_generator(params)

        # Severe amyloid accumulation
        A[0, 0] = -0.20     # Amyloid clearance overwhelmed
        A[0, 6] = 0.30      # Amyloid → chronic microglial activation
        A[0, 1] = 0.22      # Amyloid → tau phosphorylation (downstream)
        A[0, 9] = -0.15     # Direct synaptic toxicity of Aβ oligomers

        # Tau propagation (prion-like spreading)
        A[1, 1] = -0.22     # Tau clearance failing
        A[1, 9] = -0.25     # Tau tangles → neuronal death → synapse loss
        A[1, 5] = -0.15     # Tau damages mitochondria

        # Autophagy/proteasome failure
        A[8, 8] = -0.22     # Autophagy overwhelmed by aggregates
        A[8, 0] = -0.08     # Near-zero amyloid clearance
        A[8, 1] = -0.08     # Near-zero tau clearance

        # Neuroinflammation (M1 dominant)
        A[6, 6] = -0.25     # Chronic microglial overactivation
        A[6, 9] = -0.20     # Inflammation → synapse stripping
        A[6, 7] = -0.18     # Inflammation suppresses BDNF

        # BDNF collapse
        A[7, 7] = -0.42     # BDNF dramatically reduced
        A[7, 9] = 0.08      # BDNF support for synapses nearly gone

        # Synaptic devastation (correlates with cognitive decline)
        A[9, 9] = -0.40     # Massive synaptic loss
        A[5, 9] = 0.05      # Minimal mitochondrial support

        # Mitochondrial failure
        A[5, 5] = -0.45     # Bioenergetic crisis
        A[5, 3] = 0.05      # Dopamine synthesis impaired

        # Glutamate excitotoxicity
        A[4, 4] = -0.38     # Impaired glutamate uptake
        A[4, 9] = -0.18     # Excitotoxic synapse damage

        return A

    @classmethod
    def parkinsons_generator(cls, params: Optional[NeuroParams] = None) -> np.ndarray:
        """
        Parkinson's disease: α-synuclein aggregation + dopaminergic neuron loss.

        Deep attractor dominated by the dopamine-mito-α-syn triangle.

        Ref: Bloem et al. 2021, Lancet (PD review);
             Pagano et al. 2024, NEJM (prasinezumab);
             Devos et al. 2022, NEJM (deferiprone — iron chelation trial)
        """
        A = cls.healthy_aging_generator(params)

        # α-synuclein aggregation (primary driver)
        A[2, 2] = -0.20     # α-syn clearance overwhelmed (Lewy bodies)
        A[2, 5] = -0.20     # α-syn damages mitochondria (Complex I)
        A[2, 3] = -0.22     # α-syn → dopaminergic neuron death
        A[2, 8] = -0.10     # α-syn aggregates impair autophagy

        # Dopaminergic collapse
        A[3, 3] = -0.55     # Massive DA neuron loss (>60% at diagnosis)
        A[3, 9] = 0.08      # Minimal DA contribution to motor circuits
        A[5, 3] = 0.05      # Mito barely supporting DA synthesis

        # Mitochondrial dysfunction (Complex I)
        A[5, 5] = -0.48     # Severe mito dysfunction
        A[5, 9] = 0.08      # Weakened mito-synaptic support

        # Autophagy failure
        A[8, 8] = -0.28     # Impaired autophagy (GBA mutations worsen)
        A[8, 2] = -0.10     # Poor α-syn clearance

        # Neuroinflammation
        A[6, 6] = -0.32     # Chronic microglial activation
        A[6, 7] = -0.15     # Inflammation suppresses BDNF
        A[2, 6] = 0.18      # α-syn activates microglia

        # Synaptic dysfunction (motor circuits)
        A[9, 9] = -0.32     # Synaptic loss in striatum
        A[7, 7] = -0.38     # BDNF reduced

        # Glutamate contribution
        A[4, 4] = -0.42     # Mild excitotoxicity
        A[4, 9] = -0.12     # Contributes to synaptic loss

        return A

    @classmethod
    def als_generator(cls, params: Optional[NeuroParams] = None) -> np.ndarray:
        """
        ALS: motor neuron degeneration via excitotoxicity + proteostasis failure.

        Deepest neurodegeneration attractor — analogous to PDAC in cancer.
        Rapid progression, limited therapies.

        Ref: Brown & Al-Chalabi 2017, NEJM (ALS review);
             Miller et al. 2022, NEJM (tofersen for SOD1-ALS);
             Lacomblez et al. 1996, Lancet (riluzole)
        """
        A = cls.healthy_aging_generator(params)

        # Massive glutamate excitotoxicity (primary driver)
        A[4, 4] = -0.20     # EAAT2 failure → glutamate accumulation
        A[4, 9] = -0.30     # Excitotoxic motor neuron death
        A[4, 5] = -0.20     # Glutamate → calcium overload → mito damage

        # Proteostasis collapse (TDP-43, SOD1, FUS aggregation)
        A[8, 8] = -0.18     # Autophagy overwhelmed
        A[1, 1] = -0.25     # TDP-43 inclusions (mapped to tau axis)
        A[1, 9] = -0.20     # Protein aggregates → motor neuron death

        # Mitochondrial failure
        A[5, 5] = -0.50     # Severe bioenergetic failure
        A[5, 9] = 0.03      # Near-zero metabolic support for synapses

        # Severe neuroinflammation
        A[6, 6] = -0.22     # Chronic microglial/astrocyte activation
        A[6, 9] = -0.22     # Inflammatory motor neuron damage
        A[6, 7] = -0.20     # Inflammation destroys BDNF signaling

        # Synaptic/motor neuron loss (rapid)
        A[9, 9] = -0.50     # Fastest synaptic loss of any ND
        A[7, 7] = -0.45     # BDNF severely depleted

        # Motor circuit collapse
        A[3, 3] = -0.45     # Neurotransmitter systems failing
        A[2, 2] = -0.30     # Some α-syn co-pathology

        return A

    @classmethod
    def all_generators(cls) -> Dict[str, np.ndarray]:
        """Return all neurodegeneration generator matrices."""
        return {
            "Healthy_Aging": cls.healthy_aging_generator(),
            "MCI_Amyloid": cls.mci_amyloid_generator(),
            "Alzheimers": cls.alzheimers_generator(),
            "Parkinsons": cls.parkinsons_generator(),
            "ALS": cls.als_generator(),
        }


# ═══════════════════════════════════════════════════════════════
# NEURODEGENERATION THERAPEUTIC INTERVENTIONS
# ═══════════════════════════════════════════════════════════════

@dataclass
class NeuroIntervention:
    """A neurodegeneration therapeutic intervention mapped to generator correction."""
    name: str
    mechanism: str
    expected_effect: np.ndarray   # 10×10 δA correction
    category: str
    evidence_level: str
    references: List[str]
    curvature_multiplier: float = 1.0
    fatality_reduction: str = ""


def build_neuro_drug_library(n: int = 10) -> List[NeuroIntervention]:
    """Build library of neurodegeneration interventions mapped to generator corrections."""
    interventions = []

    # ────────────────────────────────────────────────────────
    # 1. LECANEMAB (Anti-amyloid monoclonal antibody — AD)
    # Ref: van Dyck et al. 2023, NEJM (CLARITY AD — 27% slowing of decline)
    # ────────────────────────────────────────────────────────
    lecan = np.zeros((n, n))
    lecan[0, 0] = -0.25     # Major amyloid removal (protofibrils)
    lecan[0, 6] = -0.08     # Reduced amyloid-driven inflammation
    lecan[0, 1] = -0.05     # Downstream tau reduction (modest)
    lecan[0, 9] = 0.05      # Reduced synaptic toxicity from Aβ oligomers
    interventions.append(NeuroIntervention(
        name="Lecanemab (Anti-Aβ mAb)",
        mechanism="Selective binding of Aβ protofibrils → amyloid clearance via Fc-mediated phagocytosis",
        expected_effect=lecan,
        category="anti-amyloid",
        evidence_level="FDA Approved (2023)",
        references=[
            "CLARITY AD 2023 NEJM (27% slowing of cognitive decline at 18 months)",
            "Amyloid PET reduction: 59 centiloids mean reduction",
        ],
        curvature_multiplier=0.78,
        fatality_reduction="27% slowing of CDR-SB decline; first anti-amyloid with clear clinical benefit",
    ))

    # ────────────────────────────────────────────────────────
    # 2. LEVODOPA/CARBIDOPA (Dopamine replacement — PD)
    # Ref: Fahn 2006, J Neural Transm (50+ years of L-DOPA)
    # ────────────────────────────────────────────────────────
    ldopa = np.zeros((n, n))
    ldopa[3, 3] = 0.30       # Direct dopamine restoration
    ldopa[3, 9] = 0.15       # Restored motor circuit signaling
    ldopa[9, 9] = 0.05       # Mild synaptic functional improvement
    interventions.append(NeuroIntervention(
        name="Levodopa/Carbidopa",
        mechanism="Exogenous DA precursor → decarboxylation to dopamine in remaining DA neurons",
        expected_effect=ldopa,
        category="dopamine_replacement",
        evidence_level="Gold Standard",
        references=[
            "Fahn 2006, J Neural Transm (L-DOPA efficacy over 5 decades)",
            "PD MED 2014, Lancet Neurol (initial therapy comparison)",
        ],
        curvature_multiplier=0.55,
        fatality_reduction="Symptomatic — does not modify disease course but transforms quality of life; motor UPDRS improvement 30-50%",
    ))

    # ────────────────────────────────────────────────────────
    # 3. RILUZOLE (Anti-excitotoxicity — ALS)
    # Ref: Lacomblez et al. 1996, Lancet (2-3 month survival extension)
    # ────────────────────────────────────────────────────────
    riluz = np.zeros((n, n))
    riluz[4, 4] = -0.15      # Reduced glutamate release (Na+ channel block)
    riluz[4, 9] = 0.08       # Reduced excitotoxic synapse damage
    riluz[4, 5] = 0.05       # Reduced calcium-mediated mito damage
    interventions.append(NeuroIntervention(
        name="Riluzole",
        mechanism="Glutamate release inhibition (Na+ channel block) + post-synaptic NMDA modulation",
        expected_effect=riluz,
        category="anti-excitotoxic",
        evidence_level="Clinical Standard",
        references=[
            "Lacomblez et al. 1996, Lancet (2-3 month survival benefit)",
            "Miller et al. 2012, Cochrane Review (ALS therapeutics)",
        ],
        curvature_multiplier=0.88,
        fatality_reduction="Modest: 2-3 month median survival extension; only FDA-approved ALS drug for decades",
    ))

    # ────────────────────────────────────────────────────────
    # 4. MEMANTINE (NMDA antagonist — AD)
    # Ref: Reisberg et al. 2003, NEJM (moderate-severe AD)
    # ────────────────────────────────────────────────────────
    meman = np.zeros((n, n))
    meman[4, 4] = -0.12      # Reduced excitotoxicity (NMDA block)
    meman[4, 9] = 0.08       # Reduced synaptic damage
    meman[9, 9] = 0.03       # Mild synaptic protection
    interventions.append(NeuroIntervention(
        name="Memantine",
        mechanism="Low-affinity NMDA receptor antagonist → reduced tonic glutamate excitotoxicity",
        expected_effect=meman,
        category="anti-excitotoxic",
        evidence_level="Clinical Standard",
        references=[
            "Reisberg et al. 2003, NEJM (significant benefit in moderate-severe AD)",
            "Tariot et al. 2004, JAMA (memantine + donepezil combination)",
        ],
        curvature_multiplier=0.82,
        fatality_reduction="Symptomatic benefit (ADCS-ADL, SIB); NNT 6-12 for clinically observable improvement",
    ))

    # ────────────────────────────────────────────────────────
    # 5. TOFERSEN (ASO — SOD1-ALS)
    # Ref: Miller et al. 2022, NEJM (VALOR trial)
    # ────────────────────────────────────────────────────────
    tofer = np.zeros((n, n))
    tofer[1, 1] = -0.20      # Reduces SOD1/TDP-43 protein (mapped to tau axis)
    tofer[1, 9] = 0.10       # Reduced motor neuron death from aggregates
    tofer[8, 8] = 0.05       # Slightly improved proteostasis
    tofer[6, 6] = -0.05      # Reduced aggregate-driven inflammation
    interventions.append(NeuroIntervention(
        name="Tofersen (Anti-SOD1 ASO)",
        mechanism="Antisense oligonucleotide → SOD1 mRNA degradation → reduced misfolded SOD1 protein",
        expected_effect=tofer,
        category="gene_therapy",
        evidence_level="FDA Approved (2023, accelerated)",
        references=[
            "Miller et al. 2022, NEJM (VALOR — 35% reduction in plasma NfL)",
            "Open-label extension: functional stabilization in early-treated patients",
        ],
        curvature_multiplier=0.72,
        fatality_reduction="35% NfL reduction (neuronal damage biomarker); disease stabilization in early treatment",
    ))

    # ────────────────────────────────────────────────────────
    # 6. EXERCISE + COGNITIVE TRAINING (Lifestyle)
    # Ref: Livingston et al. 2020, Lancet (dementia prevention)
    # ────────────────────────────────────────────────────────
    life = np.zeros((n, n))
    life[7, 7] = 0.15        # Exercise increases BDNF
    life[5, 5] = -0.10       # Improved mitochondrial biogenesis
    life[6, 6] = -0.10       # Anti-inflammatory (exercise)
    life[9, 9] = 0.08        # Synaptic maintenance (cognitive reserve)
    life[8, 8] = -0.08       # Enhanced autophagy (exercise-induced)
    interventions.append(NeuroIntervention(
        name="Exercise + Cognitive Training",
        mechanism="Multi-pathway: BDNF upregulation + mitophagy + anti-inflammation + cognitive reserve building",
        expected_effect=life,
        category="lifestyle",
        evidence_level="Meta-analytic",
        references=[
            "Livingston et al. 2020, Lancet (40% of dementia cases potentially preventable)",
            "Erickson et al. 2011, PNAS (exercise increases hippocampal volume)",
            "FINGER trial 2015, Lancet (multi-domain intervention — 25% cognitive improvement)",
        ],
        curvature_multiplier=0.60,
        fatality_reduction="Up to 40% of dementia cases potentially preventable through modifiable risk factors",
    ))

    return interventions
