"""Pydantic contracts shared across Confluence v2."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


LATENT_NAMES = (
    "T_s",
    "T_r",
    "I_act",
    "I_exh",
    "S_fib",
    "L",
    "O",
    "G",
    "C_tgfb",
    "C_ifng",
    "H",
    "T_f",
)

CONTROL_DRUG_IDS = (
    "anti_pd1",
    "tgfb_inhibitor",
    "mct1",
    "hdac",
    "targeted_kinase",
)

# Simulated therapeutic protein / biologic infusion channels.
# These are PK/PD effectors (expression-or-infusion rates), not a claim
# that fly neurons translate polypeptides.
PROTEIN_CHANNEL_IDS = (
    "protein_anti_pd1",
    "protein_tgfb_trap",
    "protein_ifng",
    "protein_il2",
)

# Fusion-directed TKI class channels. Kill terms prefer the T_f clone.
# Catalog PK is class-reference (imatinib / ALK-inhibitor), not a regimen.
FUSION_CHANNEL_IDS = (
    "tki_imatinib_like",
    "tki_alk",
)

ALL_EFFECTOR_IDS = CONTROL_DRUG_IDS + PROTEIN_CHANNEL_IDS + FUSION_CHANNEL_IDS

# Y(t) sensory vector. First five channels stay the original cancer readout;
# last two are chimeric-junction / fusion-AF proxies (research simulation).
OBS_VECTOR_NAMES = (
    "tumor_burden",
    "resistance_frequency",
    "lactate",
    "tgfb",
    "immune_competence_ratio",
    "fusion_allele_fraction",
    "junction_neoantigen",
)

# FlyWire-scale class size named by the user. Published adult FlyWire
# reconstructions are the same order (~1e5 neurons); this repo uses a
# sparse structured stub at exactly this N unless a real dump is loaded.
FULL_BRAIN_NEURONS = 166700
INTERACTIVE_BRAIN_NEURONS = 2048
DEMO_BRAIN_NEURONS = 256


class LatentCancerState(BaseModel):
    """Latent microenvironment state X ∈ R^12 (11-D TME + fusion clone)."""

    model_config = ConfigDict(extra="forbid")

    T_s: float = Field(..., description="Drug-sensitive tumor burden")
    T_r: float = Field(..., description="Drug-resistant tumor burden")
    I_act: float = Field(..., description="Active cytotoxic immune compartment")
    I_exh: float = Field(..., description="Exhausted immune compartment")
    S_fib: float = Field(..., description="Stromal fibrosis / CAF density")
    L: float = Field(..., description="Lactate")
    O: float = Field(..., description="Oxygen")
    G: float = Field(..., description="Glucose")
    C_tgfb: float = Field(..., description="TGF-β cytokine")
    C_ifng: float = Field(..., description="IFN-γ cytokine")
    H: float = Field(..., ge=0.0, le=1.0, description="Host health in [0, 1]")
    T_f: float = Field(
        0.0,
        description="Fusion-oncoprotein clone (chimeric-driver–positive)",
    )
    t: float = Field(0.0, description="Simulation time (days)")
    fusion_id: str = Field(
        "fusion_oncoprotein",
        description="Archetype fusion class id (research label, not a genotype)",
    )
    fusion_display: str = Field(
        "",
        description="Human-readable fusion class (e.g. FGFR3–TACC3-like)",
    )

    def as_vector(self) -> List[float]:
        return [getattr(self, name) for name in LATENT_NAMES]

    @classmethod
    def from_vector(cls, x, t: float = 0.0) -> "LatentCancerState":
        values = {}
        for i, name in enumerate(LATENT_NAMES):
            values[name] = float(x[i]) if i < len(x) else 0.0
        values["t"] = float(t)
        return cls(**values)

    @property
    def tumor_burden(self) -> float:
        return max(0.0, self.T_s + self.T_r + self.T_f)

    @property
    def resistance_frequency(self) -> float:
        burden = self.tumor_burden
        if burden <= 1e-12:
            return 0.0
        return self.T_r / burden

    @property
    def fusion_allele_fraction(self) -> float:
        """Fusion+ share of tumor — ctDNA-like fusion AF (latent, noiseless)."""
        burden = self.tumor_burden
        if burden <= 1e-12:
            return 0.0
        return max(0.0, self.T_f) / burden

    @property
    def junction_neoantigen(self) -> float:
        """Chimeric junction peptide / mRNA pool proxy (∝ T_f)."""
        return max(0.0, self.T_f) * 0.85

    @property
    def terminal_toxicity(self) -> bool:
        return self.H <= 0.2


class ObservationRecord(BaseModel):
    """Noisy, partial observation Y of the latent state."""

    model_config = ConfigDict(extra="forbid")

    t: float
    tumor_burden: float
    resistance_frequency: float = Field(..., description="ctDNA-like resistant fraction")
    lactate: float
    tgfb: float
    immune_competence_ratio: float
    fusion_allele_fraction: float = Field(
        0.0,
        description="Noisy ctDNA-like fusion allele fraction (research proxy, not a clinical NGS assay)",
    )
    junction_neoantigen: float = Field(
        0.0,
        description="Noisy chimeric-junction neoantigen / fusion-transcript proxy",
    )
    fusion_id: str = Field("fusion_oncoprotein", description="Archetype fusion class id")
    host_toxicity_warning: bool = False
    host_health: Optional[float] = Field(
        None, description="Optional privileged readout (not used by default controller)"
    )

    def as_vector(self) -> List[float]:
        return [float(getattr(self, name)) for name in OBS_VECTOR_NAMES]


class OrganToxicityCoeffs(BaseModel):
    liver: float = 0.0
    kidney: float = 0.0
    heart: float = 0.0
    marrow: float = 0.0
    immune: float = 0.0
    gut: float = 0.0
    nerve: float = 0.0


class DrugProvenance(BaseModel):
    doi: str
    note: str
    placeholder: bool = False


class DrugSpecification(BaseModel):
    """Pharmacology catalog entry used by the PK/PD layer."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    drug_class: str
    mechanism: str
    half_life_hours: float = Field(..., gt=0)
    ic50: float = Field(..., gt=0, description="Simulation-scaled IC50")
    hill: float = Field(..., gt=0)
    mtd: float = Field(..., gt=0, description="Maximum tolerated infusion (sim units)")
    organ_toxicity: OrganToxicityCoeffs
    control_channel: Optional[str] = Field(
        None, description="Maps catalog drug onto the 5-D control vector if applicable"
    )
    clinical_half_life_hours: Optional[float] = None
    clinical_ic50_note: Optional[str] = None
    modality: str = Field(
        "small_molecule",
        description="small_molecule or protein_biologic (simulated infusion/expression)",
    )
    tox_weight: float = Field(
        1.0,
        ge=0.0,
        description="Multiplier on C/MTD in host-health toxicity load",
    )
    provenance: DrugProvenance


class InterventionAction(BaseModel):
    """Infusion command U(t) produced by a controller or a human override."""

    model_config = ConfigDict(extra="forbid")

    t: float
    infusion: Dict[str, float]
    source: str
    clipped: bool = False
    notes: str = ""

    def vector(self, drug_ids=CONTROL_DRUG_IDS) -> List[float]:
        return [float(self.infusion.get(drug_id, 0.0)) for drug_id in drug_ids]


class FlyWireNeuron(BaseModel):
    """Neuron metadata aligned with FlyWire FAFB v783 fields."""

    model_config = ConfigDict(extra="allow")

    root_id: int
    cell_class: str
    nt_type: str
    proofread: bool = False
    supervoxel_id: Optional[int] = None
    soma_x: Optional[float] = None
    soma_y: Optional[float] = None
    soma_z: Optional[float] = None
    hemibrain_type: Optional[str] = None
    flow: Optional[str] = None
    side: Optional[str] = None


class SynapseEdge(BaseModel):
    pre_root_id: int
    post_root_id: int
    weight: float
    nt_type: str
    sign: int = Field(..., description="+1 excitatory, -1 inhibitory")
    neuropil: str = "MB"


class ConnectomeSubcircuit(BaseModel):
    """Extracted mushroom-body subcircuit (stub or FlyWire_FAFB_v783)."""

    model_config = ConfigDict(extra="forbid")

    version: str = "FlyWire_FAFB_v783"
    source: str = Field(..., description="'stub' or 'flywire'")
    n_pn: int
    n_kc: int
    n_mbon: int
    n_dan: int
    n_apl: int = 1
    neurons: List[FlyWireNeuron]
    edges: List[SynapseEdge]
    notes: str = ""
    cave_datastack: str = "flywire_fafb_production"
    materialization_version: Optional[int] = None
