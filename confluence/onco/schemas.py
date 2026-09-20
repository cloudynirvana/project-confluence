"""Pydantic contracts for OnCo refs and Confluence evidence objects."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from confluence.onco.attribution import ATTRIBUTION


Kind = str
Sign = Literal["increases", "decreases", "nonmonotonic", "unknown"]
EvidenceLevel = Literal[
    "concept",
    "in_silico",
    "cell_line",
    "organoid",
    "pdx",
    "animal",
    "clinical",
    "mixed",
    "unknown",
]
ClaimStatus = Literal["candidate", "supported", "mixed", "rejected", "wired"]
HypothesisStatus = Literal["open", "testable", "testing", "supported", "killed"]
Rung = Literal["in_silico", "cell_line", "organoid", "pdx", "animal", "clinical"]
AppliesTo = Literal["transition", "observation", "control", "translation"]
Layer = Literal["knowledge", "evidence", "mechanism", "parameter", "prediction"]
ModelId = Literal[
    "tnbc_mod_3s",
    "confluence_report_6s",
    "confluence_v2_15d",
    "confluence_v1_calibrator",
]
Provenance = Literal[
    "assumed", "identified", "transferred", "literature_point", "forbidden"
]
Identifiability = Literal[
    "unidentified", "structural", "practical", "not_applicable"
]


class OncoRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    onco_id: str
    kind: str
    name: str
    route: Optional[str] = None
    tldr: Optional[str] = None
    onco_as_of: Optional[str] = None
    retrieved_at: str
    onco_build: Optional[str] = None
    source_uri: Optional[str] = None
    content_hash: Optional[str] = None
    attribution: str = Field(default=ATTRIBUTION)


class OncoEvidenceCandidate(BaseModel):
    """Preserves OnCo semantics. Does not reinterpret confidence as P(H)."""

    model_config = ConfigDict(extra="forbid")

    ref: OncoRef
    cancer_ids: List[str] = Field(default_factory=list)
    target_ids: List[str] = Field(default_factory=list)
    mechanism_text: Optional[str] = None
    evidence_links: List[str] = Field(default_factory=list)
    onco_status: Optional[str] = None
    onco_as_of: Optional[str] = None
    onco_confidence: Optional[float] = None
    onco_idea_maturity: Optional[str] = None
    raw_record_hash: Optional[str] = None
    attribution: str = Field(default=ATTRIBUTION)


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["onco", "doi", "nct", "pmid", "url"]
    uri: str
    onco_id: Optional[str] = None


class EvidenceObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    claim: str
    mechanism_family: str
    cancer_onco_ids: List[str] = Field(default_factory=list)
    target_onco_ids: List[str] = Field(default_factory=list)
    intervention_slot: Optional[str] = None
    state_from: List[str] = Field(default_factory=list)
    state_to: List[str] = Field(default_factory=list)
    sign: Sign = "unknown"
    context_gate: Optional[str] = None
    evidence_level: EvidenceLevel = "unknown"
    effect_size: Optional[float] = None
    effect_metric: Optional[str] = None
    uncertainty: Optional[str] = None
    species: Optional[str] = None
    cell_context: Optional[str] = None
    sources: List[SourceRef] = Field(default_factory=list)
    replication_status: Literal[
        "unreplicated", "replicated", "contradicted", "unknown"
    ] = "unknown"
    status: ClaimStatus = "candidate"
    falsifier: str
    promoted_to_rhs: bool = False
    rhs_symbol: Optional[str] = None
    attribution: str = Field(default=ATTRIBUTION)


class HypothesisObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    onco_idea_id: Optional[str] = None
    onco_question: Optional[str] = None
    mechanism: str
    state: List[str] = Field(default_factory=list)
    intervention_slot: Optional[str] = None
    prediction: str
    experiment: str
    required_rung: Rung = "in_silico"
    status: HypothesisStatus = "open"
    falsifier: Optional[str] = None
    attribution: str = Field(default=ATTRIBUTION)


class ConstraintObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    onco_bottleneck_id: str
    applies_to: AppliesTo
    state_names: List[str] = Field(default_factory=list)
    implication: str
    blocks_promotion_above: Optional[Rung] = None
    attribution: str = Field(default=ATTRIBUTION)


class MechanismObject(BaseModel):
    """Causal claim. Not a parameter and not OnCo knowledge."""

    model_config = ConfigDict(extra="forbid")

    id: str
    layer: Layer = "mechanism"
    hypothesis_id: str
    evidence_ids: List[str] = Field(default_factory=list)
    intervention_slot: Optional[str] = None
    do_operator: str
    context_gate: Optional[str] = None
    state_from: List[str] = Field(default_factory=list)
    state_to: List[str] = Field(default_factory=list)
    sign: Sign = "unknown"
    model_id: ModelId = "confluence_v2_15d"
    status: Literal["proposed", "admitted", "rejected"] = "proposed"
    falsifier: str
    attribution: str = Field(default=ATTRIBUTION)


class ParameterObject(BaseModel):
    """Numeric symbol in a frozen model. OnCo cannot set this."""

    model_config = ConfigDict(extra="forbid")

    id: str
    layer: Layer = "parameter"
    symbol: str
    model_id: ModelId
    provenance: Provenance = "assumed"
    identifiability: Identifiability = "unidentified"
    dataset_id: Optional[str] = None
    bounds: Optional[List[float]] = None
    point_value: Optional[float] = None
    prior: Optional[str] = None
    posterior: Optional[str] = None
    mechanism_id: Optional[str] = None
    notes: str = ""
    attribution: str = Field(default=ATTRIBUTION)


class PredictionObject(BaseModel):
    """Simulator output. Does not write Evidence."""

    model_config = ConfigDict(extra="forbid")

    id: str
    layer: Layer = "prediction"
    model_id: ModelId
    hypothesis_id: Optional[str] = None
    parameter_ids: List[str] = Field(default_factory=list)
    intervention: Optional[str] = None
    horizon_days: float
    quantity: str
    point: Optional[float] = None
    interval: Optional[List[float]] = None
    ensemble_id: Optional[str] = None
    uncertainty_note: str = ""
    attribution: str = Field(default=ATTRIBUTION)


def refuse_knowledge_as_parameter(onco_ref: OncoRef, symbol: str) -> ParameterObject:
    """Explicit refusal object: an OnCo page is not a parameter."""
    return ParameterObject(
        id=f"refuse-{onco_ref.onco_id}-{symbol}",
        symbol=symbol,
        model_id="confluence_v2_15d",
        provenance="forbidden",
        identifiability="unidentified",
        notes=(
            f"OnCo {onco_ref.kind}:{onco_ref.onco_id} is knowledge, not a value for {symbol}."
        ),
    )
