"""Research HypothesisObject for Disease Profile work.

Distinct from confluence.onco.schemas.HypothesisObject (OnCo idea shelf).
This object never writes CancerODE parameters. Research / in-silico only.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from confluence.profiles.disease_profile import RESEARCH_DISCLAIMER

HypothesisStatus = Literal["proposed"]


class HypothesisObject(BaseModel):
    """P0 sketch: a gated, dataset-named hypothesis. Not clinical CDS."""

    model_config = ConfigDict(extra="forbid")

    id: str
    disease_profile_ref: str
    statement: str
    evidence_class: str
    named_public_dataset: str
    falsifier: str
    status: HypothesisStatus = "proposed"
    non_claims: str
    citation_ids: List[str] = Field(default_factory=list)
    disclaimer: str = RESEARCH_DISCLAIMER

    @field_validator("named_public_dataset")
    @classmethod
    def dataset_must_be_named(cls, value: str) -> str:
        if not (value or "").strip():
            raise ValueError(
                "named_public_dataset is required; unnamed cohorts cannot enter a hypothesis"
            )
        return value.strip()

    @field_validator("non_claims")
    @classmethod
    def non_claims_must_refuse_cds(cls, value: str) -> str:
        text = (value or "").lower()
        if "not" not in text or (
            "clinical" not in text and "cds" not in text and "device" not in text
        ):
            raise ValueError("non_claims must explicitly refuse clinical CDS / device use")
        return value

    @field_validator("disclaimer")
    @classmethod
    def disclaimer_is_fixed(cls, value: str) -> str:
        if value != RESEARCH_DISCLAIMER:
            raise ValueError("disclaimer must be the fixed research-only string")
        return value


def load_hypothesis(data: dict) -> HypothesisObject:
    return HypothesisObject.model_validate(data)
