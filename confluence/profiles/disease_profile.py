"""Disease Profile: thinking-lab export contract.

Research / in-silico only. This module never writes CancerODE parameters.
Knowledge ≠ Evidence ≠ Mechanism ≠ Parameter ≠ Prediction.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.0.0"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "schemas" / "disease_profile.schema.json"
)

AskerRole = Literal[
    "patient_advocate",
    "clinician",
    "researcher",
    "student",
    "other",
]

RESEARCH_DISCLAIMER = (
    "Project Confluence disease profiles are computational research artefacts. "
    "They are not a medical device, not clinical decision support, not "
    "personalized medicine as clinical CDS, and not a claim of cure, diagnosis, "
    "or dosing. In-silico / research only. See DISCLAIMER.md."
)

DEFAULT_NON_PARAMETERS: List[str] = [
    "OnCo knowledge records (pages, targets, ideas)",
    "OnCo confidence.probability",
    "OnCo Idea maturity",
    "Legacy validation/gene_to_parameter_map.json values (including LDHA → pyruvate_to_lactate)",
    "CONFLUENCE v2 symbols such as p_lactate unless independently identified",
    "Auditor classification scores",
    "Grok evidence-audit output",
    "Thinking-lab role, setting, or stuck labels",
    "Clinical intent, cure language, or dosing",
]

THINKING_ROLE_TO_ASKER = {
    "researcher": "researcher",
    "clinician": "clinician",
    "student": "student",
    "family": "patient_advocate",
    "patient_advocate": "patient_advocate",
    "engineer": "other",
    "builder": "other",
    "other": "other",
}

QUESTION_PROMPTS = {
    "role": "Who is asking?",
    "cancer": "Which disease — not cancer?",
    "setting": "What is the current regime?",
    "stuck": "Where is the system stuck?",
}

DISEASE_LABELS = {
    "all": "Childhood ALL",
    "testis": "Testicular germ-cell tumour",
    "tnbc": "Triple-negative breast cancer",
    "pdac": "Pancreatic ductal adenocarcinoma",
    "nsclc": "Non-small-cell lung cancer",
    "gbm": "Glioblastoma",
    "crc": "Colorectal cancer",
    "melanoma": "Melanoma",
    "cervix": "Cervical cancer",
    "cml": "CML",
}

GATES = (
    "Knowledge≠Evidence",
    "Evidence≠Mechanism",
    "Mechanism≠Parameter",
    "Parameter≠Prediction",
    "not_clinical_outcome",
)

_PARAMETER_LEAK = re.compile(
    r"(p_lactate|pyruvate_to_lactate|confidence\.probability|"
    r"idea\s*maturity|\btheta\b|Θ|ode\s*parameter|identified parameter)",
    re.IGNORECASE,
)
_LDHA_ONCO = re.compile(r"\b(ldha|onco)\b", re.IGNORECASE)
_CLINICAL_OVERCLAIM = re.compile(
    r"\b(cure|dosing|dose|prescribe|personalized medicine|clinical cds|"
    r"treat this patient|regimen)\b",
    re.IGNORECASE,
)


class AnswerChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    choice_id: str
    choice_label: str


class ThinkingLabAnswers(BaseModel):
    """The four thinking-lab questions and the chosen answers."""

    model_config = ConfigDict(extra="forbid")

    role: AnswerChoice
    cancer: AnswerChoice
    setting: AnswerChoice
    stuck: AnswerChoice


class Observable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str
    citation_ids: List[str] = Field(default_factory=list)


class CandidateMechanism(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str
    evidence_class: str
    citation_ids: List[str] = Field(default_factory=list)
    falsifier: str


class AdmittedHypothesis(BaseModel):
    """Hypothesis that survived the five-layer gates. Never an ODE Θ."""

    model_config = ConfigDict(extra="forbid")

    statement: str
    layer: Literal["hypothesis"] = "hypothesis"
    evidence_class: str
    citation_ids: List[str] = Field(default_factory=list)
    falsifier: str
    gates_passed: List[str]
    parameter_status: Literal["forbidden_to_enter_theta"] = "forbidden_to_enter_theta"


class Citation(BaseModel):
    """Vancouver-style bibliographic entry. Do not invent DOIs."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    url: Optional[str] = None
    doi: Optional[str] = None

    @field_validator("doi")
    @classmethod
    def doi_must_look_real(cls, value: Optional[str]) -> Optional[str]:
        if value is None or value == "":
            return None
        if not re.match(r"^10\.\d{4,9}/\S+$", value):
            raise ValueError("DOI must be a registered 10.xxxx/… string, not a placeholder")
        return value


class DiseaseProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str
    disease_id: str
    disease_label: str
    created_at: str
    schema_version: str = SCHEMA_VERSION
    asker_role: AskerRole
    answers: ThinkingLabAnswers
    observables: List[Observable] = Field(default_factory=list)
    candidate_mechanisms: List[CandidateMechanism] = Field(default_factory=list)
    non_parameters: List[str] = Field(default_factory=lambda: list(DEFAULT_NON_PARAMETERS))
    admitted_hypotheses: List[AdmittedHypothesis] = Field(default_factory=list)
    citations: List[Citation] = Field(min_length=1)
    disclaimer: str = RESEARCH_DISCLAIMER

    @field_validator("disclaimer")
    @classmethod
    def disclaimer_is_fixed(cls, value: str) -> str:
        if value != RESEARCH_DISCLAIMER:
            raise ValueError("disclaimer must be the fixed research-only string")
        return value


def map_asker_role(role_id: str) -> AskerRole:
    mapped = THINKING_ROLE_TO_ASKER.get(role_id, "other")
    return mapped  # type: ignore[return-value]


def refuse_ldha_onco_as_parameter(statement: str) -> bool:
    """True when LDHA/OnCo knowledge is being smuggled in as ODE Θ."""
    if not _LDHA_ONCO.search(statement):
        return False
    return bool(_PARAMETER_LEAK.search(statement) or re.search(r"parameter", statement, re.I))


def _gate_failures(candidate: CandidateMechanism) -> List[str]:
    text = candidate.statement
    failures: List[str] = []
    if refuse_ldha_onco_as_parameter(text) or _PARAMETER_LEAK.search(text):
        failures.append("Mechanism≠Parameter")
        failures.append("Knowledge≠Evidence")
    if not (candidate.falsifier or "").strip():
        failures.append("Evidence≠Mechanism")
    if _CLINICAL_OVERCLAIM.search(text) and "not" not in text.lower():
        failures.append("not_clinical_outcome")
        failures.append("Parameter≠Prediction")
    if candidate.evidence_class in {"knowledge", "onco_page", "confidence"}:
        failures.append("Knowledge≠Evidence")
    return failures


def admit_hypotheses(
    candidates: Sequence[CandidateMechanism],
) -> List[AdmittedHypothesis]:
    """Admit only items that pass Knowledge≠Evidence≠Mechanism≠Parameter≠Prediction."""
    admitted: List[AdmittedHypothesis] = []
    for candidate in candidates:
        failures = _gate_failures(candidate)
        if failures:
            continue
        admitted.append(
            AdmittedHypothesis(
                statement=candidate.statement,
                evidence_class=candidate.evidence_class,
                citation_ids=list(candidate.citation_ids),
                falsifier=candidate.falsifier,
                gates_passed=list(GATES),
            )
        )
    return admitted


def core_citations() -> List[Citation]:
    """Verified bibliographic entries. No fabricated DOIs."""
    return [
        Citation(
            id="1",
            text=(
                "Sung H, Filho AM, Laversanne M, et al. Global cancer statistics 2024. "
                "CA Cancer J Clin. 2026."
            ),
            url="https://doi.org/10.3322/caac.70090",
            doi="10.3322/caac.70090",
        ),
        Citation(
            id="2",
            text="World Health Organization. Cancer fact sheet. 2026.",
            url="https://www.who.int/news-room/fact-sheets/detail/cancer",
        ),
        Citation(
            id="3",
            text="National Cancer Institute. Triple-negative breast cancer.",
            url="https://www.cancer.gov/types/breast/patient/triple-negative-brochure",
        ),
        Citation(
            id="4",
            text=(
                "Gomila J, OnCo contributors. OnCo: a public, cited knowledge graph "
                "of oncology. 2026. Data CC BY-NC 4.0."
            ),
            url="https://onco.cc",
        ),
        Citation(
            id="5",
            text=(
                "Ogbonna K. Project Confluence (feat/onco-adapter-p0, pull request #9). "
                "2026."
            ),
            url="https://github.com/cloudynirvana/project-confluence/pull/9",
        ),
        Citation(
            id="6",
            text="Project Confluence. DISCLAIMER.md. Medical non-claims.",
            url="https://github.com/cloudynirvana/project-confluence/blob/main/DISCLAIMER.md",
        ),
        Citation(
            id="7",
            text=(
                "Altrock PM, Liu LL, Michor F. The mathematics of cancer: "
                "integrating quantitative models. Nat Rev Cancer. 2015."
            ),
            url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5663316/",
        ),
    ]


def _tnbc_observables() -> List[Observable]:
    return [
        Observable(
            statement=(
                "NCI describes triple-negative breast cancer as about 15% of breast "
                "cancers and notes faster growth and higher recurrence than some "
                "other invasive subtypes. This is a descriptor, not a CONFLUENCE parameter."
            ),
            citation_ids=["3"],
        ),
        Observable(
            statement=(
                "OnCo lists LDHA among lactate-metabolism targets. That record is "
                "knowledge with CC BY-NC 4.0 attribution, not identified p_lactate."
            ),
            citation_ids=["4", "5"],
        ),
    ]


def _default_observables(disease_id: str, disease_label: str) -> List[Observable]:
    if disease_id == "tnbc":
        return _tnbc_observables()
    return [
        Observable(
            statement=(
                f"{disease_label} is treated here as a disease-specific systems "
                "object. Thinking-lab answers are scaffolds, not measurements."
            ),
            citation_ids=["5", "6"],
        )
    ]


def _default_candidates(disease_id: str, disease_label: str) -> List[CandidateMechanism]:
    coupled = CandidateMechanism(
        statement=(
            f"In {disease_label}, clones, microenvironment, and treatment pressure "
            "are coupled over time; a single OnCo page does not specify the coupling."
        ),
        evidence_class="review_level",
        citation_ids=["5", "7"],
        falsifier=(
            "If a named, independent measurement shows the putative coupling is "
            "absent in that context, retire the hypothesis. Do not write Θ."
        ),
    )
    poison = CandidateMechanism(
        statement=(
            "OnCo LDHA knowledge and confidence.probability can be entered as the "
            "confluence_v2_15d parameter p_lactate (or legacy pyruvate_to_lactate)."
        ),
        evidence_class="knowledge",
        citation_ids=["4"],
        falsifier="Rejected a priori: Knowledge is not a parameter.",
    )
    if disease_id == "tnbc":
        return [
            CandidateMechanism(
                statement=(
                    "Lactate metabolism, TGF-β stroma signalling, immune exclusion "
                    "and persister states are coupled loops in TNBC and may be "
                    "stated as a testable mechanism hypothesis."
                ),
                evidence_class="review_level",
                citation_ids=["3", "7"],
                falsifier=(
                    "If uncoupling lactate from exclusion in a named assay does not "
                    "change the coupled signature, drop the coupling claim. "
                    "Never promote the claim to p_lactate."
                ),
            ),
            poison,
        ]
    return [coupled, poison]


def build_thinking_answers(
    *,
    role_id: str,
    role_label: str,
    disease_id: str,
    disease_label: str,
    setting_id: str,
    setting_label: str,
    stuck_id: str,
    stuck_label: str,
) -> ThinkingLabAnswers:
    return ThinkingLabAnswers(
        role=AnswerChoice(
            question=QUESTION_PROMPTS["role"],
            choice_id=role_id,
            choice_label=role_label,
        ),
        cancer=AnswerChoice(
            question=QUESTION_PROMPTS["cancer"],
            choice_id=disease_id,
            choice_label=disease_label,
        ),
        setting=AnswerChoice(
            question=QUESTION_PROMPTS["setting"],
            choice_id=setting_id,
            choice_label=setting_label,
        ),
        stuck=AnswerChoice(
            question=QUESTION_PROMPTS["stuck"],
            choice_id=stuck_id,
            choice_label=stuck_label,
        ),
    )


def build_disease_profile(
    *,
    role_id: str = "researcher",
    role_label: str = "Researcher",
    disease_id: str = "tnbc",
    setting_id: str = "metastatic",
    setting_label: str = "Metastatic / relapsed",
    stuck_id: str = "resistance",
    stuck_label: str = "Adaptation / persisters",
    extra_candidates: Optional[Sequence[CandidateMechanism]] = None,
    created_at: Optional[str] = None,
    profile_id: Optional[str] = None,
) -> DiseaseProfile:
    """Build a gated profile. OnCo/LDHA knowledge never becomes admitted Θ."""
    disease_label = DISEASE_LABELS.get(disease_id, disease_id)
    answers = build_thinking_answers(
        role_id=role_id,
        role_label=role_label,
        disease_id=disease_id,
        disease_label=disease_label,
        setting_id=setting_id,
        setting_label=setting_label,
        stuck_id=stuck_id,
        stuck_label=stuck_label,
    )
    candidates = list(_default_candidates(disease_id, disease_label))
    if extra_candidates:
        candidates.extend(extra_candidates)
    admitted = admit_hypotheses(candidates)
    stamp = created_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pid = profile_id or f"dp-{disease_id}-{uuid.uuid4().hex[:8]}"
    return DiseaseProfile(
        profile_id=pid,
        disease_id=disease_id,
        disease_label=disease_label,
        created_at=stamp,
        schema_version=SCHEMA_VERSION,
        asker_role=map_asker_role(role_id),
        answers=answers,
        observables=_default_observables(disease_id, disease_label),
        candidate_mechanisms=candidates,
        non_parameters=list(DEFAULT_NON_PARAMETERS),
        admitted_hypotheses=admitted,
        citations=core_citations(),
        disclaimer=RESEARCH_DISCLAIMER,
    )


def export_disease_profile(profile: DiseaseProfile) -> dict:
    return profile.model_dump(mode="json")


def load_json_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def filename_for(profile: DiseaseProfile) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", profile.disease_id.lower()).strip("-")
    return f"disease-profile-{slug}.json"


def schema_required_fields() -> Tuple[str, ...]:
    return (
        "profile_id",
        "disease_id",
        "disease_label",
        "created_at",
        "schema_version",
        "asker_role",
        "answers",
        "observables",
        "candidate_mechanisms",
        "non_parameters",
        "admitted_hypotheses",
        "citations",
        "disclaimer",
    )
