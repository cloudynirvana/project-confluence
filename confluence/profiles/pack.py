"""Build gated Disease Profile packs from a YAML/JSON case table.

Research / in-silico only. Never writes CancerODE Θ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from confluence.profiles.disease_profile import (
    RESEARCH_DISCLAIMER,
    CandidateMechanism,
    Citation,
    DiseaseProfile,
    Observable,
    build_disease_profile,
    export_disease_profile,
    refuse_ldha_onco_as_parameter,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE = REPO_ROOT / "data" / "profiles" / "cases" / "cases.yaml"
DEFAULT_OUT = REPO_ROOT / "data" / "profiles" / "cases"

REFUSE_KEYS = {
    "p_lactate",
    "pyruvate_to_lactate",
    "theta",
    "θ",
    "write_theta",
    "promote_to_rhs",
    "ode_parameter",
    "confidence.probability",
}


class RowRefused(ValueError):
    """A case table row tried to write a parameter and was rejected."""


def _as_citation(item: Dict[str, Any]) -> Citation:
    return Citation(
        id=str(item["id"]),
        text=str(item["text"]),
        url=item.get("url"),
        doi=item.get("doi"),
    )


def _as_observable(item: Dict[str, Any]) -> Observable:
    return Observable(
        statement=str(item["statement"]),
        citation_ids=[str(x) for x in (item.get("citation_ids") or [])],
    )


def _as_candidate(item: Dict[str, Any]) -> CandidateMechanism:
    return CandidateMechanism(
        statement=str(item["statement"]),
        evidence_class=str(item["evidence_class"]),
        citation_ids=[str(x) for x in (item.get("citation_ids") or [])],
        falsifier=str(item["falsifier"]),
    )


def row_attempts_parameter_write(row: Dict[str, Any]) -> Optional[str]:
    """Return a reason if the row tries to smuggle Θ / OnCo confidence as a parameter."""
    for key in row:
        lowered = str(key).lower()
        if lowered in REFUSE_KEYS or lowered.replace(" ", "_") in REFUSE_KEYS:
            return f"forbidden key {key!r}"
        if lowered in {"force_admit", "as_parameter", "write_theta"}:
            return f"forbidden key {key!r}"
    for cand in row.get("candidate_mechanisms") or []:
        if not isinstance(cand, dict):
            continue
        if cand.get("as_parameter") or cand.get("write_theta") or cand.get("force_admit"):
            return "candidate requested parameter write"
        statement = str(cand.get("statement") or "")
        if cand.get("evidence_class") == "knowledge" and refuse_ldha_onco_as_parameter(statement):
            if cand.get("admit"):
                return "knowledge candidate marked for admission as parameter"
    return None


def profile_from_row(row: Dict[str, Any]) -> DiseaseProfile:
    reason = row_attempts_parameter_write(row)
    if reason:
        raise RowRefused(f"{row.get('slug') or row.get('disease_id')}: {reason}")
    citations = [_as_citation(x) for x in (row.get("citations") or [])] or None
    observables = [_as_observable(x) for x in (row.get("observables") or [])] or None
    candidates = [_as_candidate(x) for x in (row.get("candidate_mechanisms") or [])] or None
    extra = [_as_candidate(x) for x in (row.get("extra_candidates") or [])] or None
    return build_disease_profile(
        role_id=str(row.get("role_id") or "researcher"),
        role_label=str(row.get("role_label") or "Researcher"),
        disease_id=str(row["disease_id"]),
        disease_label=row.get("disease_label"),
        setting_id=str(row.get("setting_id") or "metastatic"),
        setting_label=str(row.get("setting_label") or "Metastatic / relapsed"),
        stuck_id=str(row.get("stuck_id") or "resistance"),
        stuck_label=str(row.get("stuck_label") or "Adaptation / persisters"),
        candidate_mechanisms=candidates,
        extra_candidates=extra,
        observables=observables,
        citations=citations,
        extra_non_parameters=row.get("extra_non_parameters"),
        include_default_poison=bool(row.get("include_default_poison", True)),
        created_at=row.get("created_at") or "2026-09-20T00:00:00Z",
        profile_id=row.get("profile_id"),
    )


def load_table(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
    else:
        import json

        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("case table must be a mapping with 'cases'")
    return data


def build_pack(
    table_path: Path = DEFAULT_TABLE,
    *,
    include_refuse_examples: bool = False,
) -> Tuple[List[Tuple[str, DiseaseProfile]], List[Tuple[Dict[str, Any], str]]]:
    data = load_table(table_path)
    profiles: List[Tuple[str, DiseaseProfile]] = []
    refused: List[Tuple[Dict[str, Any], str]] = []
    rows: Sequence[Dict[str, Any]] = list(data.get("cases") or [])
    if include_refuse_examples:
        rows = list(rows) + list(data.get("refuse_examples") or [])
    for row in rows:
        try:
            profile = profile_from_row(row)
            filename = str(row.get("filename") or f"{row.get('slug') or profile.disease_id}.json")
            profiles.append((filename, profile))
        except RowRefused as err:
            refused.append((row, str(err)))
    return profiles, refused


def write_pack(
    profiles: Iterable[Tuple[str, DiseaseProfile]],
    out_dir: Path,
    *,
    refused: Optional[Sequence[Tuple[Dict[str, Any], str]]] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    import json

    rows = []
    for filename, profile in profiles:
        path = out_dir / filename
        path.write_text(json.dumps(export_disease_profile(profile), indent=2) + "\n", encoding="utf-8")
        rows.append(
            (
                filename,
                profile.disease_label,
                profile.asker_role,
                len(profile.admitted_hypotheses),
                len(profile.candidate_mechanisms),
                len(profile.citations),
            )
        )
    summary = out_dir / "SUMMARY.md"
    lines = [
        "# Disease Profile case pack",
        "",
        "Research / in-silico only. Not a medical device, not clinical CDS, not dosing, not a cure.",
        "See DISCLAIMER.md and docs/CITATION_POLICY.md.",
        "",
        "| File | Disease | Asker | Admitted hypotheses | Candidates (incl. audit traps) | Citations |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for filename, label, role, admitted, cands, cites in rows:
        lines.append(
            f"| `{filename}` | {label} | {role} | {admitted} | {cands} | {cites} |"
        )
    lines.extend(
        [
            "",
            "Admitted hypotheses passed Knowledge≠Evidence≠Mechanism≠Parameter≠Prediction.",
            "OnCo LDHA / confidence.probability / Idea maturity remain non-parameters.",
            "",
            f"Fixed disclaimer: {RESEARCH_DISCLAIMER}",
            "",
        ]
    )
    if refused:
        lines.append("## Refused rows (parameter-write attempts)")
        for row, reason in refused:
            lines.append(f"- `{row.get('slug')}`: {reason}")
        lines.append("")
    summary.write_text("\n".join(lines), encoding="utf-8")
    return summary
