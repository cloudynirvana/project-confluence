"""Map OnCo records onto Confluence slots. No ODE writes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from confluence.onco.attribution import ATTRIBUTION

DEFAULT_BINDINGS = Path(__file__).with_name("seed_bindings.yaml")


@dataclass
class Binding:
    name: str
    onco_kind: str
    onco_id_hint: Optional[str] = None
    u_slot: Optional[str] = None
    states: List[str] = field(default_factory=list)
    annotate: Optional[str] = None
    layer: str = "v2"
    note: Optional[str] = None
    attribution: str = ATTRIBUTION


def load_bindings(path: Optional[Path] = None) -> Dict[str, Any]:
    target = path or DEFAULT_BINDINGS
    return yaml.safe_load(target.read_text(encoding="utf-8"))


def bind(name_or_id: str, bindings: Optional[Dict[str, Any]] = None) -> List[Binding]:
    data = bindings or load_bindings()
    q = name_or_id.lower().strip()
    found: List[Binding] = []
    for row in data.get("targets") or []:
        hint = str(row.get("onco_id_hint") or "")
        if q in str(row.get("name", "")).lower() or (hint and q == hint.lower()):
            binds = row.get("binds") or {}
            found.append(
                Binding(
                    name=row["name"],
                    onco_kind=row.get("onco_kind", "target"),
                    onco_id_hint=row.get("onco_id_hint"),
                    u_slot=binds.get("u_slot"),
                    states=list(binds.get("states") or []),
                    annotate=binds.get("annotate"),
                    layer=binds.get("layer", "v2"),
                    note=binds.get("note"),
                )
            )
    for row in data.get("disease") or []:
        hint = str(row.get("onco_id_hint") or "")
        if q in str(row.get("name", "")).lower() or (hint and q == hint.lower()):
            found.append(
                Binding(
                    name=row["name"],
                    onco_kind=row.get("onco_kind", "cancer"),
                    onco_id_hint=row.get("onco_id_hint"),
                    layer="disease",
                    note=str((row.get("confluence") or {})),
                )
            )
    return found


def wired_claims(bindings: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    data = bindings or load_bindings()
    return list(data.get("wired_claims") or [])


def classify_legacy_gene_map_row(gene: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """Tag a validation/gene_to_parameter_map.json row. Does not identify Theta."""
    return {
        "gene": gene,
        "legacy_parameter": row.get("parameter"),
        "source_type": "legacy_mapping",
        "provenance": "assumed",
        "identifiability": "unidentified",
        "model_id": "confluence_v1_calibrator",
        "not_v2_symbol": row.get("parameter") != "p_lactate",
        "attribution": ATTRIBUTION,
        "notes": (
            "literature-inspired effect size; not an identified parameter "
            "and not a value taken from OnCo"
        ),
    }
