"""Catalog loader and organ-toxicity helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from confluence.contracts import DrugSpecification

CATALOG_PATH = Path(__file__).resolve().parent / "drug_catalog.json"


def load_drug_catalog(path: Path | None = None) -> List[DrugSpecification]:
    payload = json.loads((path or CATALOG_PATH).read_text())
    return [DrugSpecification.model_validate(item) for item in payload["drugs"]]


class ToxicityConstraints:
    """Soft organ-toxicity scores from circulating concentrations."""

    def __init__(self, catalog: List[DrugSpecification] | None = None):
        self.catalog = catalog or load_drug_catalog()
        self.by_id = {d.id: d for d in self.catalog}

    def organ_scores(self, concentrations: Dict[str, float]) -> Dict[str, float]:
        organs = ("liver", "kidney", "heart", "marrow", "immune", "gut", "nerve")
        scores = {organ: 0.0 for organ in organs}
        for drug_id, conc in concentrations.items():
            spec = self.by_id.get(drug_id)
            if spec is None:
                continue
            scale = max(0.0, float(conc)) / max(spec.mtd, 1e-8)
            tox = spec.organ_toxicity.model_dump()
            for organ in organs:
                scores[organ] += scale * float(tox.get(organ, 0.0))
        return scores

    def host_penalty(self, concentrations: Dict[str, float]) -> float:
        scores = self.organ_scores(concentrations)
        return float(sum(scores.values()))
