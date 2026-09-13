"""First-order pharmacokinetics and Hill pharmacodynamics.

    dC_k / dt = -(ln 2 / t_half_k) * C_k + U_k(t)

Concentrations are simulation-scaled. Half-lives are taken from the catalog
in hours and converted to days to match the cancer ODE clock.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from confluence.contracts import CONTROL_DRUG_IDS, DrugSpecification
from confluence.pharmacology.toxicity_constraints import load_drug_catalog


def hill_occupancy(concentration: float, ic50: float, hill: float) -> float:
    """Saturating Hill occupancy in [0, 1)."""
    c = max(0.0, float(concentration))
    if c <= 0.0:
        return 0.0
    ic50 = max(float(ic50), 1e-12)
    hill = max(float(hill), 1e-6)
    num = c ** hill
    return float(num / (num + ic50 ** hill))


class PKPDModel:
    """Vector PK for the five control-channel drugs plus catalog extras."""

    def __init__(
        self,
        catalog: Sequence[DrugSpecification] | None = None,
        drug_ids: Sequence[str] = CONTROL_DRUG_IDS,
    ):
        catalog = list(catalog) if catalog is not None else load_drug_catalog()
        by_id = {d.id: d for d in catalog}
        by_channel = {
            d.control_channel: d for d in catalog if d.control_channel
        }
        self.drug_ids = tuple(drug_ids)
        self.specs: List[DrugSpecification] = []
        for drug_id in self.drug_ids:
            spec = by_id.get(drug_id) or by_channel.get(drug_id)
            if spec is None:
                raise KeyError(f"Drug '{drug_id}' not in catalog")
            self.specs.append(spec)
        self.k_el = np.array(
            [np.log(2.0) / (spec.half_life_hours / 24.0) for spec in self.specs],
            dtype=float,
        )
        self.ic50 = np.array([spec.ic50 for spec in self.specs], dtype=float)
        self.hill = np.array([spec.hill for spec in self.specs], dtype=float)
        self.mtd = np.array([spec.mtd for spec in self.specs], dtype=float)

    @property
    def n_drugs(self) -> int:
        return len(self.drug_ids)

    def zeros(self) -> np.ndarray:
        return np.zeros(self.n_drugs, dtype=float)

    def _pad_u(self, u: np.ndarray) -> np.ndarray:
        """Accept short controller vectors (A–E are 5-D) when PK has proteins."""
        vec = np.asarray(u, dtype=float).reshape(-1)
        if vec.size < self.n_drugs:
            vec = np.pad(vec, (0, self.n_drugs - vec.size))
        elif vec.size > self.n_drugs:
            vec = vec[: self.n_drugs]
        return vec

    def rhs(self, c: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Clearance-matched infusion: U∈[0,1] targets C_ss = U · MTD.

        The catalog ODE is dC/dt = −k_el C + U_phys. We set
        U_phys = k_el · MTD · U so a unit command does not explode
        long-half-life mAbs (pembrolizumab t½ ≈ 26 d) while short
        t½ drugs (HDAC, MCT1) still equilibrate in hours.
        """
        c = np.asarray(c, dtype=float)
        u = np.clip(self._pad_u(u), 0.0, 1.0)
        return -self.k_el * np.maximum(c, 0.0) + self.k_el * self.mtd * u

    def occupancies(self, c: np.ndarray) -> Dict[str, float]:
        c = np.asarray(c, dtype=float)
        out = {}
        for i, drug_id in enumerate(self.drug_ids):
            out[drug_id] = hill_occupancy(c[i], self.ic50[i], self.hill[i])
        return out

    def as_dict(self, c: np.ndarray) -> Dict[str, float]:
        return {drug_id: float(c[i]) for i, drug_id in enumerate(self.drug_ids)}

    def clip_infusion(self, u: Mapping[str, float] | np.ndarray) -> np.ndarray:
        if isinstance(u, Mapping):
            vec = np.array([float(u.get(d, 0.0)) for d in self.drug_ids], dtype=float)
        else:
            vec = self._pad_u(u)
        return np.clip(vec, 0.0, 1.0)

    def toxicity_load(self, c: np.ndarray) -> float:
        """Weighted concentration / MTD load used by host-health dynamics."""
        c = np.maximum(np.asarray(c, dtype=float), 0.0)
        weights = np.array([float(spec.tox_weight) for spec in self.specs], dtype=float)
        return float(np.sum(weights * c / np.maximum(self.mtd, 1e-8)))
