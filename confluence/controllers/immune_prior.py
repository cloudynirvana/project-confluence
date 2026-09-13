"""Structured secretory prior: fly-brain bias toward immune / antibody readiness.

Early-warning signs (junction neoantigen, competence drop, occult AF,
dormancy-exit, surveillance) lift antibody / biologic channels **before**
bulk burden explodes. Fusion AF still lifts TKIs.

This is a documented affine prior on U, not a trained clinical policy and
not a claim that Drosophila neurons synthesize biologics.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from confluence.cancer_env.readiness import antibody_priority, early_warning_score
from confluence.contracts import ObservationRecord


_IMMUNE_CHANNELS = {
    "protein_ifng": 0.24,
    "protein_il2": 0.32,
    "protein_anti_pd1": 0.20,
    "protein_chimeric_engager": 0.40,
    "protein_tgfb_trap": 0.18,
    "protein_surveillance_igg": 0.32,
    "protein_fusion_mab": 0.24,
}
_FUSION_CHANNELS = {
    "tki_alk": 0.22,
    "tki_imatinib_like": 0.08,
}


def apply_immune_secretory_prior(
    u: np.ndarray,
    drug_ids: Sequence[str],
    observation: ObservationRecord,
) -> np.ndarray:
    """Return a clipped U with immune / chimeric-protein rescue bias."""
    out = np.clip(np.asarray(u, dtype=float).reshape(-1), 0.0, 1.0)
    immune_need = float(np.clip(1.0 - observation.immune_competence_ratio, 0.0, 1.0))
    fusion_need = float(np.clip(observation.fusion_allele_fraction, 0.0, 1.0))
    junction = float(np.clip(observation.junction_neoantigen / 1.2, 0.0, 1.0))
    early = early_warning_score(observation)
    ab_pri = antibody_priority(observation)
    tox_scale = 0.35 if observation.host_toxicity_warning else 1.0
    ids = list(drug_ids)
    for name, gain in _IMMUNE_CHANNELS.items():
        if name not in ids:
            continue
        i = ids.index(name)
        extra = gain * (0.28 + 0.40 * immune_need + 0.55 * ab_pri) * tox_scale
        if name == "protein_chimeric_engager":
            extra *= 0.55 + 0.45 * max(fusion_need, early)
        if name == "protein_fusion_mab":
            extra *= 0.40 + 0.60 * max(fusion_need, junction, early)
        if name == "protein_surveillance_igg":
            extra *= 0.45 + 0.55 * early
        if i < out.size:
            out[i] = float(np.clip(out[i] + extra, 0.0, 0.85))
    for name, gain in _FUSION_CHANNELS.items():
        if name not in ids:
            continue
        i = ids.index(name)
        extra = gain * (0.20 + 0.80 * max(fusion_need, junction, 0.5 * early)) * tox_scale
        if i < out.size:
            out[i] = float(np.clip(out[i] + extra, 0.0, 0.75))
    return out
