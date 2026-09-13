"""Early-warning / immune-readiness scores (research, not a clinical assay)."""

from __future__ import annotations

import numpy as np

from confluence.contracts import ObservationRecord


def early_warning_score(observation: ObservationRecord) -> float:
    """Scalar in [0, 1] from junction, competence drop, occult AF, dormancy-exit.

    Intended to fire **before** bulk ``Y.tumor_burden`` explodes. Not a
    screening test and not a treatment indication.
    """
    competence_drop = float(np.clip(1.0 - observation.immune_competence_ratio, 0.0, 1.0))
    junction = float(np.clip(observation.junction_neoantigen / 0.80, 0.0, 1.0))
    occult = float(np.clip(observation.occult_allele_fraction / 0.40, 0.0, 1.0))
    wake = float(np.clip(observation.dormancy_exit, 0.0, 1.0))
    surv = float(np.clip(observation.immune_surveillance, 0.0, 1.0))
    raw = (
        0.28 * competence_drop
        + 0.26 * junction
        + 0.18 * occult
        + 0.18 * wake
        + 0.10 * surv
    )
    return float(np.clip(raw, 0.0, 1.0))


def antibody_priority(observation: ObservationRecord) -> float:
    """How strongly E/F should lift antibody / biologic channels.

    High when early signs are up and bulk burden is still modest.
    """
    early = early_warning_score(observation)
    burden = float(max(0.0, observation.tumor_burden))
    still_early = float(np.clip(1.15 - burden, 0.15, 1.0))
    ready = float(np.clip(observation.antibody_readiness, 0.0, 1.0))
    return float(np.clip(early * (0.55 + 0.45 * still_early) * (0.70 + 0.30 * ready), 0.0, 1.0))
