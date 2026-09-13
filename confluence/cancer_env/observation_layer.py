"""Noisy partial observations of the latent microenvironment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from confluence.contracts import LatentCancerState, ObservationRecord


@dataclass
class ObservationNoise:
    tumor_burden: float = 0.04
    resistance_frequency: float = 0.03
    lactate: float = 0.05
    tgfb: float = 0.05
    immune_competence_ratio: float = 0.04
    fusion_allele_fraction: float = 0.035
    junction_neoantigen: float = 0.05


class ObservationLayer:
    """Y = g(X) + ε, ε ~ N(0, σ²), plus a host-toxicity warning flag."""

    def __init__(
        self,
        noise: Optional[ObservationNoise] = None,
        seed: Optional[int] = None,
        health_warn_threshold: float = 0.45,
    ):
        self.noise = noise or ObservationNoise()
        self.rng = np.random.default_rng(seed)
        self.health_warn_threshold = health_warn_threshold

    def observe(self, state: LatentCancerState, noisy: bool = True) -> ObservationRecord:
        burden = state.tumor_burden
        resist = state.resistance_frequency
        immune = state.I_act + state.I_exh
        competence = state.I_act / (immune + 1e-8)

        def _n(value: float, sigma: float) -> float:
            if not noisy or sigma <= 0:
                return max(0.0, value)
            return max(0.0, float(value + self.rng.normal(0.0, sigma)))

        n = self.noise
        resist_obs = _n(resist, n.resistance_frequency)
        resist_obs = float(np.clip(resist_obs, 0.0, 1.0))
        competence_obs = float(np.clip(_n(competence, n.immune_competence_ratio), 0.0, 1.5))

        fusion_af = float(np.clip(_n(state.fusion_allele_fraction, n.fusion_allele_fraction), 0.0, 1.0))
        junction = _n(state.junction_neoantigen, n.junction_neoantigen)
        return ObservationRecord(
            t=state.t,
            tumor_burden=_n(burden, n.tumor_burden),
            resistance_frequency=resist_obs,
            lactate=_n(state.L, n.lactate),
            tgfb=_n(state.C_tgfb, n.tgfb),
            immune_competence_ratio=competence_obs,
            fusion_allele_fraction=fusion_af,
            junction_neoantigen=junction,
            fusion_id=state.fusion_id,
            host_toxicity_warning=state.H <= self.health_warn_threshold,
            host_health=state.H,
        )
