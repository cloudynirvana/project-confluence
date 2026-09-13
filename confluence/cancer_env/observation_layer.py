"""Noisy partial observations of the latent microenvironment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from confluence.cancer_env.ode_system import ArchetypeParams
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
    occult_allele_fraction: float = 0.04
    dormancy_exit: float = 0.03
    immune_surveillance: float = 0.04
    antibody_readiness: float = 0.04


class ObservationLayer:
    """Y = g(X; class) + ε. Class changes visibility, not just the label."""

    def __init__(
        self,
        noise: Optional[ObservationNoise] = None,
        seed: Optional[int] = None,
        health_warn_threshold: float = 0.45,
        params: Optional[ArchetypeParams] = None,
    ):
        self.noise = noise or ObservationNoise()
        self.rng = np.random.default_rng(seed)
        self.health_warn_threshold = health_warn_threshold
        self.params = params
        self._prev_awake: Optional[float] = None

    def set_params(self, params: Optional[ArchetypeParams]) -> None:
        self.params = params

    def observe(self, state: LatentCancerState, noisy: bool = True) -> ObservationRecord:
        p = self.params
        burden = state.tumor_burden
        resist = state.resistance_frequency
        immune = state.I_act + state.I_exh
        competence = state.I_act / (immune + 1e-8)
        k_vis = float(getattr(p, "clinical_visibility_k", 0.12) if p is not None else 0.12)
        visibility = burden / (k_vis + burden + 1e-12)
        leak = float(getattr(p, "occult_af_leak", 0.08) if p is not None else 0.08)
        shed = float(getattr(p, "k_junction_shed", 0.85) if p is not None else 0.85)

        # Occult: bulk burden / AF stay dim; junction and occult-AF leak earlier.
        clinical_burden = visibility * burden
        clinical_af = visibility * state.fusion_allele_fraction
        occult_af = (1.0 - visibility) * leak * state.fusion_allele_fraction + clinical_af
        junction = max(0.0, state.T_f) * shed

        awake = float(state.awake)
        prev = self._prev_awake if self._prev_awake is not None else awake
        dormancy_exit = float(np.clip(awake + 1.8 * max(0.0, awake - prev), 0.0, 1.5))
        self._prev_awake = awake

        def _n(value: float, sigma: float) -> float:
            if not noisy or sigma <= 0:
                return max(0.0, value)
            return max(0.0, float(value + self.rng.normal(0.0, sigma)))

        n = self.noise
        # Occult gets extra noise on bulk Y, lower noise on leak channels.
        burden_sigma = n.tumor_burden * (1.6 if (p and p.disease_class == "occult") else 1.0)
        leak_sigma = n.junction_neoantigen * (0.55 if (p and p.disease_class == "occult") else 1.0)

        resist_obs = float(np.clip(_n(resist, n.resistance_frequency), 0.0, 1.0))
        competence_obs = float(np.clip(_n(competence, n.immune_competence_ratio), 0.0, 1.5))
        return ObservationRecord(
            t=state.t,
            tumor_burden=_n(clinical_burden, burden_sigma),
            resistance_frequency=resist_obs,
            lactate=_n(state.L, n.lactate),
            tgfb=_n(state.C_tgfb, n.tgfb),
            immune_competence_ratio=competence_obs,
            fusion_allele_fraction=float(np.clip(_n(clinical_af, n.fusion_allele_fraction), 0.0, 1.0)),
            junction_neoantigen=_n(junction, leak_sigma),
            occult_allele_fraction=float(np.clip(_n(occult_af, n.occult_allele_fraction), 0.0, 1.0)),
            dormancy_exit=float(np.clip(_n(dormancy_exit, n.dormancy_exit), 0.0, 1.5)),
            immune_surveillance=float(np.clip(_n(state.I_surv, n.immune_surveillance), 0.0, 1.5)),
            antibody_readiness=float(np.clip(_n(state.A_ready, n.antibody_readiness), 0.0, 1.5)),
            fusion_id=state.fusion_id,
            disease_class=state.disease_class,
            host_toxicity_warning=state.H <= self.health_warn_threshold,
            host_health=state.H,
        )
