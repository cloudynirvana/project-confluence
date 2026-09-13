"""Rate-based mushroom-body network suitable for real-time stepping.

Pipeline:
    Y → sensory W_in → AL/LH-style projection neurons → sparse KC expansion
    → MBON rates → motor decode U = clip(W_out · rates, 0)

Default interactive size is 256 Kenyon cells (documented). Pass n_kc=2048
for a more FlyWire-like expansion; the stub generator will scale fan-in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from confluence.connectome.circuit_extractor import CircuitExtractor, CompiledCircuit
from confluence.connectome.client_stub import FlyWireClient
from confluence.contracts import CONTROL_DRUG_IDS, FUSION_CHANNEL_IDS, OBS_VECTOR_NAMES, ObservationRecord
from confluence.neural_engine.plasticity import DopaminePlasticity, dopamine_signal


@dataclass
class NetworkConfig:
    n_kc: int = 256
    n_obs: int = len(OBS_VECTOR_NAMES)
    n_out: int = 7
    sparsity: float = 0.05
    tau_mbon: float = 0.35
    seed: int = 7
    plastic: bool = True


class MushroomBodyNetwork:
    def __init__(
        self,
        config: Optional[NetworkConfig] = None,
        compiled: Optional[CompiledCircuit] = None,
        drug_ids: Sequence[str] = CONTROL_DRUG_IDS + FUSION_CHANNEL_IDS,
    ):
        self.config = config or NetworkConfig()
        self.drug_ids = tuple(drug_ids)
        self.plasticity = DopaminePlasticity()
        rng = np.random.default_rng(self.config.seed)

        if compiled is None:
            graph = FlyWireClient().fetch_mushroom_body(n_kc=self.config.n_kc)
            compiled = CircuitExtractor().compile(graph)
        self.compiled = compiled

        n_pn = compiled.w_pn_kc.shape[1]
        n_kc = compiled.w_pn_kc.shape[0]
        n_mbon = compiled.w_kc_mbon.shape[0]
        self.n_pn = n_pn
        self.n_kc = n_kc
        self.n_mbon = n_mbon

        # Sensory → PN (AL/LH-style). Random but structured: each obs fans out.
        self.w_in = rng.normal(0.0, 0.8, size=(n_pn, self.config.n_obs))
        # Junction / fusion-AF columns (last two) get a stronger prior so E
        # can up-weight fusion TKIs when chimeric-junction signal rises.
        if self.config.n_obs >= 7:
            self.w_in[:, 5:7] += rng.normal(0.35, 0.15, size=(n_pn, 2))
        if self.config.n_obs >= 11:
            self.w_in[:, 7:] += rng.normal(0.28, 0.12, size=(n_pn, self.config.n_obs - 7))
        self.pn_bias = rng.normal(0.0, 0.1, size=n_pn)
        self.w_pn_kc = compiled.w_pn_kc.copy()
        self.w_kc_mbon = compiled.w_kc_mbon.copy()
        # Motor decode from MBON rates. GABAergic MBONs flip sign.
        # Structured prior onto excitatory MBONs so an untrained network
        # still produces a visible infusion; plasticity then reshapes it.
        self.w_out = rng.normal(0.0, 0.10, size=(self.config.n_out, n_mbon))
        self.w_out *= compiled.signs_mbon[None, :]
        excitatory = [i for i, s in enumerate(compiled.signs_mbon) if s > 0]
        if not excitatory:
            excitatory = list(range(n_mbon))
        for i in range(self.config.n_out):
            self.w_out[i, excitatory[i % len(excitatory)]] += 0.85
            self.w_out[i, excitatory[(i + 2) % len(excitatory)]] += 0.35
        # Extra prior on fusion-TKI rows (trailing channels).
        if self.config.n_out >= 7:
            for i in range(self.config.n_out - 2, self.config.n_out):
                self.w_out[i, excitatory[i % len(excitatory)]] += 0.40
        self.apl = compiled.w_apl_kc.copy()

        self.mbon_rate = np.zeros(n_mbon, dtype=float)
        self.kc_rate = np.zeros(n_kc, dtype=float)
        self.pn_rate = np.zeros(n_pn, dtype=float)
        self.last_da = 0.0
        self.last_sparsity = 0.0
        self.prev_burden: Optional[float] = None
        self.prev_resist: Optional[float] = None
        self.prev_fusion: Optional[float] = None

    def _k_winners(self, drive: np.ndarray) -> np.ndarray:
        n_kc = drive.size
        k = max(1, int(round(self.config.sparsity * n_kc)))
        thresh = np.partition(drive, n_kc - k)[n_kc - k]
        rates = np.maximum(drive - thresh, 0.0)
        if rates.max() > 0:
            rates = rates / (rates.max() + 1e-8)
        return rates

    def encode(self, observation: ObservationRecord) -> Dict[str, np.ndarray]:
        y = np.asarray(observation.as_vector(), dtype=float)
        if y.size < self.config.n_obs:
            y = np.pad(y, (0, self.config.n_obs - y.size))
        elif y.size > self.config.n_obs:
            y = y[: self.config.n_obs]
        # Gentle normalization so different archetypes stay in a similar band.
        y_n = y / (1.0 + np.abs(y))
        pn = np.tanh(self.w_in @ y_n + self.pn_bias)
        self.pn_rate = pn
        kc_drive = self.w_pn_kc @ pn - 0.35 * self.apl * np.mean(np.maximum(pn, 0.0))
        kc = self._k_winners(kc_drive)
        self.kc_rate = kc
        self.last_sparsity = float(np.mean(kc > 1e-4))
        mbon_target = np.tanh(self.w_kc_mbon @ kc)
        # First-order MBON filter (LIF-inspired rate)
        alpha = 1.0 - np.exp(-1.0 / max(self.config.tau_mbon, 1e-3))
        self.mbon_rate = (1.0 - alpha) * self.mbon_rate + alpha * mbon_target
        return {"pn": self.pn_rate, "kc": self.kc_rate, "mbon": self.mbon_rate}

    def decode(self) -> np.ndarray:
        raw = self.w_out @ self.mbon_rate
        return np.clip(raw, 0.0, 1.0)

    def apply_plasticity(self, da: float, dt: float) -> None:
        if not self.config.plastic:
            return
        self.w_kc_mbon = self.plasticity.step(
            self.w_kc_mbon, self.kc_rate, self.mbon_rate, da, dt
        )
        self.last_da = float(da)

    def reward(
        self,
        observation: ObservationRecord,
        concentrations_sum: float,
    ) -> float:
        burden = observation.tumor_burden
        resist = observation.resistance_frequency
        fusion = observation.fusion_allele_fraction
        if self.prev_burden is None:
            delta_b = 0.0
            delta_r = 0.0
            delta_f = 0.0
        else:
            delta_b = burden - self.prev_burden
            delta_r = resist - self.prev_resist
            delta_f = fusion - (self.prev_fusion or 0.0)
        self.prev_burden = burden
        self.prev_resist = resist
        self.prev_fusion = fusion
        da = dopamine_signal(delta_b, concentrations_sum, delta_r, delta_f)
        self.last_da = da
        return da

    def step(
        self,
        observation: ObservationRecord,
        concentrations_sum: float,
        dt: float,
    ) -> np.ndarray:
        self.encode(observation)
        da = self.reward(observation, concentrations_sum)
        self.apply_plasticity(da, dt)
        return self.decode()

    def telemetry(self) -> Dict[str, object]:
        # Downsample KCs for the UI heatmap.
        kc = self.kc_rate
        if kc.size > 64:
            buckets = 64
            stride = kc.size // buckets
            # Max-pool so sparse k-WTA winners stay visible in the UI heatmap.
            kc_ds = kc[: stride * buckets].reshape(buckets, stride).max(axis=1)
        else:
            kc_ds = kc
        return {
            "n_kc": int(self.n_kc),
            "n_mbon": int(self.n_mbon),
            "n_pn": int(self.n_pn),
            "kc_sparsity": float(self.last_sparsity),
            "kc_mean": float(np.mean(self.kc_rate)),
            "kc_rates": [float(v) for v in kc_ds],
            "mbon_rates": [float(v) for v in self.mbon_rate],
            "pn_mean": float(np.mean(np.abs(self.pn_rate))),
            "da": float(self.last_da),
            "plasticity_norm": float(np.linalg.norm(self.w_kc_mbon)),
            "source": self.compiled.subcircuit.source,
        }
