"""Sparse rate-based network at configurable FlyWire-class size.

N = 166,700 is the user-named whole-brain class size. This is **not** a
literal FlyWire synapse dump and **not** a multicompartment LIF model.

Memory: PN→hidden uses integer fan-in (default 7), never a dense N×N
matrix (166700² float32 ≈ 111 GB). Plasticity is confined to a small
secretory readout (n_out × n_secretory).

The secretory population maps to simulated therapeutic protein / small-
molecule infusion rates. Neurons here are not ribosomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from confluence.contracts import (
    ALL_EFFECTOR_IDS,
    CONTROL_DRUG_IDS,
    DEMO_BRAIN_NEURONS,
    FULL_BRAIN_NEURONS,
    INTERACTIVE_BRAIN_NEURONS,
    OBS_VECTOR_NAMES,
    ObservationRecord,
    PROTEIN_CHANNEL_IDS,
)
from confluence.neural_engine.plasticity import DopaminePlasticity, dopamine_signal

# Re-export scale constants for callers / tests.
__all__ = [
    "DEMO_BRAIN_NEURONS",
    "FULL_BRAIN_NEURONS",
    "FullBrainConfig",
    "FullBrainNetwork",
    "INTERACTIVE_BRAIN_NEURONS",
    "population_sizes",
]


def population_sizes(n_neurons: int) -> Dict[str, int]:
    """Split N into PN / hidden / MBON / secretory / DAN.

    Fractions are order-of-magnitude FlyWire-inspired (PN and MBON are
    small; Kenyon-like hidden cells dominate). They are **not** a census
    of a real materialization.
    """
    n = int(n_neurons)
    if n < 64:
        raise ValueError(f"n_neurons must be >= 64, got {n}")
    n_pn = max(20, int(round(0.0024 * n)))
    n_mbon = max(8, int(round(0.00015 * n)))
    n_secretory = max(16, int(round(0.002 * n)))
    n_dan = max(4, int(round(0.0001 * n)))
    reserved = n_pn + n_mbon + n_secretory + n_dan
    if reserved >= n:
        # Tiny nets: shrink reserved pops so hidden stays the majority.
        n_pn = max(12, n // 20)
        n_mbon = max(6, n // 40)
        n_secretory = max(12, n // 16)
        n_dan = max(4, n // 64)
        reserved = n_pn + n_mbon + n_secretory + n_dan
        if reserved >= n:
            raise ValueError(f"cannot allocate populations for n_neurons={n}")
    n_hidden = n - reserved
    return {
        "n_neurons": n,
        "n_pn": n_pn,
        "n_hidden": n_hidden,
        "n_mbon": n_mbon,
        "n_secretory": n_secretory,
        "n_dan": n_dan,
    }


@dataclass
class FullBrainConfig:
    n_neurons: int = INTERACTIVE_BRAIN_NEURONS
    n_obs: int = len(OBS_VECTOR_NAMES)
    n_out: int = len(ALL_EFFECTOR_IDS)
    sparsity: float = 0.05
    fan_in: int = 7
    tau_readout: float = 0.35
    seed: int = 7
    plastic: bool = True


class FullBrainNetwork:
    """Sparse PN → k-WTA hidden → secretory / MBON readout."""

    def __init__(
        self,
        config: Optional[FullBrainConfig] = None,
        drug_ids: Sequence[str] = ALL_EFFECTOR_IDS,
    ):
        self.config = config or FullBrainConfig()
        self.drug_ids = tuple(drug_ids)
        self.pops = population_sizes(self.config.n_neurons)
        self.n_neurons = self.pops["n_neurons"]
        self.n_pn = self.pops["n_pn"]
        self.n_hidden = self.pops["n_hidden"]
        self.n_mbon = self.pops["n_mbon"]
        self.n_secretory = self.pops["n_secretory"]
        self.n_dan = self.pops["n_dan"]
        self.n_out = int(self.config.n_out)
        self.plasticity = DopaminePlasticity()
        rng = np.random.default_rng(self.config.seed)

        self.w_in = rng.normal(0.0, 0.8, size=(self.n_pn, self.config.n_obs)).astype(np.float32)
        if self.config.n_obs >= 7:
            self.w_in[:, 5:7] += rng.normal(0.35, 0.15, size=(self.n_pn, 2)).astype(np.float32)
        if self.config.n_obs >= 11:
            self.w_in[:, 7:] += rng.normal(
                0.28, 0.12, size=(self.n_pn, self.config.n_obs - 7)
            ).astype(np.float32)
        self.pn_bias = rng.normal(0.0, 0.1, size=self.n_pn).astype(np.float32)
        self.hidden_idx = rng.integers(
            0, self.n_pn, size=(self.n_hidden, self.config.fan_in), dtype=np.int32
        )
        self.hidden_w = rng.normal(0.0, 0.45, size=self.hidden_idx.shape).astype(np.float32)
        self.apl = rng.uniform(0.15, 0.55, size=self.n_hidden).astype(np.float32)

        self.sec_idx = rng.choice(self.n_hidden, size=self.n_secretory, replace=False).astype(np.int32)
        self.mbon_idx = rng.choice(self.n_hidden, size=self.n_mbon, replace=False).astype(np.int32)
        self.dan_idx = rng.choice(self.n_hidden, size=self.n_dan, replace=False).astype(np.int32)
        # Dedicated PN→secretory synapses so biologic readouts are not
        # silenced when those hidden cells lose the k-WTA competition.
        sec_fan = min(4, self.n_pn)
        self.sec_pn_idx = rng.integers(0, self.n_pn, size=(self.n_secretory, sec_fan), dtype=np.int32)
        self.sec_pn_w = rng.normal(0.0, 0.55, size=self.sec_pn_idx.shape).astype(np.float32)

        self.w_sec = rng.normal(0.0, 0.10, size=(self.n_out, self.n_secretory)).astype(np.float32)
        for i in range(self.n_out):
            self.w_sec[i, i % self.n_secretory] += 0.80
            self.w_sec[i, (i + 3) % self.n_secretory] += 0.30
        n_sm = min(len(CONTROL_DRUG_IDS), self.n_out)
        self.w_mbon_u = rng.normal(0.0, 0.08, size=(n_sm, self.n_mbon)).astype(np.float32)
        for i in range(n_sm):
            self.w_mbon_u[i, i % self.n_mbon] += 0.40

        self.pn_rate = np.zeros(self.n_pn, dtype=np.float32)
        self.hidden_rate = np.zeros(self.n_hidden, dtype=np.float32)
        self.mbon_rate = np.zeros(self.n_mbon, dtype=np.float32)
        self.sec_rate = np.zeros(self.n_secretory, dtype=np.float32)
        self.dan_rate = np.zeros(self.n_dan, dtype=np.float32)
        self.u_rate = np.zeros(self.n_out, dtype=np.float32)
        self.last_da = 0.0
        self.last_sparsity = 0.0
        self.prev_burden: Optional[float] = None
        self.prev_resist: Optional[float] = None
        self.prev_fusion: Optional[float] = None

    def _k_winners(self, drive: np.ndarray) -> np.ndarray:
        n = drive.size
        k = max(1, int(round(self.config.sparsity * n)))
        if k >= n:
            rates = np.maximum(drive, 0.0)
        else:
            thresh = np.partition(drive, n - k)[n - k]
            rates = np.maximum(drive - thresh, 0.0)
        peak = float(rates.max()) if rates.size else 0.0
        if peak > 0.0:
            rates = rates / (peak + 1e-8)
        return rates.astype(np.float32, copy=False)

    def encode(self, observation: ObservationRecord) -> Dict[str, np.ndarray]:
        y = np.asarray(observation.as_vector(), dtype=np.float32)
        if y.size < self.config.n_obs:
            y = np.pad(y, (0, self.config.n_obs - y.size))
        elif y.size > self.config.n_obs:
            y = y[: self.config.n_obs]
        y_n = y / (1.0 + np.abs(y))
        pn = np.tanh(self.w_in @ y_n + self.pn_bias)
        self.pn_rate = pn
        gathered = pn[self.hidden_idx]
        drive = np.sum(self.hidden_w * gathered, axis=1)
        drive = drive - 0.35 * self.apl * float(np.mean(np.maximum(pn, 0.0)))
        hidden = self._k_winners(drive)
        self.hidden_rate = hidden
        self.last_sparsity = float(np.mean(hidden > 1e-4))
        sec_drive = np.sum(self.sec_pn_w * pn[self.sec_pn_idx], axis=1)
        sec_from_pn = np.maximum(np.tanh(sec_drive), 0.0).astype(np.float32)
        self.sec_rate = (0.55 * sec_from_pn + 0.45 * hidden[self.sec_idx]).astype(np.float32)
        mbon_target = np.tanh(hidden[self.mbon_idx] * 1.4)
        alpha = 1.0 - np.exp(-1.0 / max(self.config.tau_readout, 1e-3))
        self.mbon_rate = (1.0 - alpha) * self.mbon_rate + alpha * mbon_target
        self.dan_rate = hidden[self.dan_idx]
        return {"pn": self.pn_rate, "hidden": self.hidden_rate, "mbon": self.mbon_rate}

    def decode(self) -> np.ndarray:
        raw = self.w_sec @ self.sec_rate
        n_sm = self.w_mbon_u.shape[0]
        raw = raw.copy()
        raw[:n_sm] = 0.72 * raw[:n_sm] + 0.28 * (self.w_mbon_u @ self.mbon_rate)
        u = np.clip(raw, 0.0, 1.0).astype(np.float32)
        self.u_rate = u
        return u.astype(float)

    def apply_plasticity(self, da: float, dt: float) -> None:
        if not self.config.plastic:
            return
        p = self.plasticity.params
        da = float(np.clip(da, -p.da_clip, p.da_clip))
        # Three-factor rule on the small secretory readout only.
        hebb = np.outer(self.u_rate, self.sec_rate)
        dw = p.eta * da * hebb - p.decay * self.w_sec
        self.w_sec = np.clip(self.w_sec + dt * dw, p.w_min, p.w_max).astype(np.float32)
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

    def snapshot_weights(self) -> Dict[str, np.ndarray]:
        return {
            "w_sec": self.w_sec.copy(),
            "w_mbon_u": self.w_mbon_u.copy(),
        }

    def restore_weights(self, snap: Dict[str, np.ndarray]) -> None:
        self.w_sec = np.asarray(snap["w_sec"], dtype=np.float32).copy()
        if "w_mbon_u" in snap:
            self.w_mbon_u = np.asarray(snap["w_mbon_u"], dtype=np.float32).copy()

    def reset_rates(self) -> None:
        self.pn_rate[:] = 0.0
        self.hidden_rate[:] = 0.0
        self.mbon_rate[:] = 0.0
        self.sec_rate[:] = 0.0
        self.dan_rate[:] = 0.0
        self.u_rate[:] = 0.0
        self.prev_burden = None
        self.prev_resist = None
        self.prev_fusion = None
        self.last_da = 0.0

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            n_neurons=np.int64(self.n_neurons),
            seed=np.int64(self.config.seed),
            fan_in=np.int64(self.config.fan_in),
            w_in=self.w_in,
            pn_bias=self.pn_bias,
            hidden_idx=self.hidden_idx,
            hidden_w=self.hidden_w,
            apl=self.apl,
            sec_idx=self.sec_idx,
            sec_pn_idx=self.sec_pn_idx,
            sec_pn_w=self.sec_pn_w,
            mbon_idx=self.mbon_idx,
            dan_idx=self.dan_idx,
            w_sec=self.w_sec,
            w_mbon_u=self.w_mbon_u,
            drug_ids=np.array(self.drug_ids),
        )
        return path

    def load(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=False)
        n = int(data["n_neurons"])
        if n != self.n_neurons:
            raise ValueError(f"checkpoint n_neurons={n} != network {self.n_neurons}")
        self.w_in = np.asarray(data["w_in"], dtype=np.float32)
        self.pn_bias = np.asarray(data["pn_bias"], dtype=np.float32)
        self.hidden_idx = np.asarray(data["hidden_idx"], dtype=np.int32)
        self.hidden_w = np.asarray(data["hidden_w"], dtype=np.float32)
        self.apl = np.asarray(data["apl"], dtype=np.float32)
        self.sec_idx = np.asarray(data["sec_idx"], dtype=np.int32)
        if "sec_pn_idx" in data.files:
            self.sec_pn_idx = np.asarray(data["sec_pn_idx"], dtype=np.int32)
            self.sec_pn_w = np.asarray(data["sec_pn_w"], dtype=np.float32)
        self.mbon_idx = np.asarray(data["mbon_idx"], dtype=np.int32)
        self.dan_idx = np.asarray(data["dan_idx"], dtype=np.int32)
        self.w_sec = np.asarray(data["w_sec"], dtype=np.float32)
        self.w_mbon_u = np.asarray(data["w_mbon_u"], dtype=np.float32)

    def active_protein_channels(self, threshold: float = 0.05) -> Tuple[str, ...]:
        active = []
        id_to_i = {d: i for i, d in enumerate(self.drug_ids)}
        for pid in PROTEIN_CHANNEL_IDS:
            i = id_to_i.get(pid)
            if i is not None and float(self.u_rate[i]) > threshold:
                active.append(pid)
        return tuple(active)

    def telemetry(self) -> Dict[str, object]:
        hidden = self.hidden_rate
        if hidden.size > 64:
            buckets = 64
            stride = hidden.size // buckets
            kc_ds = hidden[: stride * buckets].reshape(buckets, stride).max(axis=1)
        else:
            kc_ds = hidden
        return {
            "n_neurons": int(self.n_neurons),
            "n_kc": int(self.n_hidden),
            "n_hidden": int(self.n_hidden),
            "n_mbon": int(self.n_mbon),
            "n_pn": int(self.n_pn),
            "n_secretory": int(self.n_secretory),
            "n_dan": int(self.n_dan),
            "kc_sparsity": float(self.last_sparsity),
            "kc_mean": float(np.mean(self.hidden_rate)),
            "kc_rates": [float(v) for v in kc_ds],
            "mbon_rates": [float(v) for v in self.mbon_rate],
            "secretory_rates": [float(v) for v in self.sec_rate[:32]],
            "pn_mean": float(np.mean(np.abs(self.pn_rate))),
            "da": float(self.last_da),
            "plasticity_norm": float(np.linalg.norm(self.w_sec)),
            "source": "sparse_stub_full_brain",
            "protein_active": list(self.active_protein_channels()),
            "scale_note": (
                "rate-based sparse stub; not FlyWire synapses and not LIF compartments"
            ),
        }
