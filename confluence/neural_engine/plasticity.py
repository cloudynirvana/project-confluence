"""Dopamine-modulated KC→MBON plasticity.

    dW_ij / dt = η · DA(t) · KC_j · MBON_i − λ W_ij
    DA(t) = −Δburden − α · Σ C_k − β · Δresistance − γ · Δfusion_AF

Weights are clipped to a compact interval so they cannot explode under
large reward transients (see tests/test_plasticity_bounds.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def dopamine_signal(
    delta_burden: float,
    concentrations_sum: float,
    delta_resistance: float,
    delta_fusion: float = 0.0,
    alpha: float = 0.08,
    beta: float = 0.35,
    gamma: float = 0.40,
    clip: float = 2.5,
) -> float:
    """Negative when burden, resistance, or fusion AF rises, or drug load is high."""
    da = (
        -float(delta_burden)
        - alpha * float(concentrations_sum)
        - beta * float(delta_resistance)
        - gamma * float(delta_fusion)
    )
    return float(np.clip(da, -clip, clip))


@dataclass
class PlasticityParams:
    eta: float = 0.08
    decay: float = 0.01
    w_min: float = -1.5
    w_max: float = 1.5
    da_clip: float = 2.5


class DopaminePlasticity:
    def __init__(self, params: PlasticityParams | None = None):
        self.params = params or PlasticityParams()

    def step(
        self,
        weights: np.ndarray,
        kc: np.ndarray,
        mbon: np.ndarray,
        da: float,
        dt: float,
    ) -> np.ndarray:
        p = self.params
        da = float(np.clip(da, -p.da_clip, p.da_clip))
        kc = np.asarray(kc, dtype=float)
        mbon = np.asarray(mbon, dtype=float)
        # Outer product MBON_i × KC_j
        hebb = np.outer(mbon, kc)
        dw = p.eta * da * hebb - p.decay * weights
        updated = weights + dt * dw
        return np.clip(updated, p.w_min, p.w_max)
