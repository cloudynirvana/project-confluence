"""JSON serializers for live frames."""

from __future__ import annotations

from typing import Any, Dict

from confluence.loop import SimFrame


def frame_to_dict(frame: SimFrame) -> Dict[str, Any]:
    latent = frame.latent
    obs = frame.observation
    return {
        "type": "frame",
        "t": frame.t,
        "terminal": frame.terminal,
        "latent": {
            "T_s": latent.T_s,
            "T_r": latent.T_r,
            "I_act": latent.I_act,
            "I_exh": latent.I_exh,
            "S_fib": latent.S_fib,
            "L": latent.L,
            "O": latent.O,
            "G": latent.G,
            "C_tgfb": latent.C_tgfb,
            "C_ifng": latent.C_ifng,
            "H": latent.H,
            "tumor_burden": latent.tumor_burden,
            "resistance_frequency": latent.resistance_frequency,
        },
        "observed": {
            "tumor_burden": obs.tumor_burden,
            "resistance_frequency": obs.resistance_frequency,
            "lactate": obs.lactate,
            "tgfb": obs.tgfb,
            "immune_competence_ratio": obs.immune_competence_ratio,
            "host_toxicity_warning": obs.host_toxicity_warning,
        },
        "drugs": {
            "U": frame.action.infusion,
            "C": frame.concentrations,
            "occupancy": frame.occupancies,
            "source": frame.action.source,
            "notes": frame.action.notes,
        },
        "connectome": frame.connectome
        or {
            "n_kc": 0,
            "kc_sparsity": 0.0,
            "kc_rates": [],
            "mbon_rates": [],
            "da": 0.0,
            "plasticity_norm": 0.0,
            "source": "none",
        },
        "embodiment": frame.embodiment,
    }
