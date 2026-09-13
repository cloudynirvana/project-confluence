"""JSON serializers for live frames."""

from __future__ import annotations

from typing import Any, Dict

from confluence.contracts import CONTROL_DRUG_IDS, FUSION_CHANNEL_IDS, PROTEIN_CHANNEL_IDS
from confluence.loop import SimFrame


def _subset(mapping: Dict[str, float], keys) -> Dict[str, float]:
    return {k: float(mapping.get(k, 0.0)) for k in keys}


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
            "T_f": latent.T_f,
            "tumor_burden": latent.tumor_burden,
            "resistance_frequency": latent.resistance_frequency,
            "fusion_allele_fraction": latent.fusion_allele_fraction,
            "junction_neoantigen": latent.junction_neoantigen,
            "fusion_id": latent.fusion_id,
            "fusion_display": latent.fusion_display,
            "disease_class": latent.disease_class,
            "I_surv": latent.I_surv,
            "A_ready": latent.A_ready,
            "awake": latent.awake,
        },
        "observed": {
            "tumor_burden": obs.tumor_burden,
            "resistance_frequency": obs.resistance_frequency,
            "lactate": obs.lactate,
            "tgfb": obs.tgfb,
            "immune_competence_ratio": obs.immune_competence_ratio,
            "fusion_allele_fraction": obs.fusion_allele_fraction,
            "junction_neoantigen": obs.junction_neoantigen,
            "occult_allele_fraction": obs.occult_allele_fraction,
            "dormancy_exit": obs.dormancy_exit,
            "immune_surveillance": obs.immune_surveillance,
            "antibody_readiness": obs.antibody_readiness,
            "fusion_id": obs.fusion_id,
            "disease_class": obs.disease_class,
            "host_toxicity_warning": obs.host_toxicity_warning,
        },
        "drugs": {
            "U": frame.action.infusion,
            "C": frame.concentrations,
            "occupancy": frame.occupancies,
            "source": frame.action.source,
            "notes": frame.action.notes,
            "small_molecule": {
                "U": _subset(frame.action.infusion, CONTROL_DRUG_IDS),
                "C": _subset(frame.concentrations, CONTROL_DRUG_IDS),
            },
            "protein": {
                "U": _subset(frame.action.infusion, PROTEIN_CHANNEL_IDS),
                "C": _subset(frame.concentrations, PROTEIN_CHANNEL_IDS),
                "active": [
                    pid
                    for pid in PROTEIN_CHANNEL_IDS
                    if float(frame.action.infusion.get(pid, 0.0)) > 0.05
                    or float(frame.occupancies.get(pid, 0.0)) > 0.05
                ],
            },
            "fusion": {
                "U": _subset(frame.action.infusion, FUSION_CHANNEL_IDS),
                "C": _subset(frame.concentrations, FUSION_CHANNEL_IDS),
                "active": [
                    fid
                    for fid in FUSION_CHANNEL_IDS
                    if float(frame.action.infusion.get(fid, 0.0)) > 0.05
                    or float(frame.occupancies.get(fid, 0.0)) > 0.05
                ],
            },
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
