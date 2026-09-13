"""Disease-class taxonomy: state signatures, not cosmetic labels.

Five research modes. Each changes latent ODE knobs **and** the observation
map ``Y = g(X) + ε`` so a controller can discriminate them. None of these
is a clinical stage, TNM label, or diagnostic claim.

Parameter mapping (class → knobs) is the dict ``CLASS_PARAM_MAP`` below.
Values are simulation-scaled; citation-tied rows are marked in
``PARAM_PROVENANCE``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from confluence.contracts import DISEASE_CLASS_IDS

CLASS_ARCHETYPE_IDS: Tuple[str, ...] = (
    "benign",
    "malignant",
    "occult",
    "dormant",
    "terminal",
)

# Documented class → parameter overlay. Tissue archetypes (GBM / PDAC /
# melanoma) are malignant by default and keep their own numeric tables.
CLASS_PARAM_MAP: Dict[str, Dict[str, Any]] = {
    "benign": {
        "r_s": 0.045,
        "r_r": 0.020,
        "r_f": 0.018,
        "k_carry": 0.85,
        "kappa_immune": 0.58,
        "p_tgfb": 0.04,
        "sigma_stroma": 0.04,
        "invasion_factor": 0.35,
        "immune_evasion": 0.30,
        "clinical_visibility_k": 0.08,
        "k_junction_shed": 0.25,
        "growth_awake_gate": False,
        "awaken_hazard": 0.0,
        "kappa_burden": 0.012,
        "notes": "Growth-limited, low invasion / immune evasion; Y is high-SNR and quiet.",
    },
    "malignant": {
        "r_s": 0.22,
        "r_r": 0.14,
        "r_f": 0.17,
        "k_carry": 4.2,
        "kappa_immune": 0.18,
        "p_tgfb": 0.18,
        "invasion_factor": 1.0,
        "immune_evasion": 1.0,
        "clinical_visibility_k": 0.12,
        "k_junction_shed": 0.85,
        "growth_awake_gate": False,
        "awaken_hazard": 0.0,
        "kappa_burden": 0.055,
        "notes": "Aggressive bulk growth + immune evasion (GBM-like malignant mode).",
    },
    "occult": {
        "r_s": 0.16,
        "r_r": 0.10,
        "r_f": 0.14,
        "k_carry": 3.4,
        "kappa_immune": 0.22,
        "clinical_visibility_k": 1.35,
        "k_junction_shed": 2.20,
        "occult_af_leak": 0.55,
        "invasion_factor": 0.85,
        "immune_evasion": 0.90,
        "growth_awake_gate": False,
        "awaken_hazard": 0.0,
        "notes": "Low bulk Y until late; junction / occult-AF leak while burden is hidden.",
    },
    "dormant": {
        "r_s": 0.18,
        "r_r": 0.12,
        "r_f": 0.15,
        "k_carry": 3.6,
        "growth_awake_gate": True,
        "awaken_hazard": 0.07,
        "awaken_jump": 0.50,
        "clinical_visibility_k": 0.20,
        "k_junction_shed": 1.10,
        "invasion_factor": 0.90,
        "immune_evasion": 0.80,
        "notes": "Near-zero net growth while awake≈0; stochastic awakening raises dormancy_exit.",
    },
    "terminal": {
        "r_s": 0.24,
        "r_r": 0.16,
        "r_f": 0.18,
        "k_carry": 5.0,
        "kappa_immune": 0.10,
        "kappa_burden": 0.14,
        "kappa_tox": 0.18,
        "r_host": 0.015,
        "invasion_factor": 1.15,
        "immune_evasion": 1.20,
        "clinical_visibility_k": 0.08,
        "k_junction_shed": 0.90,
        "growth_awake_gate": False,
        "awaken_hazard": 0.0,
        "notes": "High burden and/or H-failure dynamics from t=0.",
    },
}

PARAM_PROVENANCE: Dict[str, Dict[str, str]] = {
    "kappa_immune": {
        "status": "citation-tied-class",
        "doi": "10.1038/nri.2017.96",
        "note": "Chen & Mellman, Nat Rev Immunol 2017 (cancer-immunity cycle). Used only to justify a nonzero immune-kill term, not a numeric clinical rate.",
    },
    "p_tgfb": {
        "status": "citation-tied-class",
        "doi": "10.1038/nrc3603",
        "note": "Pickup et al. / TGF-β in TME reviews. Simulation-scaled production, not a measured pg/mL rate.",
    },
    "awaken_hazard": {
        "status": "placeholder",
        "doi": "",
        "note": "PLACEHOLDER: dormancy-exit hazard is a research knob (Aguirre-Ghiso-class intuition), not a fitted patient waiting time.",
    },
    "clinical_visibility_k": {
        "status": "placeholder",
        "doi": "",
        "note": "PLACEHOLDER: occult visibility Hill scale. Encodes 'hidden until late', not a LoD of a named assay.",
    },
    "k_junction_shed": {
        "status": "citation-tied-class",
        "doi": "10.1038/nrc.2016.97",
        "note": "Fusion / neoantigen shedding intuition (junction peptides). Coefficient is simulation-scaled.",
    },
    "rho_surv": {
        "status": "placeholder",
        "doi": "",
        "note": "PLACEHOLDER: surveillance priming rate. Not a measured vaccine or TIL expansion constant.",
    },
    "r_s": {
        "status": "placeholder",
        "doi": "",
        "note": "PLACEHOLDER: sensitive-clone growth rate. Simulation-scaled, not a fitted patient doubling time.",
    },
    "k_carry": {
        "status": "placeholder",
        "doi": "",
        "note": "PLACEHOLDER: carrying capacity in research burden units, not a clinical tumor-volume K.",
    },
    "rho_immune": {
        "status": "placeholder",
        "doi": "",
        "note": "PLACEHOLDER: immune recruitment rate. Not a measured TIL influx constant.",
    },
    "kappa_burden": {
        "status": "placeholder",
        "doi": "",
        "note": "PLACEHOLDER: host-health burden load. Not a CTCAE organ-toxicity coefficient.",
    },
}


def assert_known_class(name: str) -> str:
    key = name.strip().lower()
    if key not in DISEASE_CLASS_IDS:
        raise KeyError(f"unknown disease class '{name}'; choose from {DISEASE_CLASS_IDS}")
    return key


__all__ = [
    "CLASS_ARCHETYPE_IDS",
    "CLASS_PARAM_MAP",
    "DISEASE_CLASS_IDS",
    "PARAM_PROVENANCE",
    "assert_known_class",
]
