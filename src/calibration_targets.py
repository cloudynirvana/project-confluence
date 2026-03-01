"""
Calibration Target Profiles
============================

Target data extracted from tnbc_results.txt for use with the
calibration pipeline in calibration.py.

These targets represent known/expected outcomes that the simulation
should reproduce when parameters are correctly tuned.
"""

import numpy as np
from typing import Dict


# ─── Metabolite Endpoint Targets (from tnbc_results.txt, t=20) ────────────

METABOLITE_NAMES = [
    "Glucose", "Lactate", "Pyruvate", "ATP", "NADH",
    "Glutamine", "Glutamate", "aKG", "Citrate", "ROS",
]

TNBC_UNTREATED_ENDPOINT = np.array([
    0.240,   # Glucose
    0.002,   # Lactate
    0.056,   # Pyruvate
    0.124,   # ATP
    0.032,   # NADH
   -0.124,   # Glutamine
   -0.109,   # Glutamate
   -0.136,   # aKG
   -0.331,   # Citrate
    0.023,   # ROS
])

TNBC_TREATED_ENDPOINT = np.array([
    0.224,   # Glucose  (DN 0.016)
    0.034,   # Lactate  (UP 0.033)
    0.114,   # Pyruvate (UP 0.059)
    0.064,   # ATP      (DN 0.060)
    0.040,   # NADH     (UP 0.008)
    0.045,   # Glutamine(UP 0.169)
   -0.038,   # Glutamate(UP 0.071)
   -0.137,   # aKG      (DN 0.002)
   -0.442,   # Citrate  (DN 0.110)
    0.083,   # ROS      (UP 0.060)
])

# ─── Coherence Scores ─────────────────────────────────────────────────────

COHERENCE_TARGETS = {
    "healthy_baseline": 0.640,
    "tnbc_untreated": 0.640,
    "tnbc_treated": 0.664,
    "restoration_pct": 6.6,  # % toward healthy
}

# ─── Stability Targets ────────────────────────────────────────────────────

STABILITY_TARGETS = {
    "healthy_lyapunov": -0.2000,
    "tnbc_lyapunov": -0.0640,
    "treated_lyapunov": -0.0829,
    "coherence_deficit": 0.1360,
    "coupling_disruption": 1.1151,
}

# ─── Most Disrupted Pathways ──────────────────────────────────────────────

DISRUPTED_PATHWAYS = [
    ("ROS",      0.590),
    ("Lactate",  0.506),
    ("Pyruvate", 0.379),
]

# ─── Sparse Correction Targets ────────────────────────────────────────────

SPARSE_CORRECTIONS = {
    (9, 9): -0.5571,  # ROS → ROS
    (1, 1): -0.4893,  # Lactate → Lactate
    (2, 1): -0.2996,  # Pyruvate → Lactate
    (5, 6): -0.2764,  # Glutamine → Glutamate
    (0, 0): -0.2285,  # Glucose → Glucose
}


# ─── Scenario Distance Targets (for standalone simulation calibration) ────

def build_scenario_targets() -> Dict[str, float]:
    """
    Build target dict for use with calibration.coarse_grid_search().

    These represent the expected final distances for the standalone
    simulation under each protocol:
      Standard of Care  → FAIL   (far from healthy)
      Iatrogenic Trap   → FAIL   (even further)
      Geometric Cure    → SUCCESS (close to healthy)
    """
    return {
        "Standard":   5.76,   # Checkpoint only → stays far
        "Iatrogenic": 6.10,   # Epogen deepens → further
        "Geometric":  0.14,   # Flatten+Heat+Push → near healthy
    }


def build_metabolite_target(treated: bool = False) -> Dict[str, float]:
    """
    Build metabolite-level target dict.

    Args:
        treated: If True, return treated endpoint; else untreated.
    """
    endpoint = TNBC_TREATED_ENDPOINT if treated else TNBC_UNTREATED_ENDPOINT
    return {name: float(val) for name, val in zip(METABOLITE_NAMES, endpoint)}
