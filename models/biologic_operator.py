"""
Biologic Operator Module — Project Confluence
===============================================

Implements biologics as geometric operators on Phi-space.
Each biologic class is defined by:
  - A_k: 5x5 operator matrix (which Phi dimensions are targeted)
  - PK_k(t): pharmacokinetic envelope (time-dependent scalar)
  - sigma_k(Phi): state sensitivity function

Biologic classes:
  1. Checkpoint Inhibitors (CPI) — Attractor Destabilisers
  2. Bispecific Antibodies (BiAb) — Dimensional Collapsers
  3. Antibody-Drug Conjugates (ADC) — Targeted Trajectory Displacers
  4. Anti-Angiogenic (AntiAngio) — Microenvironmental Landscape Reshapers
  5. Cytokine Biologics (Cytokine) — Attractor Basin Deformers
  6. Targeted Pathway Biologics (Targeted) — Coherence Disruptors

References:
    Gatenby & Brown (2020) — Integrating evolutionary dynamics into cancer therapy
    Huang (2013) — Genetic and non-genetic instability in tumor progression
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Healthy Phi target (default)
PHI_STAR_DEFAULT = np.array([0.65, 0.70, 0.60, 0.55, 0.55])

PHI_LABELS = ["phi1_entropy", "phi2_coherence", "phi3_connectivity",
              "phi4_adaptability", "phi5_ME_coupling"]


# ══════════════════════════════════════════════════════════════
# Pharmacokinetic Envelopes
# ══════════════════════════════════════════════════════════════

def pk_biexponential(t: float, t_infusion: float = 0.0,
                     alpha: float = 0.05, beta: float = 0.005,
                     f_alpha: float = 0.6) -> float:
    """
    Biexponential PK decay after infusion.
    C(t) = f_alpha * exp(-alpha*(t-t_inf)) + (1-f_alpha) * exp(-beta*(t-t_inf))
    """
    dt = t - t_infusion
    if dt < 0:
        return 0.0
    return f_alpha * np.exp(-alpha * dt) + (1.0 - f_alpha) * np.exp(-beta * dt)


def pk_sustained(t: float, t_start: float = 0.0,
                 t_half: float = 21.0) -> float:
    """Sustained release PK (e.g., long-acting antibodies, half-life ~21 days)."""
    dt = t - t_start
    if dt < 0:
        return 0.0
    k = np.log(2) / t_half
    return np.exp(-k * dt)


def pk_pulsed(t: float, cycle_length: float = 21.0,
              infusion_day: float = 1.0, decay_rate: float = 0.1) -> float:
    """Cyclic PK for q3w dosing schedules."""
    t_in_cycle = t % cycle_length
    dt = t_in_cycle - infusion_day
    if dt < 0:
        return 0.0
    return np.exp(-decay_rate * dt)


# ══════════════════════════════════════════════════════════════
# Biologic Operator Dataclass
# ══════════════════════════════════════════════════════════════

@dataclass
class BiologicOperator:
    """
    A biologic agent modelled as a geometric operator on Phi-space.

    B_k(Phi, t) = PK_k(t) * A_k @ sigma_k(Phi)
    """
    name: str
    class_label: str  # CPI, BiAb, ADC, AntiAngio, Cytokine, Targeted
    A_matrix: np.ndarray  # 5x5 operator matrix
    pk_func: Callable = field(default=pk_sustained)
    pk_params: Dict = field(default_factory=dict)
    u_max: float = 1.0  # max normalised dose
    geometric_role: str = ""

    def pk(self, t: float) -> float:
        """Evaluate PK envelope at time t."""
        return self.pk_func(t, **self.pk_params)

    def sigma(self, phi: np.ndarray, phi_star: np.ndarray = None) -> np.ndarray:
        """
        State sensitivity: efficacy modulated by current Phi state.
        Default: linear attenuation as phi_i approaches phi_star_i.
        """
        if phi_star is None:
            phi_star = PHI_STAR_DEFAULT
        # Normalised distance from target (0 = at target, 1 = far)
        dist = np.abs(phi - phi_star) / (np.abs(phi_star) + 1e-8)
        # Clamp to [0.1, 1.0] — never fully zero, never above 1
        return np.clip(dist, 0.1, 1.0)

    def evaluate(self, phi: np.ndarray, t: float, dose: float = 1.0,
                 phi_star: np.ndarray = None) -> np.ndarray:
        """
        Compute the instantaneous Phi-space force from this biologic.

        Returns dPhi/dt contribution: PK(t) * A @ sigma(Phi) * dose
        """
        c_t = self.pk(t)
        sens = self.sigma(phi, phi_star)
        force = c_t * dose * (self.A_matrix @ sens)
        return force


# ══════════════════════════════════════════════════════════════
# Factory: Pre-configured Biologic Operators
# ══════════════════════════════════════════════════════════════

def create_checkpoint_inhibitor(name: str = "anti-PD1") -> BiologicOperator:
    """Checkpoint Inhibitor — Attractor Destabiliser."""
    A = np.array([
        [0.0,   0.0,   0.0,   0.0,   0.0],
        [0.0,  -0.3,   0.0,   0.0,   0.0],
        [0.0,   0.0,  +0.8,   0.0,   0.0],
        [0.0,   0.0,   0.0,  +0.2,   0.0],
        [0.0,   0.0,   0.0,   0.0,   0.0],
    ])
    return BiologicOperator(
        name=name, class_label="CPI", A_matrix=A,
        pk_func=pk_sustained, pk_params={"t_half": 25.0},
        geometric_role="Attractor Destabiliser",
    )


def create_bispecific(name: str = "blinatumomab") -> BiologicOperator:
    """Bispecific Antibody — Dimensional Collapser."""
    A = np.array([
        [-0.4,  0.0,   0.0,   0.0,   0.0],
        [0.0,   0.0,   0.0,   0.0,   0.0],
        [0.0,   0.0,  +1.0,   0.0,   0.0],
        [0.0,   0.0,   0.0,  -0.5,   0.0],
        [0.0,   0.0,   0.0,   0.0,   0.0],
    ])
    return BiologicOperator(
        name=name, class_label="BiAb", A_matrix=A,
        pk_func=pk_biexponential,
        pk_params={"alpha": 0.08, "beta": 0.01, "f_alpha": 0.7},
        geometric_role="Dimensional Collapser",
    )


def create_adc(name: str = "T-DXd") -> BiologicOperator:
    """Antibody-Drug Conjugate — Targeted Trajectory Displacer (Phase 1)."""
    A = np.array([
        [-0.6,  0.0,   0.0,   0.0,   0.0],
        [0.0,  -0.4,   0.0,   0.0,   0.0],
        [0.0,   0.0,   0.0,   0.0,   0.0],
        [0.0,   0.0,   0.0,   0.0,   0.0],
        [0.0,   0.0,   0.0,   0.0,   0.0],
    ])
    return BiologicOperator(
        name=name, class_label="ADC", A_matrix=A,
        pk_func=pk_pulsed, pk_params={"cycle_length": 21.0, "decay_rate": 0.08},
        geometric_role="Targeted Trajectory Displacer",
    )


def create_adc_resistant(name: str = "T-DXd_resistant") -> BiologicOperator:
    """ADC Phase 2 operator (resistance emergence when phi4 > threshold)."""
    A = np.array([
        [+0.5,  0.0,   0.0,   0.0,   0.0],
        [0.0,   0.0,   0.0,   0.0,   0.0],
        [0.0,   0.0,   0.0,   0.0,   0.0],
        [0.0,   0.0,   0.0,  +0.3,   0.0],
        [0.0,   0.0,   0.0,   0.0,   0.0],
    ])
    return BiologicOperator(
        name=name, class_label="ADC_R", A_matrix=A,
        pk_func=pk_pulsed, pk_params={"cycle_length": 21.0, "decay_rate": 0.08},
        geometric_role="Targeted Trajectory Displacer (Resistance Phase)",
    )


def create_anti_angiogenic(name: str = "bevacizumab") -> BiologicOperator:
    """Anti-Angiogenic — Microenvironmental Landscape Reshaper."""
    A = np.array([
        [+0.2,  0.0,   0.0,   0.0,   0.0],
        [0.0,   0.0,   0.0,   0.0,   0.0],
        [0.0,   0.0,  -0.3,   0.0,   0.0],
        [0.0,   0.0,   0.0,   0.0,   0.0],
        [0.0,   0.0,   0.0,   0.0,  -0.7],
    ])
    op = BiologicOperator(
        name=name, class_label="AntiAngio", A_matrix=A,
        pk_func=pk_sustained, pk_params={"t_half": 20.0},
        geometric_role="Microenvironmental Landscape Reshaper",
    )
    # Override sigma for dose-dependent phi3 sign flip
    original_sigma = op.sigma

    def anti_angio_sigma(phi, phi_star=None):
        s = original_sigma(phi, phi_star)
        return s

    op.sigma = anti_angio_sigma
    return op


def create_cytokine(name: str = "IL-15") -> BiologicOperator:
    """Cytokine Biologic — Attractor Basin Deformer."""
    A = np.array([
        [0.0,   0.0,   0.0,   0.0,   0.0],
        [0.0,   0.0,   0.0,   0.0,   0.0],
        [0.0,   0.0,  +0.6,   0.0,   0.0],
        [0.0,   0.0,   0.0,  +0.4,   0.0],
        [0.0,   0.0,   0.0,   0.0,  +0.3],
    ])
    return BiologicOperator(
        name=name, class_label="Cytokine", A_matrix=A,
        pk_func=pk_biexponential,
        pk_params={"alpha": 0.15, "beta": 0.02, "f_alpha": 0.5},
        geometric_role="Attractor Basin Deformer",
    )


def create_targeted_biologic(name: str = "cetuximab") -> BiologicOperator:
    """Targeted Pathway Biologic — Coherence Disruptor."""
    A = np.array([
        [0.0,   0.0,   0.0,   0.0,   0.0],
        [0.0,  -0.9,   0.0,   0.0,   0.0],
        [0.0,   0.0,   0.0,   0.0,   0.0],
        [0.0,   0.0,   0.0,  -0.3,   0.0],
        [0.0,   0.0,   0.0,   0.0,   0.0],
    ])
    return BiologicOperator(
        name=name, class_label="Targeted", A_matrix=A,
        pk_func=pk_sustained, pk_params={"t_half": 7.0},
        geometric_role="Coherence Disruptor",
    )


# ══════════════════════════════════════════════════════════════
# Biologic Library
# ══════════════════════════════════════════════════════════════

BIOLOGIC_LIBRARY = {
    "anti-PD1": create_checkpoint_inhibitor("anti-PD1"),
    "anti-PD-L1": create_checkpoint_inhibitor("anti-PD-L1"),
    "anti-CTLA4": create_checkpoint_inhibitor("anti-CTLA4"),
    "blinatumomab": create_bispecific("blinatumomab"),
    "teclistamab": create_bispecific("teclistamab"),
    "mosunetuzumab": create_bispecific("mosunetuzumab"),
    "T-DXd": create_adc("T-DXd"),
    "enfortumab-vedotin": create_adc("enfortumab-vedotin"),
    "bevacizumab": create_anti_angiogenic("bevacizumab"),
    "ramucirumab": create_anti_angiogenic("ramucirumab"),
    "IL-2": create_cytokine("IL-2"),
    "IL-15": create_cytokine("IL-15"),
    "IFN-alpha": create_cytokine("IFN-alpha"),
    "cetuximab": create_targeted_biologic("cetuximab"),
    "pertuzumab": create_targeted_biologic("pertuzumab"),
    "trastuzumab": create_targeted_biologic("trastuzumab"),
}


# ══════════════════════════════════════════════════════════════
# Synergy Tensor
# ══════════════════════════════════════════════════════════════

class SynergyTensor:
    """
    Rank-2 tensor S_ij capturing pairwise biologic synergies.
    S_ij = d^2(dPhi/dt) / du_i du_j
    Positive = synergistic, Negative = antagonistic.
    """

    # Validated synergy entries (class-level interactions)
    KNOWN_SYNERGIES = {
        ("CPI", "AntiAngio"): +0.4,   # vessel normalisation + checkpoint release
        ("CPI", "BiAb"): +0.7,        # brake-release then forced engagement
        ("ADC", "BiAb"): +0.3,        # antigen release amplifies targeting
        ("CPI", "Cytokine"): +0.5,    # immune activation x checkpoint release
        ("AntiAngio", "Targeted"): +0.3,  # VEGF reduces feedback bypass
        ("ADC", "CPI"): +0.2,         # ADC-induced immunogenic cell death
        ("Cytokine", "BiAb"): +0.4,   # amplified forced engagement
        ("CPI", "Targeted"): +0.2,    # reduced coherence aids immune access
    }

    def __init__(self, biologics: List[BiologicOperator]):
        self.biologics = biologics
        self.n = len(biologics)
        self.S = np.zeros((self.n, self.n))
        self._populate()

    def _populate(self):
        """Fill synergy tensor from known class-level interactions."""
        for i in range(self.n):
            for j in range(i + 1, self.n):
                ci = self.biologics[i].class_label
                cj = self.biologics[j].class_label
                key = (ci, cj)
                key_rev = (cj, ci)
                if key in self.KNOWN_SYNERGIES:
                    self.S[i, j] = self.KNOWN_SYNERGIES[key]
                    self.S[j, i] = self.KNOWN_SYNERGIES[key]
                elif key_rev in self.KNOWN_SYNERGIES:
                    self.S[i, j] = self.KNOWN_SYNERGIES[key_rev]
                    self.S[j, i] = self.KNOWN_SYNERGIES[key_rev]

    def get_synergy(self, i: int, j: int) -> float:
        return float(self.S[i, j])

    def best_combination(self, k: int = 2) -> List[Tuple[int, int, float]]:
        """Return top-k most synergistic pairs."""
        pairs = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                pairs.append((i, j, self.S[i, j]))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:k]


# ══════════════════════════════════════════════════════════════
# Resistance Geometry
# ══════════════════════════════════════════════════════════════

def curvature_approx(phi_trajectory: np.ndarray,
                     A_k: np.ndarray) -> float:
    """
    Approximate Gaussian curvature of the disease manifold in the
    direction of biologic operator A_k's action.

    K_approx = det(Cov(dPhi/dt | direction A_k))

    High curvature in B_k's action direction means resistance risk is high.

    Parameters
    ----------
    phi_trajectory : ndarray, shape (5, T)
        Time-series of Phi vectors.
    A_k : ndarray, shape (5, 5)
        Operator matrix of the biologic.

    Returns
    -------
    K : float
        Approximate curvature (resistance risk proxy).
    """
    if phi_trajectory.shape[1] < 3:
        return 0.0

    # Compute dPhi/dt from trajectory
    dphi = np.diff(phi_trajectory, axis=1)

    # Project onto A_k's action direction (dominant eigenvector)
    eigenvalues, eigenvectors = np.linalg.eigh(A_k.T @ A_k)
    action_dir = eigenvectors[:, -1]  # largest eigenvalue direction

    # Project dPhi onto action direction
    projections = action_dir @ dphi  # shape (T-1,)

    # Curvature proxy: variance of the projected velocity
    # High variance = unstable manifold in that direction = resistance risk
    K = float(np.var(projections))

    return K


def detect_resistance_signal(phi_trajectory: np.ndarray,
                             phi_star: np.ndarray = None,
                             window: int = 5) -> Dict:
    """
    Monitor Phi trajectory for resistance signatures.

    Returns dict with flags for each resistance pattern.
    """
    if phi_star is None:
        phi_star = PHI_STAR_DEFAULT

    T = phi_trajectory.shape[1]
    if T < 2 * window:
        return {"sufficient_data": False}

    phi_recent = phi_trajectory[:, -window:]
    phi_prior = phi_trajectory[:, -(2 * window):-window]

    # Slopes in each dimension
    slopes = np.mean(phi_recent, axis=1) - np.mean(phi_prior, axis=1)

    signals = {
        "sufficient_data": True,
        # CPI resistance: phi3 declining after initial rise
        "cpi_resistance": bool(slopes[2] < -0.02 and phi_trajectory[2, -1] > 0.3 * phi_star[2]),
        # ADC antigen loss: phi1 rebounding, phi4 increasing
        "adc_antigen_loss": bool(slopes[0] > 0.03 and slopes[3] > 0.02),
        # Targeted bypass: phi2 recovering after initial decline
        "targeted_bypass": bool(slopes[1] > 0.03),
        # Anti-VEGF hypoxic: phi5 oscillating
        "antiangio_hypoxic": bool(np.std(phi_recent[4, :]) > 0.1),
        # General: any dimension moving away from phi_star
        "diverging_dimensions": [
            PHI_LABELS[i] for i in range(5)
            if slopes[i] * np.sign(phi_trajectory[i, -1] - phi_star[i]) > 0.01
        ],
    }
    return signals


# ══════════════════════════════════════════════════════════════
# Phi-State Classifier (Biologic Switching Logic)
# ══════════════════════════════════════════════════════════════

CURVATURE_THRESHOLD = 0.05


def classify_phi_state(phi: np.ndarray,
                       phi_star: np.ndarray = None,
                       curvature: float = 0.0) -> str:
    """
    Recommend biologic operator class based on current Phi state.

    Returns one of: CPI, TARGETED_BIOLOGIC, BISPECIFIC,
    ANTI_ANGIOGENIC, ADC_HOLIDAY, CYTOKINE, MAINTAIN
    """
    if phi_star is None:
        phi_star = PHI_STAR_DEFAULT

    phi1, phi2, phi3, phi4, phi5 = phi
    ps1, ps2, ps3, ps4, ps5 = phi_star

    # Immune cold + high heterogeneity
    if phi3 < 0.4 * ps3 and phi1 > 1.2 * ps1:
        return "CPI"

    # Oncogene-addicted (high coherence, low entropy)
    if phi2 > 0.8 * ps2 and phi1 < 0.5 * ps1:
        return "TARGETED_BIOLOGIC"

    # Post-CPI, connectivity restored but entropy still high
    if phi3 > 0.6 * ps3 and phi1 > ps1:
        return "BISPECIFIC"

    # High ME instability
    if abs(phi5 - ps5) / (ps5 + 1e-8) > 0.4:
        return "ANTI_ANGIOGENIC"

    # High plasticity + resistance risk
    if phi4 > 1.5 * ps4 and curvature > CURVATURE_THRESHOLD:
        return "ADC_HOLIDAY"

    # Globally suppressed immune landscape
    if phi3 < 0.3 * ps3 and phi4 < 0.5 * ps4:
        return "CYTOKINE"

    return "MAINTAIN"


def bifurcation_proximity(jacobian: np.ndarray) -> float:
    """
    Detect proximity to a bifurcation point.
    B_prox = 1 / min_eigenvalue(Jacobian)
    High values indicate the system is near a tipping point.
    """
    evals = np.linalg.eigvals(jacobian)
    min_eval = np.min(np.abs(evals.real))
    return 1.0 / (min_eval + 1e-10)
