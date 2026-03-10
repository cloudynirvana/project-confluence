"""
ODE System Module - Project Confluence
========================================

Unified 15D nonlinear ODE system with sustained oscillatory dynamics.

Design principles:
  - Every metabolite has production + degradation (homeostatic balance)
  - Circadian and ultradian forcing prevents collapse to fixed points
  - Michaelis-Menten saturation provides nonlinearity
  - Nonlinear cross-coupling generates complex dynamics
  - Cancer shifts the balance, producing genuinely different attractors
  - Backward compatible: frozen immune/microenv recovers linear SAEM

Variables (15D):
    Metabolic (10): Glucose, Lactate, Pyruvate, ATP, NADH,
                    Glutamine, Glutamate, aKG, Citrate, ROS
    Immune (3):     I_eff, I_reg, I_exhaust
    Microenv (2):   sigma_stromal, nu_vascular

References:
    Goldberger et al. (2002) - Fractal dynamics in physiology
    Strogatz (2015) - Nonlinear Dynamics and Chaos
    Warburg (1956) - On the origin of cancer cells
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from scipy.integrate import solve_ivp

# ==========================================================================
# STATE VECTOR LAYOUT
# ==========================================================================

MET_SLICE = slice(0, 10)
I_EFF = 10
I_REG = 11
I_EXHAUST = 12
SIGMA = 13
NU = 14

METABOLITE_NAMES = [
    "Glucose", "Lactate", "Pyruvate", "ATP", "NADH",
    "Glutamine", "Glutamate", "aKG", "Citrate", "ROS",
]
STATE_NAMES = METABOLITE_NAMES + [
    "I_eff", "I_reg", "I_exhaust", "sigma_stromal", "nu_vascular",
]
METABOLITES = METABOLITE_NAMES


# ==========================================================================
# PARAMETERS
# ==========================================================================

@dataclass
class ODEParams:
    """Base metabolic ODE parameters (10D linear, backward compat)."""
    glucose_uptake: float = -0.50
    glycolysis_flux: float = 0.40
    lactate_clearance: float = -0.80
    pyruvate_to_lactate: float = 0.10
    pyruvate_to_atp: float = 0.30
    nadh_to_atp: float = 0.20
    atp_turnover: float = -0.20
    nadh_cycling: float = -0.30
    glutamine_utilization: float = -0.40
    glutaminolysis: float = 0.30
    glutamate_to_akg: float = 0.30
    akg_to_citrate: float = 0.20
    ros_clearance: float = -0.90
    nadh_ros_leak: float = 0.10
    ros_atp_damage: float = -0.10
    atp_inhibits_glucose: float = -0.10
    citrate_turnover: float = -0.30


@dataclass
class ExtendedParams(ODEParams):
    """Full parameter set for the 15D complex attractor system."""

    # Michaelis-Menten saturation
    K_saturation: float = 2.0
    K_glucose: float = 3.0
    K_ros: float = 1.5

    # Homeostatic production rates: production_i ~ |decay_i| * setpoint_i
    production_glucose: float = 1.00
    production_lactate: float = 0.40
    production_pyruvate: float = 0.30
    production_atp: float = 0.80
    production_nadh: float = 0.45
    production_glutamine: float = 0.60
    production_glutamate: float = 0.40
    production_akg: float = 0.24
    production_citrate: float = 0.30
    production_ros: float = 0.27

    # Circadian / ultradian forcing
    circadian_amplitude: float = 0.15
    circadian_period: float = 24.0
    ultradian_amplitude: float = 0.10
    ultradian_period: float = 4.0

    # Nonlinear feedback strengths
    ros_glucose_feedback: float = 0.15
    atp_ros_feedback: float = 0.12
    lactate_immune_suppression: float = 0.20
    glucose_immune_fuel: float = 0.15

    # Immune dynamics
    r_prime: float = 0.15
    k_exhaust: float = 0.08
    k_effector_decay: float = 0.03
    dc_signal_scale: float = 0.8
    r_reg: float = 0.06
    k_treg_depletion: float = 0.02
    r_rescue: float = 0.04
    eta_glucose: float = 0.08
    eta_lactate: float = -0.05
    eta_ros: float = -0.12
    eta_atp: float = 0.04

    # Microenvironment
    r_fibrosis: float = 0.02
    k_stroma_degrade: float = 0.01
    r_angio: float = 0.03
    k_vascular_prune: float = 0.04
    vegf_scale: float = 0.5
    delta_glucose: float = 0.02
    delta_drug: float = 0.05

    # Healthy baselines
    I_eff_healthy: float = 0.4
    I_reg_healthy: float = 0.25
    I_exhaust_healthy: float = 0.05
    sigma_healthy: float = 0.1
    nu_healthy: float = 0.7

    # Coupling strengths
    immune_metabolic_coupling: float = 1.0
    metabolic_immune_coupling: float = 1.0
    microenv_coupling: float = 1.0


@dataclass
class GeneratorMetadata:
    """Metadata for a cancer generator."""
    cancer_type: str
    evidence_notes: str
    confidence: str
    tags: List[str] = field(default_factory=list)
    immune_suppression: float = 0.3
    stromal_coupling: float = 0.2


# ==========================================================================
# 15D COMPLEX ATTRACTOR ODE
# ==========================================================================

class ComplexAttractorODE:
    """
    15D nonlinear ODE with sustained oscillatory dynamics.

    The system sustains complex dynamics via:
      1. Homeostatic production terms (prevent collapse to zero)
      2. Circadian + ultradian forcing (sustained external drive)
      3. Michaelis-Menten saturation (nonlinearity source)
      4. Cross-subsystem coupling (metabolic <-> immune <-> microenv)

    Cancer = shifted homeostatic balance + altered coupling.
    Treatment = partial restoration of healthy parameters.
    """

    DIM = 15

    def __init__(self, params=None, use_nonlinear=True,
                 use_immune=True, use_microenv=True):
        self.params = params or ExtendedParams()
        self.use_nonlinear = use_nonlinear
        self.use_immune = use_immune
        self.use_microenv = use_microenv
        self._A = self._build_metabolic_generator()
        self._production = self._build_production_vector()
        self._eta = np.zeros(10)
        self._eta[0] = self.params.eta_glucose
        self._eta[1] = self.params.eta_lactate
        self._eta[3] = self.params.eta_atp
        self._eta[9] = self.params.eta_ros
        self._delta = np.zeros(10)
        self._delta[0] = self.params.delta_glucose
        self._delta[5] = self.params.delta_glucose * 0.5

    def _build_metabolic_generator(self):
        """Build 10x10 metabolic generator matrix."""
        p = self.params
        A = np.zeros((10, 10))
        A[0, 0] = p.glucose_uptake
        A[1, 1] = p.lactate_clearance
        A[2, 2] = -0.30
        A[3, 3] = p.atp_turnover
        A[4, 4] = p.nadh_cycling
        A[5, 5] = p.glutamine_utilization
        A[6, 6] = -0.50
        A[7, 7] = -0.40
        A[8, 8] = p.citrate_turnover
        A[9, 9] = p.ros_clearance
        A[0, 2] = p.glycolysis_flux
        A[2, 3] = p.pyruvate_to_atp
        A[2, 1] = p.pyruvate_to_lactate
        A[4, 3] = p.nadh_to_atp
        A[5, 6] = p.glutaminolysis
        A[6, 7] = p.glutamate_to_akg
        A[7, 8] = p.akg_to_citrate
        A[3, 0] = p.atp_inhibits_glucose
        A[9, 4] = p.nadh_ros_leak
        A[9, 3] = p.ros_atp_damage
        return A

    def _build_production_vector(self):
        """Build the constant production flux vector."""
        p = self.params
        return np.array([
            p.production_glucose, p.production_lactate,
            p.production_pyruvate, p.production_atp,
            p.production_nadh, p.production_glutamine,
            p.production_glutamate, p.production_akg,
            p.production_citrate, p.production_ros,
        ])

    def get_metabolic_generator(self):
        return self._A.copy()

    def healthy_initial_state(self):
        """Return the healthy baseline state vector z0 in R^15."""
        p = self.params
        z0 = np.zeros(self.DIM)
        z0[0] = 2.0    # Glucose
        z0[1] = 0.5    # Lactate
        z0[2] = 1.0    # Pyruvate
        z0[3] = 4.0    # ATP
        z0[4] = 1.5    # NADH
        z0[5] = 1.5    # Glutamine
        z0[6] = 0.8    # Glutamate
        z0[7] = 0.6    # aKG
        z0[8] = 1.0    # Citrate
        z0[9] = 0.3    # ROS
        z0[I_EFF] = p.I_eff_healthy
        z0[I_REG] = p.I_reg_healthy
        z0[I_EXHAUST] = p.I_exhaust_healthy
        z0[SIGMA] = p.sigma_healthy
        z0[NU] = p.nu_healthy
        return z0

    def _circadian_forcing(self, t):
        """Compute circadian + ultradian forcing at time t."""
        p = self.params
        circ = p.circadian_amplitude * np.sin(2.0 * np.pi * t / p.circadian_period)
        ultra = p.ultradian_amplitude * np.sin(2.0 * np.pi * t / p.ultradian_period)
        circ2 = p.circadian_amplitude * np.cos(2.0 * np.pi * t / p.circadian_period)
        ultra2 = p.ultradian_amplitude * np.cos(2.0 * np.pi * t / p.ultradian_period + 0.7)
        return circ, ultra, circ2, ultra2

    def rhs(self, t, z):
        """
        Right-hand side of the 15D ODE: dz/dt = F(z, t)

        Structure:
          dz/dt = Production + Decay(z) + NonlinearCoupling(z) + Forcing(t)
        """
        p = self.params
        dzdt = np.zeros(self.DIM)

        x = np.maximum(z[MET_SLICE], 0.0)
        I_eff_v = np.clip(z[I_EFF], 0, 1)
        I_reg_v = np.clip(z[I_REG], 0, 1)
        I_exh_v = np.clip(z[I_EXHAUST], 0, 1)
        sigma_v = np.clip(z[SIGMA], 0, 1)
        nu_v = np.clip(z[NU], 0, 1)

        circ, ultra, circ2, ultra2 = self._circadian_forcing(t)

        # == METABOLIC DYNAMICS ==
        # 1) Constant production (homeostatic source)
        dzdt[MET_SLICE] += self._production

        # 2) Circadian modulation of production
        dzdt[0] += self._production[0] * (circ + 0.5 * ultra)
        dzdt[3] += self._production[3] * (0.5 * circ2 + ultra2)
        dzdt[5] += self._production[5] * 0.3 * circ
        dzdt[9] += self._production[9] * 0.4 * ultra

        # 3) Linear decay/flux (generator matrix)
        if self.use_nonlinear:
            K = p.K_saturation
            for i in range(10):
                for j in range(10):
                    if abs(self._A[i, j]) > 1e-10:
                        if i == j:
                            dzdt[i] += self._A[i, j] * x[i]
                        else:
                            K_ij = p.K_glucose if j == 0 else (p.K_ros if j == 9 else K)
                            dzdt[i] += self._A[i, j] * x[j] * (K_ij / (K_ij + x[j] + 1e-10))
        else:
            dzdt[MET_SLICE] += self._A @ x

        # 4) Nonlinear metabolic feedbacks
        ros_effect = x[9] / (p.K_ros + x[9] + 1e-10)
        dzdt[0] += p.ros_glucose_feedback * ros_effect * x[0]
        atp_norm = x[3] / (2.0 + x[3] + 1e-10)
        dzdt[9] -= p.atp_ros_feedback * atp_norm * x[9]
        lac_effect = x[1] / (1.0 + x[1] + 1e-10)
        dzdt[0] -= 0.08 * lac_effect * x[0]

        # == IMMUNE DYNAMICS ==
        if self.use_immune:
            coupling_mi = p.metabolic_immune_coupling
            x_h = self.healthy_initial_state()[MET_SLICE]
            tumor_load = np.linalg.norm(x - x_h) / (np.linalg.norm(x_h) + 1e-10)
            tumor_load = np.clip(tumor_load, 0, 3.0)
            ros_norm = np.clip(x[9] / 2.0, 0, 1)
            glucose_norm = np.clip(x[0] / 4.0, 0, 1)
            dc_signal = p.dc_signal_scale * ros_norm

            dI_eff = (p.r_prime * dc_signal * (1.0 - I_eff_v)
                      * (1.0 + p.glucose_immune_fuel * glucose_norm)
                      - p.k_exhaust * I_eff_v * tumor_load
                      - p.k_effector_decay * I_eff_v
                      - p.lactate_immune_suppression * lac_effect * I_eff_v
                      + 0.05 * circ2)

            dI_reg = (p.r_reg * tumor_load * (1.0 - I_reg_v)
                      - p.k_treg_depletion * I_reg_v
                      + 0.02 * circ)

            dI_exhaust = (p.k_exhaust * I_eff_v * tumor_load
                          - p.r_rescue * I_exh_v * (1.0 - tumor_load * 0.3))

            dzdt[I_EFF] = coupling_mi * dI_eff
            dzdt[I_REG] = coupling_mi * dI_reg
            dzdt[I_EXHAUST] = coupling_mi * dI_exhaust

            # Immune -> metabolic coupling
            coupling_im = p.immune_metabolic_coupling
            net_immune = I_eff_v * (1.0 - I_exh_v) * (1.0 - sigma_v * 0.5)
            dzdt[MET_SLICE] += coupling_im * self._eta * net_immune
        else:
            dzdt[I_EFF] = -0.1 * (z[I_EFF] - p.I_eff_healthy)
            dzdt[I_REG] = -0.1 * (z[I_REG] - p.I_reg_healthy)
            dzdt[I_EXHAUST] = -0.1 * (z[I_EXHAUST] - p.I_exhaust_healthy)

        # == MICROENVIRONMENT DYNAMICS ==
        if self.use_microenv:
            coupling_mic = p.microenv_coupling
            if not self.use_immune:
                x_h = self.healthy_initial_state()[MET_SLICE]
                tumor_load = np.linalg.norm(x - x_h) / (np.linalg.norm(x_h) + 1e-10)
                tumor_load = np.clip(tumor_load, 0, 3.0)

            d_sigma = (p.r_fibrosis * tumor_load * (1.0 - sigma_v)
                       - p.k_stroma_degrade * sigma_v)
            glyc_proxy = np.clip(x[0] * abs(self._A[0, 0]) / 2.0, 0, 1)
            vegf = p.vegf_scale * glyc_proxy
            d_nu = (p.r_angio * vegf * (1.0 - nu_v)
                    - p.k_vascular_prune * nu_v * I_eff_v
                    + 0.02 * circ)

            dzdt[SIGMA] = coupling_mic * d_sigma
            dzdt[NU] = coupling_mic * d_nu
            dzdt[MET_SLICE] -= sigma_v * self._delta * x
        else:
            dzdt[SIGMA] = -0.1 * (z[SIGMA] - p.sigma_healthy)
            dzdt[NU] = -0.1 * (z[NU] - p.nu_healthy)

        return dzdt

    def solve(self, z0=None, t_span=(0, 500), dt_eval=0.5,
              method="LSODA", **kwargs):
        """Integrate the ODE system."""
        if z0 is None:
            z0 = self.healthy_initial_state()
        t_eval = np.arange(t_span[0], t_span[1] + dt_eval, dt_eval)
        start = time.perf_counter()
        sol = solve_ivp(
            self.rhs, t_span, z0, method=method,
            t_eval=t_eval, rtol=1e-8, atol=1e-10, max_step=1.0,
            **kwargs,
        )
        elapsed = time.perf_counter() - start
        return {
            "t": sol.t, "z": sol.y, "success": sol.success,
            "message": sol.message, "runtime_seconds": round(elapsed, 3),
            "n_evaluations": sol.nfev, "n_timepoints": len(sol.t),
        }

    def solve_with_perturbation(self, z0=None, t_span=(0, 500),
                                 perturbation_scale=0.1, n_perturbations=5,
                                 **kwargs):
        """Solve with perturbed ICs for Lyapunov estimation."""
        if z0 is None:
            z0 = self.healthy_initial_state()
        ref = self.solve(z0, t_span, **kwargs)
        perturbed = []
        for i in range(n_perturbations):
            z0_p = z0.copy()
            z0_p += perturbation_scale * np.random.randn(self.DIM) * np.abs(z0 + 0.1)
            z0_p = np.clip(z0_p, 0, None)
            p_result = self.solve(z0_p, t_span, **kwargs)
            perturbed.append(p_result)
        return {"reference": ref, "perturbed": perturbed,
                "perturbation_scale": perturbation_scale}


# ==========================================================================
# 10D LINEAR SYSTEM (BACKWARD COMPATIBILITY)
# ==========================================================================

def validate_generator(name, A, all_generators=None):
    """Validate a cancer generator matrix. Returns list of issues."""
    issues = []
    if A.shape != (10, 10):
        issues.append(f"{name}: shape {A.shape}, expected (10, 10)")
    if np.any(np.abs(A) > 5.0):
        issues.append(f"{name}: contains entries |a_ij| > 5.0")
    if all_generators:
        for other_name, other_A in all_generators.items():
            if other_name != name and np.allclose(A, other_A, atol=1e-6):
                issues.append(f"{name}: degenerate copy of {other_name}")
    return issues


class TNBCODESystem:
    """Deterministic TNBC metabolic generator system with pan-cancer support."""
    N = 10

    @classmethod
    def healthy_generator(cls, params=None):
        p = params or ODEParams()
        A = np.zeros((cls.N, cls.N))
        A[0, 0] = p.glucose_uptake;     A[1, 1] = p.lactate_clearance
        A[2, 2] = -0.30;                A[3, 3] = p.atp_turnover
        A[4, 4] = p.nadh_cycling;       A[5, 5] = p.glutamine_utilization
        A[6, 6] = -0.50;                A[7, 7] = -0.40
        A[8, 8] = p.citrate_turnover;   A[9, 9] = p.ros_clearance
        A[0, 2] = p.glycolysis_flux;    A[2, 3] = p.pyruvate_to_atp
        A[2, 1] = p.pyruvate_to_lactate; A[4, 3] = p.nadh_to_atp
        A[5, 6] = p.glutaminolysis;     A[6, 7] = p.glutamate_to_akg
        A[7, 8] = p.akg_to_citrate;     A[3, 0] = p.atp_inhibits_glucose
        A[9, 4] = p.nadh_ros_leak;      A[9, 3] = p.ros_atp_damage
        return A

    @classmethod
    def tnbc_generator(cls, params=None):
        A = cls.healthy_generator(params)
        A[0, 0] *= 2.5; A[0, 2] *= 2.0; A[1, 1] *= 0.4
        A[5, 5] *= 2.0; A[9, 9] *= 0.5; A[8, 8] *= 0.6
        return A

    @classmethod
    def interpolated_generator(cls, alpha, params=None):
        return (1 - alpha) * cls.healthy_generator(params) + alpha * cls.tnbc_generator(params)

    @classmethod
    def bifurcation_scan(cls, n_points=50, params=None):
        alphas = np.linspace(0, 1, n_points)
        max_eigs = np.zeros(n_points)
        curvatures = np.zeros(n_points)
        for i, a in enumerate(alphas):
            A = cls.interpolated_generator(a, params)
            eigs = np.linalg.eigvals(A)
            max_eigs[i] = np.max(np.real(eigs))
            curvatures[i] = np.sum(np.real(eigs) ** 2)
        return {"alpha": alphas, "max_real_eigenvalue": max_eigs, "curvature": curvatures}

    @classmethod
    def pdac_generator(cls, params=None):
        A = cls.healthy_generator(params)
        A[0, 0] *= 3.0; A[0, 2] *= 2.5; A[5, 5] *= 2.5
        A[9, 9] *= 0.45; A[8, 8] *= 0.5; A[1, 1] *= 0.3
        return A

    @classmethod
    def nsclc_generator(cls, params=None):
        A = cls.healthy_generator(params)
        A[0, 0] *= 1.8; A[0, 2] *= 1.5; A[5, 5] *= 1.6
        A[9, 9] *= 0.65; A[2, 3] *= 1.4
        return A

    @classmethod
    def melanoma_generator(cls, params=None):
        A = cls.healthy_generator(params)
        A[0, 0] *= 1.4; A[2, 3] *= 1.8; A[4, 3] *= 1.5
        A[9, 9] *= 0.7; A[3, 3] *= 0.8
        return A

    @classmethod
    def gbm_generator(cls, params=None):
        A = cls.healthy_generator(params)
        A[0, 0] *= 2.8; A[0, 2] *= 2.2; A[1, 1] *= 0.35
        A[5, 5] *= 1.8; A[9, 9] *= 0.5; A[8, 8] *= 0.55
        return A

    @classmethod
    def crc_generator(cls, params=None):
        A = cls.healthy_generator(params)
        A[0, 0] *= 2.0; A[0, 2] *= 1.8; A[5, 5] *= 1.5
        A[9, 9] *= 0.55; A[1, 1] *= 0.5
        return A

    @classmethod
    def hgsoc_generator(cls, params=None):
        A = cls.healthy_generator(params)
        A[0, 0] *= 2.2; A[5, 5] *= 2.2; A[9, 9] *= 0.5
        A[8, 8] *= 0.6; A[4, 3] *= 1.3
        return A

    @classmethod
    def mcrpc_generator(cls, params=None):
        A = cls.healthy_generator(params)
        A[0, 0] *= 1.6; A[8, 8] *= 0.5; A[5, 5] *= 1.8
        A[9, 9] *= 0.6; A[3, 3] *= 0.85
        return A

    @classmethod
    def aml_generator(cls, params=None):
        A = cls.healthy_generator(params)
        A[0, 0] *= 2.0; A[5, 5] *= 2.5; A[9, 9] *= 0.55
        A[4, 3] *= 1.5; A[3, 3] *= 0.75
        return A

    @classmethod
    def hcc_generator(cls, params=None):
        A = cls.healthy_generator(params)
        A[0, 0] *= 2.5; A[0, 2] *= 2.0; A[8, 8] *= 0.45
        A[5, 5] *= 1.5; A[9, 9] *= 0.5
        return A

    @classmethod
    def alzheimers_generator(cls, params=None):
        A = cls.healthy_generator(params)
        A[0, 0] *= 0.6; A[0, 2] *= 0.8; A[6, 6] *= 1.5   # Hypometabolism & Glutamate excitotoxicity
        A[9, 9] *= 0.4; A[3, 3] *= 1.2                   # Oxidative stress & ATP crisis
        return A

    @classmethod
    def parkinsons_generator(cls, params=None):
        A = cls.healthy_generator(params)
        A[4, 3] *= 0.3; A[9, 9] *= 0.35                  # Intramitochondrial complex I defect & severe ROS
        A[8, 8] *= 0.7; A[5, 5] *= 0.8
        return A

    @classmethod
    def all_generators(cls, params=None):
        return {
            "Healthy": cls.healthy_generator(params),
            "TNBC": cls.tnbc_generator(params),
            "PDAC": cls.pdac_generator(params),
            "NSCLC": cls.nsclc_generator(params),
            "Melanoma": cls.melanoma_generator(params),
            "GBM": cls.gbm_generator(params),
            "CRC": cls.crc_generator(params),
            "HGSOC": cls.hgsoc_generator(params),
            "mCRPC": cls.mcrpc_generator(params),
            "AML": cls.aml_generator(params),
            "HCC": cls.hcc_generator(params),
            "Alzheimers": cls.alzheimers_generator(params),
            "Parkinsons": cls.parkinsons_generator(params),
        }


# ==========================================================================
# SIMULATION HELPERS
# ==========================================================================

def simulate_trajectory(generator, x0, t_days=60.0, dt=0.1,
                        noise_sigma=0.0, seed=None):
    """Simulate linear ODE trajectory: dx/dt = Ax + noise."""
    rng = np.random.RandomState(seed)
    n_steps = int(t_days / dt)
    N = len(x0)
    t = np.arange(n_steps) * dt
    x = np.zeros((n_steps, N))
    x[0] = x0.copy()
    for i in range(1, n_steps):
        dx = generator @ x[i - 1] * dt
        if noise_sigma > 0:
            dx += noise_sigma * np.sqrt(dt) * rng.randn(N)
        x[i] = x[i - 1] + dx
    return t, x


class TrajectoryAnalyzer:
    """Utility for basic trajectory analysis."""

    @staticmethod
    def summary_stats(z_trajectory, t):
        stats = {}
        names = STATE_NAMES if z_trajectory.shape[0] == 15 else METABOLITE_NAMES
        for i, name in enumerate(names[:z_trajectory.shape[0]]):
            series = z_trajectory[i, :]
            stats[name] = {
                "mean": float(np.mean(series)),
                "std": float(np.std(series)),
                "min": float(np.min(series)),
                "max": float(np.max(series)),
                "range": float(np.ptp(series)),
                "cv": float(np.std(series) / (np.mean(series) + 1e-10)),
            }
        return stats

    @staticmethod
    def is_bounded(z_trajectory, threshold=100.0):
        return bool(np.all(np.abs(z_trajectory) < threshold))

    @staticmethod
    def is_oscillating(z_trajectory, min_cv=0.01):
        result = {}
        names = STATE_NAMES if z_trajectory.shape[0] == 15 else METABOLITE_NAMES
        for i, name in enumerate(names[:z_trajectory.shape[0]]):
            series = z_trajectory[i, :]
            cv = np.std(series) / (np.mean(series) + 1e-10)
            result[name] = {"oscillating": bool(cv > min_cv), "cv": float(cv)}
        return result


class BasinMapper:
    """Maps attractor basins via initial-condition sampling."""

    def __init__(self, ode_system):
        self.ode = ode_system

    def map_basins(self, n_samples=50, t_settle=200, seed=42):
        rng = np.random.RandomState(seed)
        z0_base = self.ode.healthy_initial_state()
        results = []
        for i in range(n_samples):
            z0 = z0_base * (0.5 + rng.rand(self.ode.DIM))
            sol = self.ode.solve(z0, t_span=(0, t_settle), dt_eval=2.0)
            if sol["success"]:
                final_state = sol["z"][:, -1]
                results.append({"ic": z0, "final": final_state})
        return {"n_samples": len(results), "trajectories": results}
