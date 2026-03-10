"""
Clonal Dynamics Engine — Project Confluence
=============================================

Two-subpopulation Lotka-Volterra competition model for intra-tumor
clonal dynamics. Models sensitive and resistant clones competing for
shared carrying capacity under drug pressure.

The key insight: adaptive therapy maintains the sensitive clone population,
which competitively suppresses the resistant clone through resource competition.
Continuous therapy eliminates sensitive clones, removing competitive pressure
and allowing resistant clones to expand unopposed.

Mathematical Model:
    dS/dt = r_S * S * (1 - (S + α_RS * R) / K) - d_S(drug) * S
    dR/dt = r_R * R * (1 - (R + α_SR * S) / K) - d_R(drug) * R * (1 - resistance)

Where:
    S, R = sensitive and resistant population fractions
    r_S, r_R = intrinsic growth rates (r_S > r_R without drug)
    K = carrying capacity (normalized to 1.0)
    α_RS, α_SR = competition coefficients
    d_S, d_R = drug-induced death rates

References:
    - Gatenby et al. 2009, Cancer Research (adaptive therapy)
    - Enriquez-Navas et al. 2016, Science TM (clinical adaptive)
    - Zhang et al. 2022, PLoS Comput Biol (evolutionary game theory)
    - Strobl et al. 2021, Nature Comms (competition and spatial structure)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ClonalParams:
    """Parameters governing 2-clone Lotka-Volterra dynamics."""
    # Population fractions (sum to 1.0)
    sensitive_initial: float = 0.85        # Initial sensitive fraction
    resistant_initial: float = 0.15        # Initial resistant fraction (pre-existing)
    
    # Growth rates (per day)
    sensitive_growth_rate: float = 0.10    # r_S: Sensitive clone growth
    resistant_growth_rate: float = 0.07    # r_R: Resistant is slower (fitness cost)
    
    # Competition coefficients (Lotka-Volterra α)
    alpha_RS: float = 0.9                  # How much R is suppressed by S
    alpha_SR: float = 0.6                  # How much S is suppressed by R
    
    # Carrying capacity
    carrying_capacity: float = 1.0         # Normalized K
    
    # Drug sensitivity
    drug_kill_rate_sensitive: float = 0.15  # Drug-induced death rate for sensitive
    drug_kill_rate_resistant: float = 0.02  # Resistant is ~7x less sensitive
    
    # Resistance cost
    resistance_fitness_cost: float = 0.03   # Extra metabolic cost for resistant clone
    
    # Minimum population (extinction threshold)
    extinction_threshold: float = 1e-6
    
    # Stochastic noise
    noise_scale: float = 0.005             # Population fluctuation noise


@dataclass
class ClonalState:
    """Current state of the 2-clone tumor."""
    sensitive: float = 0.85
    resistant: float = 0.15
    total_tumor_burden: float = 1.0        # Normalized to initial
    
    # Trajectory tracking
    sensitive_trajectory: List[float] = field(default_factory=list)
    resistant_trajectory: List[float] = field(default_factory=list)
    burden_trajectory: List[float] = field(default_factory=list)
    time_points: List[float] = field(default_factory=list)
    
    @property
    def tumor_fraction(self) -> float:
        """Total viable tumor fraction."""
        return self.sensitive + self.resistant
    
    @property
    def resistant_fraction(self) -> float:
        """Fraction of tumor that is resistant."""
        total = self.sensitive + self.resistant
        if total < 1e-10:
            return 0.0
        return self.resistant / total
    
    @property
    def is_cured(self) -> bool:
        """Tumor is below extinction threshold."""
        return self.tumor_fraction < 0.01  # <1% viable tumor


class ClonalDynamicsEngine:
    """
    2-clone Lotka-Volterra tumor dynamics engine.
    
    Integrates with the SAEM protocol simulation to model how
    drug therapy shapes clonal composition over time.
    
    Usage:
        engine = ClonalDynamicsEngine(params)
        
        # During treatment simulation
        for day in range(total_days):
            drug_pressure = get_drug_effect(day)
            engine.step(dt=0.1, drug_active=True, drug_pressure=drug_pressure)
        
        # Get results
        state = engine.state
        print(f"Resistant fraction: {state.resistant_fraction:.1%}")
    """
    
    def __init__(self, params: Optional[ClonalParams] = None):
        self.params = params or ClonalParams()
        self.state = ClonalState(
            sensitive=self.params.sensitive_initial,
            resistant=self.params.resistant_initial,
        )
        self.t = 0.0
    
    def reset(self):
        """Reset to initial conditions."""
        self.state = ClonalState(
            sensitive=self.params.sensitive_initial,
            resistant=self.params.resistant_initial,
        )
        self.t = 0.0
    
    def step(self, dt: float, drug_active: bool, drug_pressure: float = 1.0,
             phase: str = "flatten", seed: Optional[int] = None):
        """
        Advance one timestep of clonal dynamics.
        
        Args:
            dt: Timestep in days
            drug_active: Whether drugs are being administered
            drug_pressure: Scalar 0-1 reflecting cumulative drug effect strength
            phase: Current protocol phase ("flatten", "heat", "push")
            seed: Optional RNG seed for reproducibility
        """
        p = self.params
        S = self.state.sensitive
        R = self.state.resistant
        K = p.carrying_capacity
        
        if seed is not None:
            rng = np.random.RandomState(seed + int(self.t * 100))
        else:
            rng = np.random
        
        # ── Lotka-Volterra competition ──
        # dS/dt = r_S * S * (1 - (S + α_RS*R)/K) - drug_kill_S * drug * S
        # dR/dt = r_R * R * (1 - (R + α_SR*S)/K) - drug_kill_R * drug * R - cost*R
        
        # Growth terms with competition
        S_growth = p.sensitive_growth_rate * S * (1.0 - (S + p.alpha_RS * R) / K)
        R_growth = p.resistant_growth_rate * R * (1.0 - (R + p.alpha_SR * S) / K)
        
        # Drug kill terms
        if drug_active:
            # Phase-dependent drug pressure scaling
            phase_multiplier = {
                "flatten": 1.0,    # Full metabolic drug pressure
                "heat": 0.3,       # Drug holiday (reduced)
                "push": 0.6,       # Checkpoint focus, less direct kill
            }.get(phase, 1.0)
            
            effective_pressure = drug_pressure * phase_multiplier
            S_kill = p.drug_kill_rate_sensitive * effective_pressure * S
            R_kill = p.drug_kill_rate_resistant * effective_pressure * R
        else:
            S_kill = 0.0
            R_kill = 0.0
        
        # Fitness cost for resistant clone (always active)
        R_cost = p.resistance_fitness_cost * R
        
        # Net rates
        dS = (S_growth - S_kill) * dt
        dR = (R_growth - R_kill - R_cost) * dt
        
        # Add stochastic noise
        noise_S = rng.normal(0, p.noise_scale * np.sqrt(max(S, 0))) * dt
        noise_R = rng.normal(0, p.noise_scale * np.sqrt(max(R, 0))) * dt
        
        # Update
        S_new = max(p.extinction_threshold, S + dS + noise_S)
        R_new = max(p.extinction_threshold, R + dR + noise_R)
        
        self.state.sensitive = S_new
        self.state.resistant = R_new
        self.state.total_tumor_burden = S_new + R_new
        
        # Record trajectory
        self.t += dt
        self.state.time_points.append(round(self.t, 2))
        self.state.sensitive_trajectory.append(round(S_new, 6))
        self.state.resistant_trajectory.append(round(R_new, 6))
        self.state.burden_trajectory.append(round(S_new + R_new, 6))
    
    def run_protocol(self, phase_days: Dict[str, int], 
                     drug_pressure_by_phase: Optional[Dict[str, float]] = None,
                     dt: float = 0.1, seed: int = 42) -> ClonalState:
        """
        Run full 3-phase protocol and return final clonal state.
        
        Args:
            phase_days: {"flatten": n1, "heat": n2, "push": n3}
            drug_pressure_by_phase: Optional override for drug pressure per phase
            dt: Simulation timestep
            seed: Random seed
        
        Returns:
            Final ClonalState with full trajectory
        """
        self.reset()
        
        default_pressures = {"flatten": 1.0, "heat": 0.3, "push": 0.6}
        pressures = drug_pressure_by_phase or default_pressures
        
        for phase_name in ["flatten", "heat", "push"]:
            days = phase_days.get(phase_name, 20)
            n_steps = int(days / dt)
            drug_on = phase_name != "heat"  # Drug holiday during heat
            pressure = pressures.get(phase_name, 1.0)
            
            for i in range(n_steps):
                self.step(dt, drug_active=drug_on, drug_pressure=pressure,
                         phase=phase_name, seed=seed + i)
        
        return self.state
    
    def compare_adaptive_vs_continuous(self, phase_days: Dict[str, int],
                                        dt: float = 0.1, seed: int = 42) -> Dict:
        """
        Compare adaptive (phased) vs continuous (no holiday) therapy
        on clonal composition.
        
        Returns dict with both results and comparison metrics.
        """
        # Adaptive: standard Flatten→Heat→Push
        self.reset()
        adaptive_state = self.run_protocol(phase_days, dt=dt, seed=seed)
        adaptive_result = {
            "final_sensitive": round(adaptive_state.sensitive, 6),
            "final_resistant": round(adaptive_state.resistant, 6),
            "final_resistant_fraction": round(adaptive_state.resistant_fraction, 4),
            "final_burden": round(adaptive_state.tumor_fraction, 6),
            "is_cured": adaptive_state.is_cured,
        }
        
        # Continuous: all drugs on for entire duration
        total_days = sum(phase_days.values())
        self.reset()
        n_steps = int(total_days / dt)
        for i in range(n_steps):
            self.step(dt, drug_active=True, drug_pressure=1.0,
                     phase="flatten", seed=seed + 10000 + i)
        continuous_state = self.state
        continuous_result = {
            "final_sensitive": round(continuous_state.sensitive, 6),
            "final_resistant": round(continuous_state.resistant, 6),
            "final_resistant_fraction": round(continuous_state.resistant_fraction, 4),
            "final_burden": round(continuous_state.tumor_fraction, 6),
            "is_cured": continuous_state.is_cured,
        }
        
        return {
            "adaptive": adaptive_result,
            "continuous": continuous_result,
            "adaptive_wins": (
                adaptive_result["final_resistant_fraction"] 
                < continuous_result["final_resistant_fraction"]
            ),
            "resistant_fraction_advantage": round(
                continuous_result["final_resistant_fraction"] 
                - adaptive_result["final_resistant_fraction"], 4
            ),
            "burden_advantage": round(
                continuous_result["final_burden"] 
                - adaptive_result["final_burden"], 6
            ),
        }


def get_cancer_specific_clonal_params(cancer_type: str) -> ClonalParams:
    """
    Return cancer-type-specific clonal dynamics parameters.
    
    Different cancers have different:
    - Pre-existing resistance fractions
    - Growth rates
    - Competition dynamics
    - Drug sensitivity profiles
    """
    # Base params
    params = ClonalParams()
    
    cancer_overrides = {
        "TNBC": {
            "sensitive_initial": 0.82,
            "resistant_initial": 0.18,
            "sensitive_growth_rate": 0.12,
            "resistant_growth_rate": 0.08,
            "drug_kill_rate_sensitive": 0.14,
        },
        "PDAC": {
            "sensitive_initial": 0.75,
            "resistant_initial": 0.25,      # More pre-existing resistance
            "sensitive_growth_rate": 0.08,
            "resistant_growth_rate": 0.06,
            "drug_kill_rate_sensitive": 0.10, # Harder to kill (stroma)
            "alpha_RS": 0.7,                  # Less competition (spatial barriers)
        },
        "NSCLC": {
            "sensitive_initial": 0.85,
            "resistant_initial": 0.15,
            "sensitive_growth_rate": 0.09,
            "drug_kill_rate_sensitive": 0.16,
        },
        "Melanoma": {
            "sensitive_initial": 0.88,
            "resistant_initial": 0.12,
            "sensitive_growth_rate": 0.11,
            "drug_kill_rate_sensitive": 0.18, # Immune-responsive
        },
        "GBM": {
            "sensitive_initial": 0.78,
            "resistant_initial": 0.22,
            "sensitive_growth_rate": 0.07,
            "resistant_growth_rate": 0.05,
            "drug_kill_rate_sensitive": 0.08, # BBB limits drug delivery
            "resistance_fitness_cost": 0.02,
        },
        "CRC": {
            "sensitive_initial": 0.83,
            "resistant_initial": 0.17,
            "sensitive_growth_rate": 0.10,
            "drug_kill_rate_sensitive": 0.15,
        },
        "HGSOC": {
            "sensitive_initial": 0.80,
            "resistant_initial": 0.20,
            "sensitive_growth_rate": 0.09,
            "drug_kill_rate_sensitive": 0.13,
        },
        "AML": {
            "sensitive_initial": 0.87,
            "resistant_initial": 0.13,
            "sensitive_growth_rate": 0.14,   # Fast (liquid tumor)
            "resistant_growth_rate": 0.10,
            "drug_kill_rate_sensitive": 0.20, # Accessible
            "alpha_RS": 0.95,                 # Strong competition (well-mixed)
        },
        "mCRPC": {
            "sensitive_initial": 0.80,
            "resistant_initial": 0.20,
            "sensitive_growth_rate": 0.08,
            "drug_kill_rate_sensitive": 0.12,
        },
        "HCC": {
            "sensitive_initial": 0.77,
            "resistant_initial": 0.23,
            "sensitive_growth_rate": 0.09,
            "drug_kill_rate_sensitive": 0.11,
            "resistance_fitness_cost": 0.025,
        },
    }
    
    overrides = cancer_overrides.get(cancer_type, {})
    for key, val in overrides.items():
        if hasattr(params, key):
            setattr(params, key, val)
    
    return params
