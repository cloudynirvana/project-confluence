"""
Space Physiology ODE System — 7D Coupled Dynamics
===================================================

Models the evolution of the 7-dimensional Astronaut Resilience Profile
Ψ_space under microgravity stressors, countermeasure interventions,
and cross-system physiological coupling.

The system structure mirrors the oncology ComplexAttractorODE:
    dΨ/dt = Decay(Ψ, phase) + Restoration(u, Ψ) + Coupling(Ψ) + Noise

Key differences from oncology ODE:
    - 7D instead of 15D (computational efficiency for on-board use)
    - Gravity-dependent decay instead of tumor-driven
    - Countermeasure restoration instead of drug-driven
    - Mission phase modulation of parameters
    - Circadian forcing on Ψ₁ is a PRIMARY driver, not secondary

References:
    Baevsky et al. (2017) — HRV changes in long-duration spaceflight
    Flynn-Evans et al. (2016) — Circadian clock in astronauts
    Garrett-Bakelman et al. (2019) — Twins Study multi-system changes
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.integrate import solve_ivp

from .state_vector import (
    N_DIMENSIONS, HEALTHY_GROUND_REFERENCE,
    COUPLING_MATRIX, PsiDimension, AstronautResilienceProfile,
)
from .countermeasures import (
    CountermeasureVector, COUNTERMEASURE_EFFICACY,
    compute_restoration_rate,
)
from .mission_phase import (
    MissionPhase, PhaseProfile, PHASE_PROFILES,
    get_phase_profile, gravity_factor, get_phase_at_day,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. ODE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SpaceODEParams:
    """Parameters for the 7D space physiology ODE."""

    # Global coupling strength
    coupling_strength: float = 0.1

    # Maximum restoration rate per dimension (per day)
    max_restoration_rate: float = 0.20


    # Circadian forcing parameters
    circadian_amplitude: float = 0.03
    circadian_period_hours: float = 24.0

    # Noise parameters (stochastic biological variability)
    noise_sigma: float = 0.005

    # Homeostatic pull toward individual set-points (per day)
    homeostatic_rate: float = 0.02

    # Mission-specific overrides
    mission_type: str = "iss_6month"

    # Crew-specific baseline (fitted from pre-flight data)
    crew_baseline: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.crew_baseline is None:
            self.crew_baseline = HEALTHY_GROUND_REFERENCE.copy()


# ═══════════════════════════════════════════════════════════════════════════
# 2. SPACE PHYSIOLOGY ODE
# ═══════════════════════════════════════════════════════════════════════════

class SpacePhysiologyODE:
    """
    7D coupled ODE for astronaut physiological dynamics.

    dΨ/dt = F(Ψ, u, t)

    where:
        Ψ ∈ ℝ⁷ = astronaut resilience state
        u ∈ ℝ⁷ = countermeasure scalar intensities
        t = mission elapsed time (days)

    Components:
        1. Phase-dependent gravity decay
        2. Countermeasure restoration
        3. Cross-system coupling
        4. Circadian forcing (on Ψ₁)
        5. Homeostatic pull
        6. Stochastic noise

    Usage:
        ode = SpacePhysiologyODE()
        result = ode.simulate(total_days=180, mission_type='iss_6month')
    """

    DIM = N_DIMENSIONS

    def __init__(self, params: Optional[SpaceODEParams] = None):
        self.params = params or SpaceODEParams()
        self._rng = np.random.RandomState(42)

    def rhs(self, t: float, psi: np.ndarray,
            countermeasure_scalars: np.ndarray,
            phase: MissionPhase) -> np.ndarray:
        """Right-hand side of the ODE: dΨ/dt = F(Ψ, u, t).

        Args:
            t: Time in days since launch.
            psi: Current 7D state vector.
            countermeasure_scalars: 7D countermeasure intensity [0, 1].
            phase: Current mission phase.

        Returns:
            dpsi_dt: 7D derivative vector.
        """
        p = self.params
        psi = np.clip(psi, 0.0, 1.0)  # Physical bounds
        dpsi = np.zeros(self.DIM)

        # ── 1. Phase-dependent gravity decay ──
        profile = get_phase_profile(phase)
        dpsi += profile.baseline_decay

        # ── 2. Countermeasure restoration ──
        deficit = p.crew_baseline - psi
        restoration = compute_restoration_rate(
            countermeasure_scalars, deficit,
            max_rate=p.max_restoration_rate)
        dpsi += restoration

        # ── 3. Cross-system coupling ──
        from .state_vector import compute_coupling_drift
        coupling_drift = compute_coupling_drift(
            psi, p.crew_baseline, p.coupling_strength)
        dpsi += coupling_drift

        # ── 4. Circadian forcing on Ψ₁ ──
        circ = p.circadian_amplitude * np.sin(
            2.0 * np.pi * t / (p.circadian_period_hours / 24.0))
        dpsi[PsiDimension.CIRCADIAN.value] += circ

        # ── 5. Homeostatic pull toward baseline ──
        # Weak pull toward crew baseline (biological homeostasis)
        homeostatic = p.homeostatic_rate * (p.crew_baseline - psi)
        # Homeostasis is weaker in microgravity
        g_factor = gravity_factor(profile.gravity_g)
        homeostatic *= (1.0 - 0.7 * g_factor)  # 30% of homeostasis remains at 0G
        dpsi += homeostatic

        return dpsi

    def rhs_with_noise(self, t: float, psi: np.ndarray,
                       countermeasure_scalars: np.ndarray,
                       phase: MissionPhase,
                       dt: float) -> np.ndarray:
        """RHS with stochastic noise (for Euler-Maruyama integration)."""
        dpsi = self.rhs(t, psi, countermeasure_scalars, phase)
        noise = self.params.noise_sigma * np.sqrt(dt) * self._rng.randn(self.DIM)
        return dpsi + noise / dt  # Convert to rate

    def simulate(self,
                 total_days: float = 180.0,
                 dt: float = 0.25,
                 mission_type: str = "iss_6month",
                 countermeasure_schedule: Optional[Dict[float, np.ndarray]] = None,
                 initial_psi: Optional[np.ndarray] = None,
                 use_noise: bool = True,
                 seed: int = 42) -> Dict:
        """Run a full mission simulation.

        Args:
            total_days: Mission duration in days.
            dt: Timestep in days (0.25 = 6 hours).
            mission_type: Mission type for phase determination.
            countermeasure_schedule: Dict of {day: 7D countermeasure vector}.
                                   Linearly interpolates between keyframes.
                                   If None, uses default cruise countermeasures.
            initial_psi: Starting state. Defaults to healthy baseline.
            use_noise: Include stochastic noise.
            seed: Random seed.

        Returns:
            Dict with time, trajectory, phase_history, and summary metrics.
        """
        self._rng = np.random.RandomState(seed)
        self.params.mission_type = mission_type

        n_steps = int(total_days / dt)
        t = np.arange(n_steps + 1) * dt

        if initial_psi is None:
            initial_psi = self.params.crew_baseline.copy()
        psi = initial_psi.copy()

        # Trajectory storage
        trajectory = np.zeros((n_steps + 1, self.DIM))
        trajectory[0] = psi.copy()
        phases = []
        cm_history = np.zeros((n_steps, N_DIMENSIONS))

        # Default countermeasure: standard cruise protocol
        default_cm = CountermeasureVector.default_cruise().to_scalar_vector()

        # Build interpolated countermeasure function
        if countermeasure_schedule is None:
            def get_cm(day):
                return default_cm
        else:
            sorted_days = sorted(countermeasure_schedule.keys())
            cm_arrays = [countermeasure_schedule[d] for d in sorted_days]

            def get_cm(day):
                if day <= sorted_days[0]:
                    return cm_arrays[0]
                if day >= sorted_days[-1]:
                    return cm_arrays[-1]
                for i in range(len(sorted_days) - 1):
                    if sorted_days[i] <= day < sorted_days[i + 1]:
                        alpha = ((day - sorted_days[i]) /
                                 (sorted_days[i + 1] - sorted_days[i]))
                        return (1 - alpha) * cm_arrays[i] + alpha * cm_arrays[i + 1]
                return default_cm

        # Integration loop (Euler-Maruyama for stochastic, Euler for deterministic)
        start_time = time.perf_counter()

        for step in range(n_steps):
            day = t[step]
            phase = get_phase_at_day(day, mission_type)
            phases.append(phase.name)
            cm = get_cm(day)
            cm_history[step] = cm

            if use_noise:
                dpsi = self.rhs_with_noise(day, psi, cm, phase, dt)
            else:
                dpsi = self.rhs(day, psi, cm, phase)

            psi = psi + dpsi * dt
            psi = np.clip(psi, 0.0, 1.0)  # Physical bounds
            trajectory[step + 1] = psi.copy()

        elapsed = time.perf_counter() - start_time

        # Build ARP at final timepoint
        final_arp = AstronautResilienceProfile(
            psi_circadian=float(psi[0]),
            psi_autonomic=float(psi[1]),
            psi_immune=float(psi[2]),
            psi_microbiome=float(psi[3]),
            psi_musculoskeletal=float(psi[4]),
            psi_neuro_ocular=float(psi[5]),
            psi_cognitive=float(psi[6]),
            mission_day=total_days,
            mission_phase=phases[-1] if phases else "unknown",
        )
        final_arp.archetype, final_arp.archetype_confidence = \
            final_arp.classify_archetype()
        final_arp.alerts = final_arp.check_safe_corridor()

        # Summary metrics
        min_psi = np.min(trajectory[1:], axis=0)
        mean_psi = np.mean(trajectory[1:], axis=0)
        max_distance = np.max(
            np.linalg.norm(
                trajectory[1:] - self.params.crew_baseline, axis=1))

        return {
            "t": t,
            "trajectory": trajectory,  # Shape (n_steps+1, 7)
            "phases": phases,
            "countermeasure_history": cm_history,
            "final_arp": final_arp,
            "metrics": {
                "min_per_dimension": {
                    f"Psi_{i}": round(float(min_psi[i]), 4)
                    for i in range(self.DIM)
                },
                "mean_per_dimension": {
                    f"Psi_{i}": round(float(mean_psi[i]), 4)
                    for i in range(self.DIM)
                },
                "max_distance_from_healthy": round(float(max_distance), 4),
                "final_distance": round(float(final_arp.distance_from_healthy), 4),
                "archetype": final_arp.archetype,
                "alerts_at_end": final_arp.alerts,
                "runtime_seconds": round(elapsed, 3),
            },
        }


# ═══════════════════════════════════════════════════════════════════════════
# 3. SCENARIO RUNNERS
# ═══════════════════════════════════════════════════════════════════════════

def simulate_no_countermeasures(total_days: float = 180.0,
                                mission_type: str = "iss_6month",
                                seed: int = 42) -> Dict:
    """Simulate what happens WITHOUT countermeasures.

    This is the 'worst case' baseline — pure gravity-induced
    deconditioning with cross-system coupling.
    """
    zero_cm = np.zeros(N_DIMENSIONS)
    schedule = {0.0: zero_cm, total_days: zero_cm}
    ode = SpacePhysiologyODE()
    return ode.simulate(
        total_days=total_days,
        mission_type=mission_type,
        countermeasure_schedule=schedule,
        use_noise=True,
        seed=seed,
    )


def simulate_standard_protocol(total_days: float = 180.0,
                                mission_type: str = "iss_6month",
                                seed: int = 42) -> Dict:
    """Simulate with standard ISS countermeasure protocol.

    Uses the default cruise countermeasure vector throughout.
    """
    ode = SpacePhysiologyODE()
    return ode.simulate(
        total_days=total_days,
        mission_type=mission_type,
        use_noise=True,
        seed=seed,
    )


def compare_protocols(total_days: float = 180.0,
                      mission_type: str = "iss_6month",
                      seed: int = 42) -> Dict:
    """Compare no-countermeasure vs standard protocol.

    Returns structured comparison for analysis.
    """
    result_none = simulate_no_countermeasures(total_days, mission_type, seed)
    result_standard = simulate_standard_protocol(total_days, mission_type, seed)

    return {
        "mission_type": mission_type,
        "total_days": total_days,
        "no_countermeasures": {
            "final_distance": result_none["metrics"]["final_distance"],
            "archetype": result_none["metrics"]["archetype"],
            "min_per_dimension": result_none["metrics"]["min_per_dimension"],
        },
        "standard_protocol": {
            "final_distance": result_standard["metrics"]["final_distance"],
            "archetype": result_standard["metrics"]["archetype"],
            "min_per_dimension": result_standard["metrics"]["min_per_dimension"],
        },
        "countermeasure_benefit": round(
            result_none["metrics"]["final_distance"] -
            result_standard["metrics"]["final_distance"], 4),
        "full_results": {
            "no_countermeasures": result_none,
            "standard_protocol": result_standard,
        },
    }
