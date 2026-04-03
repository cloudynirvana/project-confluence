"""
Adaptive Countermeasure Controller (ACC) — Space Medicine Module
=================================================================

Closed-loop controller that observes the astronaut's 7D Ψ state and
outputs an optimized countermeasure vector u(t).

Inherits the same design philosophy as the oncology AdaptiveController:
    π(state) → action, not a static prescription.

Three policy modes:
    1. ThresholdPolicy      — Activate countermeasures when Ψ drops below thresholds
    2. ProportionalPolicy   — Scale countermeasures proportional to deficit
    3. RobustAdaptivePolicy — Full Confluence policy with cross-system awareness

Safety invariant: ALL medication decisions require flight surgeon approval.
The controller suggests; it does not prescribe.

Architecture:
    DataIngestion → SpaceComplexityProfiler → ARP (Ψ_space)
                                              ↓
    SpaceCountermeasureController → π(Ψ, phase, constraints) → u(t)
                                              ↓
                                     SafetyAssuranceLayer → enforced u(t)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .state_vector import (
    N_DIMENSIONS, HEALTHY_GROUND_REFERENCE, SAFE_CORRIDOR,
    PsiDimension, PSI_LABELS, AstronautResilienceProfile,
)
from .countermeasures import (
    CountermeasureVector, CountermeasureConstraints,
    COUNTERMEASURE_EFFICACY, COUNTERMEASURE_NAMES,
    N_COUNTERMEASURES, TelemedicineLevel,
)
from .mission_phase import (
    MissionPhase, get_phase_profile, get_phase_at_day,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. POLICY MODES
# ═══════════════════════════════════════════════════════════════════════════

class SpacePolicyMode(Enum):
    """Available adaptive policy modes for countermeasure optimization."""
    THRESHOLD = "threshold"
    PROPORTIONAL = "proportional"
    ROBUST_ADAPTIVE = "robust_adaptive"


# ═══════════════════════════════════════════════════════════════════════════
# 2. POLICY PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SpacePolicyParams:
    """Hyperparameters of the countermeasure policy.

    These are what gets optimized for each crew member / mission type.
    """
    # Threshold policy: activate when Ψ drops below these
    activation_thresholds: np.ndarray = field(
        default_factory=lambda: np.array([
            0.55,  # Circadian
            0.55,  # Autonomic
            0.50,  # Immune
            0.50,  # Microbiome
            0.55,  # Musculoskeletal
            0.55,  # Neuro-ocular
            0.55,  # Cognitive
        ]))

    # Deactivation thresholds (hysteresis band)
    deactivation_thresholds: np.ndarray = field(
        default_factory=lambda: np.array([
            0.65,  # Circadian
            0.65,  # Autonomic
            0.60,  # Immune
            0.60,  # Microbiome
            0.65,  # Musculoskeletal
            0.65,  # Neuro-ocular
            0.65,  # Cognitive
        ]))

    # Proportional gain per dimension
    proportional_gains: np.ndarray = field(
        default_factory=lambda: np.array([
            1.5,   # Circadian — high gain (light is effective)
            1.2,   # Autonomic
            1.0,   # Immune — moderate (limited interventions)
            0.8,   # Microbiome — low gain (slow response)
            1.3,   # Musculoskeletal — high (exercise works)
            0.9,   # Neuro-ocular — moderate
            1.4,   # Cognitive — high (sleep/light responsive)
        ]))

    # Robust adaptive: uncertainty margin
    uncertainty_margin: float = 0.10

    # Emergency threshold: any Ψ below this triggers max countermeasure
    emergency_threshold: float = 0.25

    # Maximum countermeasure intensity per channel
    max_intensity: np.ndarray = field(
        default_factory=lambda: np.ones(N_COUNTERMEASURES))


# ═══════════════════════════════════════════════════════════════════════════
# 3. CONTROLLER STATE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SpaceControllerState:
    """Internal state of the countermeasure controller (memory)."""
    is_active: np.ndarray = field(
        default_factory=lambda: np.zeros(N_COUNTERMEASURES, dtype=bool))
    current_intensities: np.ndarray = field(
        default_factory=lambda: np.zeros(N_COUNTERMEASURES))
    consecutive_active_days: np.ndarray = field(
        default_factory=lambda: np.zeros(N_COUNTERMEASURES))
    changes_today: int = 0
    last_change_day: float = -1.0

    # History
    intensity_history: List[np.ndarray] = field(default_factory=list)
    decision_log: List[str] = field(default_factory=list)
    escalation_events: List[Dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# 4. ADAPTIVE COUNTERMEASURE CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════

class SpaceCountermeasureController:
    """
    Closed-loop adaptive countermeasure controller for astronaut health.

    Observes the 7D Ψ state, applies a policy function, enforces safety
    constraints, and outputs a countermeasure vector.

    Usage:
        controller = SpaceCountermeasureController(
            policy_mode=SpacePolicyMode.ROBUST_ADAPTIVE)

        # Each control cycle (e.g., every 6 hours):
        psi = current_arp.psi_vector
        u_scalar = controller.decide(psi, mission_day=42.0,
                                      phase=MissionPhase.CRUISE_LEO)
        # u_scalar is a 7D vector of countermeasure intensities [0, 1]
    """

    def __init__(self,
                 policy_mode: SpacePolicyMode = SpacePolicyMode.ROBUST_ADAPTIVE,
                 policy_params: Optional[SpacePolicyParams] = None,
                 constraints: Optional[CountermeasureConstraints] = None):
        self.mode = policy_mode
        self.params = policy_params or SpacePolicyParams()
        self.constraints = constraints or CountermeasureConstraints()
        self.state = SpaceControllerState()

    def reset(self):
        """Reset controller to initial state."""
        self.state = SpaceControllerState()

    def decide(self, psi: np.ndarray,
               mission_day: float,
               phase: MissionPhase,
               dt: float = 0.25) -> np.ndarray:
        """Core policy function: π(Ψ, phase) → u(t).

        Args:
            psi: Current 7D Ψ state vector.
            mission_day: Current mission elapsed time (days).
            phase: Current mission phase.
            dt: Control timestep (days).

        Returns:
            u_scalar: 7D countermeasure intensity vector in [0, 1].
        """
        psi = np.clip(psi, 0.0, 1.0)

        # Select policy
        if self.mode == SpacePolicyMode.THRESHOLD:
            raw_u = self._threshold_policy(psi, phase)
        elif self.mode == SpacePolicyMode.PROPORTIONAL:
            raw_u = self._proportional_policy(psi, phase)
        elif self.mode == SpacePolicyMode.ROBUST_ADAPTIVE:
            raw_u = self._robust_adaptive_policy(psi, phase)
        else:
            raw_u = np.zeros(N_COUNTERMEASURES)

        # Apply safety constraints
        u = self._apply_safety(raw_u, psi, mission_day, dt)

        # Update state
        self._update_state(u, psi, mission_day, dt)

        return u

    # ── Policy Implementations ──────────────────────────────────────────

    def _threshold_policy(self, psi: np.ndarray,
                          phase: MissionPhase) -> np.ndarray:
        """Bang-bang control with hysteresis.

        For each Ψ dimension, activate the most effective countermeasure
        when Ψ drops below threshold, deactivate when it recovers.
        """
        p = self.params
        u = np.zeros(N_COUNTERMEASURES)

        for dim_idx in range(N_DIMENSIONS):
            val = psi[dim_idx]
            if val < p.activation_thresholds[dim_idx]:
                # Find the most effective countermeasure for this dimension
                efficacies = COUNTERMEASURE_EFFICACY[dim_idx, :]
                best_cm = int(np.argmax(efficacies))
                u[best_cm] = p.max_intensity[best_cm]
            elif val > p.deactivation_thresholds[dim_idx]:
                # Above recovery threshold — maintain current
                pass

        return np.clip(u, 0, 1)

    def _proportional_policy(self, psi: np.ndarray,
                             phase: MissionPhase) -> np.ndarray:
        """Countermeasure intensity proportional to Ψ deficit.

        u_j = Σ_i (gain_i * deficit_i * efficacy_ij)
        """
        p = self.params
        deficit = np.maximum(HEALTHY_GROUND_REFERENCE - psi, 0.0)
        weighted_deficit = deficit * p.proportional_gains

        # Map dimension deficits to countermeasure intensities via efficacy matrix
        u = COUNTERMEASURE_EFFICACY.T @ weighted_deficit
        u = np.clip(u, 0, p.max_intensity)

        return u

    def _robust_adaptive_policy(self, psi: np.ndarray,
                                phase: MissionPhase) -> np.ndarray:
        """Full Confluence policy: proportional + phase-aware + emergency.

        1. Base proportional response
        2. Phase-specific priority weighting
        3. Emergency escalation for critical dimensions
        4. Cross-system awareness (preemptive action on coupled dimensions)
        """
        p = self.params

        # 1. Check for emergency
        emergency_dims = np.where(psi < p.emergency_threshold)[0]
        if len(emergency_dims) > 0:
            self.state.decision_log.append(
                f"EMERGENCY: Ψ dims {emergency_dims.tolist()} "
                f"below {p.emergency_threshold}")
            # Max countermeasures on all channels affecting emergency dims
            u = np.zeros(N_COUNTERMEASURES)
            for dim_idx in emergency_dims:
                u += COUNTERMEASURE_EFFICACY[dim_idx, :] * p.max_intensity
            return np.clip(u, 0, 1)

        # 2. Base proportional response with uncertainty margin
        adjusted_ref = HEALTHY_GROUND_REFERENCE - p.uncertainty_margin
        deficit = np.maximum(adjusted_ref - psi, 0.0)
        weighted_deficit = deficit * p.proportional_gains

        # Map to countermeasure space
        u = COUNTERMEASURE_EFFICACY.T @ weighted_deficit

        # 3. Phase-specific priority weighting
        profile = get_phase_profile(phase)
        priority_weights = 1.0 / (profile.countermeasure_priority + 1.0)
        priority_weights = priority_weights / np.max(priority_weights)
        # Only apply priority weighting to the first 7 countermeasures
        n = min(len(priority_weights), N_COUNTERMEASURES)
        u[:n] *= priority_weights[:n]

        # 4. Cross-system preemption
        # If a dimension is declining but still in safe corridor,
        # preemptively boost its top countermeasure
        for dim_idx in range(N_DIMENSIONS):
            low, _ = SAFE_CORRIDOR[PsiDimension(dim_idx)]
            if low < psi[dim_idx] < low + 0.15:
                # Approaching safe corridor boundary
                best_cm = int(np.argmax(COUNTERMEASURE_EFFICACY[dim_idx, :]))
                u[best_cm] = max(u[best_cm], 0.5)
                self.state.decision_log.append(
                    f"PREEMPTIVE: {PSI_LABELS[dim_idx]} approaching boundary "
                    f"({psi[dim_idx]:.3f}), boosting {COUNTERMEASURE_NAMES[best_cm]}")

        return np.clip(u, 0, p.max_intensity)

    # ── Safety Layer ────────────────────────────────────────────────────

    def _apply_safety(self, raw_u: np.ndarray, psi: np.ndarray,
                      mission_day: float, dt: float) -> np.ndarray:
        """Apply safety constraints (the Assurance Layer).

        These cannot be overridden by any policy.
        """
        u = raw_u.copy()
        c = self.constraints

        # 1. Rate limit: max 3 countermeasure changes per day
        current_day = int(mission_day)
        if current_day != int(self.state.last_change_day):
            self.state.changes_today = 0

        n_changes = np.sum(
            np.abs(u - self.state.current_intensities) > 0.1)
        if self.state.changes_today + n_changes > c.max_countermeasure_changes_per_day:
            # Revert to current — too many changes
            self.state.decision_log.append(
                f"RATE_LIMIT: {n_changes} changes blocked "
                f"(daily limit: {c.max_countermeasure_changes_per_day})")
            u = self.state.current_intensities.copy()

        # 2. Auto-escalate telemedicine if any Ψ critically low
        escalation = c.check_escalation(psi)
        if escalation == TelemedicineLevel.URGENT:
            u[6] = 1.0  # Telemedicine channel = max
            self.state.decision_log.append(
                f"AUTO_ESCALATE: Ψ below {c.telemedicine_auto_escalate_psi_min}")
            self.state.escalation_events.append({
                "day": mission_day,
                "level": "URGENT",
                "psi_min": float(np.min(psi)),
                "dim": int(np.argmin(psi)),
            })

        # 3. Medication channel always flagged for flight surgeon
        if u[4] > 0.1:
            self.state.decision_log.append(
                "MEDICATION_FLAG: Controller suggests medication — "
                "requires flight surgeon approval")

        # 4. Bound all channels to [0, 1]
        u = np.clip(u, 0.0, 1.0)

        return u

    # ── State Update ────────────────────────────────────────────────────

    def _update_state(self, u: np.ndarray, psi: np.ndarray,
                      mission_day: float, dt: float):
        """Update controller memory."""
        s = self.state

        n_changes = np.sum(np.abs(u - s.current_intensities) > 0.1)
        s.changes_today += int(n_changes)
        s.last_change_day = mission_day
        s.current_intensities = u.copy()
        s.is_active = u > 0.05
        s.consecutive_active_days += dt * s.is_active
        s.consecutive_active_days *= s.is_active  # Reset inactive
        s.intensity_history.append(u.copy())

    # ── Reporting ───────────────────────────────────────────────────────

    def get_summary(self) -> Dict:
        """Return a summary of the controller's behavior."""
        s = self.state
        history = np.array(s.intensity_history) if s.intensity_history else np.zeros((1, N_COUNTERMEASURES))

        return {
            "policy_mode": self.mode.value,
            "total_decisions": len(s.intensity_history),
            "mean_intensity_per_channel": {
                COUNTERMEASURE_NAMES[i]: round(float(np.mean(history[:, i])), 4)
                for i in range(N_COUNTERMEASURES)
            },
            "escalation_events": s.escalation_events,
            "decision_log_tail": s.decision_log[-15:],
        }


# ═══════════════════════════════════════════════════════════════════════════
# 5. INTEGRATED SIMULATION WITH CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════

def run_controlled_simulation(
    total_days: float = 180.0,
    dt: float = 0.25,
    mission_type: str = "iss_6month",
    policy_mode: SpacePolicyMode = SpacePolicyMode.ROBUST_ADAPTIVE,
    policy_params: Optional[SpacePolicyParams] = None,
    initial_psi: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict:
    """Run a full closed-loop simulation with the ACC.

    The controller observes Ψ at each timestep and adjusts
    countermeasures. The ODE integrates the combined effect.

    Returns:
        Dict with trajectories, controller summary, and comparison metrics.
    """
    from .space_ode import SpacePhysiologyODE, SpaceODEParams

    ode_params = SpaceODEParams(mission_type=mission_type)
    ode = SpacePhysiologyODE(ode_params)
    controller = SpaceCountermeasureController(
        policy_mode=policy_mode,
        policy_params=policy_params,
    )

    n_steps = int(total_days / dt)
    t = np.arange(n_steps + 1) * dt

    if initial_psi is None:
        initial_psi = HEALTHY_GROUND_REFERENCE.copy()
    psi = initial_psi.copy()

    trajectory = np.zeros((n_steps + 1, N_DIMENSIONS))
    trajectory[0] = psi.copy()
    cm_trajectory = np.zeros((n_steps, N_COUNTERMEASURES))

    rng = np.random.RandomState(seed)

    for step in range(n_steps):
        day = t[step]
        phase = get_phase_at_day(day, mission_type)

        # Controller decision
        u = controller.decide(psi, day, phase, dt)
        cm_trajectory[step] = u

        # ODE step
        dpsi = ode.rhs(day, psi, u, phase)
        noise = ode_params.noise_sigma * np.sqrt(dt) * rng.randn(N_DIMENSIONS)
        psi = psi + dpsi * dt + noise
        psi = np.clip(psi, 0.0, 1.0)
        trajectory[step + 1] = psi.copy()

    # Final ARP
    final_arp = AstronautResilienceProfile(
        psi_circadian=float(psi[0]),
        psi_autonomic=float(psi[1]),
        psi_immune=float(psi[2]),
        psi_microbiome=float(psi[3]),
        psi_musculoskeletal=float(psi[4]),
        psi_neuro_ocular=float(psi[5]),
        psi_cognitive=float(psi[6]),
        mission_day=total_days,
    )
    final_arp.archetype, final_arp.archetype_confidence = \
        final_arp.classify_archetype()
    final_arp.alerts = final_arp.check_safe_corridor()

    return {
        "mission_type": mission_type,
        "total_days": total_days,
        "policy_mode": policy_mode.value,
        "t": t,
        "trajectory": trajectory,
        "countermeasure_trajectory": cm_trajectory,
        "final_arp": final_arp,
        "controller_summary": controller.get_summary(),
        "metrics": {
            "final_distance": round(float(final_arp.distance_from_healthy), 4),
            "final_psi_mean": round(float(final_arp.psi_mean), 4),
            "min_dimension": final_arp.min_dimension,
            "archetype": final_arp.archetype,
            "alerts": final_arp.alerts,
        },
    }
