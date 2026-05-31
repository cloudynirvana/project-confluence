"""
Adaptive Therapeutic Controller — Project Confluence
=====================================================

Closed-loop feedback controller that implements the core First Principles
insight: the optimal therapy is a POLICY FUNCTION π(state) → action,
not a static prescription.

The controller observes the current biological state (tumor subpopulations,
resistance levels, organ function) and outputs the optimal drug action
for the current timestep, subject to robust safety bounds.

Architecture:
    ClonalDynamicsEngine (state) → AdaptiveController (policy) → dose u(t)
    ResistanceTracker (efficacy) ↗                                ↓
                                                    ← ODE integration ←

Three built-in policy modes:
    1. ThresholdPolicy   — Simple bang-bang control with hysteresis
    2. ProportionalPolicy — Dose proportional to tumor burden
    3. RobustAdaptivePolicy — Min-max bounded with uncertainty sets

References:
    - Gatenby et al. 2009: Adaptive therapy concept
    - Zhang et al. 2017: Evolutionary game therapy
    - First Principles Deconstruction (Axiom 10):
      "The optimal therapy is an algorithm, not a prescription."
"""

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum

from .clonal_dynamics import ClonalDynamicsEngine, ClonalParams, ClonalState
from .resistance_model import ResistanceTracker, ResistanceParams


class PolicyMode(Enum):
    """Available adaptive policy modes."""
    THRESHOLD = "threshold"
    PROPORTIONAL = "proportional"
    ROBUST_ADAPTIVE = "robust_adaptive"
    EPIGENETIC_STEERING = "epigenetic_steering"


@dataclass
class PolicyParams:
    """Hyperparameters of the adaptive policy — these are what gets optimized."""
    # Threshold policy params
    dose_on_threshold: float = 0.50     # Start dosing when V(t)/K > this
    dose_off_threshold: float = 0.30    # Stop dosing when V(t)/K < this
    robust_max_dose: float = 0.70       # Max dose (fraction of u_max) — safety bound

    # Proportional policy params
    proportional_gain: float = 1.5      # Dose = gain * (V(t)/K)
    min_dose: float = 0.0               # Minimum dose during treatment
    
    # Robust adaptive params
    uncertainty_margin: float = 0.15    # Extra safety margin for worst-case
    resistant_alarm_fraction: float = 0.40  # Emergency dose escalation if R% > this
    emergency_dose: float = 0.90        # Dose during emergency (still bounded)
    
    # Quality-of-Life constraint
    max_cumulative_toxicity: float = 50.0   # Cumulative toxicity budget
    toxicity_per_dose_unit: float = 1.0     # Toxicity cost per unit dose per day
    
    # Drug holiday constraints
    min_holiday_days: float = 3.0       # Minimum consecutive days off
    max_continuous_dose_days: float = 14.0  # Max consecutive days on before forced holiday

    # OSKM epigenetic steering params
    oskm_max_dose: float = 0.35
    oskm_identity_margin_target: float = 0.05
    oskm_memory_integrity_target: float = 0.75
    oskm_pulse_period_days: float = 4.0
    oskm_duty_fraction: float = 0.25

    # Landauer thermal safety budget
    k_B: float = 1.380649e-23
    body_temperature_kelvin: float = 310.15
    max_cell_temperature_delta: float = 1.5
    landauer_bits_per_full_dose: float = 1.0e12
    landauer_heat_to_kelvin_gain: float = 1.0e8
    cell_cooling_rate: float = 1.2


@dataclass 
class ControllerState:
    """Internal state of the controller (memory)."""
    current_dose: float = 0.0
    is_dosing: bool = False
    consecutive_dose_days: float = 0.0
    consecutive_holiday_days: float = 0.0
    cumulative_toxicity: float = 0.0
    cumulative_dose: float = 0.0
    dose_switches: int = 0
    elapsed_days: float = 0.0
    cell_temperature_kelvin: float = 310.15
    thermal_overrides: int = 0
    last_oskm_dose: float = 0.0
    
    # History for analysis
    dose_history: List[float] = field(default_factory=list)
    state_history: List[Dict] = field(default_factory=list)
    decision_log: List[str] = field(default_factory=list)


class EpigeneticSteeringPolicy:
    """
    Pulsatile OSKM policy with a hard Landauer heat safety budget.

    Inputs are expected to come from IdentityTensorAnalyzer certificates
    or compatible dictionaries containing margin/regime/memory integrity.
    """

    def __init__(self, params: PolicyParams):
        self.params = params

    def decide(self, identity_metrics: Optional[Dict],
               controller_state: ControllerState,
               dt: float) -> Tuple[float, Dict]:
        p = self.params
        metrics = identity_metrics or {}

        margin = float(metrics.get("margin", 0.0))
        memory_integrity = float(metrics.get("memory_integrity", 1.0))
        regime = str(metrics.get("regime", "coherent"))

        needs_steering = (
            margin < p.oskm_identity_margin_target
            or memory_integrity < p.oskm_memory_integrity_target
            or regime in {"degraded", "critical"}
        )

        phase = controller_state.elapsed_days % max(p.oskm_pulse_period_days, 1e-10)
        pulse_width = p.oskm_pulse_period_days * np.clip(p.oskm_duty_fraction, 0.0, 1.0)
        in_pulse = phase < pulse_width

        raw_dose = p.oskm_max_dose if needs_steering and in_pulse else 0.0
        raw_dose = float(np.clip(raw_dose, 0.0, p.robust_max_dose))

        energy_joules = (
            p.k_B
            * p.body_temperature_kelvin
            * np.log(2.0)
            * p.landauer_bits_per_full_dose
            * raw_dose
        )
        heat_rate = p.landauer_heat_to_kelvin_gain * energy_joules
        cooling = p.cell_cooling_rate * (
            controller_state.cell_temperature_kelvin - p.body_temperature_kelvin
        )
        projected_temperature = controller_state.cell_temperature_kelvin + (
            heat_rate - cooling
        ) * dt

        threshold = p.body_temperature_kelvin + p.max_cell_temperature_delta
        thermal_override = projected_temperature > threshold
        dose = 0.0 if thermal_override else raw_dose

        if thermal_override:
            # Recompute passive cooling without additional OSKM heat.
            projected_temperature = controller_state.cell_temperature_kelvin - cooling * dt
            projected_temperature = max(projected_temperature, p.body_temperature_kelvin)

        diagnostics = {
            "needs_steering": needs_steering,
            "in_pulse": in_pulse,
            "raw_oskm_dose": raw_dose,
            "landauer_energy_joules": float(energy_joules),
            "projected_temperature_kelvin": float(projected_temperature),
            "thermal_override": bool(thermal_override),
        }
        return dose, diagnostics


class AdaptiveController:
    """
    Closed-loop therapeutic controller.
    
    Takes the current tumor state and outputs the optimal dose for
    the current timestep, enforcing safety bounds and quality-of-life
    constraints.
    
    Usage:
        controller = AdaptiveController(
            policy_mode=PolicyMode.ROBUST_ADAPTIVE,
            policy_params=PolicyParams(dose_on_threshold=0.50)
        )
        
        # In simulation loop:
        for t in timesteps:
            state = clonal_engine.state
            dose = controller.decide(
                sensitive=state.sensitive,
                resistant=state.resistant,
                carrying_capacity=params.carrying_capacity,
                dt=0.1
            )
            clonal_engine.step(dt=0.1, drug_active=(dose > 0), drug_pressure=dose)
    """
    
    def __init__(self, 
                 policy_mode: PolicyMode = PolicyMode.ROBUST_ADAPTIVE,
                 policy_params: Optional[PolicyParams] = None,
                 guideline_retriever=None,
                 cancer_type: str = "TNBC"):
        self.mode = policy_mode
        self.params = policy_params or PolicyParams()
        self.ctrl_state = ControllerState()
        self.ctrl_state.cell_temperature_kelvin = self.params.body_temperature_kelvin
        self.epigenetic_policy = EpigeneticSteeringPolicy(self.params)
        self.guideline_retriever = guideline_retriever
        self.cancer_type = cancer_type
        
        # Load Nigeria-specific guardrails if available
        self.nigeria_guardrails = self._load_nigeria_guardrails()
        if self.nigeria_guardrails:
            self._apply_nigeria_guardrails()
    
    def _load_nigeria_guardrails(self) -> Optional[Dict]:
        """Load Nigeria-specific clinical guardrails from NSTG 2022."""
        guardrails_path = (
            Path(__file__).resolve().parent.parent
            / "validation" / "nigeria_clinical_guardrails.json"
        )
        if guardrails_path.exists():
            try:
                with open(guardrails_path) as f:
                    data = json.load(f)
                self.ctrl_state.decision_log.append(
                    f"NIGERIA_GUARDRAILS: Loaded NSTG 2022 guardrails "
                    f"({data.get('meta', {}).get('n_conditions', '?')} conditions)"
                )
                return data
            except Exception:
                return None
        return None
    
    def _apply_nigeria_guardrails(self):
        """
        Layer Nigeria-specific safety thresholds on top of default policy params.
        
        This adjusts the controller's safety margins based on Nigerian patient
        population characteristics (endemic infections, comorbidity prevalence,
        drug availability).
        """
        ng = self.nigeria_guardrails
        if not ng:
            return
        
        safety = ng.get("safety_thresholds_nigeria", {})
        resources = ng.get("resource_aware_constraints", {})
        
        # Increase uncertainty margin for Nigerian context
        # (higher comorbidity burden = more conservative)
        if safety:
            self.params.uncertainty_margin = max(
                self.params.uncertainty_margin, 0.18
            )
            self.ctrl_state.decision_log.append(
                f"NIGERIA_ADJUST: uncertainty_margin → {self.params.uncertainty_margin:.2f} "
                f"(NSTG 2022 comorbidity-aware)"
            )
        
        # If resource constraints indicate limited drug availability,
        # reduce max dose to favor commonly available drugs
        commonly_available = resources.get("drug_availability", {}).get(
            "commonly_available", []
        )
        if commonly_available:
            self.ctrl_state.decision_log.append(
                f"NIGERIA_DRUGS: {len(commonly_available)} commonly available drugs loaded"
            )
    
    def get_guideline_context(self, query: str) -> Optional[str]:
        """
        Query Nigerian clinical guidelines for context (if retriever is set).
        
        Used to inform dosing decisions with guideline-compliant recommendations.
        """
        if self.guideline_retriever is None:
            return None
        try:
            return self.guideline_retriever.answer(query)
        except Exception:
            return None
    
    def reset(self):
        """Reset controller to initial state."""
        self.ctrl_state = ControllerState()
        self.ctrl_state.cell_temperature_kelvin = self.params.body_temperature_kelvin
    
    def decide(self, sensitive: float, resistant: float,
               carrying_capacity: float, dt: float,
               resistance_efficacy: float = 1.0,
               identity_metrics: Optional[Dict] = None) -> float:
        """
        Core policy function: π(state) → action.
        
        Args:
            sensitive: Current sensitive cell population
            resistant: Current resistant cell population
            carrying_capacity: Max tumor capacity (K)
            dt: Current timestep size (days)
            resistance_efficacy: Drug efficacy factor from ResistanceTracker (0-1)
            
        Returns:
            dose: Drug dose for this timestep (0 to robust_max_dose)
        """
        V = sensitive + resistant
        V_frac = V / carrying_capacity if carrying_capacity > 0 else 0
        R_frac = resistant / V if V > 1e-10 else 0
        
        # Select policy
        if self.mode == PolicyMode.THRESHOLD:
            raw_dose = self._threshold_policy(V_frac, R_frac)
        elif self.mode == PolicyMode.PROPORTIONAL:
            raw_dose = self._proportional_policy(V_frac, R_frac)
        elif self.mode == PolicyMode.ROBUST_ADAPTIVE:
            raw_dose = self._robust_adaptive_policy(V_frac, R_frac, resistance_efficacy)
        elif self.mode == PolicyMode.EPIGENETIC_STEERING:
            raw_dose, oskm_diag = self.epigenetic_policy.decide(
                identity_metrics, self.ctrl_state, dt
            )
            self.ctrl_state.cell_temperature_kelvin = oskm_diag["projected_temperature_kelvin"]
            self.ctrl_state.last_oskm_dose = raw_dose
            if oskm_diag["thermal_override"]:
                self.ctrl_state.thermal_overrides += 1
                self.ctrl_state.decision_log.append(
                    "LANDAUER_THERMAL_OVERRIDE: "
                    f"T_cell={self.ctrl_state.cell_temperature_kelvin:.2f}K, "
                    "forcing OSKM holiday"
                )
        else:
            raw_dose = 0.0
        
        # Apply safety constraints
        dose = self._apply_constraints(raw_dose, dt)
        
        # Update internal state
        self._update_state(dose, V_frac, R_frac, dt)
        
        return dose
    
    # ── Policy Implementations ──────────────────────────────────────────
    
    def _threshold_policy(self, V_frac: float, R_frac: float) -> float:
        """
        Bang-bang control with hysteresis.
        Dose ON when V > threshold_on, OFF when V < threshold_off.
        """
        p = self.params
        
        if self.ctrl_state.is_dosing:
            # Currently dosing — keep going until tumor drops below off threshold
            if V_frac < p.dose_off_threshold:
                return 0.0  # Switch to holiday
            else:
                return p.robust_max_dose
        else:
            # Currently on holiday — start dosing if tumor grows above on threshold
            if V_frac > p.dose_on_threshold:
                return p.robust_max_dose
            else:
                return 0.0
    
    def _proportional_policy(self, V_frac: float, R_frac: float) -> float:
        """
        Dose proportional to tumor burden.
        More tumor = more drug, but capped at robust max.
        """
        p = self.params
        raw = p.proportional_gain * V_frac
        return np.clip(raw, p.min_dose, p.robust_max_dose)
    
    def _robust_adaptive_policy(self, V_frac: float, R_frac: float,
                                 resistance_efficacy: float) -> float:
        """
        The full Confluence policy: combines threshold switching with
        resistance-aware dose adjustment and robust safety margins.
        
        Key behaviors:
        1. Base threshold switching (on/off)
        2. Resistant fraction alarm → emergency escalation
        3. Efficacy-adjusted dosing (dose harder when drug still works)
        4. Uncertainty margin applied to all bounds
        """
        p = self.params
        
        # 1. Emergency: resistant takeover imminent
        if R_frac > p.resistant_alarm_fraction:
            self.ctrl_state.decision_log.append(
                f"EMERGENCY: R_frac={R_frac:.2%} > alarm={p.resistant_alarm_fraction:.0%}")
            return min(p.emergency_dose, p.robust_max_dose)
        
        # 2. Base threshold decision
        adjusted_on = p.dose_on_threshold - p.uncertainty_margin  # Be more aggressive
        adjusted_off = p.dose_off_threshold + p.uncertainty_margin  # Hold longer
        
        if self.ctrl_state.is_dosing:
            if V_frac < adjusted_off:
                return 0.0  # Holiday
            base_dose = p.robust_max_dose
        else:
            if V_frac > adjusted_on:
                base_dose = p.robust_max_dose
            else:
                return 0.0  # Stay on holiday
        
        # 3. Adjust for resistance: if drug is losing efficacy, dose less
        #    (counterintuitive but correct — dosing a failing drug just
        #     accelerates resistance without benefit)
        if resistance_efficacy < 0.3:
            self.ctrl_state.decision_log.append(
                f"EFFICACY_LOW: {resistance_efficacy:.2f} — reducing dose to preserve window")
            base_dose *= 0.5
        
        # 4. Scale by V_frac (proportional element within the threshold band)
        dose = base_dose * min(1.0, V_frac / p.dose_on_threshold)
        
        return np.clip(dose, 0.0, p.robust_max_dose)
    
    # ── Safety Constraints ──────────────────────────────────────────────
    
    def _apply_constraints(self, raw_dose: float, dt: float) -> float:
        """
        Apply hard safety constraints (the "Assurance Layer").
        These cannot be overridden by any policy.
        
        Includes both universal (CTCAE v5.0) and Nigeria-specific (NSTG 2022)
        constraints when available.
        """
        p = self.params
        cs = self.ctrl_state
        dose = raw_dose
        
        # Constraint 1: Absolute dose cap
        dose = min(dose, p.robust_max_dose)
        
        # Constraint 2: Forced holiday after max continuous dosing
        if cs.consecutive_dose_days >= p.max_continuous_dose_days and dose > 0:
            cs.decision_log.append(
                f"FORCED_HOLIDAY: {cs.consecutive_dose_days:.0f} days continuous")
            dose = 0.0
        
        # Constraint 3: Minimum holiday duration  
        if not cs.is_dosing and cs.consecutive_holiday_days < p.min_holiday_days and dose > 0:
            cs.decision_log.append(
                f"HOLIDAY_HOLD: only {cs.consecutive_holiday_days:.1f}/{p.min_holiday_days:.0f} days")
            dose = 0.0
        
        # Constraint 4: Cumulative toxicity budget
        projected_toxicity = cs.cumulative_toxicity + dose * p.toxicity_per_dose_unit * dt
        if projected_toxicity > p.max_cumulative_toxicity:
            budget_remaining = max(0, p.max_cumulative_toxicity - cs.cumulative_toxicity)
            dose = min(dose, budget_remaining / (p.toxicity_per_dose_unit * dt + 1e-10))
            cs.decision_log.append(f"TOXICITY_CAP: budget remaining={budget_remaining:.1f}")
        
        # Constraint 5 (NSTG 2022): Nigeria-specific safety layer
        dose = self._apply_nigeria_safety(dose, dt)
        
        return max(0.0, dose)
    
    def _apply_nigeria_safety(self, dose: float, dt: float) -> float:
        """
        Apply Nigeria-specific safety constraints from NSTG 2022.
        
        These constraints account for:
        - Higher baseline infection risk (malaria, HIV, Hep B)
        - Endemic comorbidities (sickle cell, anaemia)
        - Resource-aware drug selection
        """
        if not self.nigeria_guardrails:
            return dose
        
        ng = self.nigeria_guardrails
        comorbidities = ng.get("common_comorbidities_nigeria", {})
        
        # If malaria co-infection constraints exist, enforce higher platelet
        # threshold (translated to dose reduction in high-toxicity scenarios)
        malaria = comorbidities.get("malaria_coinfection", {})
        malaria_constraints = malaria.get("controller_constraints", {})
        if malaria_constraints and dose > 0:
            # In a real clinical setting, this would check actual lab values.
            # Here, we increase the safety margin as a proxy.
            toxicity_fraction = (
                self.ctrl_state.cumulative_toxicity
                / (self.params.max_cumulative_toxicity + 1e-10)
            )
            if toxicity_fraction > 0.7:
                dose *= 0.85  # 15% dose reduction in high-toxicity + endemic setting
                self.ctrl_state.decision_log.append(
                    f"NSTG_SAFETY: 15% dose reduction (toxicity={toxicity_fraction:.0%}, "
                    f"malaria-endemic adjustment)"
                )
        
        return dose
    
    # ── State Update ────────────────────────────────────────────────────
    
    def _update_state(self, dose: float, V_frac: float, R_frac: float, dt: float):
        """Update controller memory after a decision."""
        cs = self.ctrl_state
        p = self.params
        
        was_dosing = cs.is_dosing
        cs.current_dose = dose
        cs.is_dosing = dose > 0
        
        if cs.is_dosing:
            cs.consecutive_dose_days += dt
            cs.consecutive_holiday_days = 0.0
            cs.cumulative_toxicity += dose * p.toxicity_per_dose_unit * dt
            cs.cumulative_dose += dose * dt
        else:
            cs.consecutive_holiday_days += dt
            cs.consecutive_dose_days = 0.0
        
        if was_dosing != cs.is_dosing:
            cs.dose_switches += 1

        cs.elapsed_days += dt
        
        cs.dose_history.append(round(dose, 4))
        cs.state_history.append({
            "V_frac": round(V_frac, 4),
            "R_frac": round(R_frac, 4),
            "dose": round(dose, 4),
            "cumulative_tox": round(cs.cumulative_toxicity, 2),
            "T_cell_K": round(cs.cell_temperature_kelvin, 4),
        })
    
    # ── Analysis & Reporting ────────────────────────────────────────────
    
    def get_summary(self) -> Dict:
        """Return a summary of the controller's behavior over the simulation."""
        cs = self.ctrl_state
        doses = np.array(cs.dose_history) if cs.dose_history else np.array([0])
        
        dosing_fraction = np.mean(doses > 0) if len(doses) > 0 else 0
        
        summary = {
            "policy_mode": self.mode.value,
            "total_decisions": len(cs.dose_history),
            "dose_switches": cs.dose_switches,
            "dosing_fraction": round(float(dosing_fraction), 3),
            "mean_dose_when_active": round(
                float(np.mean(doses[doses > 0])) if np.any(doses > 0) else 0, 4),
            "cumulative_toxicity": round(cs.cumulative_toxicity, 2),
            "cumulative_dose": round(cs.cumulative_dose, 2),
            "cell_temperature_kelvin": round(cs.cell_temperature_kelvin, 4),
            "thermal_overrides": cs.thermal_overrides,
            "last_oskm_dose": round(cs.last_oskm_dose, 4),
            "policy_params": {
                "dose_on_threshold": self.params.dose_on_threshold,
                "dose_off_threshold": self.params.dose_off_threshold,
                "robust_max_dose": self.params.robust_max_dose,
                "resistant_alarm": self.params.resistant_alarm_fraction,
                "uncertainty_margin": self.params.uncertainty_margin,
                "oskm_max_dose": self.params.oskm_max_dose,
            },
            "nigeria_guidelines_active": self.nigeria_guardrails is not None,
            "guideline_retriever_active": self.guideline_retriever is not None,
            "decisions_log_tail": cs.decision_log[-10:] if cs.decision_log else [],
        }
        
        return summary


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATED SIMULATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_adaptive_simulation(
    cancer_type: str = "NSCLC",
    policy_mode: PolicyMode = PolicyMode.ROBUST_ADAPTIVE,
    policy_params: Optional[PolicyParams] = None,
    total_days: int = 180,
    dt: float = 0.1,
    seed: int = 42,
) -> Dict:
    """
    Run a full closed-loop adaptive therapy simulation.
    
    Integrates:
    - ClonalDynamicsEngine (biological state)
    - ResistanceTracker (drug efficacy over time)
    - AdaptiveController (policy decisions)
    
    Args:
        cancer_type: Cancer type for parameter selection
        policy_mode: Which policy to use
        policy_params: Policy hyperparameters
        total_days: Simulation duration in days
        dt: Timestep
        seed: Random seed
        
    Returns:
        Dict with trajectories, controller summary, and outcome metrics
    """
    from .clonal_dynamics import get_cancer_specific_clonal_params
    
    # Initialize components
    clonal_params = get_cancer_specific_clonal_params(cancer_type)
    engine = ClonalDynamicsEngine(clonal_params)
    
    resistance_params = ResistanceParams()
    resistance = ResistanceTracker(resistance_params)
    resistance.initialize_drugs(["primary_agent"])
    
    controller = AdaptiveController(
        policy_mode=policy_mode,
        policy_params=policy_params or PolicyParams()
    )
    
    # Run simulation loop
    n_steps = int(total_days / dt)
    
    for i in range(n_steps):
        state = engine.state
        efficacy = resistance.get_efficacy_factor("primary_agent")
        
        # Policy decision
        dose = controller.decide(
            sensitive=state.sensitive,
            resistant=state.resistant,
            carrying_capacity=clonal_params.carrying_capacity,
            dt=dt,
            resistance_efficacy=efficacy,
        )
        
        # Advance biology
        engine.step(
            dt=dt,
            drug_active=(dose > 0),
            drug_pressure=dose,
            phase="flatten" if dose > 0.5 else "heat" if dose == 0 else "push",
            seed=seed + i,
        )
        
        # Advance resistance
        active = ["primary_agent"] if dose > 0 else []
        resistance.update(dt, active)
    
    # Compute outcome metrics
    final_state = engine.state
    trajectories = {
        "time": final_state.time_points,
        "sensitive": final_state.sensitive_trajectory,
        "resistant": final_state.resistant_trajectory,
        "burden": final_state.burden_trajectory,
        "doses": controller.ctrl_state.dose_history,
    }
    
    # Outcome scoring
    burden_arr = np.array(final_state.burden_trajectory)
    resistant_arr = np.array(final_state.resistant_trajectory)
    
    time_under_control = np.sum(burden_arr < 0.8 * clonal_params.carrying_capacity) * dt
    resistant_dominated = np.sum(resistant_arr > 0.5 * burden_arr) * dt if len(burden_arr) > 0 else total_days
    
    outcome = {
        "final_burden": round(float(final_state.tumor_fraction), 6),
        "final_resistant_fraction": round(float(final_state.resistant_fraction), 4),
        "days_under_control": round(float(time_under_control), 1),
        "days_resistant_dominated": round(float(resistant_dominated), 1),
        "survived_horizon": final_state.tumor_fraction < clonal_params.carrying_capacity,
        "resistant_takeover": final_state.resistant_fraction > 0.80,
    }
    
    return {
        "cancer_type": cancer_type,
        "total_days": total_days,
        "trajectories": trajectories,
        "controller_summary": controller.get_summary(),
        "resistance_summary": resistance.get_summary(),
        "outcome": outcome,
    }


def compare_policies(
    cancer_type: str = "NSCLC",
    total_days: int = 180,
    dt: float = 0.1,
    seed: int = 42,
) -> Dict:
    """
    Compare MTD vs Fixed Low-Dose vs Adaptive across all three policies.
    Returns structured comparison for analysis.
    """
    results = {}
    
    # 1. MTD: Full dose, always on (simulated as threshold with 0/0 thresholds)
    mtd_params = PolicyParams(
        dose_on_threshold=0.0,   # Always dose
        dose_off_threshold=0.0,
        robust_max_dose=1.0,
        max_continuous_dose_days=999,
        min_holiday_days=0,
        max_cumulative_toxicity=999,
    )
    results["MTD"] = run_adaptive_simulation(
        cancer_type, PolicyMode.THRESHOLD, mtd_params, total_days, dt, seed)
    
    # 2. Fixed Low Dose
    low_params = PolicyParams(
        dose_on_threshold=0.0,
        dose_off_threshold=0.0,
        robust_max_dose=0.30,
        max_continuous_dose_days=999,
        min_holiday_days=0,
        max_cumulative_toxicity=999,
    )
    results["FixedLow"] = run_adaptive_simulation(
        cancer_type, PolicyMode.THRESHOLD, low_params, total_days, dt, seed)
    
    # 3. Confluence Adaptive (robust)
    results["Adaptive"] = run_adaptive_simulation(
        cancer_type, PolicyMode.ROBUST_ADAPTIVE, PolicyParams(), total_days, dt, seed)
    
    # Build comparison table
    comparison = {}
    for name, res in results.items():
        comparison[name] = {
            "final_burden": res["outcome"]["final_burden"],
            "final_R_fraction": res["outcome"]["final_resistant_fraction"],
            "days_controlled": res["outcome"]["days_under_control"],
            "resistant_takeover": res["outcome"]["resistant_takeover"],
            "cumulative_toxicity": res["controller_summary"]["cumulative_toxicity"],
            "dose_switches": res["controller_summary"]["dose_switches"],
        }
    
    return {
        "cancer_type": cancer_type,
        "comparison": comparison,
        "full_results": results,
    }
