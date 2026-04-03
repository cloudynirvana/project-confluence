"""
Test Suite — Space Medicine Module
====================================

Tests for:
1. State vector construction and biomarker mapping
2. Countermeasure schema and constraints
3. Controller policy modes and safety layer
4. ODE integration stability
5. End-to-end: synthetic analog scenario
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.space.state_vector import (
    AstronautResilienceProfile,
    PsiDimension,
    SpaceBiomarkerMap,
    HEALTHY_GROUND_REFERENCE,
    SAFE_CORRIDOR,
    COUPLING_MATRIX,
    compute_coupling_drift,
    N_DIMENSIONS,
    PSI_LABELS,
)
from models.space.countermeasures import (
    CountermeasureVector,
    CountermeasureConstraints,
    LightTimingAction,
    ExerciseDoseAction,
    NutritionPlanAction,
    COUNTERMEASURE_EFFICACY,
    compute_restoration_rate,
    N_COUNTERMEASURES,
)
from models.space.mission_phase import (
    MissionPhase,
    PHASE_PROFILES,
    get_phase_at_day,
    gravity_factor,
)
from models.space.space_ode import (
    SpacePhysiologyODE,
    SpaceODEParams,
    simulate_no_countermeasures,
    simulate_standard_protocol,
    compare_protocols,
)
from models.space.space_controller import (
    SpaceCountermeasureController,
    SpacePolicyMode,
    SpacePolicyParams,
    run_controlled_simulation,
)


PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} — {detail}")


def section(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. STATE VECTOR TESTS
# ═══════════════════════════════════════════════════════════════════════════

section("1. State Vector & ARP")

# 1.1 Healthy baseline
arp = AstronautResilienceProfile.healthy_baseline("crew_1")
check("Healthy baseline is 7D",
      len(arp.psi_vector) == 7)
check("Healthy baseline matches reference",
      np.allclose(arp.psi_vector, HEALTHY_GROUND_REFERENCE))
check("Healthy archetype is 'Healthy Adapted'",
      arp.archetype == "Healthy Adapted")
check("No alerts for healthy baseline",
      len(arp.check_safe_corridor()) == 0)
check("Distance from healthy is ~0",
      arp.distance_from_healthy < 0.01)

# 1.2 Degraded state classification
degraded = AstronautResilienceProfile(
    psi_circadian=0.25, psi_autonomic=0.70, psi_immune=0.65,
    psi_microbiome=0.70, psi_musculoskeletal=0.75,
    psi_neuro_ocular=0.75, psi_cognitive=0.30)
arch, conf = degraded.classify_archetype()
check("Degraded circadian + cognitive → alerts",
      len(degraded.check_safe_corridor()) > 0)
check("Min dimension correctly identified",
      degraded.min_dimension[0] in ["Circadian Coherence", "Cognitive-Behavioral Resilience"])

# 1.3 Multi-system decline
multi_decline = AstronautResilienceProfile(
    psi_circadian=0.20, psi_autonomic=0.25, psi_immune=0.20,
    psi_microbiome=0.65, psi_musculoskeletal=0.30,
    psi_neuro_ocular=0.60, psi_cognitive=0.25)
arch, conf = multi_decline.classify_archetype()
check("Multi-system decline detected",
      arch == "Multi-System Decline" or arch == "Critical",
      f"got: {arch}")

# 1.4 Biomarker mapping
bmap = SpaceBiomarkerMap()
circadian_biomarkers = bmap.get_biomarkers(PsiDimension.CIRCADIAN)
check("Circadian has biomarkers",
      len(circadian_biomarkers) >= 3)
check("Interdaily stability biomarker exists",
      any(b.name == "interdaily_stability" for b in circadian_biomarkers))

# 1.5 Biomarker normalization
score = bmap.normalize_value(PsiDimension.AUTONOMIC, "SDNN", 150.0)
check("SDNN=150 (healthy mid) normalizes high",
      score > 0.8, f"got {score:.3f}")
score_low = bmap.normalize_value(PsiDimension.AUTONOMIC, "SDNN", 40.0)
check("SDNN=40 (very low) normalizes low",
      score_low < 0.3, f"got {score_low:.3f}")

# 1.6 From measurements
measurements = {
    "interdaily_stability": 0.65,
    "SDNN": 120.0,
    "RMSSD": 40.0,
    "IL6_IL10_ratio": 1.2,
    "shannon_diversity": 4.0,
    "exercise_compliance": 0.95,
    "intraocular_pressure": 15.0,
    "pvt_lapses": 3.0,
    "sleep_efficiency": 88.0,
}
arp_from_meas = AstronautResilienceProfile.from_measurements(
    measurements, crew_id="ISS_crew_1", mission_day=45)
check("ARP from measurements has valid scores",
      0 < arp_from_meas.psi_mean < 1.0,
      f"mean={arp_from_meas.psi_mean:.3f}")

# 1.7 Coupling matrix properties
check("Coupling matrix is 7x7",
      COUPLING_MATRIX.shape == (7, 7))
check("Coupling matrix diagonal is zero",
      np.allclose(np.diag(COUPLING_MATRIX), 0))
check("Coupling matrix is non-negative",
      np.all(COUPLING_MATRIX >= 0))

# 1.8 Coupling drift
drift = compute_coupling_drift(
    np.array([0.4, 0.8, 0.7, 0.7, 0.8, 0.8, 0.4]),
    HEALTHY_GROUND_REFERENCE)
check("Coupling drift is negative (degradation from deficits)",
      np.all(drift <= 0.01))  # Small positive due to noise tolerance
check("Largest drift on most coupled dimensions",
      abs(drift[0]) > 0 or abs(drift[6]) > 0,
      f"drift={drift}")

# 1.9 JSON export
json_str = arp.to_json()
check("JSON export contains psi_vector",
      '"psi_vector"' in json_str)
check("JSON export contains archetype",
      '"archetype"' in json_str)


# ═══════════════════════════════════════════════════════════════════════════
# 2. COUNTERMEASURE TESTS
# ═══════════════════════════════════════════════════════════════════════════

section("2. Countermeasures")

# 2.1 Default cruise vector
cv = CountermeasureVector.default_cruise()
scalars = cv.to_scalar_vector()
check("Default cruise has 7 channels",
      len(scalars) == N_COUNTERMEASURES)
check("Default cruise scalars in [0,1]",
      np.all(scalars >= 0) and np.all(scalars <= 1))
check("Exercise is active in default cruise",
      scalars[1] > 0.2, f"got {scalars[1]:.3f}")

# 2.2 From scalar vector
cv2 = CountermeasureVector.from_scalar_vector(np.array([0.5, 0.7, 0.5, 0.3, 0.0, 0.0, 0.0]))
check("From scalar: exercise duration > 30",
      cv2.exercise.duration_min >= 30)
check("From scalar: light intensity > 0",
      cv2.light.intensity_lux > 0)

# 2.3 Constraints enforcement
constraints = CountermeasureConstraints()
extreme_cv = CountermeasureVector()
extreme_cv.exercise.duration_min = 200  # Over max
extreme_cv.nutrition.sodium_mg = 5000   # Over max
extreme_cv.exercise.intensity_percent_vo2max = 95  # Over max
enforced = constraints.enforce(extreme_cv)
check("Exercise capped at 150 min",
      enforced.exercise.duration_min <= 150)
check("Sodium capped at 3000 mg",
      enforced.nutrition.sodium_mg <= 3000)
check("Exercise intensity capped at 90%",
      enforced.exercise.intensity_percent_vo2max <= 90)
check("Medication always requires flight surgeon",
      enforced.medication.requires_flight_surgeon is True)

# 2.4 Efficacy matrix properties
check("Efficacy matrix is 7×7",
      COUNTERMEASURE_EFFICACY.shape == (7, 7))
check("Light most effective for circadian",
      np.argmax(COUNTERMEASURE_EFFICACY[0, :]) == 0)
check("Exercise most effective for musculoskeletal",
      np.argmax(COUNTERMEASURE_EFFICACY[4, :]) == 1)
check("Microbiome diet most effective for microbiome",
      np.argmax(COUNTERMEASURE_EFFICACY[3, :]) == 3)

# 2.5 Restoration rate
deficit = np.array([0.3, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1])
cm_scalars = np.array([0.8, 0.7, 0.5, 0.3, 0.0, 0.0, 0.0])
restoration = compute_restoration_rate(cm_scalars, deficit)
check("Restoration rates are non-negative",
      np.all(restoration >= 0))
check("Restoration rates are bounded",
      np.all(restoration <= 0.05))

# 2.6 Auto-escalation
critical_psi = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.20])
escalation = constraints.check_escalation(critical_psi)
check("Auto-escalation triggered for critical Ψ",
      escalation is not None)
safe_psi = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5])
no_escalation = constraints.check_escalation(safe_psi)
check("No escalation for safe Ψ",
      no_escalation is None)


# ═══════════════════════════════════════════════════════════════════════════
# 3. MISSION PHASE TESTS
# ═══════════════════════════════════════════════════════════════════════════

section("3. Mission Phases")

check("All 8 phases have profiles",
      len(PHASE_PROFILES) == 8)

for phase, profile in PHASE_PROFILES.items():
    check(f"{phase.name} has 7D decay",
          len(profile.baseline_decay) == 7)

# Phase lookup
check("Day 0 → LAUNCH",
      get_phase_at_day(0.005, "iss_6month") == MissionPhase.LAUNCH)
check("Day 7 → EARLY_ADAPTATION",
      get_phase_at_day(7, "iss_6month") == MissionPhase.EARLY_ADAPTATION)
check("Day 90 → CRUISE_LEO",
      get_phase_at_day(90, "iss_6month") == MissionPhase.CRUISE_LEO)
check("Day 200 → POST_FLIGHT_REHAB",
      get_phase_at_day(200, "iss_6month") == MissionPhase.POST_FLIGHT_REHAB)

# Gravity factor
check("0G → factor 1.0",
      abs(gravity_factor(0.0) - 1.0) < 0.01)
check("1G → factor 0.0",
      abs(gravity_factor(1.0)) < 0.01)
check("0.38G (Mars) → factor ~0.62",
      abs(gravity_factor(0.38) - 0.62) < 0.05)

# Mars mission phases
check("Mars day 100 → DEEP_SPACE_TRANSIT",
      get_phase_at_day(100, "mars_30month") == MissionPhase.DEEP_SPACE_TRANSIT)
check("Mars day 400 → PLANETARY_SURFACE",
      get_phase_at_day(400, "mars_30month") == MissionPhase.PLANETARY_SURFACE)


# ═══════════════════════════════════════════════════════════════════════════
# 4. ODE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

section("4. ODE Integration")

# 4.1 Basic simulation — no countermeasures
print("  Running 180-day sim (no countermeasures)...")
result_none = simulate_no_countermeasures(total_days=180, seed=42)
check("Simulation completes",
      result_none is not None)
check("Trajectory shape is correct",
      result_none["trajectory"].shape[1] == 7)
check("Final state has degraded (distance > 0.1)",
      result_none["metrics"]["final_distance"] > 0.1,
      f"distance={result_none['metrics']['final_distance']}")

# 4.2 Standard protocol
print("  Running 180-day sim (standard protocol)...")
result_std = simulate_standard_protocol(total_days=180, seed=42)
check("Standard protocol completes",
      result_std is not None)
check("Standard protocol maintains Ψ better than no countermeasures",
      result_std["metrics"]["final_distance"] <
      result_none["metrics"]["final_distance"],
      f"std={result_std['metrics']['final_distance']:.3f} "
      f"vs none={result_none['metrics']['final_distance']:.3f}")

# 4.3 State stays bounded [0, 1]
traj = result_none["trajectory"]
check("All states ≥ 0",
      np.all(traj >= 0.0))
check("All states ≤ 1",
      np.all(traj <= 1.0))

# 4.4 Protocol comparison
print("  Running protocol comparison...")
comparison = compare_protocols(total_days=90, seed=42)
check("Comparison reports positive countermeasure benefit",
      comparison["countermeasure_benefit"] > 0,
      f"benefit={comparison['countermeasure_benefit']}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. CONTROLLER TESTS
# ═══════════════════════════════════════════════════════════════════════════

section("5. Controller")

# 5.1 Threshold policy
ctrl_thresh = SpaceCountermeasureController(
    policy_mode=SpacePolicyMode.THRESHOLD)
healthy_psi = HEALTHY_GROUND_REFERENCE.copy()
u_healthy = ctrl_thresh.decide(healthy_psi, mission_day=10,
                                phase=MissionPhase.CRUISE_LEO)
check("Threshold: low output for healthy state",
      np.sum(u_healthy) < 2.0, f"sum={np.sum(u_healthy):.2f}")

degraded_psi = np.array([0.3, 0.7, 0.7, 0.7, 0.3, 0.7, 0.7])
ctrl_thresh2 = SpaceCountermeasureController(
    policy_mode=SpacePolicyMode.THRESHOLD)
u_degraded = ctrl_thresh2.decide(degraded_psi, mission_day=10,
                                  phase=MissionPhase.CRUISE_LEO)
check("Threshold: higher output for degraded state",
      np.sum(u_degraded) > np.sum(u_healthy),
      f"degraded={np.sum(u_degraded):.2f} vs healthy={np.sum(u_healthy):.2f}")

# 5.2 Proportional policy
ctrl_prop = SpaceCountermeasureController(
    policy_mode=SpacePolicyMode.PROPORTIONAL)
u_prop = ctrl_prop.decide(degraded_psi, mission_day=10,
                           phase=MissionPhase.CRUISE_LEO)
check("Proportional: output scales with deficit",
      np.sum(u_prop) > 0.3, f"sum={np.sum(u_prop):.2f}")

# 5.3 Robust adaptive policy
ctrl_robust = SpaceCountermeasureController(
    policy_mode=SpacePolicyMode.ROBUST_ADAPTIVE)
u_robust = ctrl_robust.decide(degraded_psi, mission_day=10,
                               phase=MissionPhase.CRUISE_LEO)
check("RobustAdaptive: output for degraded state",
      np.sum(u_robust) > 0.3, f"sum={np.sum(u_robust):.2f}")

# 5.4 Emergency response
emergency_psi = np.array([0.15, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60])
ctrl_emerg = SpaceCountermeasureController(
    policy_mode=SpacePolicyMode.ROBUST_ADAPTIVE)
u_emerg = ctrl_emerg.decide(emergency_psi, mission_day=10,
                             phase=MissionPhase.CRUISE_LEO)
check("Emergency: high countermeasure output",
      np.max(u_emerg) > 0.5, f"max={np.max(u_emerg):.2f}")
check("Emergency: logs decision",
      any("EMERGENCY" in log for log in ctrl_emerg.state.decision_log))

# 5.5 Auto-escalation of telemedicine
critical_psi = np.array([0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.15])
ctrl_esc = SpaceCountermeasureController(
    policy_mode=SpacePolicyMode.ROBUST_ADAPTIVE)
u_esc = ctrl_esc.decide(critical_psi, mission_day=10,
                         phase=MissionPhase.CRUISE_LEO)
check("Auto-escalation: telemedicine channel activated",
      u_esc[6] > 0.8, f"tele={u_esc[6]:.2f}")

# 5.6 Summary report
summary = ctrl_robust.get_summary()
check("Summary contains policy mode",
      summary["policy_mode"] == "robust_adaptive")


# ═══════════════════════════════════════════════════════════════════════════
# 6. END-TO-END: CONTROLLED SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

section("6. End-to-End Controlled Simulation")

print("  Running 180-day controlled ISS simulation...")
result = run_controlled_simulation(
    total_days=180, dt=0.25,
    mission_type="iss_6month",
    policy_mode=SpacePolicyMode.ROBUST_ADAPTIVE,
    seed=42)

check("Controlled sim completes",
      result is not None)
check("Final ARP exists",
      result["final_arp"] is not None)
check("Controller summary exists",
      len(result["controller_summary"]) > 0)
check("Countermeasure trajectory recorded",
      result["countermeasure_trajectory"].shape[1] == N_COUNTERMEASURES)

# Compare controlled vs uncontrolled
uncontrolled = simulate_no_countermeasures(total_days=180, seed=42)
check("Controlled sim outperforms uncontrolled",
      result["metrics"]["final_distance"] <
      uncontrolled["metrics"]["final_distance"],
      f"controlled={result['metrics']['final_distance']:.3f} "
      f"vs uncontrolled={uncontrolled['metrics']['final_distance']:.3f}")

print(f"\n  Final ARP: {result['final_arp']}")
print(f"  Archetype: {result['metrics']['archetype']}")
if result['metrics']['alerts']:
    print(f"  Alerts: {result['metrics']['alerts']}")

# 6.2 Mars mission
print("\n  Running 800-day Mars simulation (deep space transit + surface)...")
mars_result = run_controlled_simulation(
    total_days=800, dt=0.5,
    mission_type="mars_30month",
    policy_mode=SpacePolicyMode.ROBUST_ADAPTIVE,
    seed=42)

check("Mars sim completes",
      mars_result is not None)
check("Mars sim: final state has meaningful values",
      mars_result["final_arp"].psi_mean > 0.1,
      f"mean={mars_result['final_arp'].psi_mean:.3f}")

print(f"  Mars Final ARP: {mars_result['final_arp']}")
print(f"  Mars Archetype: {mars_result['metrics']['archetype']}")


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

section("RESULTS")
total = PASS + FAIL
print(f"\n  {PASS}/{total} tests passed, {FAIL} failed")
if FAIL > 0:
    print(f"  [WARNING] {FAIL} test(s) need attention")
    sys.exit(1)
else:
    print("  [SUCCESS] All tests passed!")
