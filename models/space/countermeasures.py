"""
Countermeasure Action Schema — Space Medicine Module
=====================================================

Defines the 7-element countermeasure action vector u(t) that the
Adaptive Countermeasure Controller (ACC) outputs at each timestep.

Each countermeasure has:
    - A continuous action range
    - Hard safety constraints (cannot be overridden)
    - Flight surgeon override flags
    - Evidence-linked default schedules

Countermeasure vector u(t):
    u₁ = light_timing       (wavelength, intensity, schedule)
    u₂ = exercise_dose      (duration, modality, intensity)
    u₃ = nutrition_plan     (calories, protein, sodium, hydration)
    u₄ = microbiome_diet    (prebiotic fiber, fermented foods)
    u₅ = medication         (motion sickness, sleep aids) — Rx required
    u₆ = lab_trigger        (blood draw, OCT, ultrasound frequency)
    u₇ = telemedicine_level (0=routine, 1=priority, 2=urgent)

Safety model: all medication decisions require flight surgeon approval.
The controller only suggests; it does not prescribe.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto


# ═══════════════════════════════════════════════════════════════════════════
# 1. COUNTERMEASURE ENUMERATION
# ═══════════════════════════════════════════════════════════════════════════

class CountermeasureType(Enum):
    """The 7 countermeasure channels."""
    LIGHT_TIMING = 0
    EXERCISE_DOSE = 1
    NUTRITION_PLAN = 2
    MICROBIOME_DIET = 3
    MEDICATION = 4
    LAB_TRIGGER = 5
    TELEMEDICINE = 6


N_COUNTERMEASURES = 7

COUNTERMEASURE_NAMES = [
    "light_timing",
    "exercise_dose",
    "nutrition_plan",
    "microbiome_diet",
    "medication",
    "lab_trigger",
    "telemedicine_level",
]

COUNTERMEASURE_LABELS = [
    "Light Timing & Spectrum",
    "Exercise Dose & Modality",
    "Nutrition Plan",
    "Microbiome-Directed Diet",
    "Medication (Rx Required)",
    "Lab / Diagnostic Trigger",
    "Telemedicine Escalation Level",
]


# ═══════════════════════════════════════════════════════════════════════════
# 2. INDIVIDUAL COUNTERMEASURE SPECIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LightTimingAction:
    """Light countermeasure: timing, wavelength, intensity, duration."""
    intensity_lux: float = 2500.0       # 0–10,000 lux
    peak_wavelength_nm: float = 480.0   # 460–550 nm (blue-enriched white)
    duration_min: float = 60.0          # Minutes of exposure
    phase_advance_hours: float = 0.0    # Shift wake-time earlier (positive)
    blue_cutoff_before_sleep_hours: float = 2.0  # No blue light N hours pre-sleep

    def to_scalar(self) -> float:
        """Normalize to [0, 1] scalar for the controller."""
        # Composite: higher intensity + longer duration + appropriate timing = higher
        intensity_score = np.clip(self.intensity_lux / 10000.0, 0, 1)
        duration_score = np.clip(self.duration_min / 120.0, 0, 1)
        return float((intensity_score + duration_score) / 2.0)


@dataclass
class ExerciseDoseAction:
    """Exercise countermeasure: duration, modality mix, intensity."""
    duration_min: float = 90.0          # Total daily exercise (30–150 min)
    resistive_fraction: float = 0.55    # Fraction ARED vs. aerobic
    intensity_percent_vo2max: float = 70.0  # Exercise intensity (50–90%)
    sessions_per_day: int = 2           # 1–3 sessions

    def to_scalar(self) -> float:
        """Normalize to [0, 1] scalar."""
        dur_score = np.clip((self.duration_min - 30) / 120.0, 0, 1)
        int_score = np.clip((self.intensity_percent_vo2max - 50) / 40.0, 0, 1)
        return float((dur_score + int_score) / 2.0)


@dataclass
class NutritionPlanAction:
    """Nutrition countermeasure: caloric intake, macros, sodium, hydration."""
    daily_calories: float = 2500.0      # kcal/day (1800–3200)
    protein_g_per_kg: float = 1.4       # g/kg body mass (1.0–2.0)
    sodium_mg: float = 2000.0           # mg/day (< 3000 for SANS)
    fluid_intake_ml: float = 2500.0     # ml/day (2000–3500)
    vitamin_d_iu: float = 1000.0        # IU/day (800–2000)

    def to_scalar(self) -> float:
        """Normalize to [0, 1] scalar (compliance with optimal)."""
        cal_score = 1.0 - abs(self.daily_calories - 2500) / 700
        sodium_score = 1.0 - max(0, self.sodium_mg - 2000) / 1000
        return float(np.clip((cal_score + sodium_score) / 2.0, 0, 1))


@dataclass
class MicrobiomeDietAction:
    """Microbiome-directed dietary countermeasure."""
    prebiotic_fiber_g: float = 15.0     # grams/day (0–30)
    fermented_food_servings: float = 1.0  # servings/day (0–3)
    probiotic_supplement: bool = False   # Only validated strains

    def to_scalar(self) -> float:
        """Normalize to [0, 1] scalar."""
        fiber_score = np.clip(self.prebiotic_fiber_g / 30.0, 0, 1)
        ferment_score = np.clip(self.fermented_food_servings / 3.0, 0, 1)
        return float((fiber_score + ferment_score) / 2.0)


class MedicationType(Enum):
    """Medication options from ISS formulary (requires flight surgeon)."""
    NONE = "none"
    PROMETHAZINE = "promethazine"       # Motion sickness
    MELATONIN = "melatonin"             # Sleep onset
    ZOLPIDEM = "zolpidem"               # Sleep (short-acting)
    MODAFINIL = "modafinil"             # Alertness (emergency)
    IBUPROFEN = "ibuprofen"             # Headache / pain


@dataclass
class MedicationAction:
    """Medication countermeasure — ALWAYS requires flight surgeon approval."""
    medication: MedicationType = MedicationType.NONE
    dose_fraction: float = 0.0          # 0 = no medication, 1 = standard dose
    requires_flight_surgeon: bool = True  # Always True — cannot be overridden

    def to_scalar(self) -> float:
        """Normalize to [0, 1] scalar."""
        if self.medication == MedicationType.NONE:
            return 0.0
        return float(np.clip(self.dose_fraction, 0, 1))


@dataclass
class LabTriggerAction:
    """Lab/diagnostic follow-up trigger."""
    blood_draw: bool = False
    oct_scan: bool = False
    ultrasound: bool = False
    extra_cognition_test: bool = False
    urgency: float = 0.0  # 0=routine, 0.5=priority, 1.0=urgent

    def to_scalar(self) -> float:
        """Normalize to [0, 1] scalar."""
        n_tests = sum([self.blood_draw, self.oct_scan,
                       self.ultrasound, self.extra_cognition_test])
        return float(np.clip(n_tests / 4.0 + self.urgency * 0.5, 0, 1))


class TelemedicineLevel(Enum):
    """Telemedicine escalation levels."""
    ROUTINE = 0
    PRIORITY = 1
    URGENT = 2


@dataclass
class TelemedicineAction:
    """Telemedicine escalation countermeasure."""
    level: TelemedicineLevel = TelemedicineLevel.ROUTINE
    reason: str = ""

    def to_scalar(self) -> float:
        """Normalize to [0, 1] scalar."""
        return float(self.level.value / 2.0)


# ═══════════════════════════════════════════════════════════════════════════
# 3. COMPOSITE COUNTERMEASURE VECTOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CountermeasureVector:
    """Complete 7-element countermeasure action vector u(t).

    This is what the AdaptiveCountermeasureController outputs at
    each control timestep.
    """
    light: LightTimingAction = field(default_factory=LightTimingAction)
    exercise: ExerciseDoseAction = field(default_factory=ExerciseDoseAction)
    nutrition: NutritionPlanAction = field(default_factory=NutritionPlanAction)
    microbiome: MicrobiomeDietAction = field(default_factory=MicrobiomeDietAction)
    medication: MedicationAction = field(default_factory=MedicationAction)
    lab_trigger: LabTriggerAction = field(default_factory=LabTriggerAction)
    telemedicine: TelemedicineAction = field(default_factory=TelemedicineAction)

    def to_scalar_vector(self) -> np.ndarray:
        """Convert to 7D scalar vector in [0, 1]^7 for numerical integration."""
        return np.array([
            self.light.to_scalar(),
            self.exercise.to_scalar(),
            self.nutrition.to_scalar(),
            self.microbiome.to_scalar(),
            self.medication.to_scalar(),
            self.lab_trigger.to_scalar(),
            self.telemedicine.to_scalar(),
        ])

    @classmethod
    def from_scalar_vector(cls, u: np.ndarray) -> "CountermeasureVector":
        """Construct from a 7D scalar vector (interpolated from controller).

        This creates a 'rough' countermeasure vector from scalar values.
        The controller should refine the structured actions afterward.
        """
        cv = cls()
        u = np.clip(u, 0, 1)
        cv.light.intensity_lux = u[0] * 10000
        cv.light.duration_min = 30 + u[0] * 90
        cv.exercise.duration_min = 30 + u[1] * 120
        cv.exercise.intensity_percent_vo2max = 50 + u[1] * 40
        cv.nutrition.daily_calories = 1800 + u[2] * 1400
        cv.nutrition.sodium_mg = 1500 + (1 - u[2]) * 1500
        cv.microbiome.prebiotic_fiber_g = u[3] * 30
        cv.microbiome.fermented_food_servings = u[3] * 3
        # Medication scalar > 0.5 suggests medication may be warranted
        # but always requires flight surgeon — flagged only
        cv.medication.dose_fraction = u[4]
        cv.medication.medication = (MedicationType.MELATONIN
                                    if u[4] > 0.3 else MedicationType.NONE)
        cv.lab_trigger.blood_draw = u[5] > 0.5
        cv.lab_trigger.oct_scan = u[5] > 0.7
        cv.lab_trigger.urgency = u[5]
        cv.telemedicine.level = (TelemedicineLevel.URGENT if u[6] > 0.7
                                 else TelemedicineLevel.PRIORITY if u[6] > 0.3
                                 else TelemedicineLevel.ROUTINE)
        return cv

    @classmethod
    def default_cruise(cls) -> "CountermeasureVector":
        """Default countermeasure schedule for LEO cruise phase."""
        return cls(
            light=LightTimingAction(
                intensity_lux=2500, peak_wavelength_nm=480,
                duration_min=60, blue_cutoff_before_sleep_hours=2),
            exercise=ExerciseDoseAction(
                duration_min=90, resistive_fraction=0.55,
                intensity_percent_vo2max=70, sessions_per_day=2),
            nutrition=NutritionPlanAction(
                daily_calories=2500, protein_g_per_kg=1.4,
                sodium_mg=2000, fluid_intake_ml=2500),
            microbiome=MicrobiomeDietAction(
                prebiotic_fiber_g=15, fermented_food_servings=1),
            medication=MedicationAction(),
            lab_trigger=LabTriggerAction(),
            telemedicine=TelemedicineAction(),
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. COUNTERMEASURE CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CountermeasureConstraints:
    """Hard safety constraints on all countermeasures.

    These CANNOT be overridden by the adaptive controller.
    They represent NASA medical operations minimums/maximums
    and crew safety requirements.
    """
    # Exercise
    exercise_min_daily_min: float = 30.0
    exercise_max_daily_min: float = 150.0
    exercise_max_intensity: float = 90.0  # % VO2max — prevent injury

    # Light
    light_max_intensity_lux: float = 10000.0
    light_min_daily_min: float = 30.0     # Prevent complete circadian free-run
    blue_cutoff_before_sleep_min_hours: float = 1.5

    # Nutrition
    nutrition_min_calories: float = 1800.0
    nutrition_max_calories: float = 3200.0
    sodium_max_daily_mg: float = 3000.0   # SANS risk mitigation
    protein_min_g_per_kg: float = 1.0
    fluid_min_daily_ml: float = 2000.0

    # Microbiome
    fiber_max_daily_g: float = 40.0       # GI distress prevention

    # Medication
    medication_requires_flight_surgeon: bool = True  # ALWAYS

    # Operations
    forced_rest_after_eva_hours: float = 8.0
    max_countermeasure_changes_per_day: int = 15  # Prevent oscillatory control
    telemedicine_auto_escalate_psi_min: float = 0.25  # Auto-escalate if any Ψ < this

    def enforce(self, cv: CountermeasureVector) -> CountermeasureVector:
        """Apply hard constraints to a countermeasure vector.

        Clips values to safe bounds. Returns the constrained vector.
        This method is called by the controller AFTER every policy decision.
        """
        # Exercise bounds
        cv.exercise.duration_min = np.clip(
            cv.exercise.duration_min,
            self.exercise_min_daily_min,
            self.exercise_max_daily_min)
        cv.exercise.intensity_percent_vo2max = np.clip(
            cv.exercise.intensity_percent_vo2max, 50.0,
            self.exercise_max_intensity)

        # Light bounds
        cv.light.intensity_lux = np.clip(
            cv.light.intensity_lux, 0.0, self.light_max_intensity_lux)
        cv.light.duration_min = max(
            cv.light.duration_min, self.light_min_daily_min)
        cv.light.blue_cutoff_before_sleep_hours = max(
            cv.light.blue_cutoff_before_sleep_hours,
            self.blue_cutoff_before_sleep_min_hours)

        # Nutrition bounds
        cv.nutrition.daily_calories = np.clip(
            cv.nutrition.daily_calories,
            self.nutrition_min_calories, self.nutrition_max_calories)
        cv.nutrition.sodium_mg = min(
            cv.nutrition.sodium_mg, self.sodium_max_daily_mg)
        cv.nutrition.protein_g_per_kg = max(
            cv.nutrition.protein_g_per_kg, self.protein_min_g_per_kg)
        cv.nutrition.fluid_intake_ml = max(
            cv.nutrition.fluid_intake_ml, self.fluid_min_daily_ml)

        # Microbiome bounds
        cv.microbiome.prebiotic_fiber_g = np.clip(
            cv.microbiome.prebiotic_fiber_g, 0.0, self.fiber_max_daily_g)

        # Medication ALWAYS requires flight surgeon
        cv.medication.requires_flight_surgeon = True

        return cv

    def check_escalation(self, psi: np.ndarray) -> Optional[TelemedicineLevel]:
        """Check if any Ψ dimension triggers auto-escalation.

        Returns the required telemedicine level, or None if no escalation.
        """
        if np.any(psi < self.telemedicine_auto_escalate_psi_min):
            return TelemedicineLevel.URGENT
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 5. COUNTERMEASURE EFFICACY MAP
# ═══════════════════════════════════════════════════════════════════════════

# How each countermeasure (column) affects each Ψ dimension (row).
# E[i, j] = effect of countermeasure j on Ψ_i restoration rate.
# Positive = restorative, negative = no direct effect (but safe).
COUNTERMEASURE_EFFICACY = np.array([
    # Light  Exer  Nutr  Micro Med   Lab   Tele
    [0.80,  0.15, 0.05, 0.00, 0.20, 0.00, 0.00],  # Circadian
    [0.20,  0.45, 0.10, 0.05, 0.10, 0.00, 0.00],  # Autonomic
    [0.10,  0.25, 0.15, 0.30, 0.10, 0.05, 0.00],  # Immune
    [0.00,  0.05, 0.20, 0.60, 0.00, 0.05, 0.00],  # Microbiome
    [0.00,  0.70, 0.25, 0.00, 0.00, 0.00, 0.00],  # Musculoskeletal
    [0.10,  0.10, 0.20, 0.00, 0.05, 0.30, 0.00],  # Neuro-ocular
    [0.40,  0.20, 0.10, 0.05, 0.15, 0.00, 0.00],  # Cognitive
])


def compute_restoration_rate(countermeasure_scalars: np.ndarray,
                             psi_deficit: np.ndarray,
                             efficacy_matrix: np.ndarray = COUNTERMEASURE_EFFICACY,
                             max_rate: float = 0.05) -> np.ndarray:
    """Compute the Ψ restoration rate from active countermeasures.

    Args:
        countermeasure_scalars: 7D vector of countermeasure intensities [0, 1].
        psi_deficit: 7D vector of (healthy - current) deficits (positive = degraded).
        efficacy_matrix: 7×7 matrix of countermeasure → dimension efficacy.
        max_rate: Maximum restoration rate per dimension per day.

    Returns:
        restoration: 7D vector of restoration rates (positive = improving).
    """
    # Restoration is proportional to countermeasure intensity,
    # efficacy, and current deficit (diminishing returns near healthy)
    raw = efficacy_matrix @ countermeasure_scalars
    # Scale by deficit — more restoration when further from healthy
    restoration = raw * np.clip(psi_deficit, 0, 1)
    # Cap at max rate
    return np.clip(restoration, 0, max_rate)


# ═══════════════════════════════════════════════════════════════════════════
# 6. SCHEMA EXPORT
# ═══════════════════════════════════════════════════════════════════════════

COUNTERMEASURE_SCHEMA = {
    "light_timing": {
        "type": "LightTimingAction",
        "parameters": ["intensity_lux", "peak_wavelength_nm", "duration_min",
                       "phase_advance_hours", "blue_cutoff_before_sleep_hours"],
        "risk_level": "low",
        "requires_flight_surgeon": False,
        "evidence": "ISS lighting upgrade studies, circadian photoentrainment literature",
    },
    "exercise_dose": {
        "type": "ExerciseDoseAction",
        "parameters": ["duration_min", "resistive_fraction",
                       "intensity_percent_vo2max", "sessions_per_day"],
        "risk_level": "low",
        "requires_flight_surgeon": False,
        "evidence": "SPRINT protocol (NASA), ARED effectiveness studies",
    },
    "nutrition_plan": {
        "type": "NutritionPlanAction",
        "parameters": ["daily_calories", "protein_g_per_kg", "sodium_mg",
                       "fluid_intake_ml", "vitamin_d_iu"],
        "risk_level": "low",
        "requires_flight_surgeon": False,
        "evidence": "ISS nutrition standards, SANS sodium restriction data",
    },
    "microbiome_diet": {
        "type": "MicrobiomeDietAction",
        "parameters": ["prebiotic_fiber_g", "fermented_food_servings",
                       "probiotic_supplement"],
        "risk_level": "low",
        "requires_flight_surgeon": False,
        "evidence": "NASA MAPS study, ISS microbiome monitoring",
    },
    "medication": {
        "type": "MedicationAction",
        "parameters": ["medication", "dose_fraction"],
        "risk_level": "HIGH",
        "requires_flight_surgeon": True,
        "evidence": "ISS Medical Checklist, NASA formulary",
    },
    "lab_trigger": {
        "type": "LabTriggerAction",
        "parameters": ["blood_draw", "oct_scan", "ultrasound",
                       "extra_cognition_test", "urgency"],
        "risk_level": "medium",
        "requires_flight_surgeon": False,
        "evidence": "Standard Measures cadence, consumables constraints",
    },
    "telemedicine_level": {
        "type": "TelemedicineAction",
        "parameters": ["level", "reason"],
        "risk_level": "low",
        "requires_flight_surgeon": False,
        "evidence": "ISS medical operations protocols, Polaris Dawn telemedicine",
    },
}


# Convenience type alias for external code
CountermeasureAction = CountermeasureVector
