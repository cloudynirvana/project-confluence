"""
Immune Dynamics Module — Project Confluence
=============================================

Multi-compartment immune system model acting on the metabolic phase space.
Models the immune system as interacting cell populations that generate a
net force vector pushing the metabolic state toward the healthy attractor.

Populations modeled:
  - CD8+ T-cells: Naïve → Primed → Effector → Exhausted lifecycle
  - NK cells: Innate surveillance with stress-ligand (NKG2D) activation
  - Tregs: Dynamic suppressive population that dampens all effector responses
  - Dendritic cells (DCs): Antigen presentation drives CD8+ priming rate

Tissue barriers:
  - GBM: Blood-brain barrier (BBB) reduces immune infiltration
  - PDAC: Desmoplastic stroma limits T-cell penetration
  - HGSOC: Peritoneal immune exclusion
  - General: Tumor immune evasion via PD-L1 expression

Physics:
  dF/dt = -kappa * mu(x) * F   (Force decays faster in deeper wells)
  F_net = F_cd8 + F_nk - F_treg  (Net force is sum of effector - suppressive)

References:
  - Wherry & Kurachi 2015, Nat Rev Immunol: T-cell exhaustion
  - Vivier et al. 2011, Science: NK cell biology
  - Tumeh et al. 2014, Nature: Anti-PD-1 response biomarkers
  - Spranger & Gajewski 2018, Nat Rev Cancer: Tumor immune microenvironment
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


# ═══════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ImmuneParams:
    """Parameters for the multi-compartment immune model."""
    # CD8+ T-cell parameters
    base_force: float = 1.0            # B0: Max cytotoxic T-cell pressure
    exhaustion_rate: float = 0.1       # kappa: Exhaustion per unit depth per day
    priming_rate: float = 0.05         # Rate of naïve → effector conversion per day
    effector_decay_rate: float = 0.02  # Natural effector contraction rate per day
    exhaustion_threshold: float = 2.0  # Cumulative exhaustion threshold for dysfunction
    pd1_expression_rate: float = 0.03  # PD-1 upregulation per day of antigen exposure

    # NK cell parameters
    nk_base_force: float = 0.3         # Innate surveillance pressure
    nk_stress_sensitivity: float = 0.5 # NKG2D response to ROS/stress (0-1)
    nk_exhaustion_rate: float = 0.05   # NK cells exhaust slower than T-cells

    # Treg parameters
    treg_load: float = 0.3             # gamma: Baseline suppressive coefficient (0-1)
    treg_expansion_rate: float = 0.01  # Treg expansion under tumor TGF-beta
    treg_max_load: float = 0.8         # Maximum suppressive capacity
    treg_il2_competition: float = 0.1  # IL-2 consumption reducing effector activation

    # Dendritic cell parameters
    dc_priming_efficiency: float = 0.5 # Antigen presentation efficiency (0-1)
    dc_maturation_rate: float = 0.03   # DC maturation per day in tumor
    dc_cross_presentation: float = 0.3 # Cross-presentation capability (0-1)

    # Checkpoint blockade parameters
    pd1_blockade: float = 0.0          # Anti-PD-1 effect (0-1)
    ctla4_blockade: float = 0.0        # Anti-CTLA-4 effect (0-1)
    lag3_blockade: float = 0.0         # Anti-LAG-3 effect (0-1, future)

    # Tissue-specific barrier modifiers
    tissue_barrier: float = 0.0        # 0.0 (no barrier) to 1.0 (complete exclusion)
    barrier_type: str = "none"         # "bbb", "desmoplasia", "peritoneal", "none"


# Pre-defined tissue barriers for specific cancer types
TISSUE_BARRIERS: Dict[str, Dict[str, float]] = {
    "TNBC":     {"tissue_barrier": 0.10, "barrier_type": "none"},
    "PDAC":     {"tissue_barrier": 0.65, "barrier_type": "desmoplasia"},
    "NSCLC":    {"tissue_barrier": 0.15, "barrier_type": "none"},
    "Melanoma": {"tissue_barrier": 0.05, "barrier_type": "none"},
    "GBM":      {"tissue_barrier": 0.70, "barrier_type": "bbb"},
    "CRC":      {"tissue_barrier": 0.20, "barrier_type": "none"},
    "HGSOC":    {"tissue_barrier": 0.45, "barrier_type": "peritoneal"},
    "AML":      {"tissue_barrier": 0.05, "barrier_type": "none"},       # Liquid tumor
    "mCRPC":    {"tissue_barrier": 0.25, "barrier_type": "none"},
    "HCC":      {"tissue_barrier": 0.35, "barrier_type": "none"},
}


# ═══════════════════════════════════════════════════════════════════════
# CD8+ T-CELL COMPARTMENT
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CD8State:
    """Tracks the lifecycle state of the CD8+ T-cell population."""
    naive_fraction: float = 0.7         # Fraction of naïve T-cells
    effector_fraction: float = 0.2      # Active effector T-cells
    exhausted_fraction: float = 0.1     # Dysfunctional / exhausted
    cumulative_exhaustion: float = 0.0  # Integrated exhaustion signal
    pd1_expression: float = 0.0        # PD-1 surface expression level (0-1)
    antigen_exposure_days: float = 0.0  # Days of continuous antigen stimulation

    def normalize(self):
        """Ensure fractions sum to 1."""
        total = self.naive_fraction + self.effector_fraction + self.exhausted_fraction
        if total > 0:
            self.naive_fraction /= total
            self.effector_fraction /= total
            self.exhausted_fraction /= total


# ═══════════════════════════════════════════════════════════════════════
# NK CELL COMPARTMENT
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class NKState:
    """Tracks NK cell activity state."""
    activation_level: float = 0.5       # 0 = quiescent, 1 = fully activated
    cumulative_fatigue: float = 0.0     # NK exhaustion accumulator
    nkg2d_expression: float = 0.8       # Activating receptor expression (0-1)


# ═══════════════════════════════════════════════════════════════════════
# TREG COMPARTMENT
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TregState:
    """Tracks regulatory T-cell population dynamics."""
    current_load: float = 0.3           # Current suppressive load (evolves)
    expansion_days: float = 0.0         # Days of tumor-driven expansion


# ═══════════════════════════════════════════════════════════════════════
# DENDRITIC CELL COMPARTMENT
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DCState:
    """Tracks dendritic cell maturation and antigen presentation."""
    maturation_level: float = 0.2       # 0 = immature, 1 = fully mature
    antigen_load: float = 0.0          # Accumulated tumor antigen


# ═══════════════════════════════════════════════════════════════════════
# MULTI-COMPARTMENT FORCE FIELD
# ═══════════════════════════════════════════════════════════════════════

class LymphocyteForceField:
    """
    Multi-compartment immune force field.

    Computes the net force vector applied by the immune system onto
    the metabolic state, integrating CD8+ T-cells, NK cells, Tregs,
    and dendritic cell priming.

    The force pushes the metabolic state toward the healthy attractor
    (origin in the phase space). Its magnitude depends on:
      - Effector cell populations (CD8+, NK)
      - Suppressive populations (Tregs)
      - Checkpoint status (PD-1, CTLA-4)
      - Tissue barriers (BBB, desmoplasia)
      - Exhaustion dynamics
    """

    def __init__(self, n_metabolites: int, params: ImmuneParams):
        self.n = n_metabolites
        self.params = params

        # Initialize compartments
        self.cd8 = CD8State()
        self.nk = NKState()
        self.treg = TregState(current_load=params.treg_load)
        self.dc = DCState()

        # Tissue barrier factor (reduces infiltration)
        self.barrier_factor = 1.0 - params.tissue_barrier

        # Legacy compatibility
        self.cumulative_exhaustion = 0.0

    @classmethod
    def for_cancer_type(cls, n_metabolites: int, params: ImmuneParams,
                        cancer_type: str) -> "LymphocyteForceField":
        """Factory: create a force field with tissue barriers for a cancer type."""
        barrier_info = TISSUE_BARRIERS.get(cancer_type, {"tissue_barrier": 0.1})
        params.tissue_barrier = barrier_info["tissue_barrier"]
        params.barrier_type = barrier_info.get("barrier_type", "none")
        return cls(n_metabolites, params)

    def reset(self):
        """Reset all compartments to initial state."""
        self.cd8 = CD8State()
        self.nk = NKState()
        self.treg = TregState(current_load=self.params.treg_load)
        self.dc = DCState()
        self.cumulative_exhaustion = 0.0

    # ── CD8+ T-CELL DYNAMICS ──

    def _update_cd8(self, well_depth: float, dt: float) -> float:
        """
        Update CD8+ T-cell compartment and return force magnitude.

        Lifecycle: Naïve → Effector (via DC priming) → Exhausted (via antigen)
        PD-1 blockade slows exhaustion. CTLA-4 blockade enhances priming.
        """
        p = self.params
        cd8 = self.cd8

        # DC-driven priming: naïve → effector
        priming = (p.priming_rate * cd8.naive_fraction * self.dc.maturation_level
                   * (1.0 + p.ctla4_blockade * 0.5))  # CTLA-4 blockade boosts priming
        priming = min(priming * dt, cd8.naive_fraction * 0.5)  # Don't deplete naïve too fast

        # Exhaustion: effector → exhausted (depth-dependent)
        checkpoint_protection = p.pd1_blockade * 0.8 + p.lag3_blockade * 0.3
        fatigue_rate = (p.exhaustion_rate * well_depth * (1.0 - checkpoint_protection))
        fatigue = fatigue_rate * cd8.effector_fraction * dt

        # Natural contraction
        contraction = p.effector_decay_rate * cd8.effector_fraction * dt

        # PD-1 upregulation with antigen exposure
        cd8.antigen_exposure_days += dt
        cd8.pd1_expression = min(1.0,
            p.pd1_expression_rate * cd8.antigen_exposure_days * (1.0 - p.pd1_blockade))

        # Update fractions
        cd8.naive_fraction -= priming
        cd8.effector_fraction += priming - fatigue - contraction
        cd8.exhausted_fraction += fatigue + contraction

        # Cumulative exhaustion tracking
        cd8.cumulative_exhaustion += fatigue_rate * dt
        self.cumulative_exhaustion = cd8.cumulative_exhaustion  # Legacy compat

        cd8.normalize()

        # Force = base * effector_fraction * (1 - PD-1 inhibition)
        pd1_inhibition = cd8.pd1_expression * (1.0 - p.pd1_blockade)
        force_magnitude = p.base_force * cd8.effector_fraction * (1.0 - pd1_inhibition)

        return max(0.0, force_magnitude)

    # ── NK CELL DYNAMICS ──

    def _update_nk(self, current_state: np.ndarray, well_depth: float,
                   dt: float) -> float:
        """
        Update NK cell compartment and return force magnitude.

        NK cells respond to stress signals (ROS, metabolite 9 in our model).
        They provide innate surveillance independent of antigen presentation.
        """
        p = self.params
        nk = self.nk

        # ROS-driven activation: high ROS = stressed cells = NK targets
        # Metabolite index 9 = ROS in our 10-metabolite model
        ros_level = abs(current_state[9]) if len(current_state) > 9 else 0.5
        stress_signal = min(1.0, ros_level * p.nk_stress_sensitivity)

        # NKG2D-mediated activation
        nk.activation_level = 0.3 + 0.7 * stress_signal * nk.nkg2d_expression

        # NK fatigue (slower than T-cell exhaustion)
        nk.cumulative_fatigue += p.nk_exhaustion_rate * well_depth * dt
        fatigue_factor = np.exp(-nk.cumulative_fatigue)

        # NKG2D shedding by tumor (reduces receptor expression over time)
        nk.nkg2d_expression = max(0.2, nk.nkg2d_expression - 0.005 * dt)

        force_magnitude = p.nk_base_force * nk.activation_level * fatigue_factor

        return max(0.0, force_magnitude)

    # ── TREG DYNAMICS ──

    def _update_treg(self, dt: float) -> float:
        """
        Update Treg population and return suppressive coefficient.

        Tregs expand under tumor TGF-beta signaling and compete for IL-2,
        reducing effector T-cell activation. CTLA-4 blockade partially
        reduces Treg-mediated suppression.
        """
        p = self.params
        treg = self.treg

        # Tumor-driven Treg expansion
        treg.expansion_days += dt
        expansion = p.treg_expansion_rate * dt
        treg.current_load = min(p.treg_max_load, treg.current_load + expansion)

        # CTLA-4 blockade reduces Treg suppression
        effective_load = treg.current_load * (1.0 - p.ctla4_blockade * 0.6)

        # IL-2 competition: Tregs consume IL-2, starving effectors
        il2_penalty = p.treg_il2_competition * treg.current_load

        return effective_load + il2_penalty

    # ── DENDRITIC CELL DYNAMICS ──

    def _update_dc(self, current_state: np.ndarray, dt: float):
        """
        Update dendritic cell maturation and antigen presentation.

        DCs mature in response to tumor antigens and inflammatory signals.
        Mature DCs drive CD8+ priming (naïve → effector conversion).
        """
        p = self.params
        dc = self.dc

        # Antigen accumulation from tumor metabolic state
        tumor_signal = np.linalg.norm(current_state)
        dc.antigen_load = min(1.0, dc.antigen_load + 0.01 * tumor_signal * dt)

        # DC maturation (driven by antigen + inflammatory signals)
        maturation_drive = (p.dc_maturation_rate * dc.antigen_load
                           * p.dc_cross_presentation)
        dc.maturation_level = min(1.0, dc.maturation_level + maturation_drive * dt)

    # ── NET FORCE COMPUTATION ──

    def compute_net_force(self,
                         current_state: np.ndarray,
                         well_depth: float,
                         dt: float) -> np.ndarray:
        """
        Calculate the net immune force vector pushing toward healthy attractor.

        Integrates all compartments:
          F_net = (F_cd8 + F_nk) * barrier_factor * (1 - treg_suppression)

        Args:
            current_state: Where the system is in metabolic phase space (x)
            well_depth: Current basin curvature/depth (mu)
            dt: Time step size (days)

        Returns:
            force_vector: The net immune 'kick' applied to the metabolic state
        """
        # Direction: push toward origin (healthy equilibrium)
        norm = np.linalg.norm(current_state)
        if norm < 1e-6:
            return np.zeros_like(current_state)
        direction = -current_state / norm

        # Update all compartments
        self._update_dc(current_state, dt)
        cd8_force = self._update_cd8(well_depth, dt)
        nk_force = self._update_nk(current_state, well_depth, dt)
        treg_suppression = self._update_treg(dt)

        # Net effector force
        total_effector = cd8_force + nk_force

        # Apply suppression
        friction_multiplier = max(0.05, 1.0 - treg_suppression)

        # Apply tissue barrier
        infiltration = self.barrier_factor

        # Special barrier dynamics
        if self.params.barrier_type == "bbb":
            # BBB weakens with inflammation (anti-CTLA-4 causes some BBB disruption)
            bbb_disruption = self.params.ctla4_blockade * 0.2
            infiltration = min(1.0, infiltration + bbb_disruption)
        elif self.params.barrier_type == "desmoplasia":
            # Desmoplasia can be partially overcome with prolonged immune pressure
            stromal_erosion = min(0.15, self.cd8.antigen_exposure_days * 0.003)
            infiltration = min(1.0, infiltration + stromal_erosion)

        # Final magnitude
        magnitude = total_effector * friction_multiplier * infiltration

        return direction * magnitude

    # ── DRUG HOLIDAY RECOVERY ──

    def apply_drug_holiday(self, holiday_duration_days: float,
                           recovery_fraction: float = 0.85):
        """
        Model immune recovery during Phase 2 drug holiday.

        During drug holidays:
          - CD8+ exhaustion partially recovers
          - NK NKG2D expression recovers
          - Treg expansion slows (reduced TGF-beta from tumor)
          - PD-1 expression partially downregulates

        Args:
            holiday_duration_days: Duration of the drug holiday
            recovery_fraction: Maximum fraction of exhaustion recovered (0-1)
        """
        # CD8+ recovery
        recovery = min(recovery_fraction,
                       self.cd8.cumulative_exhaustion * 0.5 * (
                           1 - np.exp(-holiday_duration_days / 12.6)))
        self.cd8.cumulative_exhaustion = max(0, self.cd8.cumulative_exhaustion - recovery)
        self.cumulative_exhaustion = self.cd8.cumulative_exhaustion

        # Partial reversion from exhausted to effector
        revert = min(self.cd8.exhausted_fraction * 0.3,
                     recovery * self.cd8.exhausted_fraction)
        self.cd8.exhausted_fraction -= revert
        self.cd8.effector_fraction += revert
        self.cd8.normalize()

        # PD-1 downregulation during holiday
        self.cd8.pd1_expression *= max(0.3, 1.0 - holiday_duration_days * 0.05)
        self.cd8.antigen_exposure_days *= 0.5  # Partial reset

        # NK recovery
        self.nk.cumulative_fatigue *= max(0.2, 1.0 - holiday_duration_days * 0.08)
        self.nk.nkg2d_expression = min(1.0,
            self.nk.nkg2d_expression + holiday_duration_days * 0.02)

        # Treg partial contraction (less TGF-beta during holiday)
        self.treg.current_load *= max(0.5, 1.0 - holiday_duration_days * 0.03)

    # ── STATUS & REPORTING ──

    def get_status(self) -> dict:
        """Return comprehensive immune status for all compartments."""
        return {
            # CD8+
            "cd8_naive": round(self.cd8.naive_fraction, 3),
            "cd8_effector": round(self.cd8.effector_fraction, 3),
            "cd8_exhausted": round(self.cd8.exhausted_fraction, 3),
            "cd8_cumulative_exhaustion": round(self.cd8.cumulative_exhaustion, 3),
            "cd8_pd1_expression": round(self.cd8.pd1_expression, 3),
            # NK
            "nk_activation": round(self.nk.activation_level, 3),
            "nk_fatigue": round(self.nk.cumulative_fatigue, 3),
            "nk_nkg2d": round(self.nk.nkg2d_expression, 3),
            # Treg
            "treg_load": round(self.treg.current_load, 3),
            # DC
            "dc_maturation": round(self.dc.maturation_level, 3),
            "dc_antigen_load": round(self.dc.antigen_load, 3),
            # Net
            "exhaustion_level": round(
                1.0 - np.exp(-self.cd8.cumulative_exhaustion), 3),
            "force_magnitude": round(
                np.exp(-self.cd8.cumulative_exhaustion)
                * (1.0 - self.treg.current_load), 3),
            "is_exhausted": self.cd8.cumulative_exhaustion > self.params.exhaustion_threshold,
            "barrier_factor": round(self.barrier_factor, 3),
        }

    def get_summary(self) -> str:
        """One-line human-readable status."""
        s = self.get_status()
        return (f"CD8: {s['cd8_effector']:.0%}eff/{s['cd8_exhausted']:.0%}exh | "
                f"NK: {s['nk_activation']:.0%} | "
                f"Treg: {s['treg_load']:.0%} | "
                f"DC: {s['dc_maturation']:.0%} | "
                f"Barrier: {1-s['barrier_factor']:.0%}")
