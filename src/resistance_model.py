"""
Resistance Evolution Module — Project Confluence
==================================================

Multi-mechanism tumor resistance modeling with:
  - Efflux pumps (P-gp/MDR1): Reduces effective drug concentration
  - Target mutation: Stochastic point mutations at drug binding sites
  - Metabolic rewiring: Cancer adapts generator in response to drug pressure
  - Clonal selection: Resistant subclone dynamics (grows during treatment,
    shrinks during drug holidays due to fitness cost)
  - Cross-resistance: Shared resistance mechanisms between drug classes

The model updates the effective drug concentration at each simulation
timestep, reducing treatment efficacy over time unless interrupted
by drug holidays (adaptive therapy).

References:
  - Gatenby et al. 2009, Cancer Research: Adaptive therapy
  - Vasan et al. 2019, Nature: Cancer resistance evolution
  - Enriquez-Navas et al. 2016, Science TM: Adaptive PDAC
  - Zhang et al. 2017, Nature Communications: Clonal dynamics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ResistanceParams:
    """Parameters governing resistance evolution."""
    # Efflux pumps (P-gp/MDR1)
    efflux_induction_rate: float = 0.02     # Pump upregulation per day of drug exposure
    efflux_max_reduction: float = 0.60      # Max drug concentration reduction from pumps
    efflux_decay_rate: float = 0.05         # Pump downregulation during drug holiday (per day)

    # Target mutations
    mutation_rate: float = 1e-6             # Point mutation probability per cell division per day
    mutation_doubling_time: float = 2.0     # Days for mutant population to double
    mutation_drug_nullification: float = 0.8  # How much a mutation reduces drug effect (0-1)

    # Metabolic rewiring
    rewiring_rate: float = 0.005            # Generator adaptation rate per day per drug
    rewiring_max: float = 0.30              # Max fraction of drug effect countered by rewiring
    rewiring_recovery_rate: float = 0.01    # Rewiring reversal during holiday (per day)

    # Clonal selection
    resistant_fraction_initial: float = 0.01   # Pre-existing resistant cells (1%)
    resistant_growth_advantage: float = 0.03   # Growth advantage under treatment (per day)
    resistant_fitness_cost: float = 0.015      # Fitness cost without drug pressure (per day)
    resistant_max_fraction: float = 0.95       # Maximum resistant fraction

    # Cross-resistance groups
    # Drugs in the same group share resistance mechanisms
    cross_resistance_transfer: float = 0.4  # Fraction of resistance transferred within group

    # Resistance time constant (legacy compatibility)
    resistance_tau: float = 18.0   # Days for exponential resistance buildup


# Drug cross-resistance groups
CROSS_RESISTANCE_GROUPS = {
    "metabolic": [
        "Dichloroacetate (DCA)",
        "Metformin",
        "2-Deoxyglucose (2-DG)",
        "Fasting-Mimicking Diet (FMD)",
    ],
    "epigenetic": [
        "Vorinostat (SAHA, HDACi)",
        "5-Azacitidine (DNMTi)",
    ],
    "oxidative_stress": [
        "High-dose Vitamin C",
        "Ferroptosis Inducer (Erastin/RSL3)",
        "N6F11 (Selective GPX4 degrader)",
    ],
    "checkpoint": [
        "Anti-PD-1 (Pembrolizumab)",
        "Anti-CTLA-4 (Ipilimumab)",
    ],
    "autophagy": [
        "Hydroxychloroquine (HCQ)",
    ],
}


@dataclass
class DrugResistanceState:
    """Resistance state for a single drug."""
    drug_name: str
    exposure_days: float = 0.0              # Cumulative days of exposure
    holiday_days: float = 0.0               # Cumulative days of holiday
    efflux_level: float = 0.0               # Current efflux pump expression (0-1)
    mutant_fraction: float = 0.0            # Fraction of cells with target mutation
    rewiring_level: float = 0.0             # Metabolic adaptation level (0-1)
    resistant_clone_fraction: float = 0.01  # Resistant subclone fraction


class ResistanceTracker:
    """
    Multi-mechanism resistance tracker for adaptive therapy simulation.

    Tracks resistance evolution for each drug independently, with
    cross-resistance transfer between drugs in the same mechanistic group.

    Usage:
        tracker = ResistanceTracker(params)
        tracker.initialize_drugs(["DCA", "Metformin", "Anti-PD-1"])

        # During treatment
        factor = tracker.get_efficacy_factor("DCA")  # Returns 0-1 multiplier
        tracker.update(dt=0.1, active_drugs=["DCA", "Metformin"])

        # During drug holiday
        tracker.apply_holiday(duration_days=7.0)
    """

    def __init__(self, params: Optional[ResistanceParams] = None):
        self.params = params or ResistanceParams()
        self.drug_states: Dict[str, DrugResistanceState] = {}
        self.total_treatment_days: float = 0.0
        self.total_holiday_days: float = 0.0

    def initialize_drugs(self, drug_names: List[str]):
        """Initialize resistance tracking for a set of drugs."""
        for name in drug_names:
            self.drug_states[name] = DrugResistanceState(
                drug_name=name,
                resistant_clone_fraction=self.params.resistant_fraction_initial,
            )

    def _get_cross_resistance_group(self, drug_name: str) -> Optional[str]:
        """Find which cross-resistance group a drug belongs to."""
        for group, drugs in CROSS_RESISTANCE_GROUPS.items():
            if drug_name in drugs:
                return group
        return None

    def update(self, dt: float, active_drugs: List[str]):
        """
        Update resistance state for one simulation timestep.

        Args:
            dt: Timestep size in days
            active_drugs: List of drug names currently being administered
        """
        p = self.params
        self.total_treatment_days += dt

        for name, state in self.drug_states.items():
            is_active = name in active_drugs

            if is_active:
                state.exposure_days += dt
                state.holiday_days = 0.0

                # 1. Efflux pump induction
                pump_growth = p.efflux_induction_rate * dt
                state.efflux_level = min(
                    p.efflux_max_reduction,
                    state.efflux_level + pump_growth)

                # 2. Target mutation accumulation
                # Mutations are stochastic but we model as deterministic rate
                div_rate = 1.0 / max(p.mutation_doubling_time, 0.1)
                new_mutants = p.mutation_rate * div_rate * dt
                state.mutant_fraction = min(
                    1.0,
                    state.mutant_fraction + new_mutants
                    + state.mutant_fraction * div_rate * dt)

                # 3. Metabolic rewiring
                rewire_growth = p.rewiring_rate * dt
                state.rewiring_level = min(
                    p.rewiring_max,
                    state.rewiring_level + rewire_growth)

                # 4. Resistant clone expansion (growth advantage under drug)
                clone_growth = (p.resistant_growth_advantage * dt
                               * state.resistant_clone_fraction
                               * (1.0 - state.resistant_clone_fraction / p.resistant_max_fraction))
                state.resistant_clone_fraction = min(
                    p.resistant_max_fraction,
                    state.resistant_clone_fraction + clone_growth)

            else:
                # Drug holiday — resistance partially decays
                state.holiday_days += dt

                # Efflux pumps downregulate
                state.efflux_level *= max(0.0, 1.0 - p.efflux_decay_rate * dt)

                # Metabolic rewiring slowly reverses
                state.rewiring_level *= max(0.0, 1.0 - p.rewiring_recovery_rate * dt)

                # Resistant clones lose fitness advantage (costly resistance)
                clone_decay = (p.resistant_fitness_cost * dt
                              * state.resistant_clone_fraction)
                state.resistant_clone_fraction = max(
                    p.resistant_fraction_initial,
                    state.resistant_clone_fraction - clone_decay)

        # Apply cross-resistance transfer
        self._transfer_cross_resistance(active_drugs, dt)

    def _transfer_cross_resistance(self, active_drugs: List[str], dt: float):
        """Transfer resistance between drugs in the same mechanistic group."""
        p = self.params

        for group_name, group_drugs in CROSS_RESISTANCE_GROUPS.items():
            # Find active drugs in this group
            active_in_group = [d for d in active_drugs if d in group_drugs]
            if len(active_in_group) < 2:
                continue

            # Find the drug with highest resistance in the group
            max_efflux = max(
                self.drug_states[d].efflux_level
                for d in active_in_group if d in self.drug_states
            )
            max_rewiring = max(
                self.drug_states[d].rewiring_level
                for d in active_in_group if d in self.drug_states
            )

            # Transfer partial resistance to other drugs in the group
            for drug_name in group_drugs:
                if drug_name not in self.drug_states:
                    continue
                state = self.drug_states[drug_name]
                transfer = p.cross_resistance_transfer * dt

                if state.efflux_level < max_efflux:
                    state.efflux_level += (max_efflux - state.efflux_level) * transfer

                if state.rewiring_level < max_rewiring:
                    state.rewiring_level += (max_rewiring - state.rewiring_level) * transfer

    def get_efficacy_factor(self, drug_name: str) -> float:
        """
        Get the net efficacy reduction factor for a drug.

        Returns a multiplier in [0, 1] where:
          1.0 = full efficacy (no resistance)
          0.0 = complete resistance

        The factor is computed as the product of all resistance mechanisms:
          factor = (1 - efflux) * (1 - mutation_impact) * (1 - rewiring) * (1 - clone_frac * cost)
        """
        if drug_name not in self.drug_states:
            return 1.0

        state = self.drug_states[drug_name]
        p = self.params

        # Efflux pump reduction
        efflux_factor = 1.0 - state.efflux_level

        # Target mutation impact
        mutation_factor = 1.0 - state.mutant_fraction * p.mutation_drug_nullification

        # Metabolic rewiring
        rewiring_factor = 1.0 - state.rewiring_level

        # Resistant clone fraction
        clone_factor = 1.0 - state.resistant_clone_fraction * 0.5

        # Combined efficacy
        combined = efflux_factor * mutation_factor * rewiring_factor * clone_factor
        return max(0.05, combined)  # Floor at 5% (never fully zero)

    def get_legacy_factor(self, exposure_days: float) -> float:
        """
        Legacy compatibility: simple exponential resistance factor.

        Used by the universal_cure_engine for backward compatibility.
        """
        return np.exp(-exposure_days / self.params.resistance_tau)

    def apply_holiday(self, duration_days: float):
        """
        Model a complete drug holiday period.
        Updates all drug states to reflect recovery.
        """
        self.total_holiday_days += duration_days

        for name, state in self.drug_states.items():
            state.holiday_days += duration_days

            # Efflux pump decay over holiday
            decay = 1.0 - np.exp(-self.params.efflux_decay_rate * duration_days)
            state.efflux_level *= (1.0 - decay)

            # Rewiring partial reversal
            rewire_decay = 1.0 - np.exp(-self.params.rewiring_recovery_rate * duration_days)
            state.rewiring_level *= (1.0 - rewire_decay)

            # Resistant clone contraction (fitness cost)
            clone_decay = (self.params.resistant_fitness_cost * duration_days
                          * state.resistant_clone_fraction)
            state.resistant_clone_fraction = max(
                self.params.resistant_fraction_initial,
                state.resistant_clone_fraction - clone_decay)

    def get_summary(self) -> Dict:
        """Return comprehensive resistance status for all drugs."""
        summary = {
            "total_treatment_days": round(self.total_treatment_days, 1),
            "total_holiday_days": round(self.total_holiday_days, 1),
            "drugs": {},
        }
        for name, state in self.drug_states.items():
            efficacy = self.get_efficacy_factor(name)
            summary["drugs"][name] = {
                "efficacy_factor": round(efficacy, 3),
                "exposure_days": round(state.exposure_days, 1),
                "efflux_level": round(state.efflux_level, 3),
                "mutant_fraction": round(state.mutant_fraction, 6),
                "rewiring_level": round(state.rewiring_level, 3),
                "resistant_clone_pct": round(state.resistant_clone_fraction * 100, 1),
            }
        return summary


def compare_adaptive_vs_continuous(
    drug_names: List[str],
    total_days: int = 60,
    cycle_length: int = 21,
    holiday_length: int = 7,
    params: Optional[ResistanceParams] = None,
) -> Dict:
    """
    Compare resistance evolution: adaptive (with holidays) vs continuous.

    Args:
        drug_names: List of drugs to track
        total_days: Total treatment duration
        cycle_length: Days of treatment per adaptive cycle
        holiday_length: Days of holiday per adaptive cycle
        params: Resistance parameters

    Returns:
        Dict with efficacy trajectories for both strategies
    """
    p = params or ResistanceParams()
    dt = 0.1  # Simulation timestep

    # Continuous therapy tracker
    cont = ResistanceTracker(p)
    cont.initialize_drugs(drug_names)

    # Adaptive therapy tracker
    adapt = ResistanceTracker(p)
    adapt.initialize_drugs(drug_names)

    cont_trajectory = []
    adapt_trajectory = []

    t = 0.0
    while t < total_days:
        # Continuous: always on
        cont.update(dt, drug_names)

        # Adaptive: cycle on/holiday
        cycle_pos = t % (cycle_length + holiday_length)
        if cycle_pos < cycle_length:
            adapt.update(dt, drug_names)
        else:
            adapt.update(dt, [])  # Holiday — no active drugs

        # Record average efficacy across all drugs
        cont_avg = np.mean([cont.get_efficacy_factor(d) for d in drug_names])
        adapt_avg = np.mean([adapt.get_efficacy_factor(d) for d in drug_names])

        cont_trajectory.append({"day": round(t, 1), "efficacy": round(cont_avg, 4)})
        adapt_trajectory.append({"day": round(t, 1), "efficacy": round(adapt_avg, 4)})

        t += dt

    return {
        "continuous": {
            "final_efficacy": cont_trajectory[-1]["efficacy"] if cont_trajectory else 0,
            "summary": cont.get_summary(),
            "trajectory": cont_trajectory[::10],  # Subsample for reporting
        },
        "adaptive": {
            "final_efficacy": adapt_trajectory[-1]["efficacy"] if adapt_trajectory else 0,
            "summary": adapt.get_summary(),
            "trajectory": adapt_trajectory[::10],
        },
        "adaptive_advantage": (
            (adapt_trajectory[-1]["efficacy"] - cont_trajectory[-1]["efficacy"])
            if cont_trajectory and adapt_trajectory else 0
        ),
    }
