"""
Tests for AdaptiveController — Project Confluence
===================================================
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.adaptive_controller import (
    AdaptiveController, PolicyParams, PolicyMode, 
    ControllerState, EpigeneticSteeringPolicy,
    run_adaptive_simulation, compare_policies,
)
from models.clonal_dynamics import ClonalDynamicsEngine, ClonalParams


class TestPolicyDecisions:
    """Test that each policy mode produces correct dosing behavior."""
    
    def test_threshold_doses_when_above_threshold(self):
        """Controller should dose when tumor fraction exceeds on-threshold."""
        params = PolicyParams(
            dose_on_threshold=0.3, dose_off_threshold=0.1, robust_max_dose=0.7,
            min_holiday_days=0,  # Disable holiday hold for this unit test
        )
        ctrl = AdaptiveController(PolicyMode.THRESHOLD, params)
        
        # V_frac = 0.6 > 0.3 → should dose
        dose = ctrl.decide(sensitive=0.5, resistant=0.1, carrying_capacity=1.0, dt=0.1)
        assert dose > 0, f"Expected dosing above threshold, got {dose}"
    
    def test_threshold_holidays_when_below_threshold(self):
        """Controller should NOT dose when tumor fraction is below off-threshold."""
        params = PolicyParams(dose_on_threshold=0.5, dose_off_threshold=0.3, robust_max_dose=0.7)
        ctrl = AdaptiveController(PolicyMode.THRESHOLD, params)
        
        # V_frac = 0.2 < 0.3 → should be off
        dose = ctrl.decide(sensitive=0.15, resistant=0.05, carrying_capacity=1.0, dt=0.1)
        assert dose == 0, f"Expected holiday below threshold, got {dose}"
    
    def test_robust_emergency_on_resistant_alarm(self):
        """Robust policy should escalate dose when resistant fraction is high."""
        params = PolicyParams(
            resistant_alarm_fraction=0.40, emergency_dose=0.90, robust_max_dose=0.90,
            min_holiday_days=0,  # Disable holiday hold for this unit test
        )
        ctrl = AdaptiveController(PolicyMode.ROBUST_ADAPTIVE, params)
        
        # R_frac = 0.6/0.7 = 85.7% → way above 40% alarm
        dose = ctrl.decide(sensitive=0.1, resistant=0.6, carrying_capacity=1.0, dt=0.1)
        assert dose >= 0.85, f"Expected emergency dose, got {dose}"
    
    def test_dose_never_exceeds_robust_max(self):
        """No policy should ever exceed the robust_max_dose."""
        params = PolicyParams(robust_max_dose=0.5)
        
        for mode in PolicyMode:
            ctrl = AdaptiveController(mode, params)
            for _ in range(100):
                dose = ctrl.decide(
                    sensitive=np.random.uniform(0, 1),
                    resistant=np.random.uniform(0, 1),
                    carrying_capacity=1.0, dt=0.1
                )
                assert dose <= params.robust_max_dose + 1e-6, \
                    f"{mode.value}: dose {dose} exceeded robust_max {params.robust_max_dose}"
    
    def test_proportional_scales_with_burden(self):
        """Proportional policy should give higher dose for higher tumor burden."""
        params = PolicyParams(
            proportional_gain=1.5, robust_max_dose=1.0,
            min_holiday_days=0,  # Disable holiday hold for this unit test
        )
        
        ctrl1 = AdaptiveController(PolicyMode.PROPORTIONAL, params)
        dose_low = ctrl1.decide(sensitive=0.1, resistant=0.05, carrying_capacity=1.0, dt=0.1)
        
        ctrl2 = AdaptiveController(PolicyMode.PROPORTIONAL, params)
        dose_high = ctrl2.decide(sensitive=0.5, resistant=0.2, carrying_capacity=1.0, dt=0.1)
        
        assert dose_high > dose_low, \
            f"Expected higher dose for higher burden: {dose_high} vs {dose_low}"


class TestSafetyConstraints:
    """Test that safety constraints are never violated."""
    
    def test_forced_holiday_after_max_continuous(self):
        """Controller must force a drug holiday after max continuous dosing days."""
        params = PolicyParams(
            dose_on_threshold=0.0,  # Always dose
            dose_off_threshold=0.0,
            robust_max_dose=1.0,
            max_continuous_dose_days=5.0,
            min_holiday_days=0,  # Don't block with holiday hold
            max_cumulative_toxicity=999,
        )
        ctrl = AdaptiveController(PolicyMode.THRESHOLD, params)
        
        # Force into dosing state first
        ctrl.ctrl_state.is_dosing = True
        
        # Dose continuously — track when forced holiday triggers
        forced_holiday_step = None
        for i in range(100):  # 100 * 0.1 = 10 days max
            dose = ctrl.decide(sensitive=0.5, resistant=0.1, carrying_capacity=1.0, dt=0.1)
            if dose == 0.0 and forced_holiday_step is None:
                forced_holiday_step = i
                break
        
        # Holiday should trigger around step 50 (5.0 days / 0.1 dt)
        assert forced_holiday_step is not None, "Forced holiday never triggered"
        assert forced_holiday_step <= 55, \
            f"Forced holiday at step {forced_holiday_step}, expected around step 50"
    
    def test_cumulative_toxicity_budget(self):
        """Cumulative toxicity must never exceed the budget."""
        params = PolicyParams(
            dose_on_threshold=0.0,
            robust_max_dose=1.0,
            max_cumulative_toxicity=5.0,
            toxicity_per_dose_unit=1.0,
            max_continuous_dose_days=999,
            min_holiday_days=0,
        )
        ctrl = AdaptiveController(PolicyMode.THRESHOLD, params)
        
        # Run for many steps
        for _ in range(1000):
            ctrl.decide(sensitive=0.5, resistant=0.1, carrying_capacity=1.0, dt=0.1)
        
        assert ctrl.ctrl_state.cumulative_toxicity <= params.max_cumulative_toxicity + 0.5, \
            f"Toxicity {ctrl.ctrl_state.cumulative_toxicity} exceeded budget {params.max_cumulative_toxicity}"
    
    def test_min_holiday_duration(self):
        """Once on holiday, controller must stay off for minimum holiday days."""
        params = PolicyParams(
            dose_on_threshold=0.3,
            dose_off_threshold=0.1,
            robust_max_dose=1.0,
            min_holiday_days=3.0,
            max_continuous_dose_days=999,
            max_cumulative_toxicity=999,
        )
        ctrl = AdaptiveController(PolicyMode.THRESHOLD, params)
        
        # Force into dosing state then drop below threshold
        ctrl.decide(sensitive=0.4, resistant=0.1, carrying_capacity=1.0, dt=0.1)
        # Now go below off threshold → holiday starts
        ctrl.decide(sensitive=0.05, resistant=0.01, carrying_capacity=1.0, dt=0.1)
        
        # Try to dose again immediately with high burden — should be blocked
        dose = ctrl.decide(sensitive=0.5, resistant=0.1, carrying_capacity=1.0, dt=0.1)
        assert dose == 0.0, \
            f"Expected min holiday to block re-dosing, got {dose}"


class TestEpigeneticSteeringPolicy:
    """Test OSKM steering with Landauer thermal safety."""

    def test_oskm_pulses_when_identity_degraded(self):
        params = PolicyParams(
            robust_max_dose=0.7,
            oskm_max_dose=0.3,
            min_holiday_days=0,
        )
        ctrl = AdaptiveController(PolicyMode.EPIGENETIC_STEERING, params)

        dose = ctrl.decide(
            sensitive=0.1,
            resistant=0.0,
            carrying_capacity=1.0,
            dt=0.1,
            identity_metrics={
                "margin": 0.01,
                "memory_integrity": 0.6,
                "regime": "degraded",
            },
        )

        assert dose > 0.0
        assert dose <= params.oskm_max_dose
        assert ctrl.ctrl_state.last_oskm_dose > 0.0

    def test_landauer_thermal_override_forces_holiday(self):
        params = PolicyParams(
            robust_max_dose=1.0,
            oskm_max_dose=1.0,
            min_holiday_days=0,
            landauer_bits_per_full_dose=1.0e18,
            landauer_heat_to_kelvin_gain=1.0e10,
        )
        ctrl = AdaptiveController(PolicyMode.EPIGENETIC_STEERING, params)

        dose = ctrl.decide(
            sensitive=0.1,
            resistant=0.0,
            carrying_capacity=1.0,
            dt=1.0,
            identity_metrics={
                "margin": -0.01,
                "memory_integrity": 0.2,
                "regime": "critical",
            },
        )

        assert dose == 0.0
        assert ctrl.ctrl_state.thermal_overrides == 1
        assert any("LANDAUER_THERMAL_OVERRIDE" in msg for msg in ctrl.ctrl_state.decision_log)

    def test_epigenetic_summary_exposes_thermal_state(self):
        params = PolicyParams(min_holiday_days=0)
        ctrl = AdaptiveController(PolicyMode.EPIGENETIC_STEERING, params)
        ctrl.decide(
            sensitive=0.1,
            resistant=0.0,
            carrying_capacity=1.0,
            dt=0.1,
            identity_metrics={"margin": 1.0, "memory_integrity": 1.0, "regime": "coherent"},
        )

        summary = ctrl.get_summary()
        assert summary["policy_mode"] == "epigenetic_steering"
        assert "cell_temperature_kelvin" in summary
        assert "thermal_overrides" in summary


class TestIntegratedSimulation:
    """Test the full simulation loop."""
    
    def test_simulation_runs_without_error(self):
        """Basic smoke test: simulation completes."""
        result = run_adaptive_simulation(
            cancer_type="NSCLC", total_days=30, dt=0.5, seed=42
        )
        assert "outcome" in result
        assert "trajectories" in result
        assert "controller_summary" in result
        assert len(result["trajectories"]["sensitive"]) > 0
    
    def test_adaptive_beats_mtd_on_resistance(self):
        """Adaptive therapy should produce lower resistant fraction than MTD."""
        comparison = compare_policies(
            cancer_type="NSCLC", total_days=60, dt=0.5, seed=42
        )
        
        mtd_R = comparison["comparison"]["MTD"]["final_R_fraction"]
        ada_R = comparison["comparison"]["Adaptive"]["final_R_fraction"]
        
        assert ada_R < mtd_R, \
            f"Expected Adaptive R%={ada_R:.2%} < MTD R%={mtd_R:.2%}"
    
    def test_controller_summary_structure(self):
        """Controller summary should contain expected keys."""
        result = run_adaptive_simulation(
            cancer_type="NSCLC", total_days=30, dt=0.5, seed=42
        )
        summary = result["controller_summary"]
        
        expected_keys = [
            "policy_mode", "total_decisions", "dose_switches",
            "dosing_fraction", "cumulative_toxicity", "policy_params"
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key '{key}' in controller summary"
    
    def test_multi_cancer_type_support(self):
        """Simulation should work across all supported cancer types."""
        cancer_types = ["NSCLC", "TNBC", "GBM", "AML", "CRC"]
        
        for ct in cancer_types:
            result = run_adaptive_simulation(
                cancer_type=ct, total_days=20, dt=0.5, seed=42
            )
            assert result["cancer_type"] == ct
            assert result["outcome"]["final_burden"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
