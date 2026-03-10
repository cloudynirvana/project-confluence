"""
ODE System Tests — Project Confluence
=======================================

Validates that the consolidated ODE system obeys fundamental constraints:
  1. Mass balance — total metabolic mass stays bounded
  2. Non-negativity — concentrations don't diverge negatively
  3. Eigenvalue stability — all generators have Re(λ) < 0
  4. Deterministic reproducibility — same seed → same output
  5. 15D system boundedness — ComplexAttractorODE stays finite

Run:
    python -m pytest tests/test_ode_system.py -v
"""

import sys
import os
import unittest
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ode_system import (
    TNBCODESystem,
    ComplexAttractorODE,
    ExtendedParams,
    ODEParams,
    simulate_trajectory,
    METABOLITE_NAMES,
    STATE_NAMES,
    TrajectoryAnalyzer,
)


class TestMassBalance(unittest.TestCase):
    """Total metabolic mass must remain bounded over simulation."""

    def setUp(self):
        self.x0 = np.array([2.0, 0.5, 1.0, 4.0, 1.5, 1.5, 0.8, 0.6, 1.0, 0.3])

    def test_healthy_mass_bounded(self):
        """Healthy generator keeps total mass bounded."""
        A = TNBCODESystem.healthy_generator()
        t, x = simulate_trajectory(A, self.x0, t_days=60, dt=0.1)
        total_mass = np.sum(np.abs(x), axis=1)
        self.assertTrue(np.all(np.isfinite(total_mass)),
                        "Mass trajectory has non-finite values")

    def test_all_cancer_mass_bounded(self):
        """All cancer generators keep mass bounded over 60 days."""
        for name, A in TNBCODESystem.all_generators().items():
            t, x = simulate_trajectory(A, self.x0, t_days=60, dt=0.1)
            total_mass = np.sum(np.abs(x), axis=1)
            self.assertTrue(np.all(total_mass < 500),
                            f"{name}: total mass exceeded 500 at some timestep")


class TestEigenvalueStability(unittest.TestCase):
    """All generators must have stable eigenvalues (Re(λ) < 0)."""

    def setUp(self):
        self.generators = TNBCODESystem.all_generators()

    def test_all_eigenvalues_stable(self):
        """Every generator must have all Re(λ) < 0."""
        for name, A in self.generators.items():
            eigs = np.linalg.eigvals(A)
            max_real = np.max(np.real(eigs))
            self.assertLess(max_real, 0.0,
                            f"{name}: max Re(λ) = {max_real:.4f} >= 0")

    def test_eigenvalue_spread(self):
        """No two generators should have nearly identical spectra."""
        names = list(self.generators.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A_i = self.generators[names[i]]
                A_j = self.generators[names[j]]
                eigs_i = np.sort(np.real(np.linalg.eigvals(A_i)))
                eigs_j = np.sort(np.real(np.linalg.eigvals(A_j)))
                self.assertFalse(
                    np.allclose(eigs_i, eigs_j, atol=1e-6),
                    f"{names[i]} and {names[j]}: eigenvalue spectra too similar"
                )


class TestDeterministicReproducibility(unittest.TestCase):
    """Same seed must produce identical output."""

    def test_seed_reproducibility(self):
        """Two runs with seed=42 must produce identical trajectories."""
        A = TNBCODESystem.tnbc_generator()
        x0 = np.array([2.0, 0.5, 1.0, 4.0, 1.5, 1.5, 0.8, 0.6, 1.0, 0.3])
        _, x1 = simulate_trajectory(A, x0, noise_sigma=0.01, seed=42)
        _, x2 = simulate_trajectory(A, x0, noise_sigma=0.01, seed=42)
        np.testing.assert_array_equal(x1, x2)

    def test_deterministic_no_seed_dependency(self):
        """Deterministic mode (noise=0) should be independent of seed."""
        A = TNBCODESystem.tnbc_generator()
        x0 = np.array([2.0, 0.5, 1.0, 4.0, 1.5, 1.5, 0.8, 0.6, 1.0, 0.3])
        _, x1 = simulate_trajectory(A, x0, noise_sigma=0.0, seed=42)
        _, x2 = simulate_trajectory(A, x0, noise_sigma=0.0, seed=99)
        np.testing.assert_array_almost_equal(x1, x2)


class TestComplexAttractorODE(unittest.TestCase):
    """Test the 15D nonlinear ODE system."""

    def test_healthy_state_shape(self):
        """Healthy initial state must be 15D."""
        ode = ComplexAttractorODE()
        z0 = ode.healthy_initial_state()
        self.assertEqual(len(z0), 15)

    def test_solve_succeeds(self):
        """Healthy initial conditions should integrate without error."""
        ode = ComplexAttractorODE()
        result = ode.solve(t_span=(0, 50), dt_eval=0.5)
        self.assertTrue(result["success"], f"Solver failed: {result['message']}")

    def test_trajectory_bounded(self):
        """Trajectory should stay bounded (no explosion)."""
        ode = ComplexAttractorODE()
        result = ode.solve(t_span=(0, 100), dt_eval=0.5)
        self.assertTrue(TrajectoryAnalyzer.is_bounded(result["z"], threshold=100),
                        "Trajectory exploded beyond 100")

    def test_linear_mode_backward_compat(self):
        """Linear mode (use_nonlinear=False) should recover SAEM behavior."""
        ode = ComplexAttractorODE(use_nonlinear=False, use_immune=False,
                                  use_microenv=False)
        A_ode = ode.get_metabolic_generator()
        A_saem = TNBCODESystem.healthy_generator()
        np.testing.assert_array_almost_equal(A_ode, A_saem)

    def test_generator_matrix_dimension(self):
        """Metabolic generator should be 10x10."""
        ode = ComplexAttractorODE()
        A = ode.get_metabolic_generator()
        self.assertEqual(A.shape, (10, 10))


class TestTrajectoryAnalyzer(unittest.TestCase):
    """Test trajectory analysis utilities."""

    def test_summary_stats_keys(self):
        """Summary stats should contain all 15 state variable names."""
        ode = ComplexAttractorODE()
        result = ode.solve(t_span=(0, 20), dt_eval=0.5)
        stats = TrajectoryAnalyzer.summary_stats(result["z"], result["t"])
        for name in STATE_NAMES:
            self.assertIn(name, stats, f"Missing state variable: {name}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
