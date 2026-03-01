"""
Generator Calibrator — Project Confluence
==========================================

Calibrates SAEM generator matrices against real metabolomics data
using Bayesian refinement: start from hand-tuned priors and adjust
entries to match observed metabolite profiles.

Key principle: do NOT extract generators from scratch (unstable).
Instead, refine existing literature-based generators within bounded
constraints to preserve theoretical structure.

Usage:
    from generator_calibrator import GeneratorCalibrator
    
    calibrator = GeneratorCalibrator()
    A_refined, report = calibrator.refine_generator(
        A_prior=A_tnbc,
        real_profiles=ccle_tnbc_profiles,  # (n_samples, 10)
        alpha=0.3  # max 30% entry deviation from prior
    )
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class CalibrationReport:
    """Report from generator calibration against real data."""
    cancer_type: str
    n_samples: int
    
    # Quality metrics
    r_squared_before: float
    r_squared_after: float
    rmse_before: float
    rmse_after: float
    
    # Per-metabolite errors
    per_metabolite_error_before: Dict[str, float] = field(default_factory=dict)
    per_metabolite_error_after: Dict[str, float] = field(default_factory=dict)
    
    # Change tracking
    entries_changed: int = 0
    max_entry_change: float = 0.0
    frobenius_distance: float = 0.0
    
    # Stability preserved?
    stable_before: bool = True
    stable_after: bool = True


METABOLITE_NAMES = [
    "Glucose", "Lactate", "Pyruvate", "ATP", "NADH",
    "Glutamine", "Glutamate", "aKG", "Citrate", "ROS",
]


class GeneratorCalibrator:
    """
    Calibrate SAEM 10×10 generator matrices against real metabolomics data.
    
    Strategy: Bayesian refinement
      1. Start from literature-based prior A_prior
      2. Simulate to steady state → get predicted metabolite profile
      3. Compare against real measured profiles
      4. Optimize A entries (bounded ±alpha) to minimize prediction error
      5. Constraints: diagonal stays negative, stability preserved
    """
    
    def __init__(self, max_iterations: int = 500, verbose: bool = True):
        self.max_iterations = max_iterations
        self.verbose = verbose
    
    def simulate_steady_state(
        self,
        A: np.ndarray,
        x0: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Extract the dominant metabolite profile from generator matrix A.
        
        For a stable linear system dx/dt = Ax (all eigenvalues < 0), the
        dominant eigenvector (least-negative eigenvalue) captures the
        slowest-decaying mode — i.e., the metabolite ratios that persist
        longest during decay to zero. This is the biologically meaningful
        "profile" since it represents the cancer cell's metabolic steady
        state before therapeutic intervention drives it to collapse.
        
        This is ~100x faster than time-stepping simulation because it's
        a single eigendecomposition O(n³) instead of O(steps × n²).
        
        Returns: 10-element normalized metabolite profile (all positive)
        """
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # Find the dominant mode: least-negative real eigenvalue
        real_parts = eigenvalues.real
        # Only consider stable modes (negative real parts)
        stable_mask = real_parts < 0
        if not np.any(stable_mask):
            # Fallback: use eigenvector with smallest |real part|
            dom_idx = np.argmin(np.abs(real_parts))
        else:
            # Among stable eigenvalues, find the one closest to 0
            masked_real = np.where(stable_mask, real_parts, -np.inf)
            dom_idx = np.argmax(masked_real)
        
        # Extract dominant eigenvector and take absolute values
        # (biological concentrations are positive)
        profile = np.abs(eigenvectors[:, dom_idx].real)
        
        # Normalize
        norm = np.linalg.norm(profile)
        if norm > 1e-9:
            profile = profile / norm
        
        return profile
    
    def compute_profile_error(
        self,
        A: np.ndarray,
        real_profiles: np.ndarray,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Compute error between simulated profile and real data.
        
        Args:
            A:             10×10 generator matrix
            real_profiles: (n_samples, 10) real metabolite profiles
        
        Returns:
            (rmse, r_squared, per_metabolite_errors)
        """
        # Mean real profile
        mean_real = np.mean(real_profiles, axis=0)
        mean_real_norm = np.linalg.norm(mean_real)
        if mean_real_norm > 1e-9:
            mean_real = mean_real / mean_real_norm
        
        # Simulated profile
        sim_profile = self.simulate_steady_state(A)
        
        # RMSE
        errors = sim_profile - mean_real
        rmse = float(np.sqrt(np.mean(errors**2)))
        
        # R²
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((mean_real - np.mean(mean_real))**2)
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-9)
        
        # Per-metabolite
        per_met = {}
        for i, name in enumerate(METABOLITE_NAMES):
            per_met[name] = float(abs(errors[i]))
        
        return rmse, r_squared, per_met
    
    def is_stable(self, A: np.ndarray) -> bool:
        """Check if all eigenvalues have negative real parts."""
        eigenvalues = np.linalg.eigvals(A)
        return bool(np.all(eigenvalues.real < 0))
    
    def refine_generator(
        self,
        A_prior: np.ndarray,
        real_profiles: np.ndarray,
        cancer_type: str = "Unknown",
        alpha: float = 0.3,
        diagonal_constraint: bool = True,
        stability_constraint: bool = True,
    ) -> Tuple[np.ndarray, CalibrationReport]:
        """
        Refine a generator matrix to better match real metabolomics data.
        
        Args:
            A_prior:       10×10 literature-based generator (our current best guess)
            real_profiles:  (n_samples, 10) real metabolite profiles from CCLE
            cancer_type:    Name for reporting
            alpha:          Max fractional deviation (0.3 = ±30%) from prior entries
            diagonal_constraint:  If True, diagonal entries must stay negative
            stability_constraint: If True, refined A must have all eigenvalues < 0
        
        Returns:
            (A_refined, CalibrationReport)
        """
        n = A_prior.shape[0]
        assert A_prior.shape == (n, n), f"Expected square matrix, got {A_prior.shape}"
        assert real_profiles.shape[1] == n, f"Profiles must have {n} columns"
        
        # Pre-refinement metrics
        rmse_before, r2_before, per_met_before = self.compute_profile_error(
            A_prior, real_profiles
        )
        stable_before = self.is_stable(A_prior)
        
        if self.verbose:
            print(f"[Calibrator] {cancer_type}: RMSE_before={rmse_before:.4f}, R²={r2_before:.4f}")
        
        # Identify which entries to optimize (non-zero entries in A_prior)
        mask = np.abs(A_prior) > 1e-6
        n_params = int(mask.sum())
        
        if self.verbose:
            print(f"[Calibrator] Optimizing {n_params} non-zero entries (±{alpha*100:.0f}%)")
        
        # Pack non-zero entries into a flat vector for optimizer
        def pack(A: np.ndarray) -> np.ndarray:
            return A[mask]
        
        def unpack(params: np.ndarray) -> np.ndarray:
            A = np.zeros_like(A_prior)
            A[~mask] = A_prior[~mask]  # Keep zero entries as zero
            A[mask] = params
            return A
        
        # Bounds: ±alpha of prior values, with min/max guards
        prior_values = pack(A_prior)
        bounds = []
        for val in prior_values:
            if abs(val) < 1e-6:
                bounds.append((-0.1, 0.1))
            else:
                delta = abs(val) * alpha
                bounds.append((val - delta, val + delta))
        
        # Apply diagonal constraint: diagonal entries must be negative
        if diagonal_constraint:
            diag_indices_in_mask = []
            flat_idx = 0
            for i in range(n):
                for j in range(n):
                    if mask[i, j]:
                        if i == j:
                            diag_indices_in_mask.append(flat_idx)
                            # Ensure upper bound is negative
                            lo, hi = bounds[flat_idx]
                            bounds[flat_idx] = (lo, min(hi, -0.01))
                        flat_idx += 1
        
        # Objective function
        mean_real = np.mean(real_profiles, axis=0)
        mean_real_norm = np.linalg.norm(mean_real)
        if mean_real_norm > 1e-9:
            mean_real_normalized = mean_real / mean_real_norm
        else:
            mean_real_normalized = mean_real
        
        def objective(params: np.ndarray) -> float:
            A = unpack(params)
            
            # Stability penalty
            if stability_constraint and not self.is_stable(A):
                return 100.0  # Heavy penalty
            
            sim = self.simulate_steady_state(A)
            error = np.sum((sim - mean_real_normalized)**2)
            
            # Regularization: don't drift too far from prior
            reg = 0.01 * np.sum((params - prior_values)**2)
            
            return float(error + reg)
        
        # Optimize
        result = minimize(
            objective,
            prior_values,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.max_iterations, 'disp': False},
        )
        
        A_refined = unpack(result.x)
        
        # Post-refinement metrics
        rmse_after, r2_after, per_met_after = self.compute_profile_error(
            A_refined, real_profiles
        )
        stable_after = self.is_stable(A_refined)
        
        # Change tracking
        diff = np.abs(A_refined - A_prior)
        frob_dist = float(np.linalg.norm(A_refined - A_prior, 'fro'))
        entries_changed = int(np.sum(diff > 1e-4))
        max_change = float(np.max(diff))
        
        if self.verbose:
            print(f"[Calibrator] {cancer_type}: RMSE_after={rmse_after:.4f}, R²={r2_after:.4f}")
            print(f"[Calibrator] Entries changed: {entries_changed}, "
                  f"max Δ={max_change:.4f}, Frobenius dist={frob_dist:.4f}")
            print(f"[Calibrator] Stability: {'✓' if stable_after else '✗'}")
        
        report = CalibrationReport(
            cancer_type=cancer_type,
            n_samples=real_profiles.shape[0],
            r_squared_before=r2_before,
            r_squared_after=r2_after,
            rmse_before=rmse_before,
            rmse_after=rmse_after,
            per_metabolite_error_before=per_met_before,
            per_metabolite_error_after=per_met_after,
            entries_changed=entries_changed,
            max_entry_change=max_change,
            frobenius_distance=frob_dist,
            stable_before=stable_before,
            stable_after=stable_after,
        )
        
        return A_refined, report
    
    def validate(
        self,
        A: np.ndarray,
        real_profiles: np.ndarray,
    ) -> Dict[str, float]:
        """
        Validate a generator against real profiles.
        
        Returns: {metric_name: value}
        """
        rmse, r2, per_met = self.compute_profile_error(A, real_profiles)
        stable = self.is_stable(A)
        
        result = {
            'rmse': rmse,
            'r_squared': r2,
            'stable': float(stable),
        }
        for name, err in per_met.items():
            result[f'error_{name}'] = err
        
        return result
    
    def calibrate_all_cancers(
        self,
        generators: Dict[str, np.ndarray],
        profiles: Dict[str, np.ndarray],
        alpha: float = 0.3,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, CalibrationReport]]:
        """
        Calibrate all generators against their corresponding real profiles.
        
        Args:
            generators: {cancer_type: A_prior}
            profiles:   {cancer_type: (n_samples, 10) real profiles}
            alpha:      Max deviation fraction
        
        Returns:
            ({cancer_type: A_refined}, {cancer_type: CalibrationReport})
        """
        refined = {}
        reports = {}
        
        for ctype in generators:
            if ctype in profiles and profiles[ctype].shape[0] > 0:
                A_ref, report = self.refine_generator(
                    generators[ctype], profiles[ctype], ctype, alpha
                )
                refined[ctype] = A_ref
                reports[ctype] = report
            else:
                if self.verbose:
                    print(f"[Calibrator] {ctype}: No real data available, keeping prior")
                refined[ctype] = generators[ctype]
        
        return refined, reports
