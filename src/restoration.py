"""
Restoration Computation Module
==============================

Compute the corrective δA needed to restore coherence.
This is the therapeutic intervention in generator space.

Key principle: Find minimal δA_correction such that
    A_healthy ≈ A_cancer + δA_correction
"""

import numpy as np
from scipy import linalg, optimize
from typing import Dict, Tuple, Optional, List


class RestorationComputer:
    """
    Compute corrective generator modifications to restore health.
    
    Given A_cancer and A_healthy (or target properties), compute
    the minimal intervention δA that restores coherent dynamics.
    """
    
    def __init__(self, sparsity_weight: float = 0.1):
        """
        Args:
            sparsity_weight: Regularization to prefer sparse corrections
                           (fewer targeted interventions)
        """
        self.sparsity_weight = sparsity_weight
        self.delta_A = None
        self.correction_targets = None
        
    def compute_direct_correction(
        self, 
        A_cancer: np.ndarray, 
        A_healthy: np.ndarray
    ) -> np.ndarray:
        """
        Direct correction: δA = A_healthy - A_cancer
        
        This is the "ideal" correction but may require
        interventions at every pathway.
        """
        self.delta_A = A_healthy - A_cancer
        return self.delta_A
    
    def compute_sparse_correction(
        self,
        A_cancer: np.ndarray,
        A_healthy: np.ndarray,
        max_interventions: int = 5
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Compute sparse correction targeting only top pathways.
        
        This is more clinically realistic: we can only intervene
        on a limited number of metabolic pathways.
        
        Args:
            A_cancer: Cancer generator
            A_healthy: Target healthy generator
            max_interventions: Maximum number of matrix entries to modify
            
        Returns:
            delta_A_sparse: Sparse correction matrix
            targets: List of (i, j) indices for intervention targets
        """
        # Full correction
        delta_full = A_healthy - A_cancer
        
        # Find largest magnitude corrections
        flat_indices = np.argsort(np.abs(delta_full.ravel()))[::-1]
        
        # Take top corrections
        n = A_cancer.shape[0]
        delta_sparse = np.zeros_like(delta_full)
        targets = []
        
        for idx in flat_indices[:max_interventions]:
            i, j = np.unravel_index(idx, delta_full.shape)
            delta_sparse[i, j] = delta_full[i, j]
            targets.append((i, j))
            
        self.delta_A = delta_sparse
        self.correction_targets = targets
        
        return delta_sparse, targets
    
    def compute_eigenvalue_correction(
        self,
        A_cancer: np.ndarray,
        target_real_bound: float = -0.1
    ) -> np.ndarray:
        """
        Correct eigenvalue spectrum to restore stability.
        
        Minimal intervention: shift eigenvalues to have
        real parts below target_real_bound.
        
        Args:
            A_cancer: Cancer generator
            target_real_bound: Maximum allowed real part (negative = stable)
            
        Returns:
            delta_A: Correction to stabilize system
        """
        eigenvalues, V = linalg.eig(A_cancer)
        
        # Find eigenvalues that need correction
        shifts = np.zeros_like(eigenvalues)
        for i, eig in enumerate(eigenvalues):
            if eig.real > target_real_bound:
                # Need to shift this eigenvalue
                shifts[i] = target_real_bound - eig.real
                
        # Construct correction in eigenspace
        # A_corrected = V @ diag(eigenvalues + shifts) @ V^(-1)
        corrected_eigenvalues = eigenvalues + shifts
        
        try:
            V_inv = linalg.inv(V)
            A_corrected = V @ np.diag(corrected_eigenvalues) @ V_inv
            self.delta_A = A_corrected.real - A_cancer
        except linalg.LinAlgError:
            # If V is singular, fall back to simple diagonal shift
            max_shift = np.max(shifts.real)
            self.delta_A = -np.eye(A_cancer.shape[0]) * max_shift
            
        return self.delta_A
    
    def compute_targeted_correction(
        self,
        A_cancer: np.ndarray,
        target_pathways: List[Tuple[int, int]],
        target_strengths: List[float]
    ) -> np.ndarray:
        """
        Apply correction to specific pathways.
        
        Useful for testing known intervention targets.
        
        Args:
            A_cancer: Cancer generator
            target_pathways: List of (source, target) metabolite indices
            target_strengths: Desired correction strength for each pathway
            
        Returns:
            delta_A: Targeted correction matrix
        """
        delta_A = np.zeros_like(A_cancer)
        
        for (i, j), strength in zip(target_pathways, target_strengths):
            delta_A[i, j] = strength
            
        self.delta_A = delta_A
        self.correction_targets = target_pathways
        
        return delta_A
    
    def analyze_correction_effect(
        self,
        A_cancer: np.ndarray,
        delta_A: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Analyze expected effect of applying correction.
        
        Returns:
            Dict with eigenvalue changes, stability improvement, etc.
        """
        if delta_A is None:
            delta_A = self.delta_A
            
        if delta_A is None:
            raise ValueError("No correction computed yet")
            
        A_corrected = A_cancer + delta_A
        
        # Original properties
        eig_original = linalg.eigvals(A_cancer)
        eig_corrected = linalg.eigvals(A_corrected)
        
        # Stability improvement
        stability_before = np.max(eig_original.real)
        stability_after = np.max(eig_corrected.real)
        
        # Contraction rate improvement
        contraction_before = -np.mean(eig_original.real)
        contraction_after = -np.mean(eig_corrected.real)
        
        # Correction magnitude
        correction_magnitude = np.linalg.norm(delta_A, 'fro')
        
        # Sparsity of correction
        sparsity = 1.0 - (np.sum(np.abs(delta_A) > 1e-10) / delta_A.size)
        
        return {
            'stability_before': stability_before,
            'stability_after': stability_after,
            'stability_improvement': stability_before - stability_after,
            'contraction_before': contraction_before,
            'contraction_after': contraction_after,
            'contraction_improvement': contraction_after - contraction_before,
            'correction_magnitude': correction_magnitude,
            'sparsity': sparsity,
            'is_stabilizing': stability_after < 0,
            'eigenvalues_before': eig_original,
            'eigenvalues_after': eig_corrected
        }
    
    def simulate_restoration(
        self,
        A_cancer: np.ndarray,
        initial_state: np.ndarray,
        time_points: np.ndarray,
        correction_time: float,
        delta_A: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate dynamics before and after applying correction.
        
        Args:
            A_cancer: Cancer generator
            initial_state: Starting metabolite concentrations
            time_points: Time array (must span before and after correction_time)
            correction_time: When to apply the correction
            delta_A: Correction to apply (uses self.delta_A if None)
            
        Returns:
            X_before: Trajectory before correction
            X_after: Trajectory after correction
        """
        if delta_A is None:
            delta_A = self.delta_A
            
        A_corrected = A_cancer + delta_A
        
        # Split time points
        t_before = time_points[time_points <= correction_time]
        t_after = time_points[time_points > correction_time]
        
        # Simulate before correction
        n_before = len(t_before)
        X_before = np.zeros((n_before, len(initial_state)))
        X_before[0] = initial_state
        
        for i in range(1, n_before):
            dt = t_before[i] - t_before[0]
            X_before[i] = linalg.expm(A_cancer * dt) @ initial_state
            
        # State at correction time
        state_at_correction = X_before[-1]
        
        # Simulate after correction (with corrected generator)
        n_after = len(t_after)
        X_after = np.zeros((n_after, len(initial_state)))
        
        for i in range(n_after):
            dt = t_after[i] - correction_time
            X_after[i] = linalg.expm(A_corrected * dt) @ state_at_correction
            
        return X_before, X_after
    
    def get_correction_report(self) -> str:
        """Generate human-readable correction report."""
        if self.delta_A is None:
            return "No correction computed yet."
            
        report = []
        report.append("=" * 50)
        report.append("RESTORATION CORRECTION REPORT")
        report.append("=" * 50)
        
        report.append(f"\nCorrection magnitude: {np.linalg.norm(self.delta_A, 'fro'):.4f}")
        
        # Non-zero corrections
        nonzero = np.argwhere(np.abs(self.delta_A) > 1e-10)
        report.append(f"Number of pathway corrections: {len(nonzero)}")
        
        if self.correction_targets:
            report.append("\nTop correction targets:")
            for i, j in self.correction_targets[:5]:
                report.append(f"  Pathway ({i} → {j}): δ = {self.delta_A[i, j]:.4f}")
                
        report.append("\n" + "=" * 50)
        
        return "\n".join(report)
