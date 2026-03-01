"""
Generator Matrix Extraction (HGI Pipeline)
==========================================

Extract the generator matrix A from metabolomics time-series data.
Based on Remisov's understanding: the generator governs system dynamics.

dX/dt = AX + noise  →  A = argmin ||dX/dt - AX||² + λ||A||
"""

import numpy as np
from scipy import linalg
from sklearn.linear_model import Ridge
from typing import Tuple, Optional


class GeneratorExtractor:
    """
    Extract generator matrix A from time-series metabolomics data.
    
    The generator A captures the fundamental dynamics of the metabolic system.
    In healthy tissue: A produces coherent, stable oscillations.
    In cancer: A shows coherence breakdown and pathological attractors.
    """
    
    def __init__(self, regularization: float = 0.01):
        """
        Initialize extractor.
        
        Args:
            regularization: Ridge regularization parameter (λ)
                           Higher values = more stable but potentially biased
        """
        self.regularization = regularization
        self.A = None
        self.confidence = None
        
    def extract(self, X: np.ndarray, time_points: np.ndarray) -> np.ndarray:
        """
        Extract generator matrix from time-series data.
        
        Args:
            X: Metabolite concentrations, shape (n_timepoints, n_metabolites)
            time_points: Time values for each observation
            
        Returns:
            A: Generator matrix, shape (n_metabolites, n_metabolites)
        """
        n_times, n_metabolites = X.shape
        
        # Compute numerical derivatives: dX/dt
        dX = self._compute_derivatives(X, time_points)
        
        # Solve for A: dX/dt ≈ AX
        # We have dX[t] = A @ X[t] for each t
        # Stack and solve via regularized least squares
        
        # Use data points excluding endpoints (derivative estimation)
        X_mid = X[1:-1, :]  # States at derivative points
        dX_mid = dX         # Derivatives
        
        # Solve column by column for A^T
        # dX[:, j] = X @ A[:, j] for each metabolite j
        A = np.zeros((n_metabolites, n_metabolites))
        
        ridge = Ridge(alpha=self.regularization, fit_intercept=False)
        
        for j in range(n_metabolites):
            ridge.fit(X_mid, dX_mid[:, j])
            A[:, j] = ridge.coef_
        
        self.A = A.T  # Transpose to get correct orientation
        self._compute_confidence(X_mid, dX_mid)
        
        return self.A
    
    def _compute_derivatives(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Compute numerical derivatives using central differences.
        
        Returns derivatives at interior points (excluding first and last).
        """
        n_times = X.shape[0]
        dX = np.zeros((n_times - 2, X.shape[1]))
        
        for i in range(1, n_times - 1):
            dt = t[i + 1] - t[i - 1]
            dX[i - 1] = (X[i + 1] - X[i - 1]) / dt
            
        return dX
    
    def _compute_confidence(self, X: np.ndarray, dX: np.ndarray):
        """
        Estimate confidence in extracted generator via residual analysis.
        """
        if self.A is None:
            return
            
        # Predicted derivatives
        dX_pred = X @ self.A.T
        
        # Residual variance
        residual = dX - dX_pred
        mse = np.mean(residual ** 2)
        
        # Signal variance
        signal_var = np.var(dX)
        
        # R² as confidence metric
        self.confidence = 1 - (mse / (signal_var + 1e-10))
        
    def get_eigendecomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get eigenvalues and eigenvectors of the generator.
        
        Returns:
            eigenvalues: Complex eigenvalues (λ)
            eigenvectors: Corresponding eigenvectors (V)
            
        Interpretation:
            - Real(λ) < 0: Stable mode (healthy)
            - Real(λ) > 0: Unstable mode (pathological)
            - Imag(λ) ≠ 0: Oscillatory dynamics
        """
        if self.A is None:
            raise ValueError("Must call extract() first")
            
        eigenvalues, eigenvectors = linalg.eig(self.A)
        
        # Sort by real part (most stable first)
        idx = np.argsort(eigenvalues.real)
        
        return eigenvalues[idx], eigenvectors[:, idx]
    
    def assess_stability(self) -> dict:
        """
        Assess system stability from generator eigenvalues.
        
        Returns dict with:
            - stable: bool, True if all eigenvalues have negative real parts
            - max_real: float, largest real part (closer to 0 = less stable)
            - oscillatory_modes: int, count of complex eigenvalue pairs
        """
        eigenvalues, _ = self.get_eigendecomposition()
        
        real_parts = eigenvalues.real
        imag_parts = eigenvalues.imag
        
        return {
            'stable': np.all(real_parts < 0),
            'max_real': np.max(real_parts),
            'min_real': np.min(real_parts),
            'oscillatory_modes': np.sum(np.abs(imag_parts) > 1e-10) // 2,
            'dominant_frequency': np.max(np.abs(imag_parts)) / (2 * np.pi)
        }


def generate_synthetic_system(
    n_metabolites: int = 10,
    stability: float = -0.5,
    coupling: float = 0.3,
    noise_level: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic generator matrix for testing.
    
    Args:
        n_metabolites: Number of metabolites
        stability: Base diagonal value (negative = stable)
        coupling: Off-diagonal coupling strength
        noise_level: Random perturbation level
        seed: Random seed for reproducibility
        
    Returns:
        A_healthy: Generator for healthy system
        A_cancer: Generator with coherence loss (cancer-like)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Base healthy generator: stable with coupled oscillations
    A_healthy = np.eye(n_metabolites) * stability
    
    # Add coupling (off-diagonal)
    coupling_matrix = np.random.randn(n_metabolites, n_metabolites) * coupling
    np.fill_diagonal(coupling_matrix, 0)
    A_healthy += coupling_matrix
    
    # Ensure stability
    eigenvalues = linalg.eigvals(A_healthy)
    max_real = np.max(eigenvalues.real)
    if max_real > 0:
        A_healthy -= np.eye(n_metabolites) * (max_real + 0.1)
    
    # Cancer generator: disrupt coherence
    A_cancer = A_healthy.copy()
    
    # 1. Destabilize some modes (shift eigenvalues toward positive)
    A_cancer += np.eye(n_metabolites) * 0.3
    
    # 2. Break coupling symmetry (coherence loss)
    disruption = np.random.randn(n_metabolites, n_metabolites) * noise_level
    A_cancer += disruption
    
    # 3. Enhance some pathological pathways
    pathological_indices = np.random.choice(n_metabolites, size=3, replace=False)
    for i in pathological_indices:
        A_cancer[i, :] *= 1.5  # Increased activity
        
    return A_healthy, A_cancer


def simulate_dynamics(
    A: np.ndarray,
    initial_state: np.ndarray,
    time_points: np.ndarray,
    noise_std: float = 0.01
) -> np.ndarray:
    """
    Simulate metabolic dynamics under generator A.
    
    Uses matrix exponential for exact solution with added noise.
    
    Args:
        A: Generator matrix
        initial_state: Initial metabolite concentrations
        time_points: Times at which to sample
        noise_std: Measurement noise standard deviation
        
    Returns:
        X: Simulated concentrations, shape (n_times, n_metabolites)
    """
    n_times = len(time_points)
    n_metabolites = len(initial_state)
    
    X = np.zeros((n_times, n_metabolites))
    X[0] = initial_state
    
    for i in range(1, n_times):
        dt = time_points[i] - time_points[0]
        # Exact solution: X(t) = exp(At) @ X(0)
        X[i] = linalg.expm(A * dt) @ initial_state
        # Add measurement noise
        X[i] += np.random.randn(n_metabolites) * noise_std
        
    return X
