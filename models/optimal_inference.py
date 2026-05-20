"""
Optimal Inference Observer — Project Confluence
===============================================

Implements the continuous-discrete Extended Kalman Filter (EKF) observer to
reconstruct the full 15D biological state vector z(t) and its corresponding
4x4 cross-scale coupling tensor C_ij(t) from sparse, noisy clinical biomarkers.

Governing equations between measurements:
    dẑ/dt = F(ẑ, u)
    dP/dt = J(ẑ) P + P J(ẑ)^T + Q

Governing update equations at discrete measurement times t_k:
    K = P H^T (H P H^T + R)^-1
    ẑ = ẑ + K (y - H ẑ)
    P = (I - K H) P
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from models.ode_system import ComplexAttractorODE
from models.coupling_tensor import CouplingTensorAnalyzer


class ExtendedKalmanFilterObserver:
    """
    Continuous-Discrete Extended Kalman Filter (EKF) observer for the 15D SAEM.
    """

    def __init__(self, ode_system: ComplexAttractorODE, 
                 Q_diagonal: Optional[np.ndarray] = None,
                 initial_covariance_scale: float = 0.1):
        """
        Parameters
        ----------
        ode_system : ComplexAttractorODE
            The 15D biological attractor model exposing `rhs(t, z)`.
        Q_diagonal : ndarray, shape (15,), optional
            Diagonal elements of the process noise covariance matrix Q.
        initial_covariance_scale : float
            Multiplier for the initial estimation uncertainty.
        """
        self.ode = ode_system
        self.dim = 15
        
        # State estimate initializing to healthy baseline
        self.z_hat = self.ode.healthy_initial_state()
        
        # Error covariance matrix P (15x15)
        self.P = np.eye(self.dim) * initial_covariance_scale
        
        # Process noise covariance Q (15x15)
        if Q_diagonal is not None:
            self.Q = np.diag(Q_diagonal)
        else:
            # Calibrated process noise (higher on cellular and metabolic scales)
            q_diag = np.ones(self.dim) * 0.01
            q_diag[0:5] *= 2.0   # Molecular fluctuations
            q_diag[5:10] *= 1.5  # Cellular metabolic fluctuations
            self.Q = np.diag(q_diag)

        self.analyzer = CouplingTensorAnalyzer()

    def predict(self, dt: float, t_current: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate the state estimate and error covariance forward in time (Prediction Step).
        Uses Euler-Maruyama equivalent deterministic flow.

        Parameters
        ----------
        dt : float
            Integration step size.
        t_current : float
            Current simulation time.
        """
        # 1. State propagation: dẑ/dt = F(ẑ, t)
        f_val = self.ode.rhs(t_current, self.z_hat)
        self.z_hat = np.clip(self.z_hat + f_val * dt, 0.0, 10.0)  # Bound to physiological limits
        
        # 2. Numerical Jacobian J(ẑ) at current state
        J = self._compute_jacobian(t_current, self.z_hat)
        
        # 3. Covariance propagation: dP/dt = J P + P J^T + Q
        dP = J @ self.P + self.P @ J.T + self.Q
        self.P += dP * dt
        
        # Symmeterise P to prevent numerical drift
        self.P = (self.P + self.P.T) / 2.0
        
        return self.z_hat.copy(), self.P.copy()

    def update(self, y_obs: np.ndarray, H: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Correct the state estimate using clinical observations (Update Step).

        Parameters
        ----------
        y_obs : ndarray, shape (M,)
            Sparse clinical observations.
        H : ndarray, shape (M, 15)
            Measurement matrix mapping 15D state to M-dimensional observations.
        R : ndarray, shape (M, M)
            Assay measurement noise covariance matrix.
        """
        # 1. Measurement residual: ỹ = y - H ẑ
        y_pred = H @ self.z_hat
        residual = y_obs - y_pred
        
        # 2. Residual covariance: S = H P H^T + R
        S = H @ self.P @ H.T + R
        
        # 3. Kalman Gain: K = P H^T S^-1
        try:
            S_inv = np.linalg.inv(S)
            K = self.P @ H.T @ S_inv
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if S is singular or ill-conditioned
            S_inv = np.linalg.pinv(S)
            K = self.P @ H.T @ S_inv
            
        # 4. Corrected State: ẑ = ẑ + K ỹ
        self.z_hat = np.clip(self.z_hat + K @ residual, 0.0, 10.0)
        
        # 5. Corrected Covariance: P = (I - K H) P
        I_KH = np.eye(self.dim) - K @ H
        self.P = I_KH @ self.P
        
        # Symmeterise P
        self.P = (self.P + self.P.T) / 2.0
        
        return self.z_hat.copy(), self.P.copy()

    def reconstruct_coupling_tensor(self, t_current: float = 0.0) -> np.ndarray:
        """
        Compute the estimated 4x4 coupling tensor Ĉ_ij from the EKF state estimate.
        """
        # Dummy trajectory of 1 step to reuse existing analyzer
        traj = self.z_hat.reshape(-1, 1)
        t_arr = np.array([t_current])
        C_series = self.analyzer.compute_from_jacobian(self.ode, traj, t_arr)
        return C_series[:, :, 0]

    def reconstruct_viability(self, entropy_rates: np.ndarray, t_current: float = 0.0) -> float:
        """
        Compute the estimated viability margin ∇(t) from the estimated coupling tensor.
        """
        C_est = self.reconstruct_coupling_tensor(t_current)
        return self.analyzer.viability(C_est, entropy_rates)

    def _compute_jacobian(self, t: float, z: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """Helper to calculate numerical Jacobian at the current state estimate."""
        J = np.zeros((self.dim, self.dim))
        for j in range(self.dim):
            z_plus = z.copy()
            z_minus = z.copy()
            z_plus[j] += h
            z_minus[j] -= h

            F_plus = self.ode.rhs(t, z_plus)
            F_minus = self.ode.rhs(t, z_minus)

            J[:, j] = (F_plus - F_minus) / (2.0 * h)
        return J


# ═══════════════════════════════════════════════════════════════════════
# PRE-CALIBRATED CLINICAL PANEL SELECTIONS
# ═══════════════════════════════════════════════════════════════════════

def get_clinical_measurement_matrix(selected_indices: List[int]) -> np.ndarray:
    """
    Constructs the selection matrix H mapping the 15D state vector to the
    biomarkers chosen in the clinical panel.

    Parameters
    ----------
    selected_indices : list of int
        Indices (0-14) of the variables measured in the clinical panel.
    """
    M = len(selected_indices)
    H = np.zeros((M, 15))
    for i, idx in enumerate(selected_indices):
        H[i, idx] = 1.0
    return H
