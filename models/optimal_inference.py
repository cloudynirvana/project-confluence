"""
Optimal Inference Observer - Project Confluence
===============================================

Continuous-discrete Extended Kalman Filter (EKF) observer for reconstructing:
  1. the biological state vector z(t),
  2. the cross-scale coupling tensor C_ij(t),
  3. the hidden neural memory kernel M(t).

The augmented EKF state is:
    x_hat = [z_hat, vec(M_neural)]

Memory dynamics:
    dM/dt = gamma * (C_neural - M)

Clinical neural observation channels:
    - DMN coherence: diagonal readout of M_neural
    - EEG PCI: broad integrated readout of M_neural
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from models.ode_system import ComplexAttractorODE
from models.coupling_tensor import CouplingTensorAnalyzer
from models.identity_tensor import IdentityTensorAnalyzer


class ExtendedKalmanFilterObserver:
    """
    Continuous-discrete EKF observer for biological state plus hidden memory.
    """

    def __init__(self,
                 ode_system: ComplexAttractorODE,
                 Q_diagonal: Optional[np.ndarray] = None,
                 initial_covariance_scale: float = 0.1,
                 memory_gamma: float = 0.02):
        """
        Parameters
        ----------
        ode_system : ComplexAttractorODE
            Biological attractor model exposing rhs(t, z).
        Q_diagonal : ndarray, optional
            Diagonal process-noise values. Can be length bio_dim or total_dim.
        initial_covariance_scale : float
            Initial uncertainty multiplier.
        memory_gamma : float
            Relaxation rate in dM/dt = gamma * (C_neural - M).
        """
        self.ode = ode_system
        self.bio_dim = getattr(self.ode, "DIM", len(self.ode.healthy_initial_state()))
        self.dim = self.bio_dim  # Backward-compatible biological dimension alias.
        self.analyzer = CouplingTensorAnalyzer()
        self.identity_analyzer = IdentityTensorAnalyzer(self.analyzer)
        self.memory_shape = (
            self.identity_analyzer.N_neural,
            self.identity_analyzer.N_neural,
        )
        self.memory_dim = int(np.prod(self.memory_shape))
        self.total_dim = self.bio_dim + self.memory_dim
        self.memory_gamma = memory_gamma

        self.z_hat = self.ode.healthy_initial_state()
        self.M_hat = self.identity_analyzer.project_neural(
            self.reconstruct_coupling_tensor_from_z(self.z_hat, 0.0)
        )
        self.x_hat = self._pack_state()

        self.P = np.eye(self.total_dim) * initial_covariance_scale

        if Q_diagonal is not None:
            q_diag = np.asarray(Q_diagonal, dtype=float)
            if len(q_diag) == self.bio_dim:
                q_diag = np.concatenate([q_diag, np.ones(self.memory_dim) * 0.005])
            if len(q_diag) != self.total_dim:
                raise ValueError(
                    f"Q_diagonal must have length {self.bio_dim} or {self.total_dim}"
                )
        else:
            q_diag = np.ones(self.total_dim) * 0.01
            q_diag[0:5] *= 2.0
            q_diag[5:10] *= 1.5
            q_diag[self.bio_dim:] *= 0.5
        self.Q = np.diag(q_diag)

    def predict(self, dt: float, t_current: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate the biological state, memory kernel, and covariance.
        """
        f_val = self.ode.rhs(t_current, self.z_hat)
        self.z_hat = np.clip(self.z_hat + f_val * dt, 0.0, 10.0)

        C_neural = self.identity_analyzer.project_neural(
            self.reconstruct_coupling_tensor_from_z(self.z_hat, t_current)
        )
        self.M_hat = self.M_hat + self.memory_gamma * (C_neural - self.M_hat) * dt
        self.M_hat = np.clip(self.M_hat, 0.0, 1.0)
        self.x_hat = self._pack_state()

        J = self._compute_augmented_jacobian(t_current, self.z_hat)
        dP = J @ self.P + self.P @ J.T + self.Q
        self.P += dP * dt
        self.P = (self.P + self.P.T) / 2.0

        return self.z_hat.copy(), self.P.copy()

    def update(self, y_obs: np.ndarray, H: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Correct the augmented state estimate using clinical observations.

        H may map either the biological state only (bio_dim columns) or the
        full augmented state (total_dim columns).
        """
        y_obs = np.asarray(y_obs, dtype=float)
        H_aug = self._ensure_augmented_measurement_matrix(H)
        R = np.asarray(R, dtype=float)

        y_pred = H_aug @ self.x_hat
        residual = y_obs - y_pred
        S = H_aug @ self.P @ H_aug.T + R

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = self.P @ H_aug.T @ S_inv

        self.x_hat = self.x_hat + K @ residual
        self.z_hat = np.clip(self.x_hat[:self.bio_dim], 0.0, 10.0)
        self.M_hat = self.x_hat[self.bio_dim:].reshape(self.memory_shape)
        self.M_hat = np.clip(self.M_hat, 0.0, 1.0)
        self.x_hat = self._pack_state()

        I_KH = np.eye(self.total_dim) - K @ H_aug
        self.P = I_KH @ self.P
        self.P = (self.P + self.P.T) / 2.0

        return self.z_hat.copy(), self.P.copy()

    def update_neuroidentity_channels(self,
                                      dmn_coherence: Optional[float] = None,
                                      eeg_pci: Optional[float] = None,
                                      R: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update from direct DMN coherence and EEG PCI clinical channels.

        Both channels are modeled as normalized values in [0, 1].
        """
        rows = []
        values = []
        if dmn_coherence is not None:
            rows.append(self._dmn_coherence_row())
            values.append(float(dmn_coherence))
        if eeg_pci is not None:
            rows.append(self._eeg_pci_row())
            values.append(float(eeg_pci))
        if not rows:
            return self.z_hat.copy(), self.P.copy()

        H = np.vstack(rows)
        y_obs = np.array(values)
        if R is None:
            R = np.eye(len(values)) * 0.03
        return self.update(y_obs, H, R)

    def reconstruct_coupling_tensor(self, t_current: float = 0.0) -> np.ndarray:
        """Compute the estimated coupling tensor C_hat from z_hat."""
        return self.reconstruct_coupling_tensor_from_z(self.z_hat, t_current)

    def reconstruct_coupling_tensor_from_z(self, z: np.ndarray,
                                           t_current: float = 0.0) -> np.ndarray:
        """Compute a coupling tensor from an arbitrary biological state."""
        traj = z.reshape(-1, 1)
        t_arr = np.array([t_current])
        C_series = self.analyzer.compute_from_jacobian(self.ode, traj, t_arr)
        return C_series[:, :, 0]

    def reconstruct_viability(self, entropy_rates: np.ndarray,
                              t_current: float = 0.0) -> float:
        """Compute the estimated viability margin from the coupling tensor."""
        C_est = self.reconstruct_coupling_tensor(t_current)
        return self.analyzer.viability(C_est, entropy_rates)

    def reconstruct_memory_kernel(self) -> np.ndarray:
        """Return the estimated hidden neural memory kernel M_hat."""
        return self.M_hat.copy()

    def reconstruct_memory_covariance(self) -> np.ndarray:
        """Return the covariance block over vec(M_hat)."""
        return self.P[self.bio_dim:, self.bio_dim:].copy()

    def identity_confidence_margin(self) -> Dict:
        """Return a compact confidence summary for identity survival estimates."""
        sigma_min_memory = float(np.linalg.svd(self.M_hat, compute_uv=False)[-1])
        covariance_trace = float(np.trace(self.reconstruct_memory_covariance()))
        return {
            "sigma_min_memory": sigma_min_memory,
            "memory_covariance_trace": covariance_trace,
            "confidence": float(1.0 / (1.0 + covariance_trace)),
        }

    def _pack_state(self) -> np.ndarray:
        return np.concatenate([self.z_hat, self.M_hat.reshape(-1)])

    def _compute_jacobian(self, t: float, z: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """Numerical biological-state Jacobian."""
        J = np.zeros((self.bio_dim, self.bio_dim))
        for j in range(self.bio_dim):
            z_plus = z.copy()
            z_minus = z.copy()
            z_plus[j] += h
            z_minus[j] -= h
            F_plus = self.ode.rhs(t, z_plus)
            F_minus = self.ode.rhs(t, z_minus)
            J[:, j] = (F_plus - F_minus) / (2.0 * h)
        return J

    def _compute_augmented_jacobian(self, t: float, z: np.ndarray) -> np.ndarray:
        """
        Approximate augmented Jacobian for [z, vec(M)].

        The M dependence on z through C_neural is intentionally treated as a
        process-noise source here; the stabilizing memory self-dynamics are
        represented directly by -gamma I.
        """
        J = np.zeros((self.total_dim, self.total_dim))
        J[:self.bio_dim, :self.bio_dim] = self._compute_jacobian(t, z)
        J[self.bio_dim:, self.bio_dim:] = -self.memory_gamma * np.eye(self.memory_dim)
        return J

    def _ensure_augmented_measurement_matrix(self, H: np.ndarray) -> np.ndarray:
        H = np.asarray(H, dtype=float)
        if H.shape[1] == self.total_dim:
            return H
        if H.shape[1] != self.bio_dim:
            raise ValueError(
                f"H has {H.shape[1]} columns; expected {self.bio_dim} or {self.total_dim}"
            )
        H_aug = np.zeros((H.shape[0], self.total_dim))
        H_aug[:, :self.bio_dim] = H
        return H_aug

    def _dmn_coherence_row(self) -> np.ndarray:
        row = np.zeros(self.total_dim)
        diag_indices = np.diag_indices(self.memory_shape[0])
        flat_diag = np.ravel_multi_index(diag_indices, self.memory_shape)
        row[self.bio_dim + flat_diag] = 1.0 / len(flat_diag)
        return row

    def _eeg_pci_row(self) -> np.ndarray:
        row = np.zeros(self.total_dim)
        row[self.bio_dim:] = 1.0 / self.memory_dim
        return row


def get_clinical_measurement_matrix(selected_indices: List[int], dim: int = 16) -> np.ndarray:
    """
    Construct a selection matrix H mapping state entries to biomarkers.
    """
    M = len(selected_indices)
    H = np.zeros((M, dim))
    for i, idx in enumerate(selected_indices):
        H[i, idx] = 1.0
    return H


def get_neuroidentity_measurement_matrix(observer: ExtendedKalmanFilterObserver,
                                        include_dmn: bool = True,
                                        include_pci: bool = True) -> np.ndarray:
    """Build direct DMN/PCI observation rows for an augmented EKF observer."""
    rows = []
    if include_dmn:
        rows.append(observer._dmn_coherence_row())
    if include_pci:
        rows.append(observer._eeg_pci_row())
    return np.vstack(rows) if rows else np.zeros((0, observer.total_dim))
