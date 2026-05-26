"""
Identity Tensor Module — Project Confluence (Φ-Unification Engine)
===================================================================

Implements the Identity Tensor I(t) = (C(t), ∇C(t), M(t)) from the
Consciousness-Complexity Bridge theory paper.

Provides:
  1. Neural subspace projection of the full coupling tensor
  2. Memory kernel computation via exponential decay convolution
  3. Identity Tensor assembly and σ_min(I) evaluation
  4. Identity Threshold certification
  5. Substrate transition rate safety bounds

The Identity Threshold Conjecture:
    An individual A persists continuously from t₀ to t₁ iff:
        σ_min(I(t)) > ε_identity   ∀ t ∈ [t₀, t₁]

Mathematical Objects:
    I(t) = (C(t), ∇ₜC(t), M(t))
    M(t) = ∫ K(t-τ) C(τ) dτ     (exponentially weighted coupling history)
    K(t-τ) = exp(-(t-τ)/τ_memory) / τ_memory
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from models.coupling_tensor import CouplingTensorAnalyzer


class IdentityTensorAnalyzer:
    """
    Computes the Identity Tensor I(t) and issues identity preservation
    certificates for neural-scale coupled dynamical systems.

    The Identity Tensor augments the biological coupling tensor with
    temporal derivatives and memory kernel to capture not just the
    instantaneous coupling state but the dynamical trajectory and
    accumulated history that constitute personal identity.
    """

    # Default neural scale indices for the 5-scale model
    # (quantum k_0, cellular k_2, organismal k_4 → mapped to 4-scale indices)
    # In the 4-scale model: molecular=0, cellular=1, organism=2, tissue=3
    # Neural subspace: cellular (1) and organism (2) carry neural-relevant coupling
    DEFAULT_NEURAL_INDICES = [1, 2]

    # Default identity threshold (calibrated from clinical DOC data)
    DEFAULT_EPSILON_IDENTITY = 0.08

    # Default memory decay timescale (in time units matching the ODE system)
    DEFAULT_TAU_MEMORY = 50.0

    def __init__(self,
                 bac_analyzer: Optional[CouplingTensorAnalyzer] = None,
                 neural_indices: Optional[List[int]] = None,
                 epsilon_identity: float = DEFAULT_EPSILON_IDENTITY,
                 tau_memory: float = DEFAULT_TAU_MEMORY):
        """
        Parameters
        ----------
        bac_analyzer : CouplingTensorAnalyzer, optional
            The underlying BAC coupling tensor analyzer.
        neural_indices : list of int, optional
            Indices of the coupling tensor rows/columns that correspond
            to neural-relevant scales. Default: [1, 2] (cellular, organism).
        epsilon_identity : float
            Critical identity threshold. Below this, the individual's
            identity pattern is no longer coherently distinguishable.
        tau_memory : float
            Memory decay timescale. Controls how fast past coupling
            configurations lose influence on current identity.
        """
        self.bac = bac_analyzer or CouplingTensorAnalyzer()
        self.neural_idx = neural_indices or self._default_neural_indices()
        self.N_neural = len(self.neural_idx)
        self.epsilon_identity = epsilon_identity
        self.tau_memory = tau_memory

    def _default_neural_indices(self) -> List[int]:
        """Choose neural-relevant scale indices for 4-scale or 5-scale BAC tensors."""
        scale_names = getattr(self.bac, "scale_names", [])
        if 'cellular' in scale_names and 'organism' in scale_names:
            return [scale_names.index('cellular'), scale_names.index('organism')]
        return self.DEFAULT_NEURAL_INDICES

    # ═══════════════════════════════════════════════════════════════════════
    # 1. NEURAL SUBSPACE PROJECTION
    # ═══════════════════════════════════════════════════════════════════════

    def project_neural(self, C: np.ndarray) -> np.ndarray:
        """
        Project the full coupling tensor onto the neural subspace.

        C_neural = Π_neural · C · Π_neural^T

        Parameters
        ----------
        C : ndarray, shape (N, N)
            Full coupling tensor.

        Returns
        -------
        C_neural : ndarray, shape (N_neural, N_neural)
            Neural subspace coupling tensor.
        """
        neural_idx = self.neural_idx
        if C.shape[0] == 4 and neural_idx == [2, 3]:
            neural_idx = self.DEFAULT_NEURAL_INDICES
        return C[np.ix_(neural_idx, neural_idx)]

    def neural_integration(self, C: np.ndarray) -> float:
        """
        Compute the neural integration strength σ_min(C_neural).

        This is the BAC analogue of Tononi's Φ_IIT — the minimum
        irreducible integration across neural scales.

        Parameters
        ----------
        C : ndarray, shape (N, N)
            Full coupling tensor.

        Returns
        -------
        phi_neural : float
            Minimum singular value of the neural subspace tensor.
        """
        C_neural = self.project_neural(C)
        return float(np.linalg.svd(C_neural, compute_uv=False)[-1])

    def phi_bridge(self, C: np.ndarray) -> float:
        """
        Compute the Φ-Bridge value — the unified consciousness measure
        linking IIT's Φ and BAC's σ_min.

        Φ_bridge = σ_min(C_neural) · log₂(det(C_neural))

        Returns 0 if the neural subspace is singular (no consciousness).

        Parameters
        ----------
        C : ndarray, shape (N, N)
            Full coupling tensor.

        Returns
        -------
        phi_bridge : float
            Unified consciousness-complexity measure.
        """
        C_neural = self.project_neural(C)
        sigma_min = np.linalg.svd(C_neural, compute_uv=False)[-1]

        det_val = np.linalg.det(C_neural)
        if det_val <= 0 or sigma_min <= 0:
            return 0.0

        return float(sigma_min * np.log2(det_val))

    # ═══════════════════════════════════════════════════════════════════════
    # 2. MEMORY KERNEL COMPUTATION
    # ═══════════════════════════════════════════════════════════════════════

    def memory_kernel(self, C_history: np.ndarray,
                      dt: float) -> np.ndarray:
        """
        Compute the memory kernel M(t) from coupling tensor history.

        M(t) = ∫₀ᵗ K(t-τ) C(τ) dτ

        where K(t-τ) = exp(-(t-τ)/τ_memory) / τ_memory is the
        exponential decay kernel.

        This represents the accumulated influence of past coupling
        configurations on current identity — learned associations,
        personality traits, and autobiographical continuity.

        Parameters
        ----------
        C_history : ndarray, shape (N, N, T)
            Time series of coupling tensors C(τ) for τ ∈ [0, t].
        dt : float
            Time step between samples.

        Returns
        -------
        M : ndarray, shape (N, N)
            Memory kernel at the final time step.
        """
        N = C_history.shape[0]
        T = C_history.shape[2]
        M = np.zeros((N, N))

        # Current time is t = (T-1) * dt
        for t_idx in range(T):
            lag = (T - 1 - t_idx) * dt
            weight = np.exp(-lag / self.tau_memory) / self.tau_memory * dt
            M += weight * C_history[:, :, t_idx]

        return M

    def memory_kernel_trajectory(self, C_history: np.ndarray,
                                  dt: float) -> np.ndarray:
        """
        Compute the memory kernel M(t) at every time step along the trajectory.

        Parameters
        ----------
        C_history : ndarray, shape (N, N, T)
            Full coupling tensor time series.
        dt : float
            Time step.

        Returns
        -------
        M_series : ndarray, shape (N, N, T)
            Memory kernel at each time step.
        """
        N = C_history.shape[0]
        T = C_history.shape[2]
        M_series = np.zeros((N, N, T))

        # Efficient recursive computation:
        # M(t+dt) = M(t) * exp(-dt/τ) + C(t+dt) * dt/τ
        decay = np.exp(-dt / self.tau_memory)
        gain = dt / self.tau_memory

        M_current = np.zeros((N, N))
        for t_idx in range(T):
            M_current = M_current * decay + C_history[:, :, t_idx] * gain
            M_series[:, :, t_idx] = M_current

        return M_series

    # ═══════════════════════════════════════════════════════════════════════
    # 3. IDENTITY TENSOR ASSEMBLY
    # ═══════════════════════════════════════════════════════════════════════

    def compute_identity_tensor(self, C_history: np.ndarray,
                                 dt: float,
                                 t_idx: int = -1) -> Dict:
        """
        Assemble the full Identity Tensor I(t) = (C(t), ∇C(t), M(t)).

        Parameters
        ----------
        C_history : ndarray, shape (N, N, T)
            Full coupling tensor time series.
        dt : float
            Time step.
        t_idx : int
            Index of the time step to evaluate. Default: -1 (final).

        Returns
        -------
        identity_tensor : dict
            - 'C': Current coupling tensor C(t)
            - 'dC_dt': Temporal derivative ∇ₜC(t)
            - 'M': Memory kernel M(t)
            - 'C_neural': Neural subspace projection
            - 'M_neural': Neural subspace memory kernel
            - 'sigma_min_identity': σ_min of the augmented identity matrix
            - 'phi_bridge': Φ-Bridge consciousness measure
            - 'memory_integrity': ||M_neural||_F / ||M_neural_max||_F
        """
        T = C_history.shape[2]
        if t_idx < 0:
            t_idx = T + t_idx

        # 1. Current coupling tensor
        C_t = C_history[:, :, t_idx]

        # 2. Temporal derivative (central difference where possible)
        if t_idx == 0:
            dC_dt = (C_history[:, :, 1] - C_history[:, :, 0]) / dt
        elif t_idx == T - 1:
            dC_dt = (C_history[:, :, -1] - C_history[:, :, -2]) / dt
        else:
            dC_dt = (C_history[:, :, t_idx + 1] - C_history[:, :, t_idx - 1]) / (2 * dt)

        # 3. Memory kernel (from history up to t_idx)
        M = self.memory_kernel(C_history[:, :, :t_idx + 1], dt)

        # 4. Neural subspace projections
        C_neural = self.project_neural(C_t)
        M_neural = self.project_neural(M)
        dC_neural = self.project_neural(dC_dt)

        # 5. Augmented identity matrix for σ_min computation
        # We construct a block matrix that captures all three components:
        #   I_aug = [ C_neural    |  dC_neural·τ  ]
        #           [ M_neural    |  C_neural     ]
        # σ_min of this block captures the weakest link across instantaneous
        # state, trajectory, and memory simultaneously.
        tau_scale = self.tau_memory  # Scale dC to same units
        I_aug = np.block([
            [C_neural, dC_neural * tau_scale],
            [M_neural, C_neural]
        ])
        sigma_min_identity = float(np.linalg.svd(I_aug, compute_uv=False)[-1])

        # 6. Φ-Bridge value
        phi_bridge = self.phi_bridge(C_t)

        # 7. Memory integrity (how preserved is the memory relative to max)
        M_neural_norm = np.linalg.norm(M_neural, 'fro')
        C_neural_norm = np.linalg.norm(C_neural, 'fro')
        memory_integrity = M_neural_norm / (C_neural_norm + 1e-12)

        return {
            'C': C_t,
            'dC_dt': dC_dt,
            'M': M,
            'C_neural': C_neural,
            'M_neural': M_neural,
            'dC_neural': dC_neural,
            'sigma_min_identity': sigma_min_identity,
            'phi_bridge': phi_bridge,
            'memory_integrity': float(memory_integrity),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # 4. IDENTITY CERTIFICATION
    # ═══════════════════════════════════════════════════════════════════════

    def certify_identity(self, C_history: np.ndarray,
                          dt: float,
                          t_idx: int = -1) -> Dict:
        """
        Issue a binary Identity Preservation Certificate.

        Checks: σ_min(I(t)) > ε_identity ?
        If YES → IDENTITY PRESERVED
        If NO  → IDENTITY DISSOLVED

        Parameters
        ----------
        C_history : ndarray, shape (N, N, T)
            Full coupling tensor time series.
        dt : float
            Time step.
        t_idx : int
            Time step to certify.

        Returns
        -------
        certificate : dict
            - 'verdict': 'IDENTITY_PRESERVED' or 'IDENTITY_DISSOLVED'
            - 'sigma_min_identity': σ_min(I(t))
            - 'epsilon_identity': threshold used
            - 'margin': σ_min - ε (positive = preserved)
            - 'phi_bridge': consciousness measure
            - 'memory_integrity': memory kernel health
            - 'regime': 'coherent', 'degraded', 'critical', or 'dissolved'
            - 'biological_viability': V(t) from BAC
        """
        I_t = self.compute_identity_tensor(C_history, dt, t_idx)

        T = C_history.shape[2]
        if t_idx < 0:
            t_idx = T + t_idx

        sigma = I_t['sigma_min_identity']
        margin = sigma - self.epsilon_identity

        # Regime classification
        if sigma > 3 * self.epsilon_identity:
            regime = 'coherent'
        elif sigma > 1.5 * self.epsilon_identity:
            regime = 'degraded'
        elif sigma > self.epsilon_identity:
            regime = 'critical'
        else:
            regime = 'dissolved'

        verdict = 'IDENTITY_PRESERVED' if margin > 0 else 'IDENTITY_DISSOLVED'

        # Also compute biological viability at this step
        C_t = C_history[:, :, t_idx]
        # Use a default entropy rate (the caller should provide real ones)
        s_default = np.ones(C_t.shape[0]) * 0.1
        V_bio = self.bac.viability(C_t, s_default)

        return {
            'verdict': verdict,
            'sigma_min_identity': float(sigma),
            'epsilon_identity': float(self.epsilon_identity),
            'margin': float(margin),
            'phi_bridge': float(I_t['phi_bridge']),
            'memory_integrity': float(I_t['memory_integrity']),
            'regime': regime,
            'biological_viability': float(V_bio),
        }

    def certify_trajectory(self, C_history: np.ndarray,
                            dt: float) -> Dict:
        """
        Issue identity certificates along an entire trajectory.

        Parameters
        ----------
        C_history : ndarray, shape (N, N, T)
            Full coupling tensor time series.
        dt : float
            Time step.

        Returns
        -------
        result : dict
            - 'certificates': list of per-step certificates
            - 'global_verdict': 'IDENTITY_PRESERVED' if all steps pass
            - 'first_dissolution_step': index of first failure (or None)
            - 'min_sigma_identity': minimum σ_min(I) across trajectory
            - 'sigma_trajectory': array of σ_min(I(t)) values
            - 'phi_trajectory': array of Φ-Bridge values
            - 'memory_trajectory': array of memory integrity values
        """
        T = C_history.shape[2]
        certificates = []
        sigma_traj = []
        phi_traj = []
        memory_traj = []
        first_failure = None

        for t in range(T):
            cert = self.certify_identity(C_history, dt, t)
            certificates.append(cert)
            sigma_traj.append(cert['sigma_min_identity'])
            phi_traj.append(cert['phi_bridge'])
            memory_traj.append(cert['memory_integrity'])

            if cert['verdict'] == 'IDENTITY_DISSOLVED' and first_failure is None:
                first_failure = t

        sigmas = np.array(sigma_traj)
        min_sigma = float(np.min(sigmas))
        global_verdict = 'IDENTITY_PRESERVED' if first_failure is None else 'IDENTITY_DISSOLVED'

        return {
            'certificates': certificates,
            'global_verdict': global_verdict,
            'first_dissolution_step': first_failure,
            'min_sigma_identity': min_sigma,
            'sigma_trajectory': sigmas,
            'phi_trajectory': np.array(phi_traj),
            'memory_trajectory': np.array(memory_traj),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # 5. SUBSTRATE TRANSITION SAFETY
    # ═══════════════════════════════════════════════════════════════════════

    def max_safe_replacement_rate(self, C_history: np.ndarray,
                                   dt: float,
                                   t_idx: int = -1) -> float:
        """
        Compute the maximum safe substrate replacement rate.

        The rate of substrate replacement must satisfy:
            ||dI/dt||_replacement < (σ_min(I) - ε_identity) / Δt_step

        This returns the maximum allowable ||dI/dt|| perturbation that
        keeps the identity tensor above threshold during each transition.

        Parameters
        ----------
        C_history : ndarray, shape (N, N, T)
            Coupling tensor history.
        dt : float
            Time step.
        t_idx : int
            Time step to evaluate.

        Returns
        -------
        max_rate : float
            Maximum safe replacement rate (perturbation norm per time unit).
            Returns 0 if already below identity threshold.
        """
        I_t = self.compute_identity_tensor(C_history, dt, t_idx)
        sigma = I_t['sigma_min_identity']
        margin = sigma - self.epsilon_identity

        if margin <= 0:
            return 0.0

        # Safety factor: use only 80% of available margin
        return float(0.8 * margin / dt)

    # ═══════════════════════════════════════════════════════════════════════
    # 6. REPORT GENERATION
    # ═══════════════════════════════════════════════════════════════════════

    def generate_report(self, certificate: Dict) -> str:
        """Generate a human-readable identity preservation report."""
        lines = [
            "=" * 60,
            "  IDENTITY PRESERVATION CERTIFICATE",
            "  (Φ-Unification Framework)",
            "=" * 60,
            "",
            f"  VERDICT:            {certificate['verdict']}",
            f"  Regime:             {certificate['regime']}",
            f"  σ_min(I):           {certificate['sigma_min_identity']:.6f}",
            f"  ε_identity:         {certificate['epsilon_identity']:.6f}",
            f"  Margin:             {certificate['margin']:.6f}",
            f"  Φ-Bridge:           {certificate['phi_bridge']:.6f}",
            f"  Memory Integrity:   {certificate['memory_integrity']:.4f}",
            f"  Bio Viability V(t): {certificate['biological_viability']:.6f}",
            "",
        ]

        if certificate['verdict'] == 'IDENTITY_PRESERVED':
            lines.append("  ✓ Individual identity is coherently maintained.")
            lines.append(f"  ✓ Identity safety margin: {certificate['margin']:.6f}")
            if certificate['regime'] == 'coherent':
                lines.append("  ✓ Deep coherence — identity is strongly anchored.")
            elif certificate['regime'] == 'degraded':
                lines.append("  ⚠ Degraded regime — identity is maintained but weakening.")
            elif certificate['regime'] == 'critical':
                lines.append("  ⚠ CRITICAL — identity is near the dissolution threshold.")
        else:
            lines.append("  ✗ Individual identity has DISSOLVED.")
            lines.append("  ✗ The coupling pattern that constituted this individual")
            lines.append("    is no longer distinguishable from noise.")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
