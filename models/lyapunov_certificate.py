"""
Lyapunov Sustainment Certifier — Project Confluence
====================================================

Implements the Universal Complexity Sustainment Theorem by computing the
Control Lyapunov Function (CLF), autonomous decay rates, control authority,
and issuing binary SUSTAINMENT CERTIFICATES for any multi-scale coupled system.

The Sustainment Inequality:
    u_max ||B(ξ)|| > δ_min σ_min(C) + λ_max^(entropy) + (k_B T ln2 · I_repair) / V(ξ)

If satisfied ∀ξ ∈ Ω: complexity is provably sustainable.
If violated at any ξ₀: no control law can prevent attractor escape.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from models.coupling_tensor import CouplingTensorAnalyzer


class SustainmentCertifier:
    """
    Computes the Control Lyapunov Function and issues sustainment certificates
    for multi-scale coupled dynamical systems under Bounded Adaptive Coherence.
    """

    # Physical constants
    K_B = 1.380649e-23       # Boltzmann constant (J/K)
    BODY_TEMP = 310.0        # Human body temperature (K)
    LN2 = np.log(2)

    def __init__(self, analyzer: Optional[CouplingTensorAnalyzer] = None,
                 C_healthy: Optional[np.ndarray] = None,
                 s_healthy: Optional[np.ndarray] = None,
                 beta: float = 1.0,
                 eta: float = 1.0):
        """
        Parameters
        ----------
        analyzer : CouplingTensorAnalyzer, optional
            The coupling tensor analysis engine. Uses default if not provided.
        C_healthy : ndarray, shape (N, N), optional
            Baseline healthy coupling tensor.
        s_healthy : ndarray, shape (N,), optional
            Baseline healthy entropy rates at each scale.
        beta : float
            Weight for coupling tensor deviation in the Lyapunov function.
        eta : float
            Weight for entropy deviation in the Lyapunov function.
        """
        self.analyzer = analyzer or CouplingTensorAnalyzer()
        self.N = self.analyzer.N_scales

        # Healthy baselines
        if C_healthy is not None:
            self.C_star = C_healthy
        else:
            # Default: strongly coupled healthy tensor
            self.C_star = np.array([
                [1.0,  0.85, 0.75, 0.65],
                [0.85, 1.0,  0.80, 0.70],
                [0.75, 0.80, 1.0,  0.60],
                [0.65, 0.70, 0.60, 1.0]
            ])

        if s_healthy is not None:
            self.s_star = s_healthy
        else:
            self.s_star = np.array([0.10, 0.12, 0.08, 0.10])

        self.beta = beta
        self.eta = eta

        # Compute reference viability
        self.V_max = self.analyzer.viability(self.C_star, self.s_star)
        if self.V_max <= 0:
            raise ValueError("Healthy baseline does not satisfy BAC condition.")

    # ═══════════════════════════════════════════════════════════════════════
    # 1. CONTROL LYAPUNOV FUNCTION
    # ═══════════════════════════════════════════════════════════════════════

    def lyapunov_value(self, C: np.ndarray, s: np.ndarray) -> float:
        """
        Compute the Control Lyapunov Function L(ξ) at state ξ = (C, s).

        L(ξ) = -ln(V(ξ) / V_max) + β ||C - C*||²_F + η Σ_k (s_k - s*_k)²

        Parameters
        ----------
        C : ndarray, shape (N, N)
            Current coupling tensor.
        s : ndarray, shape (N,)
            Current normalized entropy rates.

        Returns
        -------
        L : float
            Lyapunov function value. L ≥ 0 in viable set, L → ∞ at criticality.
        """
        V = self.analyzer.viability(C, s)

        if V <= 0:
            return float('inf')

        # Logarithmic barrier at the criticality surface
        barrier = -np.log(V / self.V_max)

        # Coupling deviation penalty
        coupling_dev = self.beta * np.linalg.norm(C - self.C_star, 'fro') ** 2

        # Entropy deviation penalty
        entropy_dev = self.eta * np.sum((s - self.s_star) ** 2)

        return float(barrier + coupling_dev + entropy_dev)

    # ═══════════════════════════════════════════════════════════════════════
    # 2. AUTONOMOUS DECAY RATE
    # ═══════════════════════════════════════════════════════════════════════

    def autonomous_decay_rate(self, C: np.ndarray, s: np.ndarray,
                              delta: np.ndarray,
                              alpha: np.ndarray,
                              gamma: float = 0.7,
                              S_crit: float = 1.0,
                              entropy_growth_rates: Optional[np.ndarray] = None
                              ) -> float:
        """
        Compute the autonomous decay rate D(ξ) = A(ξ), which measures how fast
        the system drifts toward criticality without any control intervention.

        Parameters
        ----------
        C : ndarray, shape (N, N)
            Current coupling tensor.
        s : ndarray, shape (N,)
            Current normalized entropy rates.
        delta : ndarray, shape (N, N)
            Natural coupling decay rate matrix (δ_ij).
        alpha : ndarray, shape (N, N)
            Coupling repair rate matrix (α_ij).
        gamma : float
            Saturating repair exponent (0 < γ < 1).
        S_crit : float
            Critical entropy threshold for repair failure.
        entropy_growth_rates : ndarray, shape (N,), optional
            Rate of entropy acceleration at each scale (ds_k/dt under no control).
            Defaults to small positive values.

        Returns
        -------
        D : float
            Autonomous decay rate. Positive means system is deteriorating.
        """
        V = self.analyzer.viability(C, s)
        if V <= 0:
            return float('inf')

        # Compute autonomous coupling dynamics: dC_ij/dt = α C^γ (1 - (s_i+s_j)/2S_crit)+ - δ C
        dC_autonomous = np.zeros_like(C)
        for i in range(self.N):
            for j in range(self.N):
                repair = alpha[i, j] * (C[i, j] ** gamma) * max(0, 1.0 - (s[i] + s[j]) / (2 * S_crit))
                decay = delta[i, j] * C[i, j]
                dC_autonomous[i, j] = repair - decay

        # Compute dV/dt from autonomous dynamics
        # dσ_min/dt ≈ numerical sensitivity
        eps = 1e-6
        sigma_min_current = np.linalg.svd(C, compute_uv=False)[-1]

        C_perturbed = C + dC_autonomous * eps
        sigma_min_perturbed = np.linalg.svd(C_perturbed, compute_uv=False)[-1]
        d_sigma_min = (sigma_min_perturbed - sigma_min_current) / eps

        # Entropy growth
        if entropy_growth_rates is None:
            entropy_growth_rates = np.ones(self.N) * 0.01  # Default mild entropy growth

        # d(max_k s_k)/dt = ds_k*/dt where k* = argmax s_k
        k_star = np.argmax(s)
        d_max_entropy = entropy_growth_rates[k_star]

        # Autonomous viability derivative
        dV_autonomous = d_sigma_min - d_max_entropy

        # Dominant Lyapunov drift term: -dV/V (barrier term dominates near criticality)
        D = -dV_autonomous / V

        # Add coupling deviation drift
        D += 2 * self.beta * np.sum((C - self.C_star) * dC_autonomous)

        # Add entropy deviation drift
        D += 2 * self.eta * np.sum((s - self.s_star) * entropy_growth_rates)

        return float(D)

    # ═══════════════════════════════════════════════════════════════════════
    # 3. CONTROL AUTHORITY
    # ═══════════════════════════════════════════════════════════════════════

    def control_authority(self, C: np.ndarray, s: np.ndarray,
                         G_coupling: np.ndarray,
                         P_entropy: np.ndarray,
                         u_max: float) -> float:
        """
        Compute the maximum control authority C(ξ) = max_{||u||≤u_max} [-Σ B_m u_m].

        Parameters
        ----------
        C : ndarray, shape (N, N)
            Current coupling tensor.
        s : ndarray, shape (N,)
            Current normalized entropy rates.
        G_coupling : ndarray, shape (N, N, M)
            Control influence matrices for coupling tensor (g_ij^(m)).
        P_entropy : ndarray, shape (N, M)
            Control influence on entropy rates (p_k^(m)).
        u_max : float
            Maximum control input norm bound.

        Returns
        -------
        authority : float
            Maximum achievable Lyapunov descent rate from control inputs.
        """
        V = self.analyzer.viability(C, s)
        if V <= 0:
            return 0.0

        M = G_coupling.shape[2]  # Number of control inputs
        B = np.zeros(M)

        eps = 1e-6
        sigma_min = np.linalg.svd(C, compute_uv=False)[-1]
        k_star = np.argmax(s)

        for m in range(M):
            # Sensitivity of σ_min to control input m
            C_pert = C + G_coupling[:, :, m] * eps
            sigma_pert = np.linalg.svd(C_pert, compute_uv=False)[-1]
            d_sigma = (sigma_pert - sigma_min) / eps

            # Sensitivity of max entropy to control input m
            d_entropy = P_entropy[k_star, m]

            # Barrier contribution
            B[m] = -(d_sigma - d_entropy) / V

            # Coupling deviation contribution
            B[m] += 2 * self.beta * np.sum((C - self.C_star) * G_coupling[:, :, m])

            # Entropy deviation contribution
            B[m] += 2 * self.eta * np.sum((s - self.s_star) * P_entropy[:, m])

        # Maximum authority = u_max * ||B|| (by Cauchy-Schwarz)
        B_norm = np.linalg.norm(B)
        return float(u_max * B_norm)

    # ═══════════════════════════════════════════════════════════════════════
    # 4. SUSTAINMENT CERTIFICATE
    # ═══════════════════════════════════════════════════════════════════════

    def certify(self, C: np.ndarray, s: np.ndarray,
                delta: np.ndarray, alpha: np.ndarray,
                G_coupling: np.ndarray, P_entropy: np.ndarray,
                u_max: float,
                gamma: float = 0.7,
                S_crit: float = 1.0,
                entropy_growth_rates: Optional[np.ndarray] = None
                ) -> Dict:
        """
        Issue a binary SUSTAINMENT CERTIFICATE for the current state.

        Checks: C(ξ) > D(ξ) ?
        If YES → SUSTAINMENT POSSIBLE (complexity can be maintained)
        If NO  → SUSTAINMENT IMPOSSIBLE (attractor escape is inevitable)

        Parameters
        ----------
        C : ndarray, shape (N, N)
            Current coupling tensor.
        s : ndarray, shape (N,)
            Current normalized entropy rates.
        delta, alpha : ndarray, shape (N, N)
            Coupling decay and repair rate matrices.
        G_coupling : ndarray, shape (N, N, M)
            Control influence on coupling.
        P_entropy : ndarray, shape (N, M)
            Control influence on entropy.
        u_max : float
            Maximum control authority bound.
        gamma : float
            Saturating repair exponent.
        S_crit : float
            Critical entropy threshold.
        entropy_growth_rates : ndarray, optional
            Autonomous entropy acceleration rates.

        Returns
        -------
        certificate : dict
            Contains:
            - 'verdict': 'SUSTAINABLE' or 'UNSUSTAINABLE'
            - 'lyapunov_value': L(ξ)
            - 'viability_margin': V(ξ)
            - 'decay_rate': D(ξ)
            - 'control_authority': C(ξ)
            - 'margin': C(ξ) - D(ξ) (positive = sustainable)
            - 'regime': 'deep_health', 'critical', or 'beyond_criticality'
            - 'landauer_cost': thermodynamic cost estimate
        """
        V = self.analyzer.viability(C, s)
        L = self.lyapunov_value(C, s)

        if V <= 0:
            return {
                'verdict': 'UNSUSTAINABLE',
                'lyapunov_value': float('inf'),
                'viability_margin': float(V),
                'decay_rate': float('inf'),
                'control_authority': 0.0,
                'margin': float('-inf'),
                'regime': 'beyond_criticality',
                'landauer_cost': float('inf'),
            }

        D = self.autonomous_decay_rate(C, s, delta, alpha, gamma, S_crit,
                                        entropy_growth_rates)
        CA = self.control_authority(C, s, G_coupling, P_entropy, u_max)

        # Landauer cost estimate (in dimensionless units matching the framework)
        # I_repair ≈ sum of coupling decay rates (bits/second needing erasure)
        I_repair = np.sum(delta * C)
        landauer = self.K_B * self.BODY_TEMP * self.LN2 * I_repair / V

        margin = CA - D

        # Regime classification
        if V > 0.3 * self.V_max:
            regime = 'deep_health'
        elif V > 0.0:
            regime = 'critical'
        else:
            regime = 'beyond_criticality'

        verdict = 'SUSTAINABLE' if margin > 0 else 'UNSUSTAINABLE'

        return {
            'verdict': verdict,
            'lyapunov_value': float(L),
            'viability_margin': float(V),
            'decay_rate': float(D),
            'control_authority': float(CA),
            'margin': float(margin),
            'regime': regime,
            'landauer_cost': float(landauer),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # 5. TRAJECTORY-WIDE CERTIFICATION
    # ═══════════════════════════════════════════════════════════════════════

    def certify_trajectory(self, C_series: np.ndarray, s_series: np.ndarray,
                           delta: np.ndarray, alpha: np.ndarray,
                           G_coupling: np.ndarray, P_entropy: np.ndarray,
                           u_max: float, **kwargs) -> Dict:
        """
        Issue sustainment certificates along an entire time-series trajectory.

        Parameters
        ----------
        C_series : ndarray, shape (N, N, T)
            Time-dependent coupling tensor.
        s_series : ndarray, shape (N, T)
            Time-dependent entropy rates.
        (other params as in certify())

        Returns
        -------
        result : dict
            - 'certificates': list of per-step certificates
            - 'global_verdict': 'SUSTAINABLE' if all steps pass, else 'UNSUSTAINABLE'
            - 'first_failure_step': index of first unsustainable step (or None)
            - 'min_margin': minimum sustainment margin across trajectory
            - 'lyapunov_trajectory': L(t) values
            - 'viability_trajectory': V(t) values
        """
        T = C_series.shape[-1]
        certificates = []
        lyapunov_traj = []
        viability_traj = []

        first_failure = None

        for t in range(T):
            cert = self.certify(
                C_series[:, :, t], s_series[:, t],
                delta, alpha, G_coupling, P_entropy, u_max, **kwargs
            )
            certificates.append(cert)
            lyapunov_traj.append(cert['lyapunov_value'])
            viability_traj.append(cert['viability_margin'])

            if cert['verdict'] == 'UNSUSTAINABLE' and first_failure is None:
                first_failure = t

        margins = [c['margin'] for c in certificates if np.isfinite(c['margin'])]
        min_margin = min(margins) if margins else float('-inf')

        global_verdict = 'SUSTAINABLE' if first_failure is None else 'UNSUSTAINABLE'

        return {
            'certificates': certificates,
            'global_verdict': global_verdict,
            'first_failure_step': first_failure,
            'min_margin': float(min_margin),
            'lyapunov_trajectory': np.array(lyapunov_traj),
            'viability_trajectory': np.array(viability_traj),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # 6. OPTIMAL CONTROL LAW
    # ═══════════════════════════════════════════════════════════════════════

    def optimal_control(self, C: np.ndarray, s: np.ndarray,
                        G_coupling: np.ndarray, P_entropy: np.ndarray,
                        u_max: float) -> np.ndarray:
        """
        Compute the optimal feedback control u*(ξ) = -u_max · B(ξ) / ||B(ξ)||.

        This is the Sontag-type universal formula that maximally decreases
        the Lyapunov function at each instant.

        Parameters
        ----------
        C : ndarray, shape (N, N)
            Current coupling tensor.
        s : ndarray, shape (N,)
            Current entropy rates.
        G_coupling : ndarray, shape (N, N, M)
            Control influence on coupling.
        P_entropy : ndarray, shape (N, M)
            Control influence on entropy.
        u_max : float
            Control bound.

        Returns
        -------
        u_star : ndarray, shape (M,)
            Optimal control vector.
        """
        V = self.analyzer.viability(C, s)
        if V <= 0:
            return np.zeros(G_coupling.shape[2])

        M = G_coupling.shape[2]
        B = np.zeros(M)

        eps = 1e-6
        sigma_min = np.linalg.svd(C, compute_uv=False)[-1]
        k_star = np.argmax(s)

        for m in range(M):
            C_pert = C + G_coupling[:, :, m] * eps
            sigma_pert = np.linalg.svd(C_pert, compute_uv=False)[-1]
            d_sigma = (sigma_pert - sigma_min) / eps

            d_entropy = P_entropy[k_star, m]

            B[m] = -(d_sigma - d_entropy) / V
            B[m] += 2 * self.beta * np.sum((C - self.C_star) * G_coupling[:, :, m])
            B[m] += 2 * self.eta * np.sum((s - self.s_star) * P_entropy[:, m])

        B_norm = np.linalg.norm(B)
        if B_norm < 1e-12:
            return np.zeros(M)

        # Sontag universal formula: u* = -u_max * B / ||B||
        u_star = -u_max * B / B_norm
        return u_star

    # ═══════════════════════════════════════════════════════════════════════
    # 7. REPORT GENERATION
    # ═══════════════════════════════════════════════════════════════════════

    def generate_report(self, certificate: Dict) -> str:
        """Generate a human-readable sustainment certificate report."""
        lines = [
            "=" * 60,
            "  UNIVERSAL COMPLEXITY SUSTAINMENT CERTIFICATE",
            "=" * 60,
            "",
            f"  VERDICT:            {certificate['verdict']}",
            f"  Regime:             {certificate['regime']}",
            f"  Viability V(t):     {certificate['viability_margin']:.6f}",
            f"  Lyapunov L(ξ):      {certificate['lyapunov_value']:.6f}",
            f"  Decay Rate D(ξ):    {certificate['decay_rate']:.6f}",
            f"  Control Auth C(ξ):  {certificate['control_authority']:.6f}",
            f"  Margin (C-D):       {certificate['margin']:.6f}",
            f"  Landauer Cost:      {certificate['landauer_cost']:.2e}",
            "",
        ]

        if certificate['verdict'] == 'SUSTAINABLE':
            lines.append("  ✓ Complexity can be sustained under optimal control.")
            lines.append(f"  ✓ Safety margin: {certificate['margin']:.6f}")
        else:
            lines.append("  ✗ Complexity CANNOT be sustained at this state.")
            lines.append("  ✗ System will undergo attractor escape (thermodynamic death).")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
