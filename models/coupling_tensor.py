"""
Coupling Tensor Module — Project Confluence (BAC Core Infrastructure)
=====================================================================

Computes and analyses the time-dependent, cross-scale coupling tensor C_ij(t)
along system trajectories, enabling the concrete computation of the
Bounded Adaptive Coherence (BAC) viability condition:
    V(t) = σ_min(C(t)) - max_k[ṡ_k(t)] > 0

This module provides the formal mathematical bridge from pure BAC theory
to the 15D complex attractor ODE equations and clinical multi-omic signals.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from models.complexity_profiler import sample_entropy


class CouplingTensorAnalyzer:
    """
    Computes, analyses, and classifies the BAC coupling tensor C(t) from
    ODE trajectories or multi-omic time-series data.

    Scale Partitioning (4-scale system for 15D state z):
        0: Molecular (Glucose, Lactate, Pyruvate, ATP, NADH)
        1: Cellular  (Glutamine, Glutamate, αKG, Citrate, ROS)
        2: Organism  (I_eff, I_reg, I_exhaust)
        3: Tissue    (σ_stromal, ν_vascular)
    """

    # Scale definitions
    DEFAULT_SCALES = {
        'molecular': [0, 1, 2, 3, 4],    # Glucose, Lactate, Pyruvate, ATP, NADH
        'cellular':  [5, 6, 7, 8, 9],    # Glutamine, Glutamate, αKG, Citrate, ROS
        'organism':  [10, 11, 12],        # I_eff, I_reg, I_exhaust
        'tissue':    [13, 14],            # σ_stromal, ν_vascular
    }

    # Reference baseline entropy rate for normalization (calibrated from healthy)
    S_REF_DEFAULT = 0.45

    def __init__(self, scale_indices: Optional[Dict[str, List[int]]] = None, 
                 s_ref: float = S_REF_DEFAULT):
        """
        Parameters
        ----------
        scale_indices : dict of str to list of int, optional
            Custom indices mapping biological scales to state variables.
        s_ref : float
            Normalizing constant for entropy rates (maximum sustainable entropy rate).
        """
        self.scales = scale_indices or self.DEFAULT_SCALES
        self.scale_names = list(self.scales.keys())
        self.N_scales = len(self.scales)
        self.s_ref = s_ref

    # ═══════════════════════════════════════════════════════════════════════
    # 1. CORE COUPLING TENSOR COMPUTATION
    # ═══════════════════════════════════════════════════════════════════════

    def compute_from_jacobian(self, ode_system, trajectory: np.ndarray, 
                              t_points: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """
        Compute the exact C_ij(t) tensor along a trajectory using Jacobian block norms.

        C_ij(t) = ||J_ij(t)||_F / max_kl ||J_kl(t)||_F
        where J_ij is the sub-matrix of the state Jacobian mapping scale j to scale i.

        Parameters
        ----------
        ode_system : ComplexAttractorODE or subclass
            The biological ODE model exposing a public `rhs(t, z)` method.
        trajectory : ndarray, shape (15, T) or (DIM, T)
            State trajectory along which the tensor is evaluated.
        t_points : ndarray, shape (T,)
            Time values corresponding to each column of the trajectory.
        h : float
            Finite difference perturbation step size.

        Returns
        -------
        C_series : ndarray, shape (N_scales, N_scales, T)
            Time-dependent coupling tensor normalized to [0, 1].
        """
        dim, T = trajectory.shape
        C_series = np.zeros((self.N_scales, self.N_scales, T))

        for t_idx in range(T):
            z = trajectory[:, t_idx]
            t = t_points[t_idx]

            # 1. Numerical Jacobian J at this point: J_ij = ∂F_i / ∂z_j
            J = np.zeros((dim, dim))
            for j in range(dim):
                z_plus = z.copy()
                z_minus = z.copy()
                z_plus[j] += h
                z_minus[j] -= h

                F_plus = ode_system.rhs(t, z_plus)
                F_minus = ode_system.rhs(t, z_minus)

                J[:, j] = (F_plus - F_minus) / (2.0 * h)

            # 2. Extract scale-partitioned block norms
            block_norms = np.zeros((self.N_scales, self.N_scales))
            for i, scale_i in enumerate(self.scale_names):
                for j, scale_j in enumerate(self.scale_names):
                    idx_i = self.scales[scale_i]
                    idx_j = self.scales[scale_j]
                    
                    # Slicing the Jacobian block
                    block = J[np.ix_(idx_i, idx_j)]
                    block_norms[i, j] = np.linalg.norm(block, 'fro')

            # 3. Normalize coupling tensor by the maximum block norm
            max_norm = np.max(block_norms)
            if max_norm > 1e-12:
                C_series[:, :, t_idx] = block_norms / max_norm
            else:
                C_series[:, :, t_idx] = block_norms

        return C_series

    # ═══════════════════════════════════════════════════════════════════════
    # 2. PER-SCALE ENTROPY RATES
    # ═══════════════════════════════════════════════════════════════════════

    def scale_entropy_rates(self, trajectory: np.ndarray, dt: float, 
                            window: int = 40) -> np.ndarray:
        """
        Compute rolling normalized entropy rates ṡ_k(t) for each scale.

        ṡ_k(t) = SampEn(scale_k_variables) / S_ref

        Parameters
        ----------
        trajectory : ndarray, shape (DIM, T)
            State trajectory.
        dt : float
            Sampling time interval.
        window : int
            Rolling window size (number of points) to evaluate sample entropy.

        Returns
        -------
        entropy_series : ndarray, shape (N_scales, T)
            Time-dependent normalized entropy rates.
        """
        dim, T = trajectory.shape
        entropy_series = np.zeros((self.N_scales, T))

        for t_idx in range(T):
            # Define window indices with zero-padding safety
            start_idx = max(0, t_idx - window + 1)
            end_idx = t_idx + 1
            
            # If we don't have enough history, use whatever history is available
            # (minimum window size of 5 for safety)
            if end_idx - start_idx < 5:
                # Use standard resting baseline in early steps
                for i in range(self.N_scales):
                    entropy_series[i, t_idx] = 0.1
                continue

            for i, scale_name in enumerate(self.scale_names):
                indices = self.scales[scale_name]
                # Combine variables in this scale by taking their mean profile in the window
                scale_signal = np.mean(trajectory[indices, start_idx:end_idx], axis=0)
                
                # Compute sample entropy (m=2)
                val = sample_entropy(scale_signal, m=2, r=0.2 * np.std(scale_signal))
                
                # Check for nan or infinite values
                if np.isnan(val) or np.isinf(val):
                    val = 0.0

                # Normalize against reference max entropy
                entropy_series[i, t_idx] = val / self.s_ref

        return entropy_series

    # ═══════════════════════════════════════════════════════════════════════
    # 3. BAC VIABILITY FUNCTIONALS
    # ═══════════════════════════════════════════════════════════════════════

    def viability(self, C_t: np.ndarray, entropy_rates: np.ndarray) -> float:
        """
        Compute the instantaneous viability margin.
        V(t) = σ_min(C(t)) - max_k[ṡ_k(t)]

        Parameters
        ----------
        C_t : ndarray, shape (N_scales, N_scales)
            Coupling tensor at a single time step.
        entropy_rates : ndarray, shape (N_scales,)
            Normalized scale entropy rates at the same time step.

        Returns
        -------
        viability : float
            Viability margin. Positive = stable system, Negative = critical failure.
        """
        # Smallest singular value
        sigma_min = np.linalg.svd(C_t, compute_uv=False)[-1]
        max_entropy = np.max(entropy_rates)
        return float(sigma_min - max_entropy)

    def bac_satisfied(self, C_t: np.ndarray, entropy_rates: np.ndarray) -> bool:
        """Boolean check if the BAC viability condition is satisfied."""
        return self.viability(C_t, entropy_rates) > 0.0

    def viability_trajectory(self, C_series: np.ndarray, 
                             entropy_series: np.ndarray) -> np.ndarray:
        """Compute the viability trajectory V(t) over the full time series."""
        T = C_series.shape[-1]
        return np.array([
            self.viability(C_series[:, :, t], entropy_series[:, t])
            for t in range(T)
        ])

    # ═══════════════════════════════════════════════════════════════════════
    # 4. PATHOLOGY AND FAILURE archetype CLASSIFIER
    # ═══════════════════════════════════════════════════════════════════════

    def classify_failure(self, C_current: np.ndarray, C_healthy: np.ndarray, 
                         threshold: float = 0.15) -> Tuple[str, float, Dict]:
        """
        Classify the system state as Healthy, Aging (global decay), or Cancer (scale decoupling).

        Classification heuristics:
            - Healthy: Low delta from baseline healthy tensor.
            - Aging: Uniform off-diagonal coupling decay.
            - Cancer: Selective collapse of cell-organism coupling (C_24) with elevated cell coherence.

        Parameters
        ----------
        C_current : ndarray, shape (N_scales, N_scales)
            Current coupling tensor.
        C_healthy : ndarray, shape (N_scales, N_scales)
            Baseline healthy coupling tensor.
        threshold : float
            Minimum average off-diagonal change to trigger failure classification.

        Returns
        -------
        classification : str
            'healthy', 'aging', 'cancer', or 'mixed'.
        confidence : float
            Confidence value in [0, 1].
        details : dict
            Diagnostic metrics (uniformity, selectivity, c22_c24_ratio).
        """
        delta = C_healthy - C_current
        
        # Slices for off-diagonal calculations
        N = C_current.shape[0]
        offdiag_mask = ~np.eye(N, dtype=bool)
        offdiag_change = delta[offdiag_mask]
        
        avg_offdiag_loss = np.mean(np.abs(offdiag_change))
        
        # 1. Healthy check
        if avg_offdiag_loss < threshold:
            return 'healthy', 1.0 - (avg_offdiag_loss / threshold), {
                'avg_offdiag_loss': avg_offdiag_loss
            }
        
        # 2. Assess uniform vs selective decay
        # Uniformity = standard deviation / mean of the off-diagonal changes
        uniformity = np.std(offdiag_change) / (np.mean(np.abs(offdiag_change)) + 1e-10)
        
        # Selectivity of cellular-organismal decoupling (indices: Cell=1, Organism=2)
        cell_idx, organism_idx = 1, 2
        organism_coupling_loss = np.abs(delta[cell_idx, organism_idx]) + np.abs(delta[organism_idx, cell_idx])
        other_coupling_loss = np.mean(np.abs(offdiag_change))
        selectivity = organism_coupling_loss / (other_coupling_loss + 1e-10)
        
        # Cancer marker: high cellular coherence (C_11/C_12) relative to organism connectivity (C_12)
        # Note: mapping scales: 0=molecular, 1=cellular, 2=organism, 3=tissue
        # Internal cellular coherence is C[1, 1], cellular-organismal coupling is C[1, 2]
        c22_c24_ratio = C_current[1, 1] / (C_current[1, 2] + 1e-10)

        # 3. Classify
        # Aging signature: uniform off-diagonal decay (low standard deviation of changes)
        if uniformity < 0.45 and selectivity < 1.3:
            confidence = np.clip(1.0 - uniformity, 0.5, 1.0)
            return 'aging', float(confidence), {
                'uniformity': float(uniformity),
                'selectivity': float(selectivity),
                'avg_offdiag_loss': float(avg_offdiag_loss)
            }
        
        # Cancer signature: high selectivity and high internal cell coherence compared to organism coupling
        elif selectivity > 1.6 or c22_c24_ratio > 2.5:
            confidence = np.clip(selectivity / 3.0, 0.5, 0.98)
            return 'cancer', float(confidence), {
                'selectivity': float(selectivity),
                'c22_c24_ratio': float(c22_c24_ratio),
                'organism_coupling_loss': float(organism_coupling_loss)
            }
        
        else:
            return 'mixed', 0.5, {
                'uniformity': float(uniformity),
                'selectivity': float(selectivity),
                'c22_c24_ratio': float(c22_c24_ratio)
            }

    # ═══════════════════════════════════════════════════════════════════════
    # 5. CONTROL-THEORETIC TARGETING AND LIFTING
    # ═══════════════════════════════════════════════════════════════════════

    def optimal_intervention_target(self, C_t: np.ndarray, 
                                   entropy_rates: np.ndarray, 
                                   delta: float = 0.05) -> Tuple[int, int, float]:
        """
        Identify the optimal coupling element C_ij to enhance to maximize viability.

        This computes a numerical approximation of the Hamiltonian derivative:
            ∂V / ∂C_ij = lim_{δ -> 0} [ V(C + δ E_ij) - V(C) ] / δ

        Parameters
        ----------
        C_t : ndarray, shape (N_scales, N_scales)
            Current coupling tensor.
        entropy_rates : ndarray, shape (N_scales,)
            Current scale entropy rates.
        delta : float
            Perturbation amount to add to each element.

        Returns
        -------
        i : int
            Row index of optimal targeting element.
        j : int
            Column index of optimal targeting element.
        gradient : float
            Expected derivative value (viability gain per unit increase in C_ij).
        """
        base_viability = self.viability(C_t, entropy_rates)
        best_gain = -1.0
        best_target = (0, 0)

        # We evaluate off-diagonal elements (cross-system couplings)
        for i in range(self.N_scales):
            for j in range(self.N_scales):
                if i == j:
                    continue  # Skip diagonal elements (internal scale coherence)
                
                # Apply positive perturbation to C_ij
                C_perturbed = C_t.copy()
                C_perturbed[i, j] = np.clip(C_perturbed[i, j] + delta, 0.0, 1.0)
                
                perturbed_viability = self.viability(C_perturbed, entropy_rates)
                gain = perturbed_viability - base_viability
                
                if gain > best_gain:
                    best_gain = gain
                    best_target = (i, j)

        gradient = best_gain / delta
        return best_target[0], best_target[1], float(gradient)

    def lift_biologic_to_coupling(self, biologic_operator: np.ndarray) -> np.ndarray:
        """
        Lift a 5x5 Φ-space biologic operator into a 4x4 coupling tensor perturbation.

        Maps the 5 dimensions of Φ-space to specific element additions in C:
            φ1 (temporal variability)   → ΔC_00 (molecular coherence)
            φ2 (spatial heterogeneity)  → ΔC_11 (cellular coherence)
            φ3 (immune connectivity)    → ΔC_12 (cellular-organismal coupling)
            φ4 (adaptive plasticity)    → ΔC_13 (cellular-tissue coupling)
            φ5 (microenvironmental coupling) → ΔC_23 (organismal-tissue coupling)

        Parameters
        ----------
        biologic_operator : ndarray, shape (5, 5)
            Action matrix representing the biologic in Φ-space.

        Returns
        -------
        C_perturbation : ndarray, shape (4, 4)
            Perturbation matrix to be added to C.
        """
        C_pert = np.zeros((self.N_scales, self.N_scales))
        
        # Take the diagonal entries of the biologic operator (direct action terms)
        direct_actions = np.diag(biologic_operator)
        
        # Map Φ dimensions to C elements
        C_pert[0, 0] = direct_actions[0]  # φ1 → molecular diagonal
        C_pert[1, 1] = direct_actions[1]  # φ2 → cellular diagonal
        C_pert[1, 2] = direct_actions[2]  # φ3 → cell-organism (C_24 counterpart)
        C_pert[1, 3] = direct_actions[3]  # φ4 → cell-tissue (C_25 counterpart)
        C_pert[2, 3] = direct_actions[4]  # φ5 → organism-tissue

        # Symmeterise the off-diagonals for consistency
        C_pert[2, 1] = C_pert[1, 2]
        C_pert[3, 1] = C_pert[1, 3]
        C_pert[3, 2] = C_pert[2, 3]

        return C_pert
