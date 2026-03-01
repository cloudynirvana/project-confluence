"""
Coherence Analysis Module
=========================

Measure coherence metrics that distinguish healthy from cancerous states.
Coherence = the degree to which metabolic oscillators are synchronized and
operating within stable, healthy attractor basins.

Key metrics:
- Phase coherence: Synchronization between metabolic cycles
- Spectral coherence: Eigenvalue clustering in healthy regions
- Attractor coherence: Basin structure and contractivity
"""

import numpy as np
from scipy import linalg, signal
from typing import Dict, Tuple, Optional, List


class CoherenceAnalyzer:
    """
    Analyze coherence properties of metabolic generator matrices.
    
    In Remisov's framework: loss of coherence = disease state.
    This class quantifies that loss across multiple dimensions.
    """
    
    def __init__(self):
        self.metrics = {}
        
    def analyze(self, A: np.ndarray, reference_A: Optional[np.ndarray] = None) -> Dict:
        """
        Comprehensive coherence analysis of generator matrix.
        
        Args:
            A: Generator matrix to analyze
            reference_A: Optional healthy reference for comparison
            
        Returns:
            Dictionary of coherence metrics
        """
        self.metrics = {}
        
        # Spectral analysis
        self.metrics['spectral'] = self._spectral_coherence(A)
        
        # Stability analysis
        self.metrics['stability'] = self._stability_analysis(A)
        
        # Coupling structure
        self.metrics['coupling'] = self._coupling_analysis(A)
        
        # If reference provided, compute comparative metrics
        if reference_A is not None:
            self.metrics['deficit'] = self._coherence_deficit(A, reference_A)
            
        # Overall coherence score (0-1, higher = more coherent/healthy)
        self.metrics['overall_score'] = self._compute_overall_score()
        
        return self.metrics
    
    def _spectral_coherence(self, A: np.ndarray) -> Dict:
        """
        Analyze eigenvalue spectrum for coherence signatures.
        
        Healthy systems show:
        - Clustered eigenvalues (coordinated modes)
        - Negative real parts (stability)
        - Matched imaginary pairs (oscillatory coherence)
        """
        eigenvalues, eigenvectors = linalg.eig(A)
        
        real_parts = eigenvalues.real
        imag_parts = eigenvalues.imag
        
        # Eigenvalue dispersion (lower = more coherent)
        real_dispersion = np.std(real_parts)
        
        # Stability margin (most stable eigenvalue)
        stability_margin = -np.max(real_parts)  # Positive = stable
        
        # Oscillatory coherence: how well-matched are oscillation frequencies
        nonzero_imag = imag_parts[np.abs(imag_parts) > 1e-10]
        if len(nonzero_imag) > 0:
            freq_dispersion = np.std(np.abs(nonzero_imag))
        else:
            freq_dispersion = 0.0
            
        # Spectral gap (separation between fast and slow modes)
        sorted_real = np.sort(real_parts)
        if len(sorted_real) > 1:
            spectral_gaps = np.diff(sorted_real)
            max_gap = np.max(spectral_gaps)
        else:
            max_gap = 0.0
            
        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'real_dispersion': real_dispersion,
            'stability_margin': stability_margin,
            'frequency_dispersion': freq_dispersion,
            'max_spectral_gap': max_gap,
            'n_unstable_modes': np.sum(real_parts > 0)
        }
    
    def _stability_analysis(self, A: np.ndarray) -> Dict:
        """
        Assess system stability and contractivity.
        
        Contractivity = system returns to attractor after perturbation.
        Loss of contractivity is a key cancer signature.
        """
        eigenvalues = linalg.eigvals(A)
        real_parts = eigenvalues.real
        
        # System is stable if all eigenvalues have negative real parts
        is_stable = np.all(real_parts < 0)
        
        # Lyapunov exponent (largest real part)
        lyapunov = np.max(real_parts)
        
        # Contraction rate (how fast system returns to attractor)
        # More negative = faster contraction = healthier
        contraction_rate = -np.mean(real_parts)
        
        # Condition number of eigenvector matrix (sensitivity to perturbation)
        _, V = linalg.eig(A)
        try:
            condition = np.linalg.cond(V)
        except:
            condition = np.inf
            
        return {
            'is_stable': is_stable,
            'lyapunov_exponent': lyapunov,
            'contraction_rate': contraction_rate,
            'sensitivity': min(condition, 1e10),  # Cap at large value
            'stability_score': 1.0 / (1.0 + np.exp(lyapunov * 10))  # Sigmoid transform
        }
    
    def _coupling_analysis(self, A: np.ndarray) -> Dict:
        """
        Analyze coupling structure between metabolites.
        
        Healthy systems have:
        - Balanced positive/negative coupling (homeostasis)
        - Symmetric coupling (bidirectional communication)
        - Modular structure (functional groupings)
        """
        n = A.shape[0]
        
        # Separate diagonal and off-diagonal
        diagonal = np.diag(A)
        off_diag = A - np.diag(diagonal)
        
        # Coupling symmetry
        asymmetry = np.linalg.norm(off_diag - off_diag.T, 'fro')
        symmetry_score = 1.0 / (1.0 + asymmetry)
        
        # Coupling balance (positive vs negative)
        positive_coupling = np.sum(off_diag[off_diag > 0])
        negative_coupling = np.abs(np.sum(off_diag[off_diag < 0]))
        
        if positive_coupling + negative_coupling > 0:
            balance = 1.0 - abs(positive_coupling - negative_coupling) / (positive_coupling + negative_coupling)
        else:
            balance = 1.0
            
        # Coupling density (fraction of non-zero couplings)
        coupling_density = np.sum(np.abs(off_diag) > 1e-10) / (n * (n - 1))
        
        # Coupling strength
        mean_coupling = np.mean(np.abs(off_diag))
        
        return {
            'symmetry_score': symmetry_score,
            'balance': balance,
            'density': coupling_density,
            'mean_strength': mean_coupling,
            'positive_coupling': positive_coupling,
            'negative_coupling': negative_coupling
        }
    
    def _coherence_deficit(self, A_cancer: np.ndarray, A_healthy: np.ndarray) -> Dict:
        """
        Compute coherence deficit between cancer and healthy generators.
        
        This is the core of Remisov's approach: identify exactly what
        coherence properties are lost in the disease state.
        """
        # Direct difference
        delta_A = A_cancer - A_healthy
        
        # Eigenvalue shift
        eig_cancer = linalg.eigvals(A_cancer)
        eig_healthy = linalg.eigvals(A_healthy)
        
        # Sort for comparison
        idx_c = np.argsort(eig_cancer.real)
        idx_h = np.argsort(eig_healthy.real)
        
        eigenvalue_shift = eig_cancer[idx_c] - eig_healthy[idx_h]
        
        # Stability loss
        stability_loss = np.max(eig_cancer.real) - np.max(eig_healthy.real)
        
        # Coupling disruption (Frobenius norm of difference)
        coupling_disruption = np.linalg.norm(delta_A, 'fro')
        
        # Identify most disrupted metabolites (rows with largest changes)
        row_changes = np.linalg.norm(delta_A, axis=1)
        most_disrupted = np.argsort(row_changes)[::-1]
        
        return {
            'delta_A': delta_A,
            'eigenvalue_shift': eigenvalue_shift,
            'stability_loss': stability_loss,
            'coupling_disruption': coupling_disruption,
            'most_disrupted_indices': most_disrupted,
            'disruption_magnitudes': row_changes[most_disrupted]
        }
    
    def _compute_overall_score(self) -> float:
        """
        Compute overall coherence score (0-1).
        
        Higher = more coherent = healthier.
        """
        scores = []
        
        # Stability contribution
        if 'stability' in self.metrics:
            scores.append(self.metrics['stability']['stability_score'])
            
        # Spectral contribution
        if 'spectral' in self.metrics:
            # Penalize unstable modes
            unstable_penalty = np.exp(-self.metrics['spectral']['n_unstable_modes'])
            scores.append(unstable_penalty)
            
        # Coupling contribution
        if 'coupling' in self.metrics:
            scores.append(self.metrics['coupling']['symmetry_score'])
            scores.append(self.metrics['coupling']['balance'])
            
        return np.mean(scores) if scores else 0.0
    
    def get_coherence_report(self) -> str:
        """
        Generate human-readable coherence report.
        """
        if not self.metrics:
            return "No analysis performed yet."
            
        report = []
        report.append("=" * 50)
        report.append("COHERENCE ANALYSIS REPORT")
        report.append("=" * 50)
        
        # Overall
        score = self.metrics.get('overall_score', 0)
        status = "HEALTHY" if score > 0.7 else "DISRUPTED" if score > 0.4 else "PATHOLOGICAL"
        report.append(f"\nOverall Coherence Score: {score:.3f} ({status})")
        
        # Stability
        if 'stability' in self.metrics:
            stab = self.metrics['stability']
            report.append(f"\nStability:")
            report.append(f"  - System stable: {stab['is_stable']}")
            report.append(f"  - Lyapunov exponent: {stab['lyapunov_exponent']:.4f}")
            report.append(f"  - Contraction rate: {stab['contraction_rate']:.4f}")
            
        # Coupling
        if 'coupling' in self.metrics:
            coup = self.metrics['coupling']
            report.append(f"\nCoupling Structure:")
            report.append(f"  - Symmetry: {coup['symmetry_score']:.3f}")
            report.append(f"  - Balance: {coup['balance']:.3f}")
            report.append(f"  - Density: {coup['density']:.3f}")
            
        # Deficit (if computed)
        if 'deficit' in self.metrics:
            deficit = self.metrics['deficit']
            report.append(f"\nCoherence Deficit (vs. Healthy):")
            report.append(f"  - Stability loss: {deficit['stability_loss']:.4f}")
            report.append(f"  - Coupling disruption: {deficit['coupling_disruption']:.4f}")
            report.append(f"  - Most disrupted pathways: {deficit['most_disrupted_indices'][:3].tolist()}")
            
        report.append("\n" + "=" * 50)
        
        return "\n".join(report)


def compute_phase_coherence(X: np.ndarray, fs: float = 1.0) -> np.ndarray:
    """
    Compute pairwise phase coherence between metabolite time-series.
    
    Uses Hilbert transform to extract instantaneous phase,
    then measures phase locking between all pairs.
    
    Args:
        X: Time-series data, shape (n_timepoints, n_metabolites)
        fs: Sampling frequency
        
    Returns:
        C: Phase coherence matrix, shape (n_metabolites, n_metabolites)
           Values near 1 = high coherence (phase-locked)
           Values near 0 = low coherence (independent phases)
    """
    n_times, n_metabolites = X.shape
    
    # Extract instantaneous phase via Hilbert transform
    phases = np.zeros_like(X)
    for j in range(n_metabolites):
        analytic = signal.hilbert(X[:, j])
        phases[:, j] = np.angle(analytic)
    
    # Compute phase coherence (phase locking value)
    C = np.zeros((n_metabolites, n_metabolites))
    
    for i in range(n_metabolites):
        for j in range(n_metabolites):
            phase_diff = phases[:, i] - phases[:, j]
            # Phase locking value
            C[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))
            
    return C
