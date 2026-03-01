"""
Eigenspace Visualization
========================

Visualize generator matrix eigenvalue spectra and eigenvector structure.
Key for understanding stability, oscillation modes, and coherence.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from typing import Optional, List, Tuple


class EigenspaceVisualizer:
    """
    Visualize eigenspace structure of metabolic generators.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def plot_eigenvalue_spectrum(
        self,
        A: np.ndarray,
        title: str = "Eigenvalue Spectrum",
        ax: Optional[plt.Axes] = None,
        color: str = 'blue',
        label: str = None
    ) -> plt.Axes:
        """
        Plot eigenvalues in the complex plane.
        
        Key interpretation:
        - Real part < 0: stable (left of imaginary axis)
        - Imaginary part ≠ 0: oscillatory
        - Eigenvalues near origin: slow dynamics
        """
        eigenvalues = linalg.eigvals(A)
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
        # Plot eigenvalues
        ax.scatter(eigenvalues.real, eigenvalues.imag, 
                  s=100, c=color, alpha=0.7, label=label, edgecolors='black')
        
        # Stability boundary
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Stability boundary')
        
        # Origin
        ax.scatter([0], [0], marker='+', s=200, c='black', zorder=5)
        
        ax.set_xlabel('Real Part (Stability)', fontsize=12)
        ax.set_ylabel('Imaginary Part (Oscillation)', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        if label:
            ax.legend()
            
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def compare_spectra(
        self,
        A_healthy: np.ndarray,
        A_cancer: np.ndarray,
        A_treated: Optional[np.ndarray] = None,
        metabolite_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare eigenvalue spectra of healthy, cancer, and treated systems.
        """
        fig, axes = plt.subplots(1, 3 if A_treated is not None else 2, 
                                 figsize=(18 if A_treated else 12, 6))
        
        # Healthy
        self.plot_eigenvalue_spectrum(A_healthy, "Healthy Tissue", axes[0], 
                                     color='green', label='Healthy')
        
        # Cancer
        self.plot_eigenvalue_spectrum(A_cancer, "TNBC (Cancer)", axes[1], 
                                     color='red', label='TNBC')
        
        # Treated (if provided)
        if A_treated is not None:
            self.plot_eigenvalue_spectrum(A_treated, "Post-Treatment", axes[2],
                                         color='blue', label='Treated')
        
        # Make axes equal for fair comparison
        all_eigenvalues = np.concatenate([
            linalg.eigvals(A_healthy),
            linalg.eigvals(A_cancer)
        ])
        if A_treated is not None:
            all_eigenvalues = np.concatenate([all_eigenvalues, linalg.eigvals(A_treated)])
            
        max_extent = max(np.abs(all_eigenvalues.real).max(), 
                        np.abs(all_eigenvalues.imag).max()) * 1.2
        
        for ax in axes:
            ax.set_xlim(-max_extent, max_extent)
            ax.set_ylim(-max_extent, max_extent)
            ax.set_aspect('equal')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_eigenvector_heatmap(
        self,
        A: np.ndarray,
        metabolite_names: Optional[List[str]] = None,
        title: str = "Eigenvector Structure",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Heatmap of eigenvector magnitudes.
        
        Shows which metabolites participate in each eigenmode.
        """
        eigenvalues, eigenvectors = linalg.eig(A)
        
        # Sort by real part
        idx = np.argsort(eigenvalues.real)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Magnitude of eigenvector components
        magnitudes = np.abs(eigenvectors)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        im = ax.imshow(magnitudes, aspect='auto', cmap='viridis')
        
        # Labels
        if metabolite_names is None:
            metabolite_names = [f"M{i}" for i in range(A.shape[0])]
            
        ax.set_yticks(range(len(metabolite_names)))
        ax.set_yticklabels(metabolite_names)
        ax.set_xlabel('Eigenmode (sorted by stability)')
        ax.set_ylabel('Metabolite')
        ax.set_title(title)
        
        # Mode labels
        mode_labels = [f"λ={e.real:.2f}" for e in eigenvalues]
        ax.set_xticks(range(len(eigenvalues)))
        ax.set_xticklabels(mode_labels, rotation=45, ha='right')
        
        plt.colorbar(im, label='Participation')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_phase_portrait(
        self,
        A: np.ndarray,
        metabolite_i: int = 0,
        metabolite_j: int = 1,
        metabolite_names: Optional[List[str]] = None,
        n_trajectories: int = 10,
        title: str = "Phase Portrait",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        2D phase portrait showing system dynamics.
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Generate trajectories from random initial conditions
        np.random.seed(42)
        
        for _ in range(n_trajectories):
            # Random initial state
            x0 = np.random.randn(A.shape[0]) * 0.5 + 0.5
            
            # Simulate trajectory
            t = np.linspace(0, 10, 200)
            traj = np.zeros((len(t), A.shape[0]))
            traj[0] = x0
            
            for k in range(1, len(t)):
                dt = t[k] - t[0]
                traj[k] = linalg.expm(A * dt) @ x0
                
            # Plot trajectory in 2D projection
            ax.plot(traj[:, metabolite_i], traj[:, metabolite_j], 
                   alpha=0.6, linewidth=1)
            ax.scatter(traj[0, metabolite_i], traj[0, metabolite_j], 
                      marker='o', s=30, c='green', zorder=5)
            ax.scatter(traj[-1, metabolite_i], traj[-1, metabolite_j], 
                      marker='x', s=50, c='red', zorder=5)
        
        if metabolite_names:
            ax.set_xlabel(metabolite_names[metabolite_i])
            ax.set_ylabel(metabolite_names[metabolite_j])
        else:
            ax.set_xlabel(f'Metabolite {metabolite_i}')
            ax.set_ylabel(f'Metabolite {metabolite_j}')
            
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_stability_analysis(
        self,
        A_list: List[np.ndarray],
        labels: List[str],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Bar chart comparing stability metrics across systems.
        """
        metrics = {
            'Max Real Part': [],
            'Mean Real Part': [],
            'Oscillatory Modes': [],
            'Condition Number': []
        }
        
        for A in A_list:
            eigenvalues, V = linalg.eig(A)
            metrics['Max Real Part'].append(np.max(eigenvalues.real))
            metrics['Mean Real Part'].append(np.mean(eigenvalues.real))
            metrics['Oscillatory Modes'].append(
                np.sum(np.abs(eigenvalues.imag) > 1e-10) // 2)
            try:
                metrics['Condition Number'].append(min(np.linalg.cond(V), 100))
            except:
                metrics['Condition Number'].append(100)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        colors = ['green', 'red', 'blue', 'orange'][:len(labels)]
        
        for ax, (name, values) in zip(axes.flat, metrics.items()):
            bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel(name)
            ax.set_title(name)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.2f}', ha='center', va='bottom')
                
        plt.suptitle('Stability Analysis Comparison', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
