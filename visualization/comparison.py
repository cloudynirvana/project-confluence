"""
Comparison Visualization
========================

Visualize differences between healthy and cancer generators.
Focus on identifying therapeutic targets and intervention effects.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from typing import Optional, List, Tuple


class ComparisonVisualizer:
    """
    Visualize comparisons between metabolic states.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def plot_generator_heatmaps(
        self,
        A_healthy: np.ndarray,
        A_cancer: np.ndarray,
        metabolite_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Side-by-side heatmaps of healthy and cancer generators.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        if metabolite_names is None:
            metabolite_names = [f"M{i}" for i in range(A_healthy.shape[0])]
        
        # Common color scale
        vmax = max(np.abs(A_healthy).max(), np.abs(A_cancer).max())
        vmin = -vmax
        
        # Healthy
        im1 = axes[0].imshow(A_healthy, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0].set_title('Healthy Generator (A_healthy)', fontsize=12)
        axes[0].set_xticks(range(len(metabolite_names)))
        axes[0].set_yticks(range(len(metabolite_names)))
        axes[0].set_xticklabels(metabolite_names, rotation=45, ha='right')
        axes[0].set_yticklabels(metabolite_names)
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        
        # Cancer
        im2 = axes[1].imshow(A_cancer, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1].set_title('TNBC Generator (A_cancer)', fontsize=12)
        axes[1].set_xticks(range(len(metabolite_names)))
        axes[1].set_yticks(range(len(metabolite_names)))
        axes[1].set_xticklabels(metabolite_names, rotation=45, ha='right')
        axes[1].set_yticklabels(metabolite_names)
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        # Difference (δA)
        delta_A = A_cancer - A_healthy
        im3 = axes[2].imshow(delta_A, cmap='RdBu_r')
        axes[2].set_title('Coherence Deficit (δA = A_cancer - A_healthy)', fontsize=12)
        axes[2].set_xticks(range(len(metabolite_names)))
        axes[2].set_yticks(range(len(metabolite_names)))
        axes[2].set_xticklabels(metabolite_names, rotation=45, ha='right')
        axes[2].set_yticklabels(metabolite_names)
        plt.colorbar(im3, ax=axes[2], shrink=0.8)
        
        plt.suptitle('Generator Matrix Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_intervention_targets(
        self,
        delta_A: np.ndarray,
        metabolite_names: Optional[List[str]] = None,
        top_n: int = 10,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Highlight top intervention targets from δA.
        """
        if metabolite_names is None:
            metabolite_names = [f"M{i}" for i in range(delta_A.shape[0])]
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Heatmap with highlighted targets
        ax1 = axes[0]
        im = ax1.imshow(np.abs(delta_A), cmap='Reds')
        ax1.set_title('Intervention Priority Map (|δA|)', fontsize=12)
        ax1.set_xticks(range(len(metabolite_names)))
        ax1.set_yticks(range(len(metabolite_names)))
        ax1.set_xticklabels(metabolite_names, rotation=45, ha='right')
        ax1.set_yticklabels(metabolite_names)
        plt.colorbar(im, ax=ax1)
        
        # Mark top targets
        flat_indices = np.argsort(np.abs(delta_A.ravel()))[::-1][:top_n]
        for idx in flat_indices:
            i, j = np.unravel_index(idx, delta_A.shape)
            ax1.scatter([j], [i], marker='o', s=200, facecolors='none',
                       edgecolors='blue', linewidths=2)
        
        # Bar chart of top corrections
        ax2 = axes[1]
        
        corrections = []
        for idx in flat_indices:
            i, j = np.unravel_index(idx, delta_A.shape)
            corrections.append({
                'label': f"{metabolite_names[i]}→{metabolite_names[j]}",
                'value': delta_A[i, j],
                'magnitude': abs(delta_A[i, j])
            })
        
        labels = [c['label'] for c in corrections]
        values = [c['value'] for c in corrections]
        colors = ['green' if v > 0 else 'red' for v in values]
        
        bars = ax2.barh(range(len(labels)), values, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('Correction Needed (δA_ij)')
        ax2.set_title(f'Top {top_n} Intervention Targets', fontsize=12)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax2.text(val, bar.get_y() + bar.get_height()/2,
                    f' {val:.3f}', va='center', 
                    ha='left' if val >= 0 else 'right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_coherence_scores(
        self,
        scores: dict,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Bar chart of coherence scores across conditions.
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        labels = list(scores.keys())
        values = list(scores.values())
        
        colors = []
        for v in values:
            if v > 0.7:
                colors.append('green')
            elif v > 0.4:
                colors.append('orange')
            else:
                colors.append('red')
        
        bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Threshold lines
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Healthy threshold')
        ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Warning threshold')
        
        ax.set_ylabel('Coherence Score')
        ax.set_title('Metabolic Coherence Comparison', fontsize=14)
        ax.set_ylim(0, 1.0)
        ax.legend()
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_restoration_trajectory(
        self,
        A_cancer: np.ndarray,
        A_treated: np.ndarray,
        A_healthy: np.ndarray,
        metabolite_names: Optional[List[str]] = None,
        key_metabolites: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize restoration trajectory in metabolite space.
        """
        if metabolite_names is None:
            metabolite_names = [f"M{i}" for i in range(A_cancer.shape[0])]
            
        if key_metabolites is None:
            key_metabolites = [0, 1, 3]  # Default: Glucose, Lactate, ATP
            
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Simulate steady states (or just use diagonal as proxy)
        # For visualization, use eigenvalue structure
        eig_cancer = linalg.eigvals(A_cancer)
        eig_treated = linalg.eigvals(A_treated)
        eig_healthy = linalg.eigvals(A_healthy)
        
        # Plot in 2D using first two eigenvalue dimensions
        ax.scatter(eig_healthy.real, eig_healthy.imag, 
                  s=150, c='green', label='Healthy', marker='o', edgecolors='black')
        ax.scatter(eig_cancer.real, eig_cancer.imag, 
                  s=150, c='red', label='TNBC', marker='s', edgecolors='black')
        ax.scatter(eig_treated.real, eig_treated.imag, 
                  s=150, c='blue', label='Treated', marker='^', edgecolors='black')
        
        # Draw arrows from cancer to treated for each eigenvalue
        for i in range(len(eig_cancer)):
            ax.annotate('', xy=(eig_treated[i].real, eig_treated[i].imag),
                       xytext=(eig_cancer[i].real, eig_cancer[i].imag),
                       arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5))
        
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.3, label='Stability boundary')
        ax.set_xlabel('Real Part (Stability)')
        ax.set_ylabel('Imaginary Part (Oscillation)')
        ax.set_title('Restoration Trajectory in Eigenspace', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_metabolic_dynamics(
        self,
        X_cancer: np.ndarray,
        X_treated: np.ndarray,
        time_points: np.ndarray,
        metabolite_names: Optional[List[str]] = None,
        key_metabolites: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare metabolic dynamics before and after treatment.
        """
        if metabolite_names is None:
            metabolite_names = [f"M{i}" for i in range(X_cancer.shape[1])]
            
        if key_metabolites is None:
            key_metabolites = list(range(min(4, X_cancer.shape[1])))
            
        n_mets = len(key_metabolites)
        fig, axes = plt.subplots(n_mets, 1, figsize=(12, 3 * n_mets), sharex=True)
        if n_mets == 1:
            axes = [axes]
            
        for ax, met_idx in zip(axes, key_metabolites):
            ax.plot(time_points, X_cancer[:, met_idx], 
                   'r-', linewidth=2, label='TNBC (untreated)')
            ax.plot(time_points, X_treated[:, met_idx], 
                   'b--', linewidth=2, label='Post-treatment')
            
            ax.set_ylabel(metabolite_names[met_idx])
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
        axes[-1].set_xlabel('Time')
        plt.suptitle('Metabolic Dynamics: TNBC vs Treated', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
