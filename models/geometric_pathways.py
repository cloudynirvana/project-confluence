"""
Geometric Pathways Module
=========================

Implements Freidlin-Wentzell Minimum Action Pathway (MAP) computation
for identifying therapeutic realignment trajectories between cancer and
healthy attractor states.

Includes optimizations: LRU Caching for basin attractors and Progressive 
"Lazy" Path Finding for high-dimensional efficiency.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from functools import lru_cache
import logging
from typing import Callable, Tuple, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FreidlinWentzellOptimizer:
    """
    Computes the minimum action path between attractor states in phase space
    using the String Method and progressive subspace optimization.
    """
    
    def __init__(self, ode_system, dt: float = 0.1):
        self.sys = ode_system
        self.F = ode_system.rhs
        self.dim = ode_system.DIM
        self.dt = dt
        
    @lru_cache(maxsize=32)
    def get_attractor(self, state_name: str) -> np.ndarray:
        """
        Cached retrieval of attractor states to avoid redundant ODE integrations.
        state_name can be "healthy" or a cancer type like "TNBC".
        """
        logger.info(f"Computing and caching attractor: {state_name}")
        if state_name.lower() == "healthy":
            z0 = self.sys.healthy_initial_state()
            # Integrate to steady state (attractor)
            res = self.sys.solve(z0=z0, t_span=(0, 200))
            return res['z'][:, -1]
            
        # For disease states, we would switch the params and integrate
        # Simplified here: assuming self.sys params are already set to disease
        z0 = self.sys.healthy_initial_state()
        res = self.sys.solve(z0=z0, t_span=(0, 200))
        return res['z'][:, -1]

    def _action_integrand(self, path: np.ndarray, dt: float) -> float:
        """
        Compute the Freidlin-Wentzell action for a discretized path.
        S = 1/2 * integral |dz/dt - F(z)|^2 dt
        """
        n_images = path.shape[1]
        action = 0.0
        for i in range(n_images - 1):
            z_curr = path[:, i]
            z_next = path[:, i+1]
            dzdt = (z_next - z_curr) / dt
            
            # Evaluate drift F(z) at midpoint
            z_mid = 0.5 * (z_curr + z_next)
            # Use t=0 for autonomous approximation
            drift = self.F(0.0, z_mid) 
            
            diff = dzdt - drift
            action += 0.5 * np.sum(diff**2) * dt
            
        return action

    def _reparameterize_string(self, path: np.ndarray) -> np.ndarray:
        """Enforce equal arc-length spacing between images on the string."""
        n_images = path.shape[1]
        
        # Compute arc lengths
        diffs = np.diff(path, axis=1)
        seg_lengths = np.linalg.norm(diffs, axis=0)
        s = np.insert(np.cumsum(seg_lengths), 0, 0.0)
        
        if s[-1] == 0:
            return path
            
        s_norm = s / s[-1]
        s_uniform = np.linspace(0, 1, n_images)
        
        # Interpolate each dimension
        new_path = np.zeros_like(path)
        for i in range(self.dim):
            interp_func = interp1d(s_norm, path[i, :], kind='linear')
            new_path[i, :] = interp_func(s_uniform)
            
        return new_path

    def compute_minimum_action_path(self, z_start: np.ndarray, z_end: np.ndarray,
                                     n_images: int = 40, max_iter: int = 100,
                                     active_indices: List[int] = None) -> Tuple[np.ndarray, float]:
        """
        Compute the MAP using the String Method.
        Supports lazy/subspace optimization by providing active_indices.
        """
        logger.info("Initializing string...")
        # Initialize linear string
        alphas = np.linspace(0, 1, n_images)
        path = np.zeros((self.dim, n_images))
        for i in range(self.dim):
            path[i, :] = z_start[i] + alphas * (z_end[i] - z_start[i])
            
        if active_indices is None:
            active_indices = list(range(self.dim))
            
        logger.info(f"Optimizing MAP over {len(active_indices)} active dimensions...")
        
        # String Method iteration
        tau = 0.01  # Artificial time step for string evolution
        for step in range(max_iter):
            # 1. Evolve internal images according to gradient of action
            # dz_i/dtau = -(dz_i/dt - F(z_i)) + ... (simplified gradient flow)
            new_path = np.copy(path)
            for i in range(1, n_images - 1):
                z_curr = path[:, i]
                # Forward difference approx for gradient
                drift = self.F(0.0, z_curr)
                # Only update active subspace
                for dim_idx in active_indices:
                    # Simple gradient descent step towards drift
                    # A true string method would compute the normal component of F
                    new_path[dim_idx, i] += tau * drift[dim_idx]
            
            # 2. Reparameterize
            path = self._reparameterize_string(new_path)
            
        final_action = self._action_integrand(path, self.dt)
        logger.info(f"MAP computation complete. Final Action: {final_action:.4f}")
        return path, final_action

    def get_saddle_point(self, path: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Identifies the highest-energy point on the MAP.
        Since we don't have a direct energy landscape U(z), we use the point 
        where the drift F(z) magnitude is minimal (fixed point/saddle).
        """
        n_images = path.shape[1]
        drift_mags = np.zeros(n_images)
        for i in range(n_images):
            drift = self.F(0.0, path[:, i])
            drift_mags[i] = np.linalg.norm(drift)
            
        # The saddle is typically the point with smallest drift between two attractors
        # Excluding endpoints
        saddle_idx = np.argmin(drift_mags[1:-1]) + 1
        return saddle_idx, path[:, saddle_idx]

    def get_realignment_targets(self, path: np.ndarray, metric: str = 'gradient') -> List[Tuple[int, float]]:
        """
        Ranks state variables by their importance along the path.
        'gradient': total absolute change along the path.
        """
        if metric == 'gradient':
            total_change = np.sum(np.abs(np.diff(path, axis=1)), axis=1)
            ranking = [(i, float(val)) for i, val in enumerate(total_change)]
            ranking.sort(key=lambda x: x[1], reverse=True)
            return ranking
        return []
