"""
Geometric Pathways Module — Project Confluence
===============================================

Implements Freidlin-Wentzell Minimum Action Pathway (MAP) computation
for identifying therapeutic realignment trajectories between cancer and
healthy attractor states.

The String Method (E, Ren, Vanden-Eijnden, 2007) evolves a discretized
path ("string of images") by alternating between:
  1. Evolving each image toward the minimum energy path via the negative
     gradient of the Freidlin-Wentzell action.
  2. Reparameterizing (redistributing) images to maintain equal arc-length
     spacing along the string.

Optimizations:
  - Attractor caching to avoid redundant steady-state integrations
  - Progressive "Lazy" subspace pathfinding for high-dimensional efficiency
  - Convergence tracking with early stopping

References:
    E, Ren, Vanden-Eijnden (2007) - Simplified and improved string method
    Freidlin, Wentzell (2012) - Random Perturbations of Dynamical Systems
    Heymann, Vanden-Eijnden (2008) - Geometric minimum action method
"""

import numpy as np
from scipy.interpolate import interp1d
import logging
from typing import Callable, Dict, Tuple, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attractor cache (module-level dict; numpy arrays are unhashable for lru_cache)
# ---------------------------------------------------------------------------
_attractor_cache: Dict[str, np.ndarray] = {}


class FreidlinWentzellOptimizer:
    """
    Computes the minimum action path between attractor states in phase space
    using the String Method with reparameterization.
    """

    def __init__(self, ode_system, dt: float = 0.1):
        """
        Parameters
        ----------
        ode_system : ComplexAttractorODE
            An initialised ODE system whose .rhs(t, z) defines the drift field.
        dt : float
            Discretization time step for action integral evaluation.
        """
        self.sys = ode_system
        self.F = ode_system.rhs
        self.dim = ode_system.DIM
        self.dt = dt

    # ------------------------------------------------------------------
    # Attractor retrieval with caching
    # ------------------------------------------------------------------
    def get_attractor(self, label: str, t_settle: float = 300.0) -> np.ndarray:
        """
        Integrate the ODE to steady state and cache the result.

        Parameters
        ----------
        label : str
            A unique key for this attractor (e.g. "TNBC", "healthy").
        t_settle : float
            Integration horizon (hours/days, depending on ODE time-scale).

        Returns
        -------
        z_attractor : ndarray, shape (DIM,)
            The state at the end of the settling integration.
        """
        if label in _attractor_cache:
            return _attractor_cache[label].copy()

        logger.info("Computing and caching attractor: %s", label)
        z0 = self.sys.healthy_initial_state()
        res = self.sys.solve(z0=z0, t_span=(0, t_settle), dt_eval=1.0)
        z_final = res["z"][:, -1]
        _attractor_cache[label] = z_final.copy()
        return z_final

    # ------------------------------------------------------------------
    # Freidlin-Wentzell action functional
    # ------------------------------------------------------------------
    def compute_action(self, path: np.ndarray) -> float:
        """
        Evaluate the Freidlin-Wentzell action along a discretized path.

            S[φ] = (1/2) ∫ |dφ/dt − F(φ)|² dt

        Parameters
        ----------
        path : ndarray, shape (DIM, n_images)

        Returns
        -------
        action : float
        """
        n_images = path.shape[1]
        dt = self.dt
        action = 0.0
        for k in range(n_images - 1):
            z_k = path[:, k]
            z_next = path[:, k + 1]
            velocity = (z_next - z_k) / dt

            # Drift evaluated at the midpoint for second-order accuracy
            z_mid = 0.5 * (z_k + z_next)
            drift = self.F(0.0, z_mid)

            residual = velocity - drift
            action += 0.5 * np.dot(residual, residual) * dt
        return float(action)

    # ------------------------------------------------------------------
    # Quasi-potential profile along the path
    # ------------------------------------------------------------------
    def compute_energy_profile(self, path: np.ndarray) -> np.ndarray:
        """
        Estimate a quasi-potential energy at each image by accumulating
        the local action from the start of the string.

        Returns
        -------
        energy : ndarray, shape (n_images,)
        """
        n_images = path.shape[1]
        energy = np.zeros(n_images)
        dt = self.dt
        for k in range(n_images - 1):
            z_k = path[:, k]
            z_next = path[:, k + 1]
            velocity = (z_next - z_k) / dt
            z_mid = 0.5 * (z_k + z_next)
            drift = self.F(0.0, z_mid)
            residual = velocity - drift
            energy[k + 1] = energy[k] + 0.5 * np.dot(residual, residual) * dt
        return energy

    # ------------------------------------------------------------------
    # Arc-length reparameterization
    # ------------------------------------------------------------------
    @staticmethod
    def _reparameterize(path: np.ndarray) -> np.ndarray:
        """Redistribute images to enforce equal arc-length spacing."""
        dim, n_images = path.shape
        seg = np.linalg.norm(np.diff(path, axis=1), axis=0)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = s[-1]
        if total < 1e-14:
            return path.copy()

        s_uniform = np.linspace(0.0, total, n_images)
        new_path = np.zeros_like(path)
        for d in range(dim):
            new_path[d, :] = np.interp(s_uniform, s, path[d, :])
        return new_path

    # ------------------------------------------------------------------
    # String Method core
    # ------------------------------------------------------------------
    def compute_minimum_action_path(
        self,
        z_start: np.ndarray,
        z_end: np.ndarray,
        n_images: int = 60,
        max_iter: int = 200,
        tau: float = 0.005,
        tol: float = 1e-4,
        active_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Compute the MAP using the simplified String Method.

        Parameters
        ----------
        z_start, z_end : ndarray, shape (DIM,)
            Boundary states (cancer attractor, healthy attractor).
        n_images : int
            Number of discrete images along the string.
        max_iter : int
            Maximum number of string evolution iterations.
        tau : float
            Pseudo-time step for the gradient-descent evolution of internal
            images.  Smaller values improve stability at the cost of speed.
        tol : float
            Relative change in action below which early stopping is triggered.
        active_indices : list of int, optional
            If provided, only evolve these dimensions (lazy subspace mode).
            Remaining dimensions stay on the linear interpolation.

        Returns
        -------
        path : ndarray, shape (DIM, n_images)
        action : float
        history : list of float
            Action value at each iteration (for convergence diagnostics).
        """
        if active_indices is None:
            active_indices = list(range(self.dim))

        logger.info(
            "String Method: %d images, %d active dims, max %d iters",
            n_images, len(active_indices), max_iter,
        )

        # --- Initialize: linear interpolation between endpoints ---
        alphas = np.linspace(0.0, 1.0, n_images)
        path = np.outer(z_start, 1.0 - alphas) + np.outer(z_end, alphas)

        history: List[float] = []
        prev_action = self.compute_action(path)
        history.append(prev_action)

        for step in range(max_iter):
            # 1. Evolve internal images (endpoints are pinned)
            new_path = path.copy()
            for k in range(1, n_images - 1):
                drift = self.F(0.0, path[:, k])
                for d in active_indices:
                    new_path[d, k] += tau * drift[d]

            # 2. Reparameterize to keep equal spacing
            path = self._reparameterize(new_path)

            # 3. Convergence check
            cur_action = self.compute_action(path)
            history.append(cur_action)
            rel_change = abs(cur_action - prev_action) / (abs(prev_action) + 1e-12)
            if rel_change < tol and step > 10:
                logger.info(
                    "Converged at step %d (action=%.6f, rel_change=%.2e)",
                    step, cur_action, rel_change,
                )
                break
            prev_action = cur_action

        final_action = self.compute_action(path)
        logger.info("MAP complete. Final action = %.6f", final_action)
        return path, final_action, history

    # ------------------------------------------------------------------
    # Saddle-point detection
    # ------------------------------------------------------------------
    def get_saddle_point(self, path: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """
        Identify the transition state (highest quasi-potential point)
        on the MAP.

        Returns
        -------
        idx : int
            Image index of the saddle point.
        z_saddle : ndarray, shape (DIM,)
        energy : float
            Quasi-potential at the saddle.
        """
        energy = self.compute_energy_profile(path)
        idx = int(np.argmax(energy[1:-1])) + 1  # exclude pinned endpoints
        return idx, path[:, idx].copy(), float(energy[idx])

    # ------------------------------------------------------------------
    # Realignment target ranking
    # ------------------------------------------------------------------
    def get_realignment_targets(
        self, path: np.ndarray, state_names: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Rank state variables by total absolute displacement along the MAP.

        Parameters
        ----------
        path : ndarray, shape (DIM, n_images)
        state_names : list of str, optional

        Returns
        -------
        ranking : list of dict
            Sorted by displacement (descending).  Each entry contains
            'index', 'name', and 'displacement'.
        """
        displacement = np.sum(np.abs(np.diff(path, axis=1)), axis=1)
        if state_names is None:
            state_names = [f"Var_{i}" for i in range(self.dim)]

        ranking = [
            {"index": int(i), "name": state_names[i], "displacement": float(displacement[i])}
            for i in range(self.dim)
        ]
        ranking.sort(key=lambda r: r["displacement"], reverse=True)
        return ranking

    # ------------------------------------------------------------------
    # Path tangent and per-phase directional vectors
    # ------------------------------------------------------------------
    def get_path_tangents(self, path: np.ndarray) -> np.ndarray:
        """
        Return unit tangent vectors at each internal image.

        Returns
        -------
        tangents : ndarray, shape (DIM, n_images - 2)
        """
        tangents = path[:, 2:] - path[:, :-2]  # central difference
        norms = np.linalg.norm(tangents, axis=0, keepdims=True) + 1e-14
        return tangents / norms
