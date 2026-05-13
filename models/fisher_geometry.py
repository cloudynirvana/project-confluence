"""
Fisher Information Geometry Module — Project Confluence
=======================================================

Computes the Fisher Information Matrix (FIM) for the 15D ODE system,
identifies stiff and sloppy parameter combinations via eigendecomposition,
and calculates geodesic distances on the resulting Riemannian manifold.

The FIM is computed by finite-difference sensitivity analysis of the ODE
outputs with respect to model parameters.  Each column of the Jacobian
requires two ODE integrations (central difference), yielding O(2 × N_params)
total simulations.  These are parallelised using ThreadPoolExecutor
(ProcessPoolExecutor cannot pickle bound methods).

Theoretical basis:
    Transtrum et al. (2015) — Sloppiness and emergent theories
    Transtrum & Qiu (2014) — Model reduction by manifold boundaries
    Amari & Nagaoka (2000) — Methods of Information Geometry
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm
import concurrent.futures
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FisherManifoldAnalyzer:
    """
    Analyses the Riemannian geometry of the biological model manifold
    using the Fisher Information Metric.
    """

    def __init__(self, ode_system, base_params, t_span=(0, 50), dt=1.0,
                 observable_indices: Optional[List[int]] = None):
        """
        Parameters
        ----------
        ode_system : ComplexAttractorODE
            Initialised ODE system.
        base_params : ExtendedParams (dataclass)
            Baseline parameter set around which the FIM is computed.
        t_span : tuple
            Integration window.
        dt : float
            Output sampling interval.
        observable_indices : list of int, optional
            Which state variables are "observable" for FIM computation.
            Defaults to all 15.
        """
        self.ode_system = ode_system
        self.base_params = base_params
        self.t_span = t_span
        self.t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
        self.param_dict = asdict(base_params)
        self.param_names = list(self.param_dict.keys())
        self.dim_p = len(self.param_names)
        self.observable_indices = observable_indices  # None → all

        # Cache the baseline trajectory
        self._baseline_traj: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------
    def _simulate(self, param_overrides: Dict[str, float]) -> np.ndarray:
        """Integrate the ODE with specific parameter overrides and return trajectory."""
        merged = {**self.param_dict, **param_overrides}
        new_params = type(self.base_params)(**merged)

        sys = type(self.ode_system)(
            params=new_params,
            use_nonlinear=getattr(self.ode_system, "use_nonlinear", True),
            use_immune=getattr(self.ode_system, "use_immune", True),
            use_microenv=getattr(self.ode_system, "use_microenv", True),
        )
        z0 = sys.healthy_initial_state()
        sol = solve_ivp(
            sys.rhs, self.t_span, z0,
            t_eval=self.t_eval, method="LSODA",
            rtol=1e-6, atol=1e-8,
        )
        if not sol.success:
            return np.zeros((sys.DIM, len(self.t_eval)))

        traj = sol.y
        if self.observable_indices is not None:
            traj = traj[self.observable_indices, :]
        return traj

    def _get_baseline(self) -> np.ndarray:
        if self._baseline_traj is None:
            self._baseline_traj = self._simulate({})
        return self._baseline_traj

    # ------------------------------------------------------------------
    # Per-parameter sensitivity (called inside threads)
    # ------------------------------------------------------------------
    def _sensitivity_column(self, param_name: str, h: float) -> np.ndarray:
        """Central-difference derivative of the observable trajectory."""
        val = self.param_dict[param_name]
        traj_p = self._simulate({param_name: val + h})
        traj_m = self._simulate({param_name: val - h})
        deriv = (traj_p - traj_m) / (2.0 * h)
        return deriv.flatten()

    # ------------------------------------------------------------------
    # FIM computation
    # ------------------------------------------------------------------
    def compute_fim(self, perturbation: float = 1e-4,
                    max_workers: Optional[int] = None) -> np.ndarray:
        """
        Compute the Fisher Information Matrix via parallelised finite
        differences.

            FIM_ij = Σ_t  (∂y/∂θ_i)(t) · (∂y/∂θ_j)(t)

        Parameters
        ----------
        perturbation : float
            Relative perturbation size.  The actual step for parameter θ_k is
            max(perturbation, |θ_k| × perturbation).
        max_workers : int, optional
            Number of threads.  Defaults to min(dim_p, 8).

        Returns
        -------
        FIM : ndarray, shape (dim_p, dim_p)
        """
        t0 = time.perf_counter()
        logger.info("Computing FIM for %d parameters...", self.dim_p)

        steps = {
            name: max(perturbation, abs(self.param_dict[name]) * perturbation)
            for name in self.param_names
        }

        jac_cols: Dict[str, np.ndarray] = {}

        # ThreadPoolExecutor avoids the pickle limitation of ProcessPoolExecutor
        workers = max_workers or min(self.dim_p, 8)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._sensitivity_column, name, steps[name]): name
                for name in self.param_names
            }
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                jac_cols[name] = future.result()

        # Stack columns in parameter order
        J = np.column_stack([jac_cols[n] for n in self.param_names])
        FIM = J.T @ J

        elapsed = time.perf_counter() - t0
        logger.info("FIM computed in %.1f s  (cond ≈ %.2e)",
                     elapsed, np.linalg.cond(FIM))
        return FIM

    # ------------------------------------------------------------------
    # Stiff / sloppy decomposition
    # ------------------------------------------------------------------
    def identify_stiff_sloppy(self, fim: np.ndarray,
                               stiff_frac: float = 0.15,
                               sloppy_frac: float = 0.50) -> Dict:
        """
        Eigendecompose the FIM and classify parameter directions.

        Parameters
        ----------
        fim : ndarray
        stiff_frac : float
            Fraction of eigenvalues considered "stiff" (top).
        sloppy_frac : float
            Fraction considered "sloppy" (bottom).

        Returns
        -------
        dict with keys: eigenvalues, stiff, sloppy, condition_number,
                        eigenvalue_spectrum (log10 values for plotting).
        """
        eigenvalues, eigenvectors = np.linalg.eigh(fim)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        n_stiff = max(1, int(self.dim_p * stiff_frac))
        n_sloppy = max(1, int(self.dim_p * sloppy_frac))

        def _direction_info(i: int) -> Dict:
            weights = eigenvectors[:, i]
            top3_idx = np.argsort(np.abs(weights))[::-1][:3]
            return {
                "eigenvalue": float(eigenvalues[i]),
                "log10_eigenvalue": float(np.log10(abs(eigenvalues[i]) + 1e-30)),
                "dominant_param": self.param_names[top3_idx[0]],
                "weight": float(weights[top3_idx[0]]),
                "top3_params": [
                    {"param": self.param_names[j], "weight": float(weights[j])}
                    for j in top3_idx
                ],
            }

        stiff = [_direction_info(i) for i in range(n_stiff)]
        sloppy = [_direction_info(i) for i in range(self.dim_p - n_sloppy, self.dim_p)]

        spectrum = np.log10(np.abs(eigenvalues) + 1e-30)

        return {
            "eigenvalues": eigenvalues,
            "eigenvalue_spectrum": spectrum,
            "stiff": stiff,
            "sloppy": sloppy,
            "condition_number": float(eigenvalues[0] / (abs(eigenvalues[-1]) + 1e-12)),
            "n_stiff": n_stiff,
            "n_sloppy": n_sloppy,
        }

    # ------------------------------------------------------------------
    # Geodesic distance (Mahalanobis approximation)
    # ------------------------------------------------------------------
    def geodesic_distance(self, fim: np.ndarray,
                          state_a: np.ndarray,
                          state_b: np.ndarray) -> float:
        """
        Approximate geodesic distance between two states using the FIM
        as a Riemannian metric tensor.

        For states that live in the observation space (R^{n_obs × n_t}),
        the linearised geodesic distance is:

            d² = (a − b)ᵀ · G · (a − b)

        where G = J^T J = FIM, evaluated in the *observation-space*
        inner product.  When states are given as 15D vectors (not full
        trajectories), we fall back to the Mahalanobis distance using
        the top eigenvalues of the FIM to weight the dimensions.

        Parameters
        ----------
        fim : ndarray, shape (dim_p, dim_p)
        state_a, state_b : ndarray, shape (DIM,) or (dim_p,)

        Returns
        -------
        distance : float
        """
        diff = state_a - state_b

        # If state dimension matches FIM dimension, use full Mahalanobis
        if len(diff) == fim.shape[0]:
            # Regularise to ensure positive-definiteness
            fim_reg = fim + 1e-8 * np.eye(fim.shape[0])
            return float(np.sqrt(diff @ fim_reg @ diff))

        # Otherwise, use Euclidean (state-space) distance
        return float(np.linalg.norm(diff))

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    def generate_report(self, analysis: Dict) -> str:
        """Human-readable summary of stiff/sloppy analysis."""
        lines = [
            "=" * 60,
            "FISHER INFORMATION GEOMETRY REPORT",
            "=" * 60,
            f"Parameters analysed : {self.dim_p}",
            f"Condition number    : {analysis['condition_number']:.2e}",
            f"Stiff directions    : {analysis['n_stiff']}",
            f"Sloppy directions   : {analysis['n_sloppy']}",
            "",
            "STIFF (high therapeutic leverage):",
        ]
        for s in analysis["stiff"]:
            lines.append(
                f"  λ={s['eigenvalue']:.4e}  →  {s['dominant_param']} "
                f"(w={s['weight']:.3f})"
            )
        lines.append("")
        lines.append("SLOPPY (low therapeutic leverage):")
        for s in analysis["sloppy"][:5]:
            lines.append(
                f"  λ={s['eigenvalue']:.4e}  →  {s['dominant_param']} "
                f"(w={s['weight']:.3f})"
            )
        lines.append("=" * 60)
        return "\n".join(lines)
