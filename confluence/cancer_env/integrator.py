"""Thin integrator over the existing ``CancerODE.rhs`` — not a new vector field.

All trajectories come from ``CancerODE.pack`` / ``rhs`` / ``unpack``.
Host-death and near-eradication are ``solve_ivp`` events defined on that RHS.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from confluence.cancer_env.ode_system import (
    DEFAULT_SOLVER,
    CancerODE,
    host_death_event,
    near_eradication_event,
)

__all__ = [
    "DEFAULT_SOLVER",
    "host_death_event",
    "integrate_cancer_ode",
    "near_eradication_event",
]


def integrate_cancer_ode(
    ode: CancerODE,
    x0: np.ndarray,
    c0: np.ndarray,
    u_of_t,
    t_span: Tuple[float, float],
    n_eval: int = 200,
    method: str = DEFAULT_SOLVER,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_step: Optional[float] = None,
    events: bool = True,
) -> Dict[str, np.ndarray]:
    """Integrate ``ode.rhs`` only. Do not substitute a parallel F(t,X,U)."""
    return ode.integrate(
        x0,
        c0,
        u_of_t,
        t_span,
        n_eval=n_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        events=events,
    )
