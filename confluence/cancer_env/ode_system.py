"""Coupled nonlinear ODE for the 11-D cancer microenvironment.

Latent state X:
    T_s, T_r, I_act, I_exh, S_fib, L, O, G, C_tgfb, C_ifng, H

Design:
    * logistic tumor growth with Lotka–Volterra competition
    * phenotypic switch ε_switch(C_drugs, L) attenuated by HDAC occupancy
    * immune kill attenuated by stroma
    * exhaustion γ_exh(TGF-β, L, PD1 occupancy)
    * lactate production / MCT1-modulated clearance
    * stroma driven by TGF-β
    * host health H ∈ [0, 1]; H ≤ 0.2 is terminal toxicity

The RHS is Lipschitz on a compact box (clip + saturating Hill terms) so
explicit RK4 and scipy LSODA stay finite for ≥1000 interactive steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from confluence.contracts import LATENT_NAMES, LatentCancerState
from confluence.pharmacology.pk_pd_model import PKPDModel, hill_occupancy

STATE_NAMES = LATENT_NAMES
STATE_INDEX = {name: i for i, name in enumerate(STATE_NAMES)}
DIM = 11


def _sat(x: float, k: float) -> float:
    x = max(0.0, float(x))
    return x / (k + x + 1e-12)


@dataclass
class ArchetypeParams:
    """Biophysical knobs that distinguish cancer archetypes."""

    name: str
    display_name: str
    r_s: float = 0.18
    r_r: float = 0.12
    k_carry: float = 4.0
    alpha_rs: float = 0.85
    alpha_sr: float = 0.55
    kappa_immune: float = 0.35
    kappa_kinase: float = 0.45
    resist_immune_factor: float = 0.45
    resist_kinase_factor: float = 0.12
    eps0: float = 0.015
    lambda_lactate: float = 1.4
    k_lactate: float = 1.0
    rho_immune: float = 0.20
    gamma0: float = 0.08
    delta_act: float = 0.04
    delta_exh: float = 0.03
    sigma_stroma: float = 0.12
    mu_stroma: float = 0.04
    p_lactate: float = 0.22
    cl_lactate: float = 0.18
    supply_o2: float = 0.35
    consume_o2: float = 0.12
    supply_glc: float = 0.40
    consume_glc: float = 0.14
    p_tgfb: float = 0.16
    cl_tgfb: float = 0.12
    p_ifng: float = 0.22
    cl_ifng: float = 0.20
    r_host: float = 0.05
    kappa_burden: float = 0.04
    kappa_tox: float = 0.10
    x0: Tuple[float, ...] = (
        0.85, 0.15, 0.35, 0.12, 0.25, 0.40, 0.70, 1.10, 0.35, 0.30, 0.92,
    )
    notes: str = ""


class CancerODE:
    """11-D microenvironment + 5-D PK, integrated together."""

    def __init__(
        self,
        params: ArchetypeParams,
        pk: Optional[PKPDModel] = None,
    ):
        self.params = params
        self.pk = pk or PKPDModel()
        self.dim = DIM + self.pk.n_drugs

    def pack(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(x, dtype=float), np.asarray(c, dtype=float)])

    def unpack(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return z[:DIM].copy(), z[DIM:].copy()

    def initial_state(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.array(self.params.x0, dtype=float)
        return x, self.pk.zeros()

    def clip_state(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=float).copy()
        y[0] = max(y[0], 0.0)  # T_s
        y[1] = max(y[1], 0.0)  # T_r
        y[2] = np.clip(y[2], 0.0, 1.2)  # I_act
        y[3] = np.clip(y[3], 0.0, 1.2)  # I_exh
        y[4] = np.clip(y[4], 0.0, 1.0)  # S_fib
        y[5] = max(y[5], 0.0)  # L
        y[6] = np.clip(y[6], 0.0, 2.5)  # O
        y[7] = np.clip(y[7], 0.0, 3.5)  # G
        y[8] = max(y[8], 0.0)  # C_tgfb
        y[9] = max(y[9], 0.0)  # C_ifng
        y[10] = np.clip(y[10], 0.0, 1.0)  # H
        return y

    def occupancies(self, c: np.ndarray) -> Dict[str, float]:
        return self.pk.occupancies(c)

    def rhs_cancer(self, x: np.ndarray, occ: Mapping[str, float], tox_load: float) -> np.ndarray:
        p = self.params
        T_s, T_r, I_act, I_exh, S_fib, L, O, G, C_tgfb, C_ifng, H = [float(v) for v in x]
        T_s = max(T_s, 0.0)
        T_r = max(T_r, 0.0)
        I_act = max(I_act, 0.0)
        I_exh = max(I_exh, 0.0)
        S_fib = float(np.clip(S_fib, 0.0, 1.0))
        L = max(L, 0.0)
        O = max(O, 0.0)
        G = max(G, 0.0)
        C_tgfb = max(C_tgfb, 0.0)
        C_ifng = max(C_ifng, 0.0)
        H = float(np.clip(H, 0.0, 1.0))

        e_pd1 = float(occ.get("anti_pd1", 0.0))
        e_tgfbi = float(occ.get("tgfb_inhibitor", 0.0))
        e_mct1 = float(occ.get("mct1", 0.0))
        e_hdac = float(occ.get("hdac", 0.0))
        e_kin = float(occ.get("targeted_kinase", 0.0))

        burden = T_s + T_r
        nutrient = _sat(G, 0.35) * (0.35 + 0.65 * _sat(O, 0.25))
        crowding_s = (T_s + p.alpha_rs * T_r) / max(p.k_carry, 1e-8)
        crowding_r = (T_r + p.alpha_sr * T_s) / max(p.k_carry, 1e-8)

        stroma_shield = 1.0 - 0.85 * S_fib
        ifng_boost = 1.0 + 0.6 * _sat(C_ifng, 0.4)
        immune_kill = p.kappa_immune * I_act * stroma_shield * ifng_boost

        # Phenotypic switch: lactate + cytotoxic pressure, attenuated by HDAC.
        drug_pressure = 0.35 * e_kin + 0.15 * e_pd1
        eps_switch = (
            p.eps0
            * (1.0 + p.lambda_lactate * _sat(L, p.k_lactate))
            * (1.0 + 1.2 * drug_pressure)
            * (1.0 - 0.85 * e_hdac)
        )
        eps_switch = float(np.clip(eps_switch, 0.0, 0.25))

        dT_s = (
            p.r_s * T_s * (1.0 - crowding_s) * nutrient
            - immune_kill * T_s
            - p.kappa_kinase * e_kin * T_s
            - eps_switch * T_s
        )
        dT_r = (
            p.r_r * T_r * (1.0 - crowding_r) * nutrient
            - immune_kill * p.resist_immune_factor * T_r
            - p.kappa_kinase * e_kin * p.resist_kinase_factor * T_r
            + eps_switch * T_s
        )

        # Exhaustion γ_exh(TGF-β, lactate, PD-1 occupancy)
        gamma_exh = (
            p.gamma0
            * (1.0 + 1.6 * _sat(C_tgfb, 0.35) * (1.0 - 0.75 * e_tgfbi))
            * (1.0 + 1.1 * _sat(L, 0.8))
            * (1.0 - 0.80 * e_pd1)
        )
        room = max(0.0, 1.15 - I_act - I_exh)
        dI_act = p.rho_immune * _sat(C_ifng, 0.3) * room - gamma_exh * I_act - p.delta_act * I_act
        dI_exh = gamma_exh * I_act - p.delta_exh * I_exh

        # Stroma driven by TGF-β (reduced when TGF-β is inhibited)
        tgfb_drive = _sat(C_tgfb, 0.3) * (1.0 - 0.7 * e_tgfbi)
        dS = p.sigma_stroma * tgfb_drive * (1.0 - S_fib) - p.mu_stroma * S_fib

        # Lactate: production from tumor (Warburg-like, worse when hypoxic),
        # clearance boosted by MCT1 occupancy (export + systemic sink).
        warburg = 0.45 + 0.55 * (1.0 - _sat(O, 0.4))
        dL = p.p_lactate * burden * warburg - p.cl_lactate * (1.0 + 1.8 * e_mct1) * L

        dO = p.supply_o2 * (1.0 - 0.55 * S_fib) - p.consume_o2 * burden * O - 0.08 * O
        dG = p.supply_glc - p.consume_glc * burden * G - 0.06 * G

        dTgf = p.p_tgfb * (0.4 * burden + 0.8 * S_fib) - p.cl_tgfb * (1.0 + 2.0 * e_tgfbi) * C_tgfb
        dIfn = p.p_ifng * I_act - p.cl_ifng * C_ifng

        tox = max(0.0, tox_load)
        dH = p.r_host * (1.0 - H) - p.kappa_burden * burden * H - p.kappa_tox * tox * H
        if H <= 0.2:
            # Terminal toxicity: freeze recovery, allow residual decay only.
            dH = min(dH, -0.01 * H)

        return np.array(
            [dT_s, dT_r, dI_act, dI_exh, dS, dL, dO, dG, dTgf, dIfn, dH],
            dtype=float,
        )

    def rhs(self, t: float, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        x, c = self.unpack(z)
        x = self.clip_state(x)
        c = np.maximum(c, 0.0)
        occ = self.occupancies(c)
        tox = self.pk.toxicity_load(c)
        dx = self.rhs_cancer(x, occ, tox)
        dc = self.pk.rhs(c, u)
        return np.concatenate([dx, dc])

    def step(
        self,
        x: np.ndarray,
        c: np.ndarray,
        u: np.ndarray,
        dt: float,
        method: str = "RK45",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance one interval with scipy; fallback RK4 if the solver rejects."""
        z0 = self.pack(self.clip_state(x), np.maximum(c, 0.0))
        u = np.asarray(u, dtype=float)

        def fun(t, z):
            return self.rhs(t, z, u)

        try:
            sol = solve_ivp(
                fun,
                (0.0, dt),
                z0,
                method=method,
                rtol=1e-6,
                atol=1e-8,
                max_step=max(dt / 2.0, 1e-3),
            )
            if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                z1 = sol.y[:, -1]
            else:
                z1 = self._rk4(z0, u, dt)
        except Exception:
            z1 = self._rk4(z0, u, dt)

        x1, c1 = self.unpack(z1)
        return self.clip_state(x1), np.maximum(c1, 0.0)

    def _rk4(self, z: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.rhs(0.0, z, u)
        k2 = self.rhs(0.0, z + 0.5 * dt * k1, u)
        k3 = self.rhs(0.0, z + 0.5 * dt * k2, u)
        k4 = self.rhs(0.0, z + dt * k3, u)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def integrate(
        self,
        x0: np.ndarray,
        c0: np.ndarray,
        u_of_t,
        t_span: Tuple[float, float],
        n_eval: int = 200,
    ) -> Dict[str, np.ndarray]:
        z0 = self.pack(self.clip_state(x0), np.maximum(c0, 0.0))
        t_eval = np.linspace(t_span[0], t_span[1], n_eval)

        def fun(t, z):
            return self.rhs(t, z, np.asarray(u_of_t(t), dtype=float))

        sol = solve_ivp(fun, t_span, z0, t_eval=t_eval, method="LSODA", rtol=1e-6, atol=1e-8)
        xs = np.array([self.clip_state(sol.y[:DIM, i]) for i in range(sol.y.shape[1])]).T
        cs = np.maximum(sol.y[DIM:, :], 0.0)
        return {"t": sol.t, "x": xs, "c": cs, "success": bool(sol.success)}

    def to_latent(self, x: np.ndarray, t: float = 0.0) -> LatentCancerState:
        return LatentCancerState.from_vector(self.clip_state(x), t=t)
