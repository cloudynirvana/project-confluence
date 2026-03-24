"""
Project Confluence — Clonal Evolution Proof of Concept Simulation
=================================================================
Compares three therapeutic strategies on a Lotka-Volterra
competition model of Sensitive (S) vs Resistant (R) tumor clones:

  1. MTD (Maximum Tolerated Dose)  — Standard of Care
  2. Fixed Low Dose                — Naive reduction
  3. Confluence Adaptive           — Robust pulse therapy

Outputs a matplotlib figure with S(t), R(t), V(t), and u(t) for each.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Biological Parameters ──────────────────────────────────────────
r_s = 0.03       # Sensitive growth rate (per day)
r_r = 0.015      # Resistant growth rate  (slower but unkillable)
K   = 1e6        # Carrying capacity (cells)
alpha_rs = 0.5   # Competitive effect of R on S
alpha_sr = 0.9   # Competitive effect of S on R  ← the KEY lever
delta    = 0.8   # Drug kill-rate efficiency on S
u_max    = 1.0   # Max normalised dose

# ── Initial Conditions ─────────────────────────────────────────────
S0 = 8e5         # 80% sensitive
R0 = 2e5         # 20% resistant
T  = 180         # Simulation horizon (days)

# ── Dosing Strategies ──────────────────────────────────────────────

def dose_mtd(t):
    """Maximum tolerated dose — constant full blast."""
    return u_max

def dose_low(t):
    """Fixed low dose — 30% of max, continuous."""
    return 0.3 * u_max

def dose_confluence_adaptive(t, S_current, R_current):
    """
    Confluence Adaptive Strategy:
    - Dose ON  when tumor volume exceeds 50% of K  (need to push it down)
    - Dose OFF when tumor volume drops below 30% of K (let S recover to suppress R)
    Robust bound: never exceed 70% of u_max to preserve renal safety margin.
    """
    V = S_current + R_current
    if V > 0.50 * K:
        return 0.70 * u_max   # Robustly bounded dose
    elif V < 0.30 * K:
        return 0.0            # Drug holiday — let S regrow
    else:
        return 0.35 * u_max   # Maintenance

# ── ODE System ─────────────────────────────────────────────────────

def tumor_ode(t, y, dose_fn, adaptive=False):
    S, R = y
    S = max(S, 0)
    R = max(R, 0)

    if adaptive:
        u = dose_fn(t, S, R)
    else:
        u = dose_fn(t)

    dSdt = r_s * S * (1 - (S + alpha_rs * R) / K) - delta * u * S
    dRdt = r_r * R * (1 - (R + alpha_sr * S) / K)
    return [dSdt, dRdt]

# ── Run Simulations ────────────────────────────────────────────────

t_eval = np.linspace(0, T, 2000)

strategies = {
    "MTD (Standard Care)": (dose_mtd, False),
    "Fixed Low Dose":      (dose_low, False),
    "Confluence Adaptive":  (dose_confluence_adaptive, True),
}

results = {}
for name, (dfn, is_adaptive) in strategies.items():
    sol = solve_ivp(
        tumor_ode, [0, T], [S0, R0],
        args=(dfn, is_adaptive),
        t_eval=t_eval, method="RK45", max_step=0.5
    )
    # Reconstruct dosage curve
    doses = []
    for i, ti in enumerate(sol.t):
        Si, Ri = sol.y[0][i], sol.y[1][i]
        if is_adaptive:
            doses.append(dfn(ti, Si, Ri))
        else:
            doses.append(dfn(ti))
    results[name] = {
        "t": sol.t,
        "S": sol.y[0],
        "R": sol.y[1],
        "V": sol.y[0] + sol.y[1],
        "u": np.array(doses),
    }

# ── Plotting ───────────────────────────────────────────────────────

colors = {
    "MTD (Standard Care)": "#e74c3c",
    "Fixed Low Dose":      "#f39c12",
    "Confluence Adaptive":  "#2ecc71",
}

fig = plt.figure(figsize=(16, 12))
fig.suptitle(
    "Project Confluence — Clonal Evolution PoC\n"
    "Adaptive Therapy vs Standard Care under Biological Uncertainty",
    fontsize=15, fontweight="bold", y=0.98
)

gs = GridSpec(3, 3, hspace=0.35, wspace=0.30)

for idx, (name, data) in enumerate(results.items()):
    col = idx
    c = colors[name]

    # Row 1: Subpopulations
    ax1 = fig.add_subplot(gs[0, col])
    ax1.plot(data["t"], data["S"], label="Sensitive (S)", color=c, linewidth=2)
    ax1.plot(data["t"], data["R"], label="Resistant (R)", color=c, linewidth=2, linestyle="--")
    ax1.set_title(name, fontsize=12, fontweight="bold", color=c)
    ax1.set_ylabel("Cell Count")
    ax1.set_ylim(bottom=0, top=K * 1.05)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Row 2: Total Volume
    ax2 = fig.add_subplot(gs[1, col])
    ax2.fill_between(data["t"], data["V"], alpha=0.3, color=c)
    ax2.plot(data["t"], data["V"], color=c, linewidth=2)
    ax2.axhline(0.5 * K, color="red", linestyle=":", alpha=0.6, label="$V_{max}$ threshold")
    ax2.set_ylabel("Total Tumor Volume")
    ax2.set_ylim(bottom=0, top=K * 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Row 3: Dosage
    ax3 = fig.add_subplot(gs[2, col])
    ax3.fill_between(data["t"], data["u"], alpha=0.4, color=c, step="mid")
    ax3.step(data["t"], data["u"], color=c, linewidth=2, where="mid")
    ax3.set_ylabel("Dose u(t)")
    ax3.set_xlabel("Days")
    ax3.set_ylim(-0.05, u_max * 1.1)
    ax3.grid(alpha=0.3)

fig.savefig(
    r"C:\Users\Kelechi\.gemini\antigravity\brain\3a007900-82e4-4191-98bc-5c0fc710ae33\clonal_evolution_results.png",
    dpi=150, bbox_inches="tight"
)
print("\n✅ Simulation complete. Figure saved to artifacts directory.")
