"""
Monte Carlo Uncertainty Simulation — Project Confluence
========================================================

Runs the adaptive therapy simulation across N random samples from
the biological uncertainty set to demonstrate that the Robust
Adaptive Policy maintains tumor control across the ENTIRE parameter
space, not just the nominal case.

Outputs:
  1. A figure showing S(t), R(t), V(t) envelopes across all samples
  2. A comparison table: MTD vs Adaptive survival rates under uncertainty

This is the Proof of Concept for the "Assurance Layer" of the thesis.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.clonal_dynamics import ClonalDynamicsEngine, ClonalParams
from models.adaptive_controller import (
    AdaptiveController, PolicyParams, PolicyMode
)


# ── Uncertainty Set Definition ──────────────────────────────────────────
# These are the ranges of biological parameters that are "uncertain"
UNCERTAINTY_SET = {
    "sensitive_growth_rate": (0.06, 0.14),
    "resistant_growth_rate": (0.04, 0.10),
    "alpha_RS": (0.3, 0.9),         # Competition of R on S
    "alpha_SR": (0.5, 1.2),         # Competition of S on R — THE critical lever
    "drug_kill_rate_sensitive": (0.08, 0.20),
}

N_SAMPLES = 200       # Number of Monte Carlo samples
T_DAYS = 180          # Simulation horizon
DT = 0.5             # Timestep (coarser for speed)


def sample_params(rng: np.random.RandomState) -> ClonalParams:
    """Sample one set of biological parameters from the uncertainty set."""
    params = ClonalParams()
    for key, (lo, hi) in UNCERTAINTY_SET.items():
        setattr(params, key, rng.uniform(lo, hi))
    return params


def run_single(params: ClonalParams, policy_mode: PolicyMode,
               policy_params: PolicyParams, seed: int) -> dict:
    """Run one simulation and return trajectory + outcome."""
    engine = ClonalDynamicsEngine(params)
    ctrl = AdaptiveController(policy_mode, policy_params)
    
    n_steps = int(T_DAYS / DT)
    for i in range(n_steps):
        state = engine.state
        dose = ctrl.decide(
            sensitive=state.sensitive,
            resistant=state.resistant,
            carrying_capacity=params.carrying_capacity,
            dt=DT,
        )
        engine.step(dt=DT, drug_active=(dose > 0), drug_pressure=dose,
                    phase="flatten", seed=seed + i)
    
    final = engine.state
    return {
        "S": final.sensitive_trajectory,
        "R": final.resistant_trajectory,
        "V": final.burden_trajectory,
        "t": final.time_points,
        "final_burden": final.tumor_fraction,
        "final_R_frac": final.resistant_fraction,
        "controlled": final.tumor_fraction < 0.9 * params.carrying_capacity,
        "R_takeover": final.resistant_fraction > 0.80,
    }


def run_monte_carlo():
    """Run full Monte Carlo comparison: MTD vs Adaptive under uncertainty."""
    rng = np.random.RandomState(42)
    
    # MTD policy: always max dose
    mtd_params = PolicyParams(
        dose_on_threshold=0.0, dose_off_threshold=0.0,
        robust_max_dose=1.0, max_continuous_dose_days=999,
        min_holiday_days=0, max_cumulative_toxicity=999,
    )
    
    # Confluence Adaptive policy
    adaptive_params = PolicyParams()
    
    mtd_results = []
    adaptive_results = []
    
    print(f"Running {N_SAMPLES} Monte Carlo samples...")
    for i in range(N_SAMPLES):
        params = sample_params(rng)
        seed = rng.randint(0, 100000)
        
        mtd_res = run_single(params, PolicyMode.THRESHOLD, mtd_params, seed)
        ada_res = run_single(params, PolicyMode.ROBUST_ADAPTIVE, adaptive_params, seed)
        
        mtd_results.append(mtd_res)
        adaptive_results.append(ada_res)
        
        if (i + 1) % 50 == 0:
            print(f"  Completed {i+1}/{N_SAMPLES}")
    
    return mtd_results, adaptive_results


def plot_results(mtd_results, adaptive_results, save_path):
    """Create the Monte Carlo comparison figure."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Project Confluence — Monte Carlo Uncertainty Analysis\n"
        f"({N_SAMPLES} random biological parameter sets from uncertainty set)",
        fontsize=14, fontweight="bold", y=0.98
    )
    
    gs = GridSpec(3, 2, hspace=0.35, wspace=0.25)
    
    strategies = {
        "MTD (Standard Care)": (mtd_results, "#e74c3c"),
        "Confluence Adaptive": (adaptive_results, "#2ecc71"),
    }
    
    for col, (name, (results, color)) in enumerate(strategies.items()):
        # Find common time length
        min_len = min(len(r["V"]) for r in results)
        
        V_matrix = np.array([r["V"][:min_len] for r in results])
        S_matrix = np.array([r["S"][:min_len] for r in results])
        R_matrix = np.array([r["R"][:min_len] for r in results])
        t = np.array(results[0]["t"][:min_len])
        
        # Row 1: Tumor Volume Envelope
        ax1 = fig.add_subplot(gs[0, col])
        V_median = np.median(V_matrix, axis=0)
        V_p5 = np.percentile(V_matrix, 5, axis=0)
        V_p95 = np.percentile(V_matrix, 95, axis=0)
        V_p25 = np.percentile(V_matrix, 25, axis=0)
        V_p75 = np.percentile(V_matrix, 75, axis=0)
        
        ax1.fill_between(t, V_p5, V_p95, alpha=0.15, color=color, label="5th-95th %ile")
        ax1.fill_between(t, V_p25, V_p75, alpha=0.3, color=color, label="25th-75th %ile")
        ax1.plot(t, V_median, color=color, linewidth=2, label="Median")
        ax1.axhline(1.0, color="red", linestyle=":", alpha=0.5, label="Carrying Capacity")
        ax1.set_title(name, fontsize=13, fontweight="bold", color=color)
        ax1.set_ylabel("Total Tumor Volume")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # Row 2: Resistant Fraction Envelope
        ax2 = fig.add_subplot(gs[1, col])
        # Compute R fraction safely
        R_frac_matrix = np.divide(R_matrix, V_matrix, where=V_matrix > 1e-10,
                                   out=np.zeros_like(R_matrix))
        R_median = np.median(R_frac_matrix, axis=0)
        R_p5 = np.percentile(R_frac_matrix, 5, axis=0)
        R_p95 = np.percentile(R_frac_matrix, 95, axis=0)
        
        ax2.fill_between(t, R_p5, R_p95, alpha=0.2, color=color)
        ax2.plot(t, R_median, color=color, linewidth=2)
        ax2.axhline(0.80, color="red", linestyle="--", alpha=0.5, label="Takeover threshold")
        ax2.set_ylabel("Resistant Fraction (R/V)")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel("Days")
    
    # Row 3: Summary Statistics
    ax3 = fig.add_subplot(gs[2, :])
    ax3.axis("off")
    
    mtd_controlled = sum(1 for r in mtd_results if r["controlled"]) / len(mtd_results)
    ada_controlled = sum(1 for r in adaptive_results if r["controlled"]) / len(adaptive_results)
    mtd_takeover = sum(1 for r in mtd_results if r["R_takeover"]) / len(mtd_results)
    ada_takeover = sum(1 for r in adaptive_results if r["R_takeover"]) / len(adaptive_results)
    mtd_final_V = np.mean([r["final_burden"] for r in mtd_results])
    ada_final_V = np.mean([r["final_burden"] for r in adaptive_results])
    
    table_data = [
        ["Metric", "MTD", "Adaptive", "Advantage"],
        ["Tumor Controlled at Day 180",
         f"{mtd_controlled:.0%}", f"{ada_controlled:.0%}",
         f"+{(ada_controlled - mtd_controlled)*100:.0f}pp"],
        ["Resistant Takeover Rate",
         f"{mtd_takeover:.0%}", f"{ada_takeover:.0%}",
         f"-{(mtd_takeover - ada_takeover)*100:.0f}pp"],
        ["Mean Final Tumor Burden",
         f"{mtd_final_V:.3f}", f"{ada_final_V:.3f}",
         f"{((mtd_final_V - ada_final_V)/mtd_final_V)*100:.0f}% lower" if mtd_final_V > ada_final_V else "—"],
    ]
    
    table = ax3.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)
    
    # Color code the cells
    for (row, col_idx), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif col_idx == 3:
            cell.set_facecolor("#d5f5e3")
    
    ax3.set_title(
        f"Aggregate Results Across {N_SAMPLES} Uncertain Biological Parameter Sets",
        fontsize=12, fontweight="bold", pad=20
    )
    
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Monte Carlo figure saved to: {save_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"MONTE CARLO RESULTS ({N_SAMPLES} samples)")
    print(f"{'='*60}")
    print(f"  MTD — Controlled: {mtd_controlled:.0%} | R-Takeover: {mtd_takeover:.0%}")
    print(f"  Adaptive — Controlled: {ada_controlled:.0%} | R-Takeover: {ada_takeover:.0%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    save_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "brain",
        "3a007900-82e4-4191-98bc-5c0fc710ae33",
        "monte_carlo_results.png"
    )
    # Use absolute path
    save_path = r"C:\Users\Kelechi\.gemini\antigravity\brain\3a007900-82e4-4191-98bc-5c0fc710ae33\monte_carlo_results.png"
    
    mtd, adaptive = run_monte_carlo()
    plot_results(mtd, adaptive, save_path)
