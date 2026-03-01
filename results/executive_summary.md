# SAEM Pan-Cancer Hypothetical Cure Engine — Executive Summary

> ⚠️ **COMPUTATIONAL HYPOTHESIS ONLY** — All results below are from in silico simulations. No experimental validation has been performed. See [DISCLAIMER.md](../DISCLAIMER.md).

## Thesis

The **Spectral Attractor Escape Model (SAEM)** treats cancer metabolic states as attractor basins in a 10-dimensional phase space. The **Geometric Achievement Protocol** — a 3-phase adaptive therapy (Flatten → Heat → Push) — systematically destabilizes these basins to achieve therapeutic escape toward a healthy metabolic equilibrium.

## Simulation Results

**All 10 cancer types achieve 100% simulated escape rate** across 100 Monte Carlo trials per cancer with 95% bootstrap confidence intervals of [100%, 100%]. Note: a 100% rate across all cancers may indicate the model is under-constrained — see Limitations.

| Gate | Result |
|---|---|
| **Cure Threshold** (≥90% per cancer) | ✅ 10/10 |
| **Adaptive Superiority** (vs continuous) | ✅ 10/10 |
| **Protocol Diversity** (≥5 unique combos) | ✅ 5/10 |
| **Non-Uniform Outcomes** (distance range >0.05) | ✅ 0.271–0.543 |

## How the Protocol Works

### Phase 1: Flatten (18–32 days)
Metabolic drugs (2-DG, CB-839, DCA, Metformin) reduce the cancer attractor's basin curvature. This makes the "well" shallower and easier to escape. Drug effectiveness decays exponentially with resistance (τ = 18 days).

### Phase 2: Heat (5–9 days)
Drug holiday — metabolic drugs are reduced to 30%, allowing:
- Tumor resistance to partially decay (recovery_tau = 12.6d)
- Immune T-cell exhaustion to recover by 85%
- Entropic drivers (hyperthermia, Vitamin C) inject noise to destabilize the flattened state

### Phase 3: Push (20–32 days)
Checkpoint blockade (Anti-PD-1, Anti-CTLA-4) + refreshed T-cells push the system toward the healthy attractor. The immune system is the primary escape driver in this phase.

## Why Adaptive Beats Continuous

The adaptive protocol outperforms continuous therapy in **all 10 cancers** because:

1. **Immune Recovery**: Continuous therapy exhausts T-cells without recovery. Adaptive's drug holiday restores 85% of immune competence.
2. **Resistance Reset**: Drug holidays allow partial resistance decay, restoring drug efficacy for Phase 3.
3. **Checkpoint Timing**: Strategic PD-1/CTLA-4 deployment in Phase 3 (after recovery) maximizes immune force.

Average adaptive escape distance: **0.40** vs continuous: **0.73** (lower = closer to healthy = better).

## Calibrated Parameters

| Parameter | Value | Biological Basis |
|---|---|---|
| BASE_FORCE | 0.55 | T-cell cytotoxic pressure (calibrated to CPI clinical response rates) |
| EXHAUSTION_RATE | 0.150 | Exhaustion accumulation per unit depth per day |
| NOISE_SCALE | 0.15 | Stochastic metabolic fluctuations |
| RESISTANCE_TAU | 18.0 days | Time constant for exponential resistance buildup |
| TREG_LOAD | 0.50 | Regulatory T-cell friction coefficient |

## Cancer-Specific Seriousness Ranking

1. **PDAC** (0.494) — Highest: extreme stromal barrier (0.85), immune suppression (0.80)
2. **HCC** (0.482) — ROS-adaptive, hepatic metabolic rewiring
3. **GBM** (0.435) — BBB barrier, immune privilege
4. **HGSOC** (0.427) — Peritoneal immune exclusion
5. **TNBC** (0.390) — Glycolytic, moderate immune suppression
6. **AML** (0.386) — Liquid tumor, BH3-dependent
7. **mCRPC** (0.384) — Androgen-independent metabolic switch
8. **CRC** (0.347) — Wnt-driven, butyrate-sensitive
9. **Melanoma** (0.335) — Lowest immune suppression (0.20)
10. **NSCLC** (0.329) — Most tractable: diverse drug response

## Limitations

- **10D approximation** — real tumor ecosystems have spatial gradients and clonal heterogeneity
- **Linear drug additivity** — synergy/antagonism modeled only at first order
- **Simplified PK** — constant-dose phases rather than time-weighted PK curves
- **Single-compartment immune model** — does not capture T-cell trafficking or tissue-specific barriers
- **No patient-specific calibration** — generators based on population-level metabolomics (CCLE)

## Next Steps for Translation

1. **Wet-lab validation**: Test Flatten→Heat→Push sequencing in TNBC organoid models
2. **PK integration**: Replace constant-dose with pharmacokinetic curves for each drug
3. **Patient stratification**: Calibrate generators from individual tumor metabolomics
4. **Adaptive scheduling**: Clinical protocol with real-time curvature monitoring via metabolomics panels
