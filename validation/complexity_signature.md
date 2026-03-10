# Complexity Signature — What Confluence Measures

> Exact definitions of the 5D Φ vector, therapeutic thresholds, and the complexity recovery score.

---

## 1. The Φ Vector: Five Dimensions of Dynamical Complexity

Each dimension is computed from a patient's **time-series data** (not a static snapshot). The input is a multivariate trajectory **z(t)** ∈ ℝⁿ observed at discrete timepoints.

### 1.1 Φ_temporal — Multiscale Entropy (MSE)

**Question:** Does the system maintain structured variability across time scales?

| Property | Value |
|---|---|
| **Algorithm** | Coarse-grain at scales τ = 1…20, then Sample Entropy at each scale |
| **Implementation** | [`complexity_profiler.py → multiscale_entropy()`](../models/complexity_profiler.py) |
| **Parameters** | m=2 (template length), r=0.2×std (tolerance) |
| **Output** | Mean MSE across all scales (scalar) |
| **Healthy range** | 0.6–0.8 |
| **Disease signatures** | Cancer (chaotic): >0.85 · Autoimmune (rigid): <0.4 · Cachexia: <0.3 |

### 1.2 Φ_spatial — Correlation Dimension (D₂)

**Question:** How many independent degrees of freedom does the system occupy?

| Property | Value |
|---|---|
| **Algorithm** | Grassberger-Procaccia on time-delay embedded phase space |
| **Implementation** | [`complexity_profiler.py → correlation_dimension()`](../models/complexity_profiler.py) |
| **Parameters** | emb_dim=10, tau=1, Theiler window=10 |
| **Output** | Scaling exponent of correlation integral C(ε) ~ ε^D₂ |
| **Healthy range** | 3.0–6.0 |
| **Disease signatures** | Cancer: >7.0 (hyperdimensional chaos) · Rigid: <2.0 (collapsed to limit cycle) |

### 1.3 Φ_functional — Perturbation Recovery Rate

**Question:** How quickly does the system return to its attractor after a perturbation?

| Property | Value |
|---|---|
| **Algorithm** | Exponential fit to deviation-from-mean after identified perturbation events |
| **Implementation** | [`complexity_profiler.py → ComplexityProfiler._compute_functional()`](../models/complexity_profiler.py) |
| **Parameters** | Perturbation detection threshold = 2×std |
| **Output** | Recovery rate constant (1/time units) |
| **Healthy range** | 0.5–0.8 |
| **Disease signatures** | Exhausted: <0.2 (no recovery) · Chaotic: variable/unpredictable |

### 1.4 Φ_informational — Lyapunov Exponent + Spectral Slope

**Question:** Is the system's information production balanced between chaos and order?

| Property | Value |
|---|---|
| **Algorithm** | Rosenstein et al. (1993) for λ_max; log-log regression of power spectrum for β |
| **Implementation** | [`complexity_profiler.py → largest_lyapunov_exponent()`](../models/complexity_profiler.py), [`power_spectral_slope()`](../models/complexity_profiler.py) |
| **Parameters** | emb_dim=7, tau=2, min_separation=20 |
| **Output** | Normalized composite: (λ_max_norm + β_norm) / 2 |
| **Healthy range** | 0.5–0.7 |
| **Disease signatures** | Cancer: >0.8 (positive λ_max, white-noise spectrum) · Rigid: <0.3 (negative λ_max, steep 1/f) |

### 1.5 Φ_coupling — Inter-System Correlation

**Question:** Are the subsystems (metabolic, immune, microenvironment) synchronized?

| Property | Value |
|---|---|
| **Algorithm** | Mean absolute Pearson correlation across all variable pairs |
| **Implementation** | [`complexity_profiler.py → ComplexityProfiler._compute_coupling()`](../models/complexity_profiler.py) |
| **Parameters** | Computed on windowed segments (window = trajectory_length / 4) |
| **Output** | Mean |r| across all unique variable pairs |
| **Healthy range** | 0.4–0.7 |
| **Disease signatures** | Decoupled (cancer): <0.2 · Locked (autoimmune): >0.85 |

---

## 2. Pathology Archetypes

The 5D Φ vector maps to three canonical disease attractors:

```
                          Φ_temporal
                          HIGH (>0.8)
                             │
              ┌──────────────┼──────────────┐
              │   CHAOTIC /  │              │
              │   DECOUPLED  │              │
              │   Φ_c < 0.2  │              │
              │              │              │
    LOW       │         ● HEALTHY           │  HIGH
  Φ_coupling ─┤        COMPLEX              ├─ Φ_coupling
   (<0.3)     │       (0.4-0.7)             │  (>0.8)
              │              │              │
              │  COLLAPSED / │   RIGID /    │
              │  EXHAUSTED   │   LOCKED     │
              │  all Φ low   │  Φ_c > 0.85  │
              └──────────────┼──────────────┘
                             │
                          LOW (<0.4)
```

| Archetype | Φ_t | Φ_s | Φ_f | Φ_i | Φ_c | Example |
|---|---|---|---|---|---|---|
| Chaotic / Decoupled | >0.85 | >7.0 | variable | >0.8 | <0.2 | Metastatic TNBC |
| Rigid / Locked | <0.4 | <2.0 | moderate | <0.3 | >0.85 | Autoimmune, T2D insulin resistance |
| Collapsed / Exhausted | <0.3 | <2.0 | <0.2 | <0.3 | <0.2 | Cachexia, late-stage multi-organ |
| **Healthy Complex** | 0.6–0.8 | 3–6 | 0.5–0.8 | 0.5–0.7 | 0.4–0.7 | — |

---

## 3. Complexity Recovery Score (CRS)

The primary outcome metric for therapeutic validation:

```
CRS = Σᵢ wᵢ × (Φᵢ_post − Φᵢ_pre) / (Φᵢ_healthy − Φᵢ_pre)
```

Where:
- `Φᵢ_pre` = Φ dimension i measured before intervention
- `Φᵢ_post` = Φ dimension i measured after intervention
- `Φᵢ_healthy` = midpoint of healthy range for dimension i
- `wᵢ` = weight per dimension (default: equal, 0.2 each)

| CRS Value | Interpretation |
|---|---|
| **> 0.8** | Strong therapeutic recovery — system approaching healthy attractor |
| **0.4–0.8** | Partial recovery — trajectory bending toward health |
| **0.0–0.4** | Minimal effect — system remains in disease basin |
| **< 0.0** | Deterioration — intervention pushed system further from health |

---

## 4. Minimum Data Requirements for Φ Computation

| Φ Dimension | Minimum Timepoints | Minimum Variables | Minimum Sampling Rate |
|---|---|---|---|
| Φ_temporal (MSE) | 200 per series | 1 | ≥ 2× characteristic frequency |
| Φ_spatial (D₂) | 500 per series | 1 | any regular |
| Φ_functional | 50 post-perturbation | 1 | ≤ recovery time / 10 |
| Φ_informational (λ_max) | 300 per series | 1 (multivariate preferred) | regular |
| Φ_coupling | 100 per series | ≥ 3 | synchronized |

> [!NOTE]
> For clinical datasets with sparse sampling (e.g., monthly blood draws), we can compute reduced Φ vectors using only Φ_coupling and a variance-based Φ_temporal proxy. Contact us for guidance.
