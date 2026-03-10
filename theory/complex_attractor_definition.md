# Proposal 1 — Defining the Complex Attractor Mathematically

> **Core Claim:** The healthy biological state is not a fixed point but a *strange attractor* — a bounded, deterministic, aperiodic trajectory through metabolic-immune phase space. Cancer represents dimensional collapse of this attractor. Therapy should restore attractor complexity, not merely shift the equilibrium.

---

## 1. State Space Definition

### 1.1 Extending the SAEM Phase Space

The existing SAEM framework operates on a 10-dimensional metabolic state space:

```
x(t) ∈ ℝ¹⁰ = {Glucose, Lactate, Pyruvate, ATP, NADH, Glutamine, Glutamate, αKG, Citrate, ROS}
```

governed by `dx/dt = A·x` where `A ∈ ℝ¹⁰ˣ¹⁰` is the generator matrix (`TNBCODESystem`).

This is a **linear** system. Its attractors are necessarily fixed points (stable node/focus) or limit cycles (if complex eigenvalues dominate). It *cannot* produce strange attractors.

**Extension:** Augment the state space with **immune** and **microenvironmental** variables to create a coupled nonlinear system:

```
State vector:  z(t) ∈ ℝ¹⁵

z = [x₁, ..., x₁₀, I_eff, I_reg, I_exhaust, σ_stromal, ν_vascular]
     └─ metabolic ─┘  └──── immune ────────┘  └── microenvironment ──┘
```

| Dimension | Variable | Biological Meaning | Source |
|-----------|----------|-------------------|--------|
| z₁–z₁₀ | x₁–x₁₀ | 10 SAEM metabolites | `tnbc_ode.py` (existing) |
| z₁₁ | I_eff | Effector immune activity (CD8⁺ + NK) | `immune_dynamics.py` |
| z₁₂ | I_reg | Regulatory T-cell load | `immune_dynamics.py` |
| z₁₃ | I_exhaust | Immune exhaustion index | `immune_dynamics.py` |
| z₁₄ | σ_stromal | Stromal density / desmoplasia | `spatial_dynamics.py` |
| z₁₅ | ν_vascular | Vascular integrity / angiogenic index | Literature |

### 1.2 Biological Bounds

Each dimension has physiological constraints:

```
z(t) ∈ Ω = {z ∈ ℝ¹⁵ : z_min ≤ z ≤ z_max}
```

| Variable | Lower Bound | Upper Bound | Rationale |
|----------|------------|-------------|-----------|
| Glucose | 0 | 10 mM | Physiological range |
| ATP | 0.5 mM | 8 mM | Below 0.5 = necrosis |
| ROS | 0 | 5 (normalized) | Above 5 = oxidative catastrophe |
| I_eff | 0 | 1 (normalized) | Fraction of max effector capacity |
| I_reg | 0 | 1 | Treg fraction |
| I_exhaust | 0 | 1 | Exhaustion progression |
| σ_stromal | 0 | 1 | 0 = no stroma; 1 = complete desmoplasia |
| ν_vascular | 0 | 1 | 0 = avascular; 1 = fully vascularized |

---

## 2. The Nonlinear Extended System

### 2.1 Governing Equations

The coupled system evolves under:

```
dz/dt = F(z, θ, u(t))
```

where:
- `F : ℝ¹⁵ × Θ × U → ℝ¹⁵` is the nonlinear vector field
- `θ ∈ Θ` are biological parameters (cancer type, patient genetics)
- `u(t) ∈ U` is the therapeutic control input (drug doses, immunotherapy)

**Decomposition** (preserves compatibility with existing SAEM):

```
F(z, θ, u) = F_met(x, I, σ, θ)     ← metabolic dynamics (modified from Ax)
            + F_imm(I, x, u, θ)     ← immune dynamics
            + F_mic(σ, ν, x, θ)     ← microenvironment dynamics
            + G(z, u)               ← therapeutic forcing
```

### 2.2 Metabolic Subsystem (Modified)

The linear SAEM system `dx/dt = Ax` becomes nonlinear via:
1. **Michaelis-Menten saturation** on uptake/clearance terms
2. **Immune coupling** — immune activity modulates metabolic rates
3. **Stromal gating** — desmoplasia attenuates drug and immune access

```
F_met(x, I, σ, θ):
  dx_i/dt = Σⱼ A_ij · x_j · (K_ij / (K_ij + x_j))     [saturating kinetics]
          + η_i · I_eff · (1 - σ)                         [immune metabolic effect]
          - δ_i · σ · x_i                                  [stromal sequestration]
```

Where:
- `K_ij` = Michaelis constant for the (i,j) interaction
- `η_i` = immune sensitivity of metabolite i
- `δ_i` = stromal sequestration rate for metabolite i

**Key consequence:** Michaelis-Menten terms introduce the nonlinearity necessary for strange attractor dynamics. The linear SAEM `Ax` is recovered when `x_j ≪ K_ij` (first-order regime).

### 2.3 Immune Subsystem

Derived from the existing `LymphocyteForceField` but written as continuous ODEs:

```
dI_eff/dt     = r_prime · DC_signal · (1 - I_eff) - k_exhaust · I_eff · tumor_load - k_decay · I_eff
dI_reg/dt     = r_reg · tumor_load · (1 - I_reg) - k_depletion · I_reg · u_anti_treg
dI_exhaust/dt = k_exhaust · I_eff · tumor_load - r_rescue · u_checkpoint · I_exhaust
```

Where:
- `tumor_load = ‖x(t) - x_healthy‖ / ‖x_cancer - x_healthy‖` (normalized metabolic distance)
- `DC_signal = f(ROS, necrotic_fraction)` — damage-associated molecular pattern signaling
- `u_checkpoint`, `u_anti_treg` — immunotherapy control inputs

### 2.4 Microenvironment Subsystem

```
dσ/dt = r_fibro · tumor_load - k_degrade · σ · (1 + u_stroma_disrupt)
dν/dt = r_angio · VEGF(x) · (1 - ν) - k_prune · ν · I_eff
```

Where `VEGF(x)` is a function of metabolic state (high glycolysis → high VEGF).

---

## 3. Strange Attractor Characterization

### 3.1 Why a Strange Attractor?

The 15D nonlinear system `dz/dt = F(z)` can exhibit:

| Attractor Type | Dimension | Condition | Biological Interpretation |
|---------------|-----------|-----------|---------------------------|
| **Fixed point** | 0 | All Lyapunov exponents < 0 | Homeostatic equilibrium (oversimplification) |
| **Limit cycle** | 1 | One zero exponent, rest < 0 | Circadian / cell-cycle oscillation |
| **Torus** | 2 | Two zero exponents | Coupled oscillators (circadian × cell cycle) |
| **Strange attractor** | Non-integer (fractal) | At least one positive exponent | Healthy adaptive complexity |

**Claim:** The healthy state is a strange attractor because:
1. Biological oscillators operate at incommensurate frequencies (cell cycle ~24h, immune ~7d, metabolic ~seconds) → quasi-periodic dynamics
2. The coupling between subsystems introduces sensitivity to initial conditions (positive Lyapunov exponent)
3. Despite deterministic chaos, the system is bounded (physiological limits)
4. The attractor is structurally stable under perturbation (health is robust)

### 3.2 Lyapunov Spectrum Targets

The Lyapunov spectrum `{λ₁ ≥ λ₂ ≥ ... ≥ λ₁₅}` characterizes the attractor:

**Healthy attractor target:**
```
λ₁ > 0          (bounded chaos — adaptive unpredictability)
λ₂ ≈ 0          (slow manifold — quasi-periodic base rhythm)
λ₃, ..., λ₁₅ < 0  (contractivity — perturbations decay)

Σᵢ λᵢ < 0       (dissipative — phase-space volume contracts globally)
```

**Cancer attractor signature:**
```
λ₁ → 0⁺ or 0    (loss of chaos — stereotyped, rigid dynamics)
All λ close to 0  (loss of time-scale separation)
OR
λ₁ ≫ 0           (pathological chaos — explosive instability)
```

**Quantitative bounds (to be calibrated):**

| Metric | Healthy Range | Cancer (Rigid) | Cancer (Explosive) |
|--------|---------------|----------------|-------------------|
| λ_max | 0.01 – 0.10 bits/day | < 0.005 | > 0.50 |
| Σλ | −0.5 to −0.1 | > −0.05 (weak contraction) | < −2.0 (hyper-contraction) |
| Kaplan-Yorke dim D_KY | 3.0 – 6.0 | < 2.0 | > 8.0 |

> **Note:** These ranges are initial hypotheses. Calibration against TCGA data (Proposal 2) and the existing SAEM bifurcation scan will refine them.

### 3.3 Correlation Dimension D₂

The correlation dimension provides a model-free estimate of attractor dimensionality:

```
D₂ = lim(ε→0) [log C(ε) / log ε]
```

where `C(ε) = (2/N²) Σᵢ<ⱼ Θ(ε - ‖zᵢ - zⱼ‖)` is the correlation integral.

**Targets:**

| State | Expected D₂ | Interpretation |
|-------|-------------|----------------|
| Healthy tissue | 3.0 – 6.0 | Rich, multiscale dynamics |
| Early cancer | 2.0 – 3.0 | Complexity loss begins |
| Advanced cancer | 1.0 – 2.0 | Near-periodic, rigid |
| Treated (recovering) | Rising toward 3.0+ | Complexity restoration |

### 3.4 Multiscale Entropy Profile

The MSE profile `S(τ)` at coarse-graining scale `τ` distinguishes:
- **Healthy:** High entropy maintained across scales (complexity)
- **Random noise:** High entropy at fine scales, drops at coarse scales
- **Cancer (rigid):** Low entropy at all scales (stereotyped dynamics)

**Target MSE profile shape:**

```
S(τ) for healthy:   ████████████████████   (flat, high)
S(τ) for cancer:    ██▓▓░░░░░░░░░░░░░░░   (drops with scale)
S(τ) for noise:     ████████▓▓▓░░░░░░░░   (drops with scale, but starts high)
```

The healthy MSE reference profile will be established by:
1. Simulating the 15D system with parameter values drawn from healthy tissue literature
2. Computing `S(τ)` for `τ = 1, 2, 4, ..., 2¹⁰` across multiple realizations
3. Establishing mean ± 2σ bounds at each scale

---

## 4. The Coherence Restoration Functional

### 4.1 Composite Metric C(t)

Extending the existing `CoherenceAnalyzer.overall_score` (which operates on the generator matrix `A`) into a **trajectory-based** metric:

```
C(t) = w_D · f_D(D₂(t)) + w_S · f_S(MSE(t)) + w_λ · f_λ(λ_max(t)) + w_β · f_β(β(t))
```

Where:
- `D₂(t)` = correlation dimension estimated from a trailing window of `z(t)`
- `MSE(t)` = mean multiscale entropy over scales τ = 1..20
- `λ_max(t)` = largest Lyapunov exponent from trailing window
- `β(t)` = power spectral density slope (log-log)
- `f_*(·)` = normalization functions mapping raw values to [0, 1] based on healthy reference

**Normalization functions:**

```
f_D(D₂) = exp(−((D₂ − D₂_healthy)² / (2·σ_D²)))     [Gaussian, centered on healthy]
f_S(MSE) = tanh(MSE / MSE_healthy)                     [saturating at healthy level]
f_λ(λ)  = exp(−(λ − λ_target)² / (2·σ_λ²))           [Gaussian, penalizes deviation]
f_β(β)  = exp(−(β − β_healthy)² / (2·σ_β²))           [1/f noise target: β ≈ 1.0]
```

**Weights** (default, tunable):

| Component | Weight | Rationale |
|-----------|--------|-----------|
| w_D (dimension) | 0.30 | Primary indicator of attractor structure |
| w_S (entropy) | 0.25 | Scale-free complexity measure |
| w_λ (Lyapunov) | 0.25 | Sensitivity / adaptability |
| w_β (spectral) | 0.20 | Long-range temporal correlations |

### 4.2 Therapeutic Objective

The SAEM protocol, reframed as an optimal control problem:

```
minimize  J[u] = ∫₀ᵀ  [ (1 − C(t))² + ρ·‖u(t)‖² ] dt
  u(t)

subject to:
  dz/dt = F(z, θ, u(t))
  z(0) = z_cancer
  z(t) ∈ Ω                                [physiological bounds]
  C(t) ∈ [C_min, C_max]   ∀t             [safe corridor constraint]
  ‖u(t)‖ ≤ u_max                          [dosage limits]
```

Where:
- `(1 − C(t))²` penalizes deviation from perfect coherence
- `ρ·‖u(t)‖²` penalizes aggressive treatment (toxicity regularizer)
- The safe corridor `[C_min, C_max]` prevents rigidity undershoot and chaos overshoot

### 4.3 Relationship to Existing SAEM Metrics

| Existing Metric | Complex Coherence Extension |
|----------------|----------------------------|
| `escape_distance` | `‖z(t) - z_cancer‖` — escaping the cancer basin |
| `cure_rate` | `P(C(T) > C_threshold)` — probability of coherence restoration |
| `CoherenceAnalyzer.overall_score` | `C(t)` — trajectory-based, time-varying |
| `seriousness` | Initial `1 - C(0)` — how far from healthy at diagnosis |

---

## 5. Connecting to the SAEM Codebase

### 5.1 Backward Compatibility

The new formalism is a **strict superset** of the existing SAEM:

```
Existing:  dx/dt = Ax,           health = ‖x − x*‖ < ε
New:       dz/dt = F(z, θ, u),   health = C(t) > C_threshold

When F is linearized around x* and immune/microenvironment variables are frozen:
  F(z) ≈ A·x  →  recovers the existing system exactly
```

### 5.2 Implementation Bridge

| New Component | Builds On | File |
|---------------|-----------|------|
| `ComplexAttractorODE.__init__` | `TNBCODESystem` generators | `models/dynamical_systems.py` |
| `ComplexAttractorODE.rhs` | `ODEParams` + `ImmuneParams` | `models/dynamical_systems.py` |
| `coherence_metric()` | `CoherenceAnalyzer._compute_overall_score` | `models/complexity_metrics.py` |
| `lyapunov_spectrum()` | `CoherenceAnalyzer._spectral_coherence` | `models/complexity_metrics.py` |
| `WeakConstraint` | `toxicity_constraints.py` | `protocols/weak_constraint_impl.py` |

### 5.3 Computational Feasibility (i5-3380M, 8GB RAM)

| Operation | Estimated Time | Memory | Notes |
|-----------|---------------|--------|-------|
| 15D ODE integration (1000 days) | ~0.1s | <10 MB | `scipy.solve_ivp`, RK45 |
| Lyapunov spectrum (Wolf algorithm) | ~5s per trajectory | <50 MB | Use `nolds` library |
| Correlation dimension | ~2s per trajectory | <20 MB | Theiler window saves time |
| MSE (20 scales) | ~1s per signal | <5 MB | Coarse-graining is cheap |
| Full C(t) computation | ~10s per trajectory | <80 MB | All 4 sub-metrics |
| Monte Carlo (200 trials) | ~30 min | <100 MB | Parallelizable embarrassingly |

All within hardware constraints. No GPU required.

---

## 6. Open Questions

1. **What is the healthy attractor's topology?** — Is it a single strange attractor, or a chaotic itinerancy between multiple quasi-stable states? (Affects D₂ interpretation)

2. **How does clonal heterogeneity map to attractor dimension?** — Does polyclonal tumor = higher D₂, or does clonal cooperation reduce effective dimension?

3. **Is β ≈ 1.0 (1/f noise) truly the healthy target?** — Strong evidence from cardiac/neural systems; weaker evidence from metabolic time-series

4. **Calibration data:** The healthy reference values for all metrics require simulation of the extended 15D system with healthy parameters — this is the immediate next computational task

---

## 7. Next Steps

1. **Implement `ComplexAttractorODE.rhs`** in `models/dynamical_systems.py` using the nonlinear formulation from §2
2. **Calibrate healthy reference** by running the 15D system with healthy parameters and computing {D₂, MSE, λ_max, β}
3. **Validate backward compatibility** by showing that the linearized 15D system reproduces existing SAEM results
4. **Map cancer signatures** by running with cancer generator parameters and comparing complexity metrics

---

*Document version: 1.0 | Date: 2026-03-05 | Author: Agent-assisted formalization*
*Parent: [README.md](../README.md) | Next: [Biomarker Mapping](../validation/biomarker_mapping.md)*
