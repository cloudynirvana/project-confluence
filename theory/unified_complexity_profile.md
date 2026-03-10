# Unified Complexity Profile (UCP) — Theoretical Framework

## 1. Core Hypothesis

Health is not a fixed point but a **Complex Attractor State** characterized by adaptive variability, fractal rhythms, and moderate inter-system coupling. Disease (cancer, autoimmunity, cachexia) represents a transition to pathological attractors. Therapy should steer the system back to the **Healthy Complex Attractor**.

> "We do not kill the disease — we restore the complexity."

## 2. Two-Dimensional Clinical Complexity

### 2.1 Clinical Complexity Ψ (Psi)

Derived from electronic health records and standard clinical data:

| Component | Source | Interpretation |
|-----------|--------|----------------|
| Disease burden | TNM staging, tumor markers | Primary pathology severity |
| Comorbidity index | Charlson/Elixhauser scores | Multi-system fragility |
| Treatment history | Prior lines, resistance patterns | Therapeutic landscape |
| Genomic complexity | TMB, MSI status, pathway alterations | Mutational pressure |

### 2.2 Dynamical Complexity Φ (Phi) — 5D Vector

Derived from time-series physiological measurements and computational modeling:

| Dimension | Metric | Healthy Target | Biomarker Proxy |
|-----------|--------|----------------|-----------------|
| Φ_temporal | Multiscale Entropy (MSE) | 0.6–0.8 | HRV, glucose variability |
| Φ_spatial | Correlation Dimension (D₂) | 3.0–6.0 | Cell diversity, tumor architecture |
| Φ_functional | Perturbation recovery rate | 0.5–0.8 | Stress response, resilience |
| Φ_informational | λ_max + β (spectral slope) | 0.5–0.7 | Signal entropy, EEG/ECG |
| Φ_coupling | Inter-system correlation | 0.4–0.7 | IL-6, immune-metabolic sync |

## 3. Pathology Archetypes

Three canonical attractors that disease trajectories converge toward:

```
                    High Chaos
                       │
            ┌──────────┼──────────┐
            │  Chaotic/ │          │
            │  Decoupled│          │
            │  (Cancer) │          │
            │          ─┼─── Healthy Complex
            │           │   (Target)│
            │           │          │
            │  Collapsed│  Rigid/  │
            │  /Exhaust │  Locked  │
            │  (Cachexia)(Autoimmune)
            └──────────┼──────────┘
                       │
                  Low Chaos
          Low Coupling ←→ High Coupling
```

### Archetype Properties

| Archetype | Φ_temporal | Φ_coupling | Cancer Type Example | Therapeutic Strategy |
|-----------|-----------|-----------|---------------------|---------------------|
| Chaotic/Decoupled | High | Low | Metastatic, TNBC | Reduce chaos, restore coupling |
| Rigid/Locked | Low | High | Autoimmune tumors | Increase variability |
| Collapsed/Exhausted | Low | Low | Cachexia, late-stage | Rebuild all dimensions |
| Healthy Complex | Moderate | Moderate | — | Maintain |

## 4. Mathematical Core: 15D SAEM Model

The system state **z** ∈ ℝ¹⁵ evolves according to:

```
dz/dt = F(z, θ, u)

z = [x₁...x₁₀, I_eff, I_reg, I_exhaust, σ_stromal, ν_vascular]

where:
  x₁...x₁₀  : Metabolic concentrations (Glucose, Lactate, ..., ROS)
  I_eff      : Effector immune activity
  I_reg      : Regulatory T-cell load
  I_exhaust  : Immune exhaustion index
  σ_stromal  : Stromal density
  ν_vascular : Vascular integrity
  θ          : Patient-specific parameters (from PatientFitter)
  u          : Drug input (from RADO engine)
```

### Key Properties

- **Nonlinearity**: Michaelis-Menten saturation kinetics provide the nonlinearity needed for strange attractor behavior
- **Backward Compatible**: When immune/microenvironment are frozen, the system reduces to the original 10D linear SAEM (dx/dt = Ax)
- **Bounded**: All state variables remain physiologically bounded through soft clamping

## 5. Computational Pipeline

```
┌─────────────────┐     ┌──────────────┐     ┌───────────┐     ┌──────────────┐
│  Clinical Data  │────▶│  Complexity  │────▶│  Patient  │────▶│    RADO      │
│  + Omics Data   │     │  Profiler    │     │  Fitter   │     │   Engine     │
│  (Module 4)     │     │  (Module 1)  │     │ (Module 2)│     │  (Module 3)  │
└─────────────────┘     └──────────────┘     └───────────┘     └──────────────┘
                              │                     │                  │
                        phi_profile.json    digital_twin.json   protocol.json
```

## 6. Regulatory Alignment

All Φ dimensions map to existing clinical measurement categories:

| Φ Dimension | LOINC Code | SNOMED-CT | FDA MIDD Category |
|-------------|----------|-----------|-------------------|
| Φ_temporal | 8867-4 | 251670001 | Biomarker, pharmacodynamic |
| Φ_spatial | 33747-3 | 371469007 | Prognostic biomarker |
| Φ_functional | 30525-0 | 165109007 | Response biomarker |
| Φ_informational | LP99691-0 | 251629003 | Monitoring biomarker |
| Φ_coupling | 26881-3 | 52988006 | Safety biomarker |
