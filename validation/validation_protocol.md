# Validation Protocol — Three-Arm Complexity Validation

> Testing whether Confluence identifies shared mathematical signatures of therapeutic response across distinct pathologies.

---

## Hypothesis Under Test

**H₀ (Null):** Disease-specific models are required — no universal complexity signature exists across pathologies.

**H₁ (Confluence):** Therapeutic success corresponds to a measurable restoration of dynamical complexity (Φ shift toward healthy attractor), and this signature is mathematically identical across diseases, even when the underlying biology differs.

---

## Arm 1: Separate Pathology Validation (Internal Validity)

### Goal
Prove Confluence works on Cancer **and** Diabetes independently using the same underlying equations.

### Method

```
                    ONCOLOGY ARM                    METABOLIC ARM
                    ────────────                    ─────────────
Input:              Longitudinal tumor data          Longitudinal glucose/HbA1c
                    (e.g., TCGA subset, PDX)         (e.g., NIDDK-DPP, CGM data)

Step 1:             Compute Φ_pre (5D vector)        Compute Φ_pre (5D vector)
Step 2:             Identify intervention point       Identify intervention point
Step 3:             Compute Φ_post (5D vector)       Compute Φ_post (5D vector)
Step 4:             Calculate CRS                    Calculate CRS

Output:             Therapeutic tipping point         Therapeutic tipping point
```

### Success Criteria
| Metric | Threshold | Meaning |
|---|---|---|
| CRS correlation with clinical outcome | r > 0.5, p < 0.05 | Complexity recovery predicts real outcomes |
| Archetype classification accuracy | > 70% | Φ vector correctly identifies disease state |
| Same ODE equations used for both | Yes/No | Framework is truly disease-agnostic |

### Candidate Datasets

**Oncology:**
- TCGA longitudinal subset (GDC portal) — multi-omics + survival
- Patient-derived xenograft (PDX) time-series — controlled interventions
- Liquid biopsy cohorts (ctDNA tracking through treatment)

**Metabolic:**
- NIDDK Diabetes Prevention Program (DPP) — lifestyle + metformin intervention
- Continuous glucose monitoring (CGM) cohorts — high-frequency time-series
- UK Biobank metabolic panel (repeat assessments)

---

## Arm 2: Conjoined Pathology Validation (Comorbidity Stress Test)

### Goal
Prove Confluence handles **coupled disease systems** — where intervention in one domain shifts the trajectory of another.

### Method

```
Input:    Patients with Cancer + Diabetes (comorbidity cohort)
          e.g., breast cancer patients on metformin

Step 1:   Compute joint Φ vector from BOTH disease domains simultaneously
Step 2:   Model as coupled attractor system:
            dΦ_cancer/dt = f(Φ_cancer, Φ_metabolic, u_treatment)
            dΦ_metabolic/dt = g(Φ_metabolic, Φ_cancer, u_treatment)
Step 3:   Perturb one domain (e.g., start metformin)
Step 4:   Predict effect on other domain (does cancer trajectory shift?)
Step 5:   Compare prediction to observed outcome
```

### Success Criteria
| Metric | Threshold | Meaning |
|---|---|---|
| Cross-domain prediction accuracy | > 60% | Model captures coupling between diseases |
| Coupling coefficient (Φ_coupling) | Significant change post-intervention | Intervention measurably alters cross-system coherence |
| Better than independent models | ΔAIC < -2 | Coupled model outperforms treating diseases independently |

### Candidate Datasets
- Metformin in breast cancer trials (metabolic drug → cancer effect)
- Diabetes onset post-chemotherapy cohorts (cancer treatment → metabolic effect)
- Pancreatic cancer with glucose dysregulation (intrinsically coupled)
- All of Us Research Program (has both cancer registry + metabolic EHR data)

---

## Arm 3: Universality Validation (The Math Proof)

### Goal
Prove the **mathematics is universal** — that "complexity recovery" looks identical across diseases when reduced to Φ profiles.

### Method

```
Step 1:   Take validated Arm 1 datasets (cancer + diabetes)
Step 2:   Reduce each to raw Φ trajectories only (strip all biological labels)
Step 3:   Compute per-dimension distributions:
            - Φ_pre distributions for cancer vs. diabetes
            - Φ_post distributions for cancer vs. diabetes
            - ΔΦ (recovery) distributions for cancer vs. diabetes
Step 4:   Statistical test: Are recovery distributions indistinguishable?
```

### Success Criteria
| Metric | Threshold | Meaning |
|---|---|---|
| KS test on ΔΦ distributions | p > 0.05 (fail to reject H₀ of same distribution) | Recovery profiles are statistically indistinguishable |
| Φ trajectory clustering | Disease labels not recoverable from Φ alone | Complexity profiles don't encode disease type |
| Archetype overlap | Same archetypes appear in both diseases | Pathology classification is topology-based |

### Analysis Pipeline

```python
from scipy.stats import ks_2samp
from models.complexity_profiler import ComplexityProfiler

# Compute Φ recovery vectors for both disease arms
delta_phi_cancer  = phi_post_cancer  - phi_pre_cancer    # shape: (n_cancer, 5)
delta_phi_diabetes = phi_post_diabetes - phi_pre_diabetes  # shape: (n_diabetes, 5)

# Per-dimension KS test
for dim in range(5):
    stat, pval = ks_2samp(delta_phi_cancer[:, dim], delta_phi_diabetes[:, dim])
    print(f"Φ_{dim}: KS={stat:.3f}, p={pval:.3f}")

# Universality holds if p > 0.05 for all dimensions
```

---

## Execution Timeline

| Phase | Description | Dependencies | Target |
|---|---|---|---|
| **1a** | Ingest first public oncology dataset | Data access | April 2026 |
| **1b** | Ingest first public metabolic dataset | Data access | April 2026 |
| **1c** | Arm 1 analysis (separate) | 1a + 1b | June 2026 |
| **2a** | Identify comorbidity cohort | Community submissions or All of Us | July 2026 |
| **2b** | Arm 2 analysis (conjoined) | 2a | August 2026 |
| **3** | Arm 3 analysis (universality) | 1c complete | September 2026 |
| **4** | Preprint submission | Arms 1+3 minimum | Q4 2026 |

---

## Reporting

Each validation arm produces a **Validation Report** containing:

1. **Dataset description** — source, size, sampling, variables
2. **Φ profiles** — pre/post/Δ for each patient (anonymized)
3. **CRS distribution** — histogram + summary statistics
4. **Archetype classification** — confusion matrix vs. clinical labels
5. **Statistical tests** — KS, correlation, AIC comparison
6. **Attractor visualizations** — phase portraits, Φ trajectory plots
7. **Failure analysis** — where the framework fails and why

All reports will be deposited in `validation/reports/` and referenced in future publications.
