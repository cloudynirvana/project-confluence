# 📢 Call for Longitudinal Pathology & Omics Data

> **Benchmarking Dynamical Disease Frameworks — Project Confluence**

[![Status: Accepting Submissions](https://img.shields.io/badge/Status-Accepting%20Submissions-brightgreen.svg)](#how-to-submit)

---

## Objective

Project Confluence is validating a **universal dynamical systems framework** that models disease progression and therapeutic response based on **complexity profiles** rather than disease-specific labels.

We seek wet-lab validation datasets to test whether shared mathematical structures underlie therapeutic success across distinct pathologies — **Oncology**, **Metabolic Disease**, and **Comorbidities**.

> **Core Hypothesis:** Disease is a loss of regulatory complexity (a shift to a pathological attractor). Therapeutic success is a restoration of healthy dynamics. If true, the *same mathematics* should identify therapeutic windows in both a tumor regression dataset and a glycemic control dataset.

---

## What We Mean by "Complexity"

Confluence does not analyze individual genes or proteins. It analyzes **state-space trajectories** — how a patient's measurable variables evolve over time. From these trajectories, we compute a 5-dimensional **Φ (Phi) vector**:

| Φ Dimension | What It Measures | Computation | Healthy Range |
|---|---|---|---|
| **Φ_temporal** | Temporal regularity | Multiscale Entropy (MSE) | 0.6–0.8 |
| **Φ_spatial** | State-space dimensionality | Correlation Dimension (D₂) | 3.0–6.0 |
| **Φ_functional** | Perturbation recovery | Recovery rate after stress | 0.5–0.8 |
| **Φ_informational** | Predictability structure | Lyapunov exponent + spectral slope | 0.5–0.7 |
| **Φ_coupling** | Cross-system coherence | Inter-variable correlation | 0.4–0.7 |

**A "therapeutic success" in Confluence terms** is a measurable shift in Φ from a pathological archetype back toward the healthy complex attractor.

For full definitions, see [`validation/complexity_signature.md`](validation/complexity_signature.md).

---

## Data We Are Seeking

### Required: Time-Series / Longitudinal Data

> [!IMPORTANT]
> Static snapshots (single biopsy, single blood draw) **cannot** reveal complexity profiles. We need **multiple timepoints** per patient, ideally spanning pre-intervention → intervention → post-intervention.

### 1. Separate Pathologies

**Oncology Datasets**
- Longitudinal tumor measurements (imaging, liquid biopsy, ctDNA)
- Multi-timepoint omics (RNA-seq, proteomics, metabolomics)
- Treatment response outcomes (RECIST, pathological response, survival)
- Any cancer type — we are disease-agnostic

**Metabolic Disease Datasets**
- Longitudinal metabolic markers (HbA1c, fasting glucose, insulin, C-peptide)
- Continuous glucose monitoring (CGM) traces (ideal for entropy calculation)
- Intervention outcomes (medication changes, lifestyle modifications, bariatric surgery)
- Type 1, Type 2, MODY, or gestational diabetes

### 2. Conjoined Pathologies (Comorbidity)

- Patients with **both** malignancy and metabolic dysfunction
- Data tracking how intervention in one domain affects the other
- Examples: metformin use in breast cancer, diabetes onset post-chemotherapy, pancreatic cancer with glucose dysregulation

### 3. Negative Results

> [!TIP]
> Data where therapeutic value was **not** found is equally valuable. Failed interventions define the boundaries of pathological attractor basins and help us characterize treatment-resistant complexity states.

---

## Ideal Dataset Features

| Feature | Priority | Why It Matters |
|---|---|---|
| **≥3 timepoints per patient** | 🔴 Critical | Minimum for trajectory reconstruction |
| **Pre- and post-intervention** | 🔴 Critical | Needed to measure Φ shift |
| **Sampling frequency recorded** | 🟡 High | Required for entropy calculation |
| **Multi-modal** (omics + clinical) | 🟡 High | Enables cross-domain Φ_coupling |
| **Intervention metadata** (dose, timing, type) | 🟡 High | Required for perturbation modeling |
| **Raw measurements** (not just summary stats) | 🟢 Preferred | Enables full complexity profiling |
| **≥20 patients per cohort** | 🟢 Preferred | Statistical power for archetype classification |

---

## Metadata Schema

For each submitted dataset, please provide these fields (see [`validation/data_submission_template.json`](validation/data_submission_template.json) for the full schema):

```json
{
  "dataset_name": "...",
  "pathology_class": "oncology | metabolic | comorbidity",
  "sampling_frequency": "daily | weekly | monthly | irregular",
  "n_timepoints_per_patient": "median value",
  "intervention_timestamp": "relative day/hour of intervention start",
  "outcome_trajectory": "continuous | categorical",
  "variables_measured": ["list of measured quantities"],
  "variable_variance_available": true
}
```

---

## What We Offer in Return

### 🔬 Complexity Analysis (Free)
We will run the full Confluence pipeline on your dataset and return:
- **Φ profile** for each patient (5D complexity vector over time)
- **Archetype classification** (Chaotic/Decoupled, Rigid/Locked, Collapsed/Exhausted)
- **Complexity recovery score** (pre→post intervention Φ shift)
- **Attractor landscape visualization** (phase portraits, bifurcation diagrams)

### 📝 Validation Partnership
- Co-authorship on publications demonstrating cross-disease dynamical universality
- Named acknowledgment in the Confluence validation registry
- Priority access to framework updates and analytical tools

### 📦 FAIR Data Support
- Assistance depositing data in compliant repositories (GEO, NCI IDC, NIDDK Central Repository)
- DOI generation for your contribution
- Metadata standardization to LOINC/SNOMED-CT codes

---

## How to Submit

### Option A: Full Dataset Sharing
Share your dataset directly (CSV, HDF5, or database export) with metadata using our [submission template](validation/data_submission_template.json).

📧 **Contact:** [kelechi@projectconfluence.org]  
🔗 **Upload:** [GitHub Issues](https://github.com/cloudynirvana/project-confluence/issues/new?template=data_submission.md)

### Option B: Federated Complexity Analysis
If data cannot leave your institution, we provide the **Confluence Complexity Script** — a self-contained Python package. You run it locally and send us only the **anonymized Φ profiles** (no patient-level data leaves your system).

```bash
# Install
pip install confluence-profiler  # (coming soon)

# Run
python -m confluence.federated --input your_data.csv --output phi_profiles.json
```

### Option C: Existing Public Data Nomination
Know of a public dataset (GEO, TCGA, NIDDK, All of Us) that meets our criteria? Open an issue or email us the accession number — we'll handle the rest.

---

## Target Data Sources

| Repository | Disease Area | URL |
|---|---|---|
| NCI Imaging Data Commons (IDC) | Cancer pathology + imaging | [portal.imaging.datacommons.cancer.gov](https://portal.imaging.datacommons.cancer.gov) |
| NIDDK Central Repository | Diabetes & metabolic | [repository.niddk.nih.gov](https://repository.niddk.nih.gov) |
| All of Us Research Program | Comorbidity (cancer + metabolic) | [researchallofus.org](https://researchallofus.org) |
| GEO / ArrayExpress | Multi-omics time-series | [ncbi.nlm.nih.gov/geo](https://www.ncbi.nlm.nih.gov/geo) |
| TCGA (via GDC) | Pan-cancer omics | [portal.gdc.cancer.gov](https://portal.gdc.cancer.gov) |
| PhysioNet | Physiological time-series | [physionet.org](https://physionet.org) |

---

## Validation Arms

This data call supports a **three-arm validation strategy** (full protocol: [`validation/validation_protocol.md`](validation/validation_protocol.md)):

| Arm | Goal | Success Metric |
|---|---|---|
| **Separate** | Confluence works on Cancer *and* Diabetes individually | Same equations identify therapeutic tipping points in both |
| **Conjoined** | Confluence handles coupled disease systems | Model predicts cross-domain interaction effects |
| **Universality** | The mathematics is disease-agnostic | Complexity recovery profiles are statistically indistinguishable across diseases |

---

## Timeline

| Milestone | Target Date |
|---|---|
| Call for Data posted | March 2026 |
| First public dataset ingested | April 2026 |
| Separate Arm validation complete | June 2026 |
| Conjoined Arm pilot | August 2026 |
| Universality analysis preprint | Q4 2026 |

---

## Framework Preview

- 🧬 **GitHub:** [github.com/cloudynirvana/project-confluence](https://github.com/cloudynirvana/project-confluence)
- 📄 **Theory:** [Unified Complexity Profile](theory/unified_complexity_profile.md)
- 🔧 **Code:** [Complexity Profiler](models/complexity_profiler.py) — 854 lines, fully implemented

---

## Citation

If you use Project Confluence or reference this Call for Data:

```bibtex
@software{ogbonna2026confluence,
  author = {Ogbonna, Kelechi},
  title = {Project Confluence: Universal Dynamical Framework for Therapeutic Validation},
  year = {2026},
  url = {https://github.com/cloudynirvana/project-confluence}
}
```

---

*"The measure of health is not the absence of disease, but the presence of complexity."*
