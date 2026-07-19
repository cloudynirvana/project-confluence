# Project Confluence

**A Phi-vector framework for modeling shared metabolic dynamics across cancer types**

## Abstract

Cancer cells across tissue types converge on shared metabolic reprogramming
patterns (the Warburg effect and its extensions), but most models validate
against synthetic or single-cancer-type data, limiting claims of generality.
Project Confluence models this convergence directly using an ODE-based
state-space system anchored in a five-component Phi-vector - Phi_temporal,
Phi_informational, Phi_functional, Phi_spatial, and Phi_coupling - representing
distinct facets of metabolic-regulatory state.

Six enzymes central to glycolytic and oxidative metabolism (HK2, PKM2, LDHA,
IDH1/2, PDK1, G6PD) are mapped to specific channels in the ODE system as
grounded, biologically-interpretable state variables rather than abstract
parameters.

**Key result:** Replacing synthetic-data validation with six real CCLE
metabolomics channels (`CCLE_metabolomics_20190502.csv`; 225 metabolites x
928 cell lines) raises structurally identifiable parameters from 7/17 to
15/17, evaluated across three cancer types chosen for maximal biological
diversity - AML (blood), osteosarcoma (bone), and NSCLC (lung) - to support
generalizability claims beyond a single tissue context.

An adaptive controller built on this framework extends structurally to
theranostic applications (radioligand diagnostic-therapeutic pairing).

## Citation

See `CITATION.cff`. DOI badge added below once Zenodo publishes.

---

# 🧬 Project Confluence

> ⚠️ Status: Phase 1 computational validation only. No real patient data used.

> **Redefining Precision Oncology: From Tumor Killing to Complexity Restoration**

> Citation and attribution: this repository is MIT-licensed for open review and collaboration. If you use the code, theory, figures, or documentation, please cite the repository and credit Kelechi Ogbonna / cloudynirvana.

> Expert review invited: oncology, systems biology, control theory, clinical trial design, mathematical biology, and research-software reviewers are encouraged to audit assumptions, reproduce simulations, and challenge the validation plan before any translational claims are made.

> External validation preparation: see [validation/external_validation_pipeline.md](validation/external_validation_pipeline.md) for the PhysioNet, GDC, cBioPortal, and Hugging Face data-readiness plan.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-green.svg)](https://python.org)
[![Status: Computational Validation](https://img.shields.io/badge/Status-Computational%20Validation-orange.svg)](#validation-roadmap)

---

## Core Hypothesis

Health is not a fixed point but a **Complex Attractor State** characterized by adaptive variability, fractal rhythms, and moderate inter-system coupling. Disease is a transition to pathological attractors. **Therapy should restore the complexity, not just kill the tumor.**

```
Traditional Oncology:   Kill Cancer Cells → Measure Tumor Shrinkage
Project Confluence:     Restore Complexity → Measure Φ Improvement
```

## Theoretical Foundation: Bounded Adaptive Coherence (BAC)

> *A biological system sustains viable complexity if and only if the minimum singular value of its cross-scale coupling tensor exceeds the maximum normalised rate of local entropy production at any organisational scale.*

The BAC framework provides a **first-principles unification** of aging, cancer, and health as states of a single mathematical object — the **coupling tensor** $C(t)$, which governs causal coordination across biological scales (molecular → cellular → tissue → organism → evolutionary).

| Failure Mode | Coupling Tensor Signature | BAC Violation Type |
|---|---|---|
| **Aging** | Global off-diagonal decay of $C_{ij}$ | $\sigma_{\min}(C) \to 0$ uniformly |
| **Cancer** | Selective collapse of organism-scale pairs | $\sigma_{\min}(C) \to 0$ in specific sectors |
| **Health** | BAC condition satisfied with positive margin | $V(t) = \sigma_{\min}(C) - \max_k[\dot{s}_k] > 0$ |

The Φ vector is a **partial measurement** of the coupling tensor — the elements most relevant to cancer pathology. Biologics act as **coupling restoration operators** on specific $C_{ij}$ elements.

📄 **Full derivation:** [theory/bounded_adaptive_coherence.md](theory/bounded_adaptive_coherence.md)

## Unified Complexity Profile (UCP)

The framework operates on two complexity dimensions:

| Dimension | Symbol | Source | Purpose |
|-----------|--------|--------|---------|
| **Clinical Complexity** | Ψ (Psi) | EHR data, staging, genomics | Treatment difficulty |
| **Dynamical Complexity** | Φ (Phi) | Time-series physiology, modeling | Optimization target |

**Φ is a 5D vector:**

| Φ Dimension | Metric | Healthy Range | Biomarker |
|-------------|--------|---------------|-----------|
| Φ_temporal | Multiscale Entropy | 0.6–0.8 | HRV, glucose variability |
| Φ_spatial | Correlation Dimension D₂ | 3.0–6.0 | Cell diversity |
| Φ_functional | Recovery rate | 0.5–0.8 | Stress response |
| Φ_informational | λ_max + spectral slope | 0.5–0.7 | Signal entropy |
| Φ_coupling | Cross-system correlation | 0.4–0.7 | Immune-metabolic sync |

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌────────────────┐
│ Bioinformatics  │────▶│   Complexity     │────▶│   Patient    │────▶│     RADO       │
│     Miner       │     │   Profiler       │     │   Fitter     │     │    Engine      │
│   (Module 4)    │     │   (Module 1)     │     │  (Module 2)  │     │   (Module 3)   │
└─────────────────┘     └──────────────────┘     └──────────────┘     └────────────────┘
   TCGA/cBioPortal         5D Φ vector            Digital Twin         Optimized Protocol
   Omics extraction        Archetype ID           Bayesian MCMC        Complexity restoration
```

## Adaptive Therapy Controller (NEW)

> *"The optimal therapy is an algorithm, not a prescription."* — First Principles Deconstruction, Axiom 10

Project Confluence now includes a **closed-loop adaptive therapy controller** that treats dosing as a real-time policy decision, not a fixed protocol.

### Key Innovation

Instead of optimizing for a static dose (e.g., "DCA at 25mg for 60 days"), the system optimizes the **hyperparameters of an adaptive policy** — when to dose, when to hold, and how to respond to resistance signals.

```
Traditional:  Optimizer → Fixed Dose Schedule → Patient
Confluence:   Optimizer → Adaptive Policy π(state) → Dynamic Dosing → Patient
```

### Three Policy Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Threshold** | Bang-bang control with hysteresis | Simple on/off dosing |
| **Proportional** | Dose scales with tumor burden | Continuous dose adjustment |
| **RobustAdaptive** | Threshold + resistance-aware + uncertainty margins | Full Confluence policy |

### Safety Constraints (Assurance Layer)

All policies are bounded by hard safety constraints that **cannot be overridden**:
- Absolute dose cap (robust_max_dose)
- Forced drug holidays after max continuous dosing
- Minimum holiday duration
- Cumulative toxicity budget

### Monte Carlo Validation: 200 Uncertain Biological Scenarios

| Metric | MTD (Standard Care) | Confluence Adaptive |
|--------|---------------------|---------------------|
| **Resistant Takeover Rate** | 178/200 (89.0%) | **1/200 (0.5%)** |
| **Tumor Controlled at Day 180** | 200/200 (100%) | 36/200 (18.0%) |
| **Mean Final Tumor Burden** | 0.271 | 0.952 |
| **Mean Final Resistant Fraction** | 91.5% | **11.9%** |

The adaptive policy achieves near-zero resistant takeover (1/200 scenarios) across 200 random biological parameter sets sampled from the uncertainty set. The tradeoff is explicit: it preserves evolutionary containment at the cost of short-horizon tumor shrinkage. MTD keeps burden smaller but selects for resistance in 89% of scenarios. The adaptive controller maintains sensitive-cell competitive suppression of resistant clones — the ecological mechanism adaptive therapy is designed to exploit.

```bash
# Run the comparison
python validate_controller.py

# Run full Monte Carlo analysis (200 samples, ~5 min)
python scripts/monte_carlo_uncertainty.py
```

### Mathematical Core — 15D SAEM Model

The patient state **z** ∈ ℝ¹⁵ evolves under:

```
dz/dt = F(z, θ, u)

Metabolic (10D):     Glucose, Lactate, Pyruvate, ATP, NADH,
                     Glutamine, Glutamate, αKG, Citrate, ROS
Immune (3D):         I_eff, I_reg, I_exhaust
Microenvironment (2D): σ_stromal, ν_vascular
```

Nonlinearity via Michaelis-Menten kinetics → **strange attractor dynamics**.

```mermaid
graph TD
    subgraph Scale 0: Molecular (z0-z4)
        M1[Glucose/Lactate Flux] <--> M2[ATP/NADH Energetics]
    end

    subgraph Scale 1: Cellular (z5-z9)
        C1[Glutamine/alpha-KG] <--> C2[ROS Accumulation]
    end

    subgraph Scale 2: Organismal (z10-z12)
        O1[Effector T-Cells] <--> O2[Tregs / Exhaustion]
    end

    subgraph Scale 3: Tissue (z13-z14)
        T1[Stromal Density] <--> T2[Vascular Integrity]
    end

    %% Cross-Scale Coupling Tensor Channels C_ij
    M2 -- "C_01 (Metabolic feedback)" --> C2
    C2 -- "C_12 (Stress-immune gating)" --> O1
    O2 -- "C_23 (Immune-stroma pruning)" --> T1
    T2 -- "C_30 (Vascular glucose supply)" --> M1
```

## Quick Start

```bash
# Clone
git clone https://github.com/cloudynirvana/project-confluence.git
cd project-confluence

# Install dependencies
pip install -r requirements.txt

# Run complexity profiling
python -c "
from models.complexity_profiler import ComplexityProfiler
from models.ode_system import ComplexAttractorODE

ode = ComplexAttractorODE()
result = ode.solve(t_span=(0, 200), dt_eval=0.5)
profiler = ComplexityProfiler()
phi = profiler.profile(result['z'], dt=0.5)
print(phi.to_json())
"
```

## Convergence Implementation Status

The current computational stack now implements the four Codex convergence prompts:

| Layer | Implementation | Verification |
|-------|----------------|--------------|
| Quantum scale k0 | `ComplexAttractorODE` is extended to 16D with `psi_coherent`; `CouplingTensorAnalyzer` computes a 5-scale tensor and direct `C_02` quantum-to-cellular coupling. | `tests/test_ode_system.py`, `tests/test_coupling_tensor.py` |
| OSKM steering | `PolicyMode.EPIGENETIC_STEERING` emits pulsatile OSKM dosing from identity metrics with Landauer thermal override holidays. | `tests/test_adaptive_controller.py` |
| Curvature bottlenecks | `scripts/detect_curvature_bottlenecks.py` exports a Forman-Ricci JSON report and network plot for cellular-organismal bottlenecks. | `results/curvature_bottlenecks/` |
| Memory-kernel EKF | `ExtendedKalmanFilterObserver` estimates `[z, vec(M_neural)]` and accepts DMN coherence plus EEG PCI measurement channels. | `tests/test_optimal_inference.py` |

Focused validation:

```bash
python -B -m pytest tests/test_adaptive_controller.py tests/test_ode_system.py tests/test_coupling_tensor.py tests/test_optimal_inference.py -q
python -B scripts/detect_curvature_bottlenecks.py
```

## PDAC Rogue Closure Model

Project Confluence now includes a disease-specific executable scaffold for pancreatic ductal adenocarcinoma (PDAC):

```text
PDAC persistence = KRAS/RAS driver closure
                 + EGFR/STAT3 bypass recovery
                 + stromal/glycocalyx shielding
                 + immune exclusion
                 + therapy-selected resistance
```

Run the synthetic workflow:

```bash
python scripts/run_pdac_rogue_closure.py --all-scenarios
```

Validation data links and the real-data plan are in [`validation/pdac_data_sources.md`](validation/pdac_data_sources.md). The committed PDAC time series in `results/pdac_rogue_closure/` is synthetic and exists for reproducibility; raw public datasets should be fetched from GDC, cBioPortal, GEO, DepMap, PDMR, PDX Finder, GlyGen, and GlyConnect rather than stored directly in the repository.


## 🦞 AutoResearchClaw Integration

Generate a full conference paper from Project Confluence's models with one command:

```bash
python scripts/run_autoresearch.py phi-universality
python scripts/run_autoresearch.py --list-topics
```

**Pre-built topics:** phi-universality · drug-scheduling · immune-metabolic · ferroptosis-complexity · digital-twin

AutoResearchClaw runs 23 stages autonomously — literature review, hypothesis debate, experiments using Confluence's ODE system, peer review, and LaTeX paper. No GPU required.

**Config:** config.arc.yaml | **Prompts:** prompts.confluence.yaml

## Repository Structure

```
project-confluence/
├── models/                          # Core computational modules
│   ├── adaptive_controller.py       # Closed-loop adaptive therapy controller
│   ├── clonal_dynamics.py           # Lotka-Volterra clonal competition engine
│   ├── resistance_model.py          # Multi-mechanism resistance tracker
│   ├── complexity_profiler.py       # Module 1: 5D Φ vector
│   ├── patient_fitter.py            # Module 2: Bayesian digital twin
│   ├── drug_optimization_engine.py  # Module 3: RADO engine
│   ├── ode_system.py                # 15D SAEM ODE
│   ├── immune_dynamics.py           # Immune force field
│   ├── intervention.py              # Drug library (20+ drugs)
│   ├── geometric_optimization.py    # Basin curvature, Kramers escape, Flatten-Heat-Push
│   ├── geometric_pathways.py        # Freidlin-Wentzell MAP via String Method
│   ├── fisher_geometry.py           # Fisher Information Matrix / stiff-sloppy (MBAM)
│   ├── network_curvature.py         # Forman-Ricci curvature bottleneck detection
│   ├── realistic_failure.py         # Stochastic failure model
│   ├── ferroptosis.py               # Iron-dependent cell death
│   ├── coupling_tensor.py           # Block Jacobian cross-scale C_ij tensor
│   ├── optimal_inference.py         # EKF state & coupling tensor observer
│   ├── lyapunov_certificate.py      # Universal Complexity Sustainment — CLF certifier
│   └── identity_tensor.py           # Φ-Unification Identity Tensor — consciousness preservation
├── scripts/
│   ├── monte_carlo_uncertainty.py   # 200-sample uncertainty validation
│   ├── test_pathways.py             # Geometric calibration integration test
│   ├── test_sustainment.py          # Sustainment Theorem validation (4 scenarios)
│   ├── test_identity.py             # Identity Tensor validation (5 scenarios)
│   ├── clonal_evolution_sim.py      # Adaptive vs MTD comparison
│   ├── confluence_runner.py         # Full pipeline runner
│   ├── optimize_biomarker_panel.py  # EKF biomarker selection optimization
│   └── ...                          # Data agents, validation scripts
├── agents/                          # Data agents
│   └── bioinformatics_miner.py      # Module 4: TCGA/cBioPortal
├── validation/                      # Safety & reference data
│   ├── clinical_guardrails.json     # CTCAE v5.0 constraints
│   └── gene_to_parameter_map.json   # Omics → ODE mapping
├── theory/                          # Mathematical framework
│   ├── age_reversal_transfer.md          # Scaling BAC & C_ij framework to biogerontology
│   ├── bounded_adaptive_coherence.md     # BAC first-principles theory
│   ├── complexity_sustainment.md         # Optimal complexity maintenance (cancer vs aging)
│   ├── optimal_inference_design.md       # Inference of C_ij from sparse clinical observations
│   ├── sustained_complexity_and_death.md # Biophysics & thermodynamics of death
│   ├── deepmind_executive_brief.md       # Proposal for DeepMind & Isomorphic Labs integration
│   ├── geometric_calibration_research.md  # Geometric calibration research proposal
│   ├── quantum_criticality_and_unison.md  # Penrose Orch OR × BAC quantum-classical integration
│   ├── universal_sustainment_theorem.md   # Control Lyapunov proof for indefinite sustainment
│   └── consciousness_complexity_bridge.md  # IIT × BAC Φ-Unification — identity preservation theory
├── tests/                           # Test suite (11 test files)
├── docs/                            # User documentation
└── notebooks/                       # Validation pipelines
```

## 🗺️ Mathematical-to-Code Mapping Registry

To bridge abstract biophysical theory with verified computational executions, use the following translation map linking the mathematical papers to their Python modules:

| Biophysical Equation / Concept | Mathematical Theory Paper | Executable Python Module | Verification Test Suite |
| :--- | :--- | :--- | :--- |
| **16D Spectral Attractor (SAEM + k0)** | [`theory/optimal_inference_design.md`](theory/optimal_inference_design.md), [`theory/quantum_criticality_and_unison.md`](theory/quantum_criticality_and_unison.md) | [`models/ode_system.py`](models/ode_system.py) | `tests/test_ode_system.py` |
| **5x5 Cross-Scale Coupling Tensor $C_{ij}$** | [`theory/complexity_sustainment.md`](theory/complexity_sustainment.md) | [`models/coupling_tensor.py`](models/coupling_tensor.py) | [`tests/test_coupling_tensor.py`](tests/test_coupling_tensor.py) |
| **EKF Observer + Memory Kernel $M(t)$** | [`theory/optimal_inference_design.md`](theory/optimal_inference_design.md), [`theory/consciousness_complexity_bridge.md`](theory/consciousness_complexity_bridge.md) | [`models/optimal_inference.py`](models/optimal_inference.py) | [`tests/test_optimal_inference.py`](tests/test_optimal_inference.py) |
| **OED Sensor Selection Matrix $H$** | [`theory/optimal_inference_design.md`](theory/optimal_inference_design.md) | [`scripts/optimize_biomarker_panel.py`](scripts/optimize_biomarker_panel.py) | *Runs combinatorial validation* |
| **Stochastic Laboratory Calibration** | [`theory/problem_statement_and_justification.md`](theory/problem_statement_and_justification.md) | [`scripts/stochastic_noise_sweep.py`](scripts/stochastic_noise_sweep.py) | *Assay noise sweeps* |
| **Universal Sustainment Theorem (CLF)** | [`theory/universal_sustainment_theorem.md`](theory/universal_sustainment_theorem.md) | [`models/lyapunov_certificate.py`](models/lyapunov_certificate.py) | [`scripts/test_sustainment.py`](scripts/test_sustainment.py) |
| **Φ-Unification (IIT × BAC Bridge)** | [`theory/consciousness_complexity_bridge.md`](theory/consciousness_complexity_bridge.md) | [`models/identity_tensor.py`](models/identity_tensor.py) | [`scripts/test_identity.py`](scripts/test_identity.py) |
| **Bioinformatics Parameter Mapping** | [`theory/geometric_calibration_research.md`](theory/geometric_calibration_research.md) | [`agents/bioinformatics_miner.py`](agents/bioinformatics_miner.py) | `tests/test_bioinformatics.py` |
| **Genomic Cohort Reconstructor** | [`theory/problem_statement_and_justification.md`](theory/problem_statement_and_justification.md) | [`scripts/reconstruct_tcga_patients.py`](scripts/reconstruct_tcga_patients.py) | *TCGA diagnostic outputs* |

## Pan-Cancer Support

| Cancer Type | Metabolic Profile | Key Vulnerability |
|-------------|-------------------|-------------------|
| TNBC | Warburg + glutamine addiction | Glycolysis inhibition |
| PDAC | Extreme glycolysis + stromal barrier | Stromal depletion |
| NSCLC | Moderate glycolysis | OXPHOS targeting |
| Melanoma | OXPHOS-dependent | ETC inhibition |
| GBM | High glycolysis + neurotransmitter crosstalk | Glucose deprivation |
| CRC | MSI-H, moderate Warburg | Immunotherapy + metabolic |
| HGSOC | Glutamine-dependent | GLS1 inhibition |
| mCRPC | Lipogenesis from citrate | Citrate diversion block |
| AML | OXPHOS + glutamine | Combined metabolic attack |
| HCC | Extreme Warburg + lipogenesis | Multi-pathway inhibition |

## 📢 Call for Data

We are seeking **longitudinal pathology and omics datasets** to validate Confluence across oncology, metabolic disease, and comorbidities. Static snapshots are insufficient — we need time-series data that allows reconstruction of complexity profiles.

**We welcome:** Cancer time-series · Diabetes/metabolic longitudinal data · Comorbidity cohorts · **Negative results**

📄 **Full details:** [CALL_FOR_DATA.md](CALL_FOR_DATA.md)
📋 **Submission template:** [data_submission_template.json](validation/data_submission_template.json)
🔬 **What we measure:** [complexity_signature.md](validation/complexity_signature.md)

### Three-Arm Validation Strategy

| Arm | Goal | Success Metric |
|-----|------|----------------|
| **Separate** | Confluence works on Cancer *and* Diabetes individually | Same equations identify tipping points in both |
| **Conjoined** | Handles coupled comorbidity systems | Predicts cross-domain interaction effects |
| **Universality** | Mathematics is disease-agnostic | Φ recovery profiles statistically indistinguishable |

📋 **Full protocol:** [validation_protocol.md](validation/validation_protocol.md)

## Validation Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Computational validation (1000-trial Monte Carlo) | ✅ Complete |
| **Phase 1b** | Adaptive therapy Monte Carlo (200 uncertain scenarios) | ✅ Complete |
| **Phase 2** | Retrospective validation (TCGA complexity vs. survival) | 🔄 In Progress |
| **Phase 2b** | Cross-disease complexity validation (3-arm protocol) | 📢 Call for Data posted |
| **Phase 3** | Prospective wet-lab (collaborator-dependent) | ⏳ Planned |

## Validation Walkthrough (Phase 1 Snapshot)

Results below are from `scripts/disease_poc.py` with output captured in `poc_results.txt`.

| Disease | |Phi| | Coherence | Dist. from Healthy |
|---------|------|-----------|--------------------|
| Healthy | 1.3199 | 0.2628 | -- |
| Glioblastoma | 1.5577 | 0.5593 | 0.6732 |
| TNBC | 1.4490 | 0.5014 | 0.5794 |
| Alzheimers | 1.3028 | 0.3650 | 0.3792 |
| Nephroblastoma | 1.3486 | 0.3473 | 0.2932 |
| Diabetes | 1.3547 | 0.3649 | 0.2404 |
| Parkinsons | 1.2171 | 0.2355 | 0.2059 |
| Lupus | 1.2522 | 0.2015 | 0.1574 |
| ALS | 1.2769 | 0.1982 | 0.1507 |

In this snapshot, Glioblastoma is the furthest from healthy (0.6732), exceeding TNBC.
Lupus shows the lowest coherence (0.2015), aligned with the autoimmune hyperactivation settings in `LupusParams`.
ALS and Lupus are closest to healthy (0.1507 and 0.1574), indicating subtle early-stage deviations in this model.

TNBC vs Nephroblastoma distance: 0.3076.

Per-dimension divergence (TNBC vs Nephroblastoma):

| Dimension | Healthy | TNBC | Nephro | D(TNBC-Nephro) |
|-----------|---------|------|--------|----------------|
| Phi_temporal | 0.4897 | 0.3901 | 0.3684 | 0.0217 |
| Phi_spatial | 0.2786 | 0.3224 | 0.3191 | 0.0033 |
| Phi_functional | 0.9757 | 0.9829 | 0.9871 | 0.0042 |
| Phi_informational | 0.2882 | 0.8267 | 0.5434 | 0.2833 |
| Phi_coupling | 0.6242 | 0.4403 | 0.5581 | 0.1178 |

Therapeutic simulation (Nephroblastoma):

| Intervention | Phi-distance (pre) | Phi-distance (post) | Restoration | Notes |
|--------------|--------------------|---------------------|-------------|-------|
| IGF2R monotherapy (IGF2_signaling: 0.75 -> 0.30) | 0.2932 | 0.2302 | 21.5% | 3/5 dimensions shift toward healthy |
| IGF2R + WT1 mRNA (WT1_activity: 0.20 -> 0.55) | 0.2932 | 0.2186 | 25.5% | 4.0% synergy gain vs mono |

TCGA retrospective (Track A, synthetic cohort):

| Disease | Phi-dist | Survival (d) | Spearman rho | HR |
|---------|----------|--------------|--------------|----|
| TNBC | 0.4869 | 275 | -0.8220 | 9.83 |
| Alzheimers | 0.3871 | 1005 | -0.9181 | 1.29 |
| ALS | 0.1769 | 1078 | -0.8358 | 1.24 |
| Diabetes | 0.2042 | 1532 | -0.7753 | 1.17 |
| Parkinsons | 0.1796 | 1486 | -0.7904 | 1.07 |
| Nephroblastoma | 0.2811 | 1338 | -0.3437 | 1.04 |
| Lupus | 0.1934 | 1612 | -0.7566 | 1.13 |
| Glioblastoma | 0.6195 | 387 | -0.8376 | 1.95 |

Overall Spearman rho (240 patients): -0.7937.
Glioblastoma now shows a strong negative rho after scaling, consistent with its aggressiveness.

Reproduce locally:

```bash
python scripts/disease_poc.py > poc_results.txt 2>&1
```

```bash
python scripts/tcga_retrospective.py > tcga_output.txt 2>&1
```

TCGA retrospective results are saved to `results/tcga_val/retrospective_metrics.json`.

Track B ingestion (longitudinal cohort):

```bash
python scripts/tcga_track_b.py --input data/track_b/mock_cohort.json
```

Track B results are saved to `results/tcga_val/track_b_metrics.json`.
Use `--use-neural-ode` to reconstruct trajectories if torchdiffeq is installed.
Note: With fewer than 3 patients, Spearman rho is not statistically meaningful (2-point rho will be ±1 by definition).

To generate a pinned lockfile (`requirements.lock.txt`) on a machine with Python installed:

```powershell
powershell -File scripts/pin_requirements.ps1
```

## Safety & Regulatory

- All protocols constrained by `clinical_guardrails.json` (CTCAE v5.0)
- Φ dimensions mapped to LOINC / SNOMED-CT codes
- FDA MIDD (Model-Informed Drug Development) aligned
- See [DISCLAIMER.md](DISCLAIMER.md) for medical use limitations

## 🇳🇬 Nigeria Clinical Guidelines Integration

Project Confluence integrates the **Nigeria Standard Treatment Guidelines (NSTG 2022)** — 270 structured clinical conditions published by the Federal Ministry of Health, Nigeria — as a RAG (Retrieval-Augmented Generation) layer for guideline-aware precision oncology.

> **Data Source**: [chisomrutherford/nigeria-clinical-guidelines-dataset](https://huggingface.co/datasets/chisomrutherford/nigeria-clinical-guidelines-dataset)
> **License**: CC-BY-4.0 | **Curated by**: Chisom Rutherford

### What This Adds

| Feature | Description |
|---------|-------------|
| **NigeriaGuidelineRetriever** | Semantic search (RAG) over all 270 NSTG conditions with FAISS + sentence-transformers |
| **Nigeria-Specific Guardrails** | Adjusted safety thresholds for malaria, HIV, sickle cell, anaemia comorbidities |
| **Guideline-Aware Controller** | Adaptive therapy controller with NSTG 2022 safety layer |
| **Resource-Aware Dosing** | Drug availability tiers (commonly/intermittently/rarely available in Nigeria) |
| **Clinical Query API** | FastAPI endpoints for real-time guideline retrieval |

### Quick Start

```python
from agents.nigeria_guideline_retriever import NigeriaGuidelineRetriever

# Initialize (downloads from HuggingFace on first run, or uses built-in mock data)
retriever = NigeriaGuidelineRetriever()

# Semantic search
results = retriever.retrieve("first-line treatment for breast cancer in Nigeria")
for r in results:
    print(f"[{r.score:.3f}] {r.chunk.condition_name}: {r.chunk.text[:100]}")

# Structured clinical answer
print(retriever.answer("What is the dosing for cisplatin in cervical cancer?"))

# Direct protocol lookup
protocol = retriever.get_treatment_protocol("BREAST CANCER")

# Drug-specific constraints
constraints = retriever.get_dosing_constraints("doxorubicin")
```

### Guideline-Aware Adaptive Controller

```python
from models.adaptive_controller import AdaptiveController, PolicyMode

# Controller auto-loads Nigeria guardrails if JSON exists
controller = AdaptiveController(
    policy_mode=PolicyMode.ROBUST_ADAPTIVE,
    guideline_retriever=retriever,
    cancer_type="TNBC",
)

# Summary includes Nigeria guidelines status
print(controller.get_summary())
# → {"nigeria_guidelines_active": true, ...}
```

### API Endpoints

```bash
# Query guidelines (semantic search)
curl -X POST http://localhost:8000/guideline_query \
  -H "Content-Type: application/json" \
  -d '{"query": "management of neutropenia during chemotherapy", "top_k": 5}'

# List all 270 conditions
curl http://localhost:8000/guideline_conditions

# Get specific protocol
curl http://localhost:8000/guideline_protocol/breast%20cancer

# Get drug constraints
curl http://localhost:8000/guideline_drug/doxorubicin
```

### Install Optional Dependencies

```bash
pip install sentence-transformers faiss-cpu datasets
```

Without these, the retriever falls back to TF-IDF/keyword matching (still functional, lower accuracy).

## Contributing

We welcome contributions from computational biologists, oncologists, and dynamical systems researchers. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{ogbonna2026confluence,
  author = {Ogbonna, Kelechi},
  title = {Project Confluence: Complexity-Restoring Precision Oncology Framework},
  year = {2026},
  url = {https://github.com/cloudynirvana/project-confluence}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Disclaimer

This is a **research framework** for computational exploration. It is **not** a medical device, clinical decision support system, or diagnostic tool. See [DISCLAIMER.md](DISCLAIMER.md).

---

*"The measure of health is not the absence of disease, but the presence of complexity."*
