# 🧬 Project Confluence

> **Redefining Precision Oncology: From Tumor Killing to Complexity Restoration**

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

## Repository Structure

```
project-confluence/
├── models/                          # Core computational modules
│   ├── complexity_profiler.py       # Module 1: 5D Φ vector
│   ├── patient_fitter.py            # Module 2: Bayesian digital twin
│   ├── drug_optimization_engine.py  # Module 3: RADO engine
│   ├── ode_system.py                # 15D SAEM ODE
│   ├── immune_dynamics.py           # Immune force field
│   ├── intervention.py              # Drug library (20+ drugs)
│   ├── realistic_failure.py         # Stochastic failure model
│   └── ferroptosis.py               # Iron-dependent cell death
├── agents/                          # Data agents
│   └── bioinformatics_miner.py      # Module 4: TCGA/cBioPortal
├── validation/                      # Safety & reference data
│   ├── clinical_guardrails.json     # CTCAE v5.0 constraints
│   └── gene_to_parameter_map.json   # Omics → ODE mapping
├── theory/                          # Mathematical framework
│   ├── unified_complexity_profile.md
│   ├── complex_attractor_definition.md
│   └── phase_transitions.md
├── tests/                           # Test suite
├── docs/                            # User documentation
└── notebooks/                       # Validation pipelines
```

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
| **Phase 2** | Retrospective validation (TCGA complexity vs. survival) | 🔄 In Progress |
| **Phase 2b** | Cross-disease complexity validation (3-arm protocol) | 📢 Call for Data posted |
| **Phase 3** | Prospective wet-lab (collaborator-dependent) | ⏳ Planned |

## Safety & Regulatory

- All protocols constrained by `clinical_guardrails.json` (CTCAE v5.0)
- Φ dimensions mapped to LOINC / SNOMED-CT codes
- FDA MIDD (Model-Informed Drug Development) aligned
- See [DISCLAIMER.md](DISCLAIMER.md) for medical use limitations

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
