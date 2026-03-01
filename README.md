# 🧬 Project Confluence — A Hypothetical Universal Cancer Cure Framework

> **Computational research prototype** exploring cancer therapy optimization through geometric attractor dynamics. **Not for clinical use.** See [DISCLAIMER.md](DISCLAIMER.md).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![Status: Hypothesis](https://img.shields.io/badge/status-computational%20hypothesis-orange.svg)]()

## What Is This?

Project Confluence is a computational framework that models cancer metabolic states as **stochastic attractors** in a 10-dimensional phase space and explores whether a structured 3-phase therapy protocol can systematically destabilize these attractors — pushing the system toward a healthy equilibrium.

The core idea: if cancer is a "trapped" metabolic state (an attractor basin), then therapy can be designed as a **geometric escape operation**:

```
Phase 1: FLATTEN  — Reduce attractor basin depth with metabolic drugs
Phase 2: HEAT     — Inject noise to destabilize the flattened basin  
Phase 3: PUSH     — Apply immune force toward the healthy attractor
```

This is a **hypothetical framework** — all results are in silico simulations that have not been experimentally validated.

## ⚠️ Important Limitations (Read First)

Before diving into the code or results, understand what this project is and isn't:

| What This Is | What This Is NOT |
|---|---|
| A computational hypothesis generator | A validated clinical tool |
| An in silico proof-of-concept | A substitute for real wet-lab experiments |
| A framework for thinking about therapy design | A treatment recommendation system |
| An invitation for expert critique | A finished, peer-reviewed study |

**Key assumptions that may not hold in reality:**
1. **10-metabolite approximation** — Real tumors have spatial gradients, clonal heterogeneity, and epigenetic dimensions beyond 10 metabolites
2. **Linear drug additivity** — Real drug interactions are non-linear and dose/schedule-dependent
3. **Simplified immune model** — Single-compartment; doesn't capture T-cell trafficking or tissue barriers fully
4. **Population-level generators** — Not calibrated to individual patients
5. **No experimental validation** — Zero wet-lab, animal, or clinical data supports these computational results

## Architecture

```
src/                            # Core computational engine (12 modules)
├── tnbc_ode.py                 # 10-metabolite ODE systems (10 cancer types + healthy)
├── geometric_optimization.py   # Basin curvature, Kramers escape rate, optimizer
├── intervention.py             # 19-drug library with PK engine and synergy matrix
├── immune_dynamics.py          # Multi-compartment immune force field
├── coherence.py                # Spectral coherence analysis
├── clonal_dynamics.py          # Lotka-Volterra 2-clone competition model
├── toxicity_constraints.py     # Clinical safety constraints (MTD, organ overlap)
├── protocol_translator.py      # Simulation → wet-lab protocol conversion
├── resistance_model.py         # Multi-mechanism resistance tracking
├── spatial_dynamics.py         # 3-compartment tumor model (core/rim/stroma)
├── calibration.py              # Parameter calibration engine
└── restoration.py              # Generator correction computation

confluence_runner.py             # Main pipeline: runs all 10 cancers end-to-end
universal_cure_engine.py         # Enhanced engine with sensitivity analysis

results/                         # Simulation outputs
├── executive_summary.md         # Start here — one-page overview
├── confluence_report.md         # Full pan-cancer results
├── gaps_and_limitations.md      # Honest risk assessment
├── latiff.md                    # Mathematical formalism (LATIFF)
├── protocols/                   # Per-cancer lab-ready protocols (10 × md + json)
└── data/                        # Raw JSON simulation data

tests/                           # Test suites
├── test_universal_cure.py       # Unit tests (generators, interventions, resistance)
├── test_adversarial.py          # Adversarial stress tests (break the model)
└── test_optimization.py         # Protocol optimization tests

examples/                        # Runnable demos
tools/                           # Development utilities (calibration, sweeps)
visualization/                   # Interactive dashboard
docs/                            # Community posting drafts
```

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy scipy scikit-learn matplotlib
```

### 2. Run the Full Pipeline

```bash
python confluence_runner.py --all --lab-protocols
```

This runs the Geometric Achievement Protocol across all 10 cancer types and generates:
- Per-cancer cure simulations (100 Monte Carlo trials each)
- Safety-screened drug protocols
- Lab-executable protocol documents
- Comprehensive validation report

### 3. Run Tests

```bash
# Standard tests
python -m pytest tests/test_universal_cure.py -v

# Adversarial stress tests (deliberately break the model)
python -m pytest tests/test_adversarial.py -v
```

### 4. View Results

Results are written to `results/`. Key files:
- `confluence_report.md` — Pan-cancer summary with validation gates
- `executive_summary.md` — High-level overview
- `*_lab_protocol.md` — Per-cancer lab protocols
- `gaps_and_limitations.md` — Honest risk assessment

## Pan-Cancer Coverage

| Cancer | Metabolic Signature | Seriousness Rank |
|---|---|---|
| **PDAC** | Extreme glycolysis, desmoplastic stroma | Highest |
| **HCC** | ROS-adaptive, hepatic rewiring | 2 |
| **GBM** | BBB barrier, lipid-dependent | 3 |
| **HGSOC** | Peritoneal immune exclusion | 4 |
| **TNBC** | Enhanced Warburg, glutamine addiction | 5 |
| **AML** | Liquid tumor, BH3-dependent | 6 |
| **mCRPC** | Androgen-independent metabolic switch | 7 |
| **CRC** | Wnt-driven glycolysis, butyrate-sensitive | 8 |
| **NSCLC** | Metabolic flexibility (glycolysis + OXPHOS) | 9 |
| **Melanoma** | OXPHOS-dependent, ROS-adaptive | Most tractable |

## Drug Library (19 Interventions)

**Curvature Reducers:** DCA, Metformin, 2-DG, CB-839, Olaparib, Vorinostat, FMD, HCQ, 5-Azacitidine  
**Entropic Drivers:** Hyperthermia, High-dose Vitamin C, Ferroptosis Inducers, N6F11  
**Vector Rectifiers:** Anti-PD-1, Anti-CTLA-4, Bevacizumab, CAR-T  
**Supportive:** NAD+ Precursors  
**Negative Control:** Epogen (iatrogenic — excluded from all protocols)

## Validation Gates

The framework evaluates itself against 6 validation gates:

| Gate | Criterion |
|---|---|
| Cure Threshold | ≥90% simulated escape rate in ≥8/10 cancers |
| Adaptive Superiority | Adaptive protocol beats continuous in ≥8/10 cancers |
| Protocol Diversity | ≥5 unique drug combinations across cancers |
| Non-Uniform Outcomes | Escape distance range > 0.05 |
| Safety Clearance | ≥8/10 protocols pass toxicity screening |
| Clonal Dynamics | Adaptive suppresses resistant clones in ≥8/10 cancers |

## Mathematical Foundation

The framework is grounded in **Kramers escape theory** applied to metabolic phase space:

- **Generator matrix A** (10×10): governs metabolic dynamics dx/dt = Ax
- **Basin curvature μ(A)**: minimum eigenvalue modulus (attractor depth)
- **Kramers escape rate κ**: κ ∝ exp(-μ/σ²), where σ is noise scale
- **Therapeutic correction δA**: A_treated = A_cancer + Σ δA_drug

See `results/latiff.md` for the full lattice-theoretic formalism (LATIFF).

## Call for Expert Review

**We actively seek criticism.** This project is most valuable when experts identify where it's wrong. See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- What to critique
- How to reproduce results
- How to submit feedback

Priority review areas: biological plausibility of generator matrices, drug mechanism modeling, immune dynamics assumptions, and clinical translatability.

## Key References

- Kramers 1940, Physica (Escape rate theory)
- Gatenby et al. 2009, Cancer Research (Adaptive therapy)
- Vander Heiden et al. 2009, Science (Warburg effect)
- Bonnet et al. 2007, Cancer Cell (DCA and Warburg reversal)
- DeBerardinis et al. 2007, PNAS (Glutaminolysis)
- Gorrini et al. 2013, Nat Rev Drug Discov (ROS dynamics)

## License

[MIT License](LICENSE). See [DISCLAIMER.md](DISCLAIMER.md) — this is a research prototype, not a clinical tool.
