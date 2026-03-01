# Source Modules (`src/`)

The core computational engine. Each module handles one aspect of the framework.

## Reading Order (for understanding the framework)

| # | Module | Purpose | Key Concepts |
|---|--------|---------|-------------|
| 1 | `tnbc_ode.py` | 10-metabolite ODE systems for 10 cancers + healthy | Generator matrices, metabolic axes |
| 2 | `geometric_optimization.py` | Basin curvature and Kramers escape rate | Attractor depth, escape probability |
| 3 | `intervention.py` | 19-drug library with PK engine | Drug effects as generator corrections (δA) |
| 4 | `immune_dynamics.py` | Multi-compartment immune force field | CD8+/NK/Treg/DC dynamics, exhaustion |
| 5 | `coherence.py` | Spectral coherence analysis | Generator health scoring |
| 6 | `resistance_model.py` | Multi-mechanism resistance tracking | Efflux, mutations, rewiring, clonal selection |
| 7 | `clonal_dynamics.py` | Lotka-Volterra 2-clone competition | Sensitive vs resistant population dynamics |
| 8 | `toxicity_constraints.py` | Clinical safety screening | MTD, organ overlap, G3/4 risk |
| 9 | `spatial_dynamics.py` | 3-compartment tumor model | Core/rim/stroma drug penetration |
| 10 | `protocol_translator.py` | Simulation → wet-lab conversion | Concentrations, cell lines, endpoints |
| 11 | `calibration.py` | Parameter calibration engine | Grid search + local refinement |
| 12 | `restoration.py` | Generator correction computation | Sparse eigenvalue-targeted δA |

## How They Connect

```
tnbc_ode.py (generators) ──→ geometric_optimization.py (curvature)
                          ──→ coherence.py (health scoring)
                          ──→ intervention.py (drug library)
                                    │
                                    ▼
                          immune_dynamics.py (force field)
                          resistance_model.py (drug decay)
                          clonal_dynamics.py (population)
                                    │
                                    ▼
                          toxicity_constraints.py (safety)
                          protocol_translator.py (lab-ready)
```
