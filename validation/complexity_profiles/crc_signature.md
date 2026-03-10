# Complexity Signature: Colorectal Cancer (CRC)

**Generated via:** Project Confluence SAEM (Systemic Attractor Endotype Model)
**Base Model:** 15D Nonlinear Complex Attractor ODE

---

## 1. Disease Context & Pathophysiology Modeled
Colorectal cancer introduces a unique pathological steady state defined by:
- **Strong Warburg Effect:** High glucose uptake (x2.0) and severe glycolysis flux due to hypoxia early in tumorigenesis.
- **TME Acidification:** Severe accumulation of lactate due to poor clearance (x0.5).
- **Mucosal/Stromal Barrier & Dysbiosis:** Modeled via accelerated fibrosis (`r_fibrosis=0.08`) reflecting the physical desmoplastic response in CRC.
- **Angiogenesis:** Extremely high expression/drive of VEGF (`r_angio=0.08`).
- **Immune Evasion:** Rapid exhaustion of effector cells (`k_exhaust=0.15`) and pronounced Treg accumulation.

## 2. Complexity Profile (Φ Vector)
*Extracted via `ComplexityProfiler` combining functional coherence, spectral slope, state-space volume, and Lyapunov estimation.*

| Dimension | Healthy Baseline | CRC Attractor | Delta | Direction |
| :--- | :--- | :--- | :--- | :--- |
| **Φ_temporal** (Spectral slope) | 1.0125 | 0.9926 | -0.0199 | ~ Similar |
| **Φ_spatial** (Correlation dim) | 2.5694 | 1.8159 | -0.7535 | v Depleted |
| **Φ_functional** (Lyapunov max) | 0.0091 | 0.0245 | +0.0154 | ^ Elevated |
| **Φ_informational** (Entropy) | 0.9859 | 0.7761 | -0.2098 | v Depleted |
| **Φ_coupling** (Network MSE) | 0.8105 | 0.6559 | -0.1545 | v Depleted |

### Summary Metrics
- **Healthy Magnitude (\|Φ\|):** 2.9698
- **CRC Magnitude (\|Φ\|):** 2.3168
- **Coherence (C):** 0.1983 (Healthy) -> 0.1654 (CRC)
- **Archetype Shift:** `Healthy -> Pathological_Rigid`

## 3. Therapeutic Intervention Protocol (Cure Test)
**Regimen Simulated:** Trimodal Target (FOLFIRI + Bevacizumab + Anti-PD1)
- *FOLFIRI:* General cytotoxic disruption of the hyper-metabolic state.
- *Bevacizumab:* Strong anti-VEGF targeting the massive angiogenesis parameter.
- *Anti-PD1:* Immune rescue to restore effector priming and slow exhaustion.

### Restoration Results
| Metric | Healthy | CRC | Treated | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Φ_temporal** | 1.0125 | 0.9926 | 0.9996 | [OK] 65% |
| **Φ_spatial** | 2.5694 | 1.8159 | 2.5484 | [OK] 97% |
| **Φ_functional**| 0.0091 | 0.0245 | 0.0120 | [OK] 81% |
| **Φ_informational**| 0.9859 | 0.7761 | 0.9825 | [OK] 98% |
| **Φ_coupling** | 0.8105 | 0.6559 | 0.8033 | [OK] 95% |

**OVERALL COMPLEXITY RESTORATION: 95.9%**
- **Phi-distance (Cancer -> Healthy):** 0.8252
- **Phi-distance (Treated -> Healthy):** 0.0336
- **Status:** [PASS] THERAPEUTIC HYPOTHESIS SUPPORTED
