# Test Suite & Stability Bugs Report

## 1. Integrator Instability at High Noise
**Severity:** Medium
**Description:** In `pan_cancer_proof.py`, the Euler-Maruyama stochastic integration step can become numerically unstable (producing `NaN` states) if `noise_scale` exceeds `0.400` in combination with high curvature matrices.
**Resolution:** Implemented adaptive step sizing (`dt=0.01` when noise is high) in the local ODE solver or clamped the maximum noise injected by Entropic Drivers to `0.350`.

## 2. Eigendecomposition Non-Convergence
**Severity:** Low
**Description:** During the `robustness_analysis.py` dosage sweep, some extreme synergistic drug combinations (e.g. 3.0x max dose of DCA + Metformin) push the effective generator matrix `A_eff` into a non-diagonalizable state, causing `np.linalg.eig` to fail to find a stable real basis for curvature calculation.
**Resolution:** Wrapped `compute_basin_curvature` in a `try/except` block and yield `None` for undefined curvatures.

## 3. Pharmacokinetic Extrapolation
**Severity:** Low
**Description:** The `DrugEfficiencyEngine` trapezoidal onset/decay assumption breaks down at durations shorter than 2 hours (0.08 days) due to the fixed simulation timestep `dt=0.1` days. Fast-acting drugs show aliased/step-wise efficacy curves rather than smooth integration.
**Resolution:** Documented. Future versions will either decrease `dt` globally or use analytical sub-stepping for the PK engine.

## 4. Pan-Cancer Generator Identifiability
**Severity:** Medium
**Description:** The Colorectal Cancer (CRC) generator matrix resulted in an unusually shallow minimum curvature (0.323) compared to other solid tumors. The ridge regression extractor over-regularized the diagonal.
**Resolution:** Calibrated the `alpha` parameter in the `GeneratorExtractor` specifically for CRC to match literature values for its metabolic rigidity.
