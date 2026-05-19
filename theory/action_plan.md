# Action Plan: Codifying the BAC Coupling Tensor (C_ij)
## From Theoretical Semantics to Executable Code

This document outlines the concrete, non-semantic engineering roadmap to implement the Bounded Adaptive Coherence (BAC) coupling tensor $C_{ij}$ directly into the Project Confluence computational engine.

---

## 1. Concrete Code Deliverables

We will implement three specific additions to the codebase:

```
┌────────────────────────────────────────────────────────┐
│ 1. Create: models/coupling_tensor.py                   │
│    - CouplingTensorAnalyzer class                      │
│    - Block Jacobian, sample entropy, and viability     │
│    - Failure mode classifier                           │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ 2. Create: tests/test_coupling_tensor.py               │
│    - Unit tests for bounds, dimensions, & SVD          │
│    - Classifier validation (aging vs. cancer)          │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ 3. Modify: scripts/confluence_runner.py                │
│    - Integrate C_ij analysis along trajectories        │
│    - Add Gate 7: Coupling Tensor Viability Gate        │
│    - Write metrics to JSON and Markdown reports        │
└────────────────────────────────────────────────────────┘
```

---

## 2. Deliverable Details

### Deliverable 1: `models/coupling_tensor.py`
The new module will implement the `CouplingTensorAnalyzer` class with the following core mathematical methods:

1. **`compute_from_jacobian(ode_system, trajectory, t_points)`**:
   - Computes the $15 \times 15$ Jacobian matrix along the trajectory via finite differences (reusing `fisher_geometry` sensitivity logic).
   - Partitions the Jacobian into blocks corresponding to the four biological scales:
     - **Molecular**: `[0, 1, 2, 3, 4]` (Glucose, Lactate, Pyruvate, ATP, NADH)
     - **Cellular**: `[5, 6, 7, 8, 9]` (Glutamine, Glutamate, αKG, Citrate, ROS)
     - **Organism**: `[10, 11, 12]` (I_eff, I_reg, I_exhaust)
     - **Tissue**: `[13, 14]` (σ_stromal, ν_vascular)
   - Computes $C_{ij}(t) = \frac{\|J_{ij}\|_F}{\max_{k,l} \|J_{kl}\|_F}$, outputting a $4 \times 4 \times T$ array of normalized coupling elements.

2. **`scale_entropy_rates(trajectory, dt, window_size)`**:
   - Computes rolling sample entropy (`sample_entropy`) for each of the four biological scales, utilizing the existing high-performance function in `complexity_profiler.py`.
   - Normalizes local entropy rates against a calibrated maximum sustainable entropy rate $\dot{S}_{\text{ref}}$: $\dot{s}_k(t) = \text{SampEn}(z_k(t)) / \dot{S}_{\text{ref}}$.

3. **`viability(C_t, entropy_rates)`**:
   - Computes $V(t) = \sigma_{\min}(C(t)) - \max_k [\dot{s}_k(t)]$.
   - Returns a scalar viability margin for each time step.

4. **`classify_failure(C_t, C_healthy)`**:
   - Compares the patient's current coupling tensor with a baseline healthy tensor.
   - Triggers **Aging** if there is a uniform off-diagonal decline ($\text{std}(\Delta C_{\text{off-diag}}) / \text{mean}(\Delta C_{\text{off-diag}}) < 0.5$).
   - Triggers **Cancer** if there is selective decoupling of cellular-organismal scales ($C_{24} \to 0$).

5. **`optimal_intervention_target(C_t, entropy_rates)`**:
   - Computes the gradient $\partial V / \partial C_{ij}$ to determine which coupling pathway must be therapeutically bolstered to yield the highest increase in viability.

---

### Deliverable 2: `tests/test_coupling_tensor.py`
We will write a standard python unit test file to ensure the numerical integrity of the analyzer:
- **`test_tensor_bounds`**: Asserts that computed $C_{ij}(t)$ is always within $[0, 1]$ and $C_{ii}(t) \approx 1$ under homeostatic conditions.
- **`test_viability_calculation`**: Validates $V(t)$ values against expected mathematical boundaries.
- **`test_classifier_aging`**: Feeds the classifier a synthetic globally degraded tensor and asserts that it correctly logs `'aging'`.
- **`test_classifier_cancer`**: Feeds the classifier a synthetic tensor where the organismal block $C_{24}$ is set to zero and asserts that it logs `'cancer'`.

---

### Deliverable 3: `scripts/confluence_runner.py`
We will modify the universal runner to incorporate the new coupling tensor parameters:
1. **Simulation Integration**: During the 3-phase simulation loop (Flatten $\to$ Heat $\to$ Push), track the time-series of $C_{ij}(t)$ and $V(t)$ along with the traditional state variables.
2. **New Validation Gate**: Add **Gate 7: Coupling Tensor Viability Gate**. This gate demands that:
   - The final viability margin under the adaptive protocol must be positive: $V(T) > 0$ (remission).
   - The average viability margin under the adaptive protocol must be at least 15% higher than the continuous protocol (validating the superiority of adaptive containment).
3. **Data Export**: Save the complete coupling tensor dynamics directly into the output json (`results/confluence_results.json`) and report (`results/confluence_report.md`).

---

## 3. Immediate Next Steps

1. **Write the core implementation**: Author `models/coupling_tensor.py`.
2. **Write the unit tests**: Author `tests/test_coupling_tensor.py`.
3. **Verify locally**: Since the Python Microsoft Store alias is broken on this OS, we will write a custom test runner script or directly launch Node.js / Python tests by finding a valid python interpreter path.
4. **Integrate into the runner**: Update `scripts/confluence_runner.py` and run the universal pipeline to generate the clinical results.
