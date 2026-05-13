# Project Confluence — Biologics Integration Framework

**Module:** confluence-biologics
**Version:** 0.1.0-theoretical
**Status:** Pre-validation | Awaiting Phase 2 data (Gatenby/NCI ITCR pipeline)
**Author:** Kelechi Emeka Ogbonna · cloudynirvana

---

## 1. Theoretical Premise

Project Confluence operationalises cancer as complexity collapse — a disease state in which the Φ vector diverges from the healthy complexity manifold Φ*. Therapy is framed as complexity restoration: the application of controlled perturbations that return the system trajectory toward Φ*.

Previous formulations of the adaptive therapy controller focused primarily on chemotherapeutic and radiotherapeutic agents, which act as bulk killers — they reduce tumour burden but do not selectively reshape the disease manifold. Biologics are fundamentally different. They are precision geometric operators: they act with specificity on defined molecular axes, which correspond to specific dimensions of the Φ vector and specific topological features of the disease manifold.

This document formalises the integration of biologics into the Confluence framework as a first-class intervention class.

---

## 2. The Biologic Operator Formalism

A biologic agent B_k is modelled as an operator acting on Φ-space:

```
B_k(Φ, t) = PK_k(t) · A_k · σ_k(Φ)
```

Where PK_k(t) is the pharmacokinetic envelope, A_k ∈ ℝ⁵ˣ⁵ is the operator matrix encoding which Φ dimensions are targeted, and σ_k(Φ) is the state sensitivity function.

The extended ODE system becomes:

```
dX/dt = f(X, Φ, t) + Σ_k C_k(X, Φ) · u_k(t)
dΦ/dt = g(X, Φ, t) + Σ_k B_k(Φ, t) · u_k(t)
```

---

## 3. Biologic Class Operator Profiles

| Class | Geometric Role | Primary Φ Action | Key Drugs |
|-------|---------------|------------------|-----------|
| Checkpoint Inhibitors | Attractor Destabiliser | ↑φ₃, ↓φ₂ | anti-PD1, anti-CTLA4 |
| Bispecific Antibodies | Dimensional Collapser | ↑↑φ₃, ↓φ₁, ↓φ₄ | blinatumomab, teclistamab |
| ADCs | Targeted Trajectory Displacer | ↓φ₁, ↓φ₂ (phase-dependent) | T-DXd, enfortumab vedotin |
| Anti-Angiogenic | Landscape Reshaper | ↓φ₅, dose-dependent φ₃ | bevacizumab, ramucirumab |
| Cytokine Biologics | Basin Deformer | ↑φ₃, ↑φ₄ (toxicity via φ₅) | IL-2, IL-15, IFN-α |
| Targeted Pathway | Coherence Disruptor | ↓↓φ₂, ↓φ₄ | cetuximab, pertuzumab |

---

## 4. Synergy Tensor

Pairwise synergies are captured by S_ij = ∂²(dΦ/dt) / ∂u_i ∂u_j. Validated cases include CPI + bispecific (super-additive on φ₃), CPI + anti-VEGF (dose-constrained), and ADC + bispecific (antigen release amplifies targeting).

---

## 5. Resistance Geometry

Acquired resistance is defined as the emergence of a new stable attractor outside the effective action radius of B_k. Resistance risk is proportional to the Gaussian curvature K(Φ) in the direction of A_k's action. High curvature means small displacements are strongly opposed — promoting rapid resistance emergence.

---

## 6. Implementation

See `models/biologic_operator.py` for the full computational implementation including all 6 biologic classes, PK envelopes, synergy tensor, resistance geometry, and Phi-state classifier.
