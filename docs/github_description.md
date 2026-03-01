# GitHub Repository Setup

## Repository Name

`project-confluence`

## Description (Short — 250 chars)

Computational framework for hypothetical cancer therapy optimization using geometric attractor dynamics and Kramers escape theory. 10 cancers, 19 drugs, adaptive protocols. Research prototype — not for clinical use.

## Topics / Tags

```
cancer-research
computational-biology
stochastic-dynamics
mathematical-modeling
kramers-escape-theory
attractor-dynamics
adaptive-therapy
drug-optimization
systems-biology
bioinformatics
```

## About Section

Project Confluence — A hypothetical universal cancer cure framework that models tumor metabolic states as stochastic attractors and optimizes therapeutic escape using a 3-phase Flatten→Heat→Push protocol. Built on Kramers escape theory with a 19-drug PK library, multi-mechanism resistance modeling, and toxicity constraints. Seeking expert review.

## Pinned Issues to Create

### Issue 1: "Expert Review Wanted: Biological Plausibility of Generator Matrices"
Label: `biology`, `help wanted`
Body: Are the 10×10 generator matrices in `src/tnbc_ode.py` biologically defensible representations of metabolic dynamics for these cancer types? What parameters seem unrealistic?

### Issue 2: "Expert Review Wanted: Is Kramers Escape Theory the Right Formalism?"
Label: `math`, `help wanted`
Body: The framework treats therapeutic cure as a noise-assisted escape from a metabolic attractor basin. Is this a valid application of Kramers theory, or is the mapping too loose?

### Issue 3: "Expert Review Wanted: Drug Library Pharmacological Accuracy"
Label: `pharmacology`, `help wanted`
Body: The `expected_effect` matrices in `src/intervention.py` model drug mechanisms as generator corrections. Do these corrections match known drug mechanisms at the metabolic level?

### Issue 4: "Known Limitation: Model May Be Under-Constrained"
Label: `known-issue`, `discussion`
Body: Some cancers achieve suspiciously high simulated escape rates (near 100%). This may indicate the model has too many degrees of freedom. Discussion on adding constraints welcome.
