# Knowledge Gates for Dynamical Oncology Models: Findings from an OnCo × CONFLUENCE Integration

**Document type:** Research findings chapter (working thesis format)  
**Status:** Specification and architectural findings. Not a clinical result. Not evidence that listed couplings are true in patients.  
**Date:** 18 September 2026  
**Author context:** Project Confluence (`cloudynirvana/project-confluence`)  
**Companion code:** pull request #9, branch `feat/onco-adapter-p0`  
**External corpus:** OnCo (`judegomila/OnCo`, https://onco.cc), data CC BY-NC 4.0, code MIT  

---

## ABSTRACT

Oncology now has a public, cited knowledge graph (OnCo) and a separate computational research simulator (CONFLUENCE v2). The former enumerates cancers, targets, drugs, trials, bottlenecks and ideas. The latter integrates a noisy observation vector through a connectome-style controller into a 15-dimensional tumour-microenvironment ordinary differential equation. The working hypothesis of this chapter is that those two artefacts are complementary rather than redundant, *provided they are not collapsed into each other*.

The central finding is epistemic rather than biological. The hard problem is not collecting more cancer facts. It is converting facts into causal, testable, uncertainty-aware dynamical relationships without treating a web page, a gene symbol, or a legacy numeric map as an identified parameter. This chapter records the invariants, the verified repository facts, the P0 adapter that implements the gates, and the explicit refusal to treat a universal cure as the success criterion of the model.

Three frozen lineages must remain distinct: the 3-state TNBC metabolic notebook (`tnbc_mod_3s`), the 6-state architecture of an earlier integration report (`confluence_report_6s`), and the live 15-D simulator (`confluence_v2_15d`). A fourth executable but unidentified map, `validation/gene_to_parameter_map.json`, is classified as `confluence_v1_calibrator`. OnCo Ideas (1,139 records with hypothesis, rationale, test and maturity) constitute an external hypothesis shelf, not a CONFLUENCE hypothesis library.

This document does not claim disease eradication, clinical efficacy, or parameter identification.

---

## TABLE OF CONTENTS

- [1. Introduction](#1-introduction)
- [2. Specific aims of this research pass](#2-specific-aims-of-this-research-pass)
- [3. Materials and verified sources](#3-materials-and-verified-sources)
- [4. Findings](#4-findings)
- [5. Discussion](#5-discussion)
- [6. Limitations](#6-limitations)
- [7. Next work](#7-next-work)
- [8. References](#8-references)

---

## 1. INTRODUCTION

A public oncology knowledge graph and a closed-loop dynamical simulator answer different questions. OnCo answers *what is named, linked, dated and cited*. CONFLUENCE v2 answers *how a frozen state vector moves under a control and a pharmacokinetic layer*. Mixing the answers produces a familiar failure mode in computational biology: a target page for LDHA is treated as a value for `p_lactate`, or a literature-inspired mutation effect is treated as an identified parameter.

CONFLUENCE v2 already implements the 15-D latent state

X = (T_s, T_r, I_act, I_exh, S_fib, L, O, G, C_tgfb, C_ifng, H, T_f, I_surv, A_ready, awake)

with observations Y, control U, and a PK/PD concentration layer C. The repository labels outputs as computational research quantities, not clinical outcomes.

The integration report that preceded this chapter proposed sitting OnCo beside the loop as knowledge, not inside the right-hand side. That proposal is adopted here as a specification, not as evidence that any listed coupling is true.

---

## 2. SPECIFIC AIMS OF THIS RESEARCH PASS

**Aim 1.** State a conversion contract that forbids skip-level promotion from OnCo knowledge to model parameters.

**Aim 2.** Verify the live CONFLUENCE and OnCo artefacts against GitHub and the public API, including `gene_to_parameter_map.json`.

**Aim 3.** Implement a read-only P0 adapter and nine gate tests without editing CancerODE.

**Aim 4.** Separate the external OnCo idea shelf from any future CONFLUENCE hypothesis library.

**Aim 5.** Record the scientific scope: control and disease-specific clearance, not a universal cure.

---

## 3. MATERIALS AND VERIFIED SOURCES

| Source | What was verified |
|---|---|
| `cloudynirvana/project-confluence` (`main`) | 15-D CancerODE, contracts, controllers, gene-to-parameter map |
| `feat/onco-adapter-p0` / PR #9 | Adapter package, fixtures, nine tests, spec |
| `https://onco.cc` and `/api/v1/` | Static JSON API, idea records, CC BY-NC 4.0 data licence |
| `judegomila/OnCo` `src/lib/schema.ts` | IdeaSchema: hypothesis, rationale, test, maturity, actor, cost, horizonYears |
| `cloudynirvana/TNBC-Metabolic-Strain-MOD` | Separate 3-state notebook lineage; ROS term not imported |
| GLOBOCAN / IARC 2024 estimates (published 2026) | Global burden context only; not a model input |

No patient-level data were used. No ODE coefficients were re-fit.

---

## 4. FINDINGS

### 4.1 Complementary architecture

OnCo is a knowledge layer. CONFLUENCE is a dynamical layer. Single-cell perturbation models are a cellular-transition layer. Experimental data remain the reality check.

Intended pipeline (specified, not executed end-to-end):

OnCo knowledge -> Evidence -> Hypothesis -> Mechanism -> optional cellular transition -> frozen 15-D dynamics -> controller -> prediction -> experiment -> evidence.

Adding facts directly to the ODE right-hand side is not a step on that path.

### 4.2 The five-layer invariant

Knowledge != Evidence != Causal mechanism != Parameter != Prediction.

OnCo confidence.probability is not P(H). Idea maturity is not CONFLUENCE evidence_level. Wired RHS terms are parameters with provenance=assumed until identified.

`refuse_knowledge_as_parameter(OncoRef("ldha"), "p_lactate")` returns provenance=forbidden.

### 4.3 Frozen model lineages

| ID | Artefact | Status |
|---|---|---|
| `tnbc_mod_3s` | TNBC-Metabolic-Strain-MOD notebooks | Frozen pending ROS audit. Do not port into v2. |
| `confluence_report_6s` | Report sketch X=[T,I,S,L,R,H] | Paper only. |
| `confluence_v2_15d` | Live CancerODE | Adapter sits around it. |
| `confluence_v1_calibrator` | gene_to_parameter_map.json | Executable, not identified. |

OnCo integration does not justify changing the dynamics.

### 4.4 Legacy gene-to-parameter map

File `validation/gene_to_parameter_map.json` maps LDHA to `pyruvate_to_lactate` (+0.10 / +0.30). That map is not identified Theta.

Classification: source_type=legacy_mapping, provenance=assumed, identifiability=unidentified, model_id=confluence_v1_calibrator.

Alias trap: `pyruvate_to_lactate` != v2 `p_lactate` (default 0.22).

### 4.5 OnCo idea shelf

OnCo Ideas (~1,139) carry hypothesis, rationale, test and maturity. Most are policy or infrastructure claims. They are Knowledge. They are not 15-D mechanisms. A CONFLUENCE hypothesis library, if added, is an admit-list only. Bulk ingest of ideas.json is forbidden.

### 4.6 P0 adapter

Read-only client, cache envelope (CC BY-NC 4.0 payloads gitignored), seed bindings, nine gate tests. `bind("ldha")` annotates `p_lactate` and does not write it. Cache has no TTL yet and does not pin invalidation to OnCo buildDate.

### 4.7 Deterministic testing

Lane A: fixtures, merge-blocking, no network. Lane B: frozen-model invariants. Lane C: live OnCo, not merge-blocking. Do not assert retrieved_at. Do not golden-file OnCo tldr.

### 4.8 Cure is the wrong unit of success

Cancer is a family of diseases. Some are already curable in defined settings. Global 2024 burden was on the order of 20.6 million new cases and 9.8 million deaths. CONFLUENCE and OnCo do not cure patients. Success for this stack is better questions, killed couplings, and identified parameters disease by disease. Durable control and interception remain first-class goals beside clearance.

---

## 5. DISCUSSION

The safeguard is the refusal to let representation become observation, or observation become a parameter. The default fly circuit in CONFLUENCE is already a structured stub, not a FlyWire dump. The same provenance rule should cover OnCo, connectome graphs, genomic observations and wet-lab evidence.

The 15-D model is lumped by design. OnCo listing more cell types is not a reason to expand X.

The integration report remains spec://. Citing it as Evidence would violate Aim 1.

---

## 6. LIMITATIONS

1. Specification plus a thin adapter. No new biological measurement.
2. OnCo object counts move with buildDate.
3. Cache TTL and buildDate invalidation are not in running code.
4. The 3-state ROS audit is not done.
5. confluence-evidence was not used as a hypothesis catalogue.
6. Global burden figures are context, not Theta.
7. Not medical advice. Not a treatment path.

---

## 7. NEXT WORK

1. Merge or iterate PR #9 without RHS edits.
2. Envelope-hash and params-unchanged-after-bind tests.
3. Independent audit of tnbc_mod_3s.
4. Hypothesis library as an admit-list.
5. Optional Lane C nightly meta.json drift check.
6. Identification workflow for p_lactate only after Aims 1-3 stay green.

---

## 8. REFERENCES

1. Gomila J. and contributors. OnCo. https://onco.cc and https://github.com/judegomila/OnCo (data CC BY-NC 4.0; code MIT).
2. Ogbonna K. Project Confluence. https://github.com/cloudynirvana/project-confluence PR #9 (18 September 2026).
3. OnCo IdeaSchema, src/lib/schema.ts, judegomila/OnCo.
4. Sung H. et al. Global cancer statistics 2024. CA Cancer J Clin (2026). IARC/ACS.
5. World Health Organization. Cancer fact sheet.
6. American Cancer Society. Survival rates for childhood leukemia.

**Attribution:** Data from OnCo (onco.cc), CC BY-NC 4.0; commercial use needs a licence.

**Disclaimer:** Computational research instrument. Not a medical device. Does not claim clinical efficacy, disease eradication, or a treatment path.
