# OnCo → CONFLUENCE ontology and evidence-ingestion spec

**Status:** draft v0.3 — specification, not evidence.
**Date:** 18 September 2026
**Repos:** cloudynirvana/project-confluence · judegomila/OnCo

This PR implements the P0 adapter. It does not change `CancerODE`.

---

## Invariant

The integration report and this document are specifications. They are not evidence that any listed relationship is true in a patient, a cell line, or the current ODE.

Keep Knowledge, Evidence, Causal mechanism, Parameter, and Prediction as separate objects. A record may point at the next layer. It may not collapse into it.

| Layer | Object | Allowed question | Forbidden leap |
|---|---|---|---|
| Knowledge | `OncoRef` | What does OnCo name and link? | therefore k = ... |
| Evidence | `EvidenceObject` | What was measured, where, in what system? | therefore this term belongs in F |
| Causal mechanism | `MechanismObject` | do(U) changes which state, in which context? | therefore identifiable from CCLE |
| Parameter | `ParameterObject` | Which symbol in which frozen model? | silent write into rhs_cancer |
| Prediction | `PredictionObject` | Frozen-model output under U, with uncertainty | clinical advice |

Conversion is one-way and gated:

```
OnCo knowledge
  --cite--> Evidence          (source URI + rung required)
    --interpret--> Hypothesis (falsifier required)
      --propose--> Mechanism  (context + sign + do-operator)
        --identify--> Parameter  (model_id + symbol + identifiability != unidentified)
          --simulate--> Prediction
            --test--> Experiment
              --write--> Evidence
```

No arrow may skip a box. Wired RHS terms are still Parameters with provenance=assumed until identified.

OnCo confidence.probability and Idea maturity stay on OncoEvidenceCandidate. They are not P(H) and not CONFLUENCE evidence_level.

---

## Frozen lineages

| ID | What | Status |
|---|---|---|
| `tnbc_mod_3s` | TNBC-Metabolic-Strain-MOD notebooks | frozen pending ROS audit |
| `confluence_report_6s` | report architecture X=[T,I,S,L,R,H] | paper only |
| `confluence_v2_15d` | live 15-D CancerODE | adapter sits around it |
| `confluence_v1_calibrator` | validation/gene_to_parameter_map.json | executable, not identified |

OnCo integration does not justify changing the dynamics.

legacy gene-to-parameter map != identified parameter mapping.

Alias trap: legacy `pyruvate_to_lactate` != v2 `p_lactate` (default 0.22).

---

## Rules

1. Do not vendor the OnCo corpus. Data is CC BY-NC 4.0. Cache under `data/onco/cache/` (gitignored). Attribution on every export: `Data from OnCo (onco.cc), CC BY-NC 4.0; commercial use needs a licence.`
2. OnCo is not a parameter source. `refuse_knowledge_as_parameter(OncoRef("ldha"), "p_lactate")` returns provenance=forbidden.
3. OnCo is not a controller prior.
4. Adapter code is MIT. Cached payloads remain CC BY-NC 4.0.
5. `bind()` returns slot annotations. It never writes parameters.

---

## P0 boundary

Read-only client + cache envelope + bindings + nine gates.

Not in this PR: RHS edits, controller edits, 3-state ROS import, 6-state expansion, automatic fitting, reverse writes to OnCo.

Tests: `tests/test_onco_adapter.py`.
Code: `confluence/onco/`.
Usage: `docs/ONCO_ADAPTER.md`.
