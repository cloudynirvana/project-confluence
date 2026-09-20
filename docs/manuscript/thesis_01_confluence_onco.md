# CONFLUENCE × OnCo: An Evidence-Gated Dynamical Framework for Integrating Oncology Knowledge Graphs with Adaptive Cancer-State Models

**Document type:** Thesis #1 — working manuscript (computational research)  
**Author:** Kelechi Emeka Ogbonna  
**Affiliation:** Independent computational research / Project Confluence (GitHub cloudynirvana)  
**Correspondence:** https://github.com/cloudynirvana/project-confluence  
**Date:** 20 September 2026  
**Public HTML:** https://confluence-research.vercel.app/thesis  
**Citeable PDF:** https://confluence-research.vercel.app/thesis.pdf  
**Status:** Architectural and methodological findings. Not a clinical result.  
**Citation style:** numbered Vancouver [n] matching the References list and the public thesis page.  
**DOI:** none registered for this document. Do not reuse the software Zenodo record.

This manuscript expands `ONCO_CONFLUENCE_THESIS_FINDINGS.md`, `docs/ONCO_CONFLUENCE_ONTOLOGY_SPEC.md`, and `docs/ONCO_ADAPTER.md`. It does not add wet-lab measurements, patient-level results, or invented identifiers.

---

## Abstract

Cancer research now produces knowledge graphs, multi-omic assays and dynamical simulators in parallel. The failure mode is collapsing those layers. This study asks whether a provenance-controlled pipeline can keep the layers apart while still allowing testable predictions. Scientific success is a staged chain — traceability, mathematical validity, identifiability, out-of-sample prediction, experimental falsification — not disease eradication.

CONFLUENCE v2 is a frozen 15-dimensional computational cancer-state model with an optional connectome-style controller. OnCo is a public, cited oncology knowledge graph [10]. They are complementary only if knowledge is not written into parameters. Pull request #9 shipped a read-only OnCo adapter and nine refusal tests. It is architectural evidence for the gates, not therapeutic efficacy [11].

This document records the gates, the P0 adapter, the LDHA / `p_lactate` refusal, and the honesty rule that OnCo confidence is not P(H). It does not claim a cure, a dose, a clinical decision-support system, or an identified Θ.

---

## Keywords

computational oncology; knowledge graphs; dynamical systems; evidence gates; identifiability; OnCo; CONFLUENCE; research-only; not a medical device

---

## Introduction

Oncology knowledge graphs name genes, diseases, ideas and citations [10]. Dynamical simulators ask how a state moves under control. Mixing the two without provenance treats a web page as an identified parameter. The problem stated on the public thesis page, and restated here, is: how can heterogeneous oncology knowledge and molecular observations be incorporated into a cancer dynamical model without collapsing evidence, mechanism, parameterisation and prediction into unsupported assumptions?

GLOBOCAN 2024 estimates, published 2026, report about 20.6 million diagnoses and 9.8 million deaths [1,2]. Female breast cancer accounted for about 2.43 million new cases in that estimate series [1,2]. WHO states that many cancers can be cured if detected early and treated effectively, while access remains uneven [3]. That sentence is a policy statement about staged, treatable disease — not a claim that CONFLUENCE or OnCo cures patients [3,10,11]. The WHO global status report on cancer is cited as context only [4]. Nigeria GLOBOCAN *2022* estimates (127,763 new cases; 79,542 deaths; breast 32,278) are setting context only; they are not 2026 incidence [5].

Tumours evolve and adapt under therapy [7]. Multi-omics increase resolution without automatically producing a causal model. Mathematical oncology supplies in-silico laboratories [8]. Adaptive therapy treats treatment as a process under selection [9]. The gap this thesis addresses is connecting knowledge to dynamical hypotheses without dropping provenance.

TNBC is used as a test case because NCI describes it as roughly 15% of breast cancers, typically faster-growing and more recurrent, and heterogeneous [6]. That description is not a CONFLUENCE parameter.

---

## Specific aims

1. Forbid skip-level promotion from OnCo knowledge to ODE parameters.
2. Classify legacy artefacts, including `validation/gene_to_parameter_map.json`, as assumed / unidentified rather than as identified Θ.
3. Ship a read-only P0 adapter without editing `CancerODE`.
4. Keep the OnCo Ideas shelf separate from any CONFLUENCE hypothesis library.
5. Define success as gated, falsifiable computational work — control and disease-specific clearance — not a universal cure [3,12].

---

## Background

### Two artefacts, two questions

OnCo answers what is named, linked, dated and cited [10]. CONFLUENCE v2 answers how a frozen 15-D state moves under an in-silico controller [11]. The working hypothesis of the findings chapter is that those artefacts are complementary provided they are not collapsed into each other.

Frozen lineages in this repository are:

| ID | Role | Status |
|---|---|---|
| `tnbc_mod_3s` | TNBC-Metabolic-Strain-MOD notebooks | frozen; ROS audit pending |
| `confluence_report_6s` | report architecture X = [T, I, S, L, R, H] | paper only |
| `confluence_v2_15d` | live 15-D CancerODE | adapter sits around it |
| `confluence_v1_calibrator` | `gene_to_parameter_map.json` | executable, not identified |

OnCo integration does not justify changing the dynamics.

OnCo Ideas carry hypothesis, rationale, test and maturity fields. Bulk ingest into CONFLUENCE is forbidden. Idea maturity is not evidence level. OnCo `confidence.probability` is not P(H).

### Layer objects (ontology spec v0.3)

The ingestion spec keeps five objects. A record may point at the next layer. It may not collapse into it.

| Layer | Object | Allowed question | Forbidden leap |
|---|---|---|---|
| Knowledge | `OncoRef` | What does OnCo name and link? | therefore k = … |
| Evidence | `EvidenceObject` | What was measured, where, in what system? | therefore this term belongs in F |
| Causal mechanism | `MechanismObject` | do(U) changes which state, in which context? | therefore identifiable from CCLE |
| Parameter | `ParameterObject` | Which symbol in which frozen model? | silent write into rhs_cancer |
| Prediction | `PredictionObject` | Frozen-model output under U, with uncertainty | clinical advice |

### Conversion ladder

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

Every arrow is a failure point. A page may motivate a hypothesis; it must not become Θ. No arrow may skip a box. Wired RHS terms remain Parameters with provenance `assumed` until identified.

legacy gene→parameter map ≠ identified parameter mapping. Alias trap: legacy `pyruvate_to_lactate` is not v2 `p_lactate` (default 0.22).

---

## Methods

### Honesty gates

The public protocol is sequential and falsifiable. It is not a treatment path.

| Gate | Requirement |
|---|---|
| 0 | Research-only scope. Not a medical device. Not clinical decision support. |
| 1 | Provenance: source, URI, date, population. |
| 2 | Evidence class named. |
| 3 | Mechanism: intervention, state, sign, context, falsifier. |
| 4 | Parameter: model id, symbol, units, estimator, uncertainty, identifiability. Unidentified ≠ established. |
| 5 | Prediction labelled as computational. |
| 6 | Held-out / external / baseline validation. |
| 7 | An experiment that could show the prediction is wrong. |
| 8 | Translation only after biological and clinical validation this repository does not claim. |

Scientific success is: traceability → mathematical validity → identifiability → out-of-sample prediction → experimental falsification.

### Adapter P0

`confluence/onco/` is a read-only client: cache envelope, bindings, schemas. Commands documented in `docs/ONCO_ADAPTER.md` include fixture `meta`, `bind --id ldha`, and `wired`. `bind()` returns slot annotations. It never writes `p_lactate` or `pyruvate_to_lactate`. `refuse_knowledge_as_parameter` returns provenance `forbidden`. Tests live in `tests/test_onco_adapter.py`.

OnCo data is CC BY-NC 4.0. Adapter code in this repository is MIT. Cached payloads are not vendored; live cache stays under `data/onco/cache/` (gitignored). Attribution on every export: Data from OnCo (onco.cc), CC BY-NC 4.0; commercial use needs a licence.

P0 ships: read-only client + cache envelope + bindings + nine gates. P0 does not ship: RHS edits, controller edits, 3-state ROS import, 6-state expansion, automatic fitting, reverse writes to OnCo.

### Refusal rules

1. Do not vendor the OnCo corpus.
2. OnCo is not a parameter source. `refuse_knowledge_as_parameter(OncoRef("ldha"), "p_lactate")` returns provenance `forbidden`.
3. OnCo is not a controller prior.
4. Wired RHS terms remain Parameters with provenance `assumed` until identified.
5. No reverse writes to OnCo.
6. No ODE right-hand-side edits in the P0 / thesis-#1 scope.
7. Public claims need a ledger id or a numbered reference (`docs/CITATION_POLICY.md`).
8. Do not fabricate DOIs.

This manuscript uses the twelve Vancouver entries of the public thesis page. The only DOI in that list that is verified in-repo is `10.3322/caac.70090` [1].

### What was not done

No patient data. No ODE refit. No automatic fitting. No 3-state ROS import. No 6-state expansion. No clinical trial, RECIST adjudication, or dosing table.

---

## Results / architectural findings

These are specification and software-architecture findings. They are not experimental oncology results and not patient outcomes.

1. **Honesty ladder holds as an object model.** Knowledge, Evidence, Causal mechanism, Parameter and Prediction are separate types. A record may point at the next layer. It may not collapse into it.

2. **LDHA is not `p_lactate` (refuse-as-parameter).** `refuse_knowledge_as_parameter(OncoRef("ldha"), "p_lactate")` is `forbidden` [11]. The auditor demo claim — that LDHA expression can be entered as `p_lactate` because OnCo lists LDHA as a lactate-metabolism target — is the claim the gate is built to reject.

3. **Legacy map is assumed / unidentified.** `validation/gene_to_parameter_map.json` maps LDHA → `pyruvate_to_lactate` (+0.10 / +0.30). That row is a legacy numeric hint, not an identified v2 parameter.

4. **No ODE writes.** PR #9 and the P0 adapter do not edit `CancerODE.rhs_cancer` [11].

5. **OnCo is a cited graph, not Θ [10].** Counts move with `buildDate`.

6. **Three-arm programme remains a protocol.** `validation/validation_protocol.md` states H0 versus H1. Thresholds such as r > 0.5 are criteria, not findings [12].

7. **Public ledger keeps year and population.** Claims `EVID-001`–`EVID-010` on the thesis page cite [1–12] without promoting burden statistics into model parameters.

No wet-lab measurement is reported here. No new parameter was identified.

---

## Discussion

Keeping OnCo and CONFLUENCE apart is the result. Integration is valuable only as a gated pipeline: cite → interpret → propose → identify → simulate → test. Collapsing any two boxes produces a confident but unidentifiable model.

WHO language that many cancers can be cured if found early and treated well is retained as a health-system statement [3]. It is not transferred to CONFLUENCE as a product claim.

Disease profiles and the thinking lab are scaffolds for questions. They are not protocols and are not the Scholar landing page. The Scholar article URL is `/thesis` with the same-directory PDF `/thesis.pdf`.

Do not expand the state vector merely because OnCo lists more cell types. Cache has no TTL yet. Those are engineering limits, not clinical limits.

---

## Limitations

- No new wet-lab measurement.
- PR #9 does not identify `p_lactate` or any other Θ.
- The 3-state ROS notebook remains unaudited.
- The three-arm protocol can fail; that would still be a result.
- OnCo counts move with `buildDate`.
- The optional Grok auditor may miss sources; “insufficient” is preferred to invented citations.
- Nigeria 2022 estimates are setting context only [5].
- This document has no DOI.

---

## Future work

1. Iterate the adapter without RHS edits.
2. Admit-list a hypothesis library separate from OnCo Ideas.
3. Identify `p_lactate` only after gates 1–4 stay green and an estimator with uncertainty exists.
4. External / held-out validation (Gate 6) before any experimental falsifier (Gate 7).
5. Optional later: deposit this PDF on Zenodo or a preprint server and only then add `citation_doi` to the HTML landing page.

Translation (Gate 8) is out of scope until biological and clinical validation that this repository does not claim.

---

## References

1. Sung H et al. Global cancer statistics 2024. CA Cancer J Clin. 2026. doi:10.3322/caac.70090. https://doi.org/10.3322/caac.70090
2. IARC. GLOBOCAN 2024 news. 8 July 2026. https://www.iarc.who.int/news-events/global-cancer-statistics-2024-globocan-estimates-of-incidence-and-mortality-worldwide-for-34-cancers-in-186-countries/
3. World Health Organization. Cancer fact sheet. 2026. https://www.who.int/news-room/fact-sheets/detail/cancer
4. World Health Organization. Global status report on cancer 2026. https://www.who.int/publications/i/item/9789240123977
5. IARC. Nigeria fact sheet (GLOBOCAN 2022). https://gco.iarc.who.int/media/globocan/factsheets/populations/566-nigeria-fact-sheet.pdf
6. National Cancer Institute. Triple-negative breast cancer. https://www.cancer.gov/types/breast/patient/triple-negative-brochure
7. National Cancer Institute. Tumour heterogeneity. https://www.cancer.gov/about-nci/organization/dcb/research-programs/tumor-heterogeneity
8. Altrock PM, Liu LL, Michor F. The mathematics of cancer. Nat Rev Cancer. 2015. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5663316/
9. Gatenby RA et al. Adaptive therapy. Cancer Res. 2009. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2676449/
10. Gomila J et al. OnCo. https://onco.cc
11. Ogbonna K. Project Confluence pull request #9. 2026. https://github.com/cloudynirvana/project-confluence/pull/9
12. Project Confluence. Validation protocol. https://github.com/cloudynirvana/project-confluence/blob/main/validation/validation_protocol.md

Only reference 1 carries a DOI present and verified in this repository. PMC, WHO, IARC, NCI and GitHub URLs are used where no DOI is on hand. No DOI is invented.

---

## Disclaimer

**Research technical report.** Project Confluence is a computational research framework. It is not a medical device, not a clinical decision-support system, not a diagnostic or therapeutic product, and not a protocol. This document makes no cure claim, no dosing recommendation, and no claim of patient benefit. Simulated trajectories are not patient outcomes. Current results are computational unless an external experiment is cited.

OnCo data (onco.cc) is cited under CC BY-NC 4.0; commercial use needs a licence. CONFLUENCE software is MIT.

Public HTML: https://confluence-research.vercel.app/thesis  
Citeable PDF: https://confluence-research.vercel.app/thesis.pdf
