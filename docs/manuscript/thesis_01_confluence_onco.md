# CONFLUENCE × OnCo: An Evidence-Gated Dynamical Framework for Integrating Oncology Knowledge Graphs with Adaptive Cancer-State Models

**Document type:** Thesis #1 — working manuscript (computational research)  
**Author:** Kelechi Emeka Ogbonna  
**Affiliation:** Independent computational research / Project Confluence (GitHub cloudynirvana)  
**Correspondence:** https://github.com/cloudynirvana/project-confluence  
**Date:** 20 September 2026  
**Public HTML:** https://confluence-research.vercel.app/thesis  
**Citeable PDF:** https://confluence-research.vercel.app/thesis.pdf  
**Status:** Architectural and methodological findings. Not a clinical result.  
**Citation style:** numbered Vancouver [n] matching the References list and the public thesis page. See `docs/CITATION_STYLE.md`.  
**DOI:** none registered for this document. Do not reuse the software Zenodo record.

This manuscript expands `ONCO_CONFLUENCE_THESIS_FINDINGS.md`, `docs/ONCO_CONFLUENCE_ONTOLOGY_SPEC.md`, and `docs/ONCO_ADAPTER.md`. It does not add wet-lab measurements, patient-level results, or invented identifiers.

---

## Abstract

Cancer research now produces knowledge graphs, multi-omic assays and dynamical simulators in parallel [10,53,61]. The failure mode is collapsing those layers. This study asks whether a provenance-controlled pipeline can keep the layers apart while still allowing testable predictions. Scientific success is a staged chain — traceability, mathematical validity, identifiability, out-of-sample prediction, experimental falsification — not disease eradication [43-52].

CONFLUENCE v2 is a frozen 15-dimensional computational cancer-state model with an optional connectome-style controller [11,58,60]. OnCo is a public, cited oncology knowledge graph [10,61]. They are complementary only if knowledge is not written into parameters [57,58]. Pull request #9 shipped a read-only OnCo adapter and nine refusal tests. It is architectural evidence for the gates, not therapeutic efficacy [11,56].

GLOBOCAN 2024 estimates, published 2026, place the contemporary burden at about 20.6 million diagnoses and 9.8 million deaths [1,2,64]. That statistic motivates computational oncology; it is not a CONFLUENCE parameter [1,59]. This document records the gates, the P0 adapter, the LDHA / `p_lactate` refusal, and the honesty rule that OnCo confidence is not P(H) [10,11,36,63]. It does not claim a cure, a dose, a clinical decision-support system, or an identified Θ [3,56].

---

## Keywords

computational oncology; knowledge graphs; dynamical systems; evidence gates; identifiability; tumour heterogeneity; tumour microenvironment; OnCo; CONFLUENCE; research-only; not a medical device

---

## Introduction

Oncology knowledge graphs name genes, diseases, ideas and citations [10,53,61]. Dynamical simulators ask how a state moves under control [8,39,43,44]. Mixing the two without provenance treats a web page as an identified parameter [45-52,57]. The problem stated on the public thesis page, and restated here, is: how can heterogeneous oncology knowledge and molecular observations be incorporated into a cancer dynamical model without collapsing evidence, mechanism, parameterisation and prediction into unsupported assumptions [57,59]?

GLOBOCAN 2024 estimates, published 8 July 2026, report about 20.6 million diagnoses and 9.8 million deaths across 34 cancer types in 186 countries [1,2,64]. Female breast cancer accounted for about 2.43 million new cases in that estimate series [1,2]. Those figures continue a modelled GLOBOCAN series previously reported for 2022 and 2020 [54,55]. WHO states that many cancers can be cured if detected early and treated effectively, while access remains uneven [3]. That sentence is a policy statement about staged, treatable disease — not a claim that CONFLUENCE or OnCo cures patients [3,10,11,56]. The WHO global status report on cancer 2026 is cited as health-system context only [4]. Nigeria GLOBOCAN *2022* estimates (127,763 new cases; 79,542 deaths; breast 32,278) are setting context only; they are not 2026 incidence [5].

Cancer is organised as a set of acquired capabilities — the hallmarks — rather than as a single targetable defect [13-15]. Tumours evolve clonally and adapt under therapy [7,16,17]. Intra-tumour heterogeneity, branched evolution and therapy resistance are documented empirical features of that process [18-21]. Multi-omics increase resolution without automatically producing a causal model [20,43,49]. The tumour microenvironment recruits stromal and immune cells that regulate progression, exclusion and metastasis [22-27]. Mathematical oncology supplies in-silico laboratories that couple those layers as dynamical systems [8,39]. Adaptive therapy treats treatment as a process under selection rather than as a fixed maximum-tolerated pulse [9,40,41]. The gap this thesis addresses is connecting knowledge to dynamical hypotheses without dropping provenance [10,11,57].

TNBC is used as a test case because NCI describes it as roughly 15% of breast cancers, typically faster-growing and more recurrent, and heterogeneous [6]. Independent reviews describe the same clinical heterogeneity and recurrence pattern [28-31]. That description is not a CONFLUENCE parameter [6,56,63].

---

## Specific aims

1. Forbid skip-level promotion from OnCo knowledge to ODE parameters [10,11,57,58].
2. Classify legacy artefacts, including `validation/gene_to_parameter_map.json`, as assumed / unidentified rather than as identified Θ [47,51,52,63].
3. Ship a read-only P0 adapter without editing `CancerODE` [11,58].
4. Keep the OnCo Ideas shelf separate from any CONFLUENCE hypothesis library [10,61].
5. Define success as gated, falsifiable computational work — control and disease-specific clearance — not a universal cure [3,12,56].

---

## Background

### Two artefacts, two questions

OnCo answers what is named, linked, dated and cited [10,61]. Biomedical knowledge graphs more generally integrate heterogeneous assertions; they remain literature objects until an identification step is performed [53]. CONFLUENCE v2 answers how a frozen 15-D state moves under an in-silico controller [11,58,60]. The working hypothesis of the findings chapter is that those artefacts are complementary provided they are not collapsed into each other [57].

Frozen lineages in this repository are:

| ID | Role | Status |
|---|---|---|
| `tnbc_mod_3s` | TNBC-Metabolic-Strain-MOD notebooks | frozen; ROS audit pending [62] |
| `confluence_report_6s` | report architecture X = [T, I, S, L, R, H] | paper only [60] |
| `confluence_v2_15d` | live 15-D CancerODE | adapter sits around it [11,58] |
| `confluence_v1_calibrator` | `gene_to_parameter_map.json` | executable, not identified [63] |

OnCo integration does not justify changing the dynamics [11,58].

OnCo Ideas carry hypothesis, rationale, test and maturity fields [10,61]. Bulk ingest into CONFLUENCE is forbidden [57,58]. Idea maturity is not evidence level. OnCo `confidence.probability` is not P(H) [57,59].

### Hallmarks, heterogeneity and the microenvironment

Hanahan and Weinberg organised neoplastic growth as a small set of acquired capabilities, later expanded and re-dimensioned [13-15]. That organising frame is why a knowledge graph of named targets is not a closed dynamical explanation: the same hallmark can be implemented by different molecular routes [14,15,20]. Clonal evolution and ecological competition supply the time axis [16,17]. Intra-tumour heterogeneity is both a looking-glass for those processes and a documented source of therapeutic failure [7,18-21].

The tumour microenvironment is not a passive backdrop. Recruited stromal and immune populations regulate progression and metastasis [22-24]. T-cell exclusion and oncogenic-pathway-driven immune evasion are review-level mechanisms for why a metabolic or antigenic fact does not become a control law [25,26]. Exhaustion is a distinct T-cell state, not a synonym for “the immune variable is low” [27]. None of those reviews is treated here as an identified CONFLUENCE coupling.

### Warburg metabolism and why LDHA is still not `p_lactate`

Aerobic glycolysis is a classical observation in cancer cells [32]. Later work reframed the Warburg effect as a proliferation-linked metabolic programme rather than a simple mitochondrial defect [33-35]. Lactate is not only a waste product: tumour-derived lactic acid can polarise macrophages and blunt T- and NK-cell surveillance [36-38]. A microenvironmental model of carcinogenesis treats acidification and selection as coupled processes [42].

OnCo lists LDHA among lactate-metabolism targets [10]. The CONFLUENCE v2 symbol `p_lactate` is a frozen-model parameter. The P0 adapter refuses to write the OnCo record into that symbol [11,36,58,63]. The literature explains why the gene is scientifically interesting. It does not identify Θ.

### Mathematical oncology and adaptive control as research context

Ordinary differential equations, stochastic models, agent-based approaches and control-theoretic methods can represent tumour growth, treatment response and resistance as in-silico laboratories [8,39]. Adaptive therapy, introduced as an evolutionary alternative to continuous maximum-tolerated dosing, treats sensitive clones as competitive suppressors of resistant clones [9,40]. A later clinical-research programme in metastatic castrate-resistant prostate cancer is cited only as context that the control idea has a literature, not as a CONFLUENCE protocol or a patient result [41]. CONFLUENCE controllers that implement a Gatenby-style treat-and-halt heuristic remain in-silico research arms [9,12,56].

### Layer objects (ontology spec v0.3)

The ingestion spec keeps five objects [57]. A record may point at the next layer. It may not collapse into it.

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

Every arrow is a failure point. A page may motivate a hypothesis; it must not become Θ [45,47,51,57]. No arrow may skip a box. Wired RHS terms remain Parameters with provenance `assumed` until identified [52,63].

legacy gene→parameter map ≠ identified parameter mapping. Alias trap: legacy `pyruvate_to_lactate` is not v2 `p_lactate` (default 0.22) [58,63].

---

## Methods

### Honesty gates

The public protocol is sequential and falsifiable [12,56,59]. It is not a treatment path.

| Gate | Requirement |
|---|---|
| 0 | Research-only scope. Not a medical device. Not clinical decision support. [56] |
| 1 | Provenance: source, URI, date, population. [59] |
| 2 | Evidence class named. [57] |
| 3 | Mechanism: intervention, state, sign, context, falsifier. [57] |
| 4 | Parameter: model id, symbol, units, estimator, uncertainty, identifiability. Unidentified ≠ established. [45-52] |
| 5 | Prediction labelled as computational. [56] |
| 6 | Held-out / external / baseline validation. [12] |
| 7 | An experiment that could show the prediction is wrong. [12] |
| 8 | Translation only after biological and clinical validation this repository does not claim. [3,56] |

Scientific success is: traceability → mathematical validity → identifiability → out-of-sample prediction → experimental falsification [43,44,51,52].

### Identifiability as a methods claim, not a result

Systems biology treats a model as a formal object whose parameters may or may not be recoverable from observations [43,44]. Structural identifiability asks whether a unique parameter vector is consistent with noise-free input–output data [45,46,48,50]. Practical identifiability asks whether finite, noisy data actually constrain those parameters [47,52]. Nonlinear biological systems are often only partially observable; unidentified symbols must remain labelled as such [51]. CONFLUENCE Gate 4 is that literature applied as a refusal rule: OnCo confidence and the legacy numeric map are not estimators [10,47,63]. This manuscript does not report a new identifiability computation.

### Adapter P0

`confluence/onco/` is a read-only client: cache envelope, bindings, schemas [11,58]. Commands documented in `docs/ONCO_ADAPTER.md` include fixture `meta`, `bind --id ldha`, and `wired` [58]. `bind()` returns slot annotations. It never writes `p_lactate` or `pyruvate_to_lactate` [11,63]. `refuse_knowledge_as_parameter` returns provenance `forbidden` [11,57]. Tests live in `tests/test_onco_adapter.py` [11,60].

OnCo data is CC BY-NC 4.0 [10,61]. Adapter code in this repository is MIT [60]. Cached payloads are not vendored; live cache stays under `data/onco/cache/` (gitignored) [58]. Attribution on every export: Data from OnCo (onco.cc), CC BY-NC 4.0; commercial use needs a licence [10,59].

P0 ships: read-only client + cache envelope + bindings + nine gates [11,58]. P0 does not ship: RHS edits, controller edits, 3-state ROS import, 6-state expansion, automatic fitting, reverse writes to OnCo [11,58,62].

### Refusal rules

1. Do not vendor the OnCo corpus [10,61].
2. OnCo is not a parameter source. `refuse_knowledge_as_parameter(OncoRef("ldha"), "p_lactate")` returns provenance `forbidden` [11,36,58].
3. OnCo is not a controller prior [9,40,58].
4. Wired RHS terms remain Parameters with provenance `assumed` until identified [47,52,63].
5. No reverse writes to OnCo [10,61].
6. No ODE right-hand-side edits in the P0 / thesis-#1 scope [11,58].
7. Public claims need a ledger id or a numbered reference (`docs/CITATION_POLICY.md`) [59].
8. Do not fabricate DOIs [59].

This manuscript uses the Vancouver entries generated from `docs/manuscript/thesis_01_bibliography.json`. Every `doi:` field was checked against Crossref or PubMed on 20 September 2026. Agency and GitHub items are URL citations without a DOI [2-7,10-12,56-64].

### What was not done

No patient data. No ODE refit. No automatic fitting. No 3-state ROS import. No 6-state expansion. No clinical trial, RECIST adjudication, or dosing table [12,56].

---

## Results / architectural findings

These are specification and software-architecture findings. They are not experimental oncology results and not patient outcomes [56].

1. **Honesty ladder holds as an object model.** Knowledge, Evidence, Causal mechanism, Parameter and Prediction are separate types [57]. A record may point at the next layer. It may not collapse into it.

2. **LDHA is not `p_lactate` (refuse-as-parameter).** `refuse_knowledge_as_parameter(OncoRef("ldha"), "p_lactate")` is `forbidden` [11,58]. The auditor demo claim — that LDHA expression can be entered as `p_lactate` because OnCo lists LDHA as a lactate-metabolism target — is the claim the gate is built to reject [10,32,36,63].

3. **Legacy map is assumed / unidentified.** `validation/gene_to_parameter_map.json` maps LDHA → `pyruvate_to_lactate` (+0.10 / +0.30) [63]. That row is a legacy numeric hint, not an identified v2 parameter [47,51,52].

4. **No ODE writes.** PR #9 and the P0 adapter do not edit `CancerODE.rhs_cancer` [11,58].

5. **OnCo is a cited graph, not Θ [10,61].** Counts move with `buildDate`.

6. **Three-arm programme remains a protocol.** `validation/validation_protocol.md` states H0 versus H1. Thresholds such as r > 0.5 are criteria, not findings [12].

7. **Public ledger keeps year and population.** Claims `EVID-001`–`EVID-010` on the thesis page cite [1-12] without promoting burden statistics into model parameters [1,5,59,64].

No wet-lab measurement is reported here. No new parameter was identified [45,52,56].

---

## Discussion

Keeping OnCo and CONFLUENCE apart is the result [10,11,57]. Integration is valuable only as a gated pipeline: cite → interpret → propose → identify → simulate → test [49,57]. Collapsing any two boxes produces a confident but unidentifiable model [45,47,51].

The hallmarks, clonal-evolution and microenvironment literatures explain why a named gene, a lactate observation, or a TNBC subtype label is not a sufficient statistic for Θ [13-15,16-21,22-27,28-31]. Warburg and lactate-immune papers explain why LDHA appears in OnCo and why lactate is biologically non-trivial; they still do not license a silent write into `p_lactate` [32-38,42]. Mathematical oncology and adaptive-therapy papers explain why a controller layer can exist around a frozen model; they are not CONFLUENCE efficacy [8,9,39-41].

WHO language that many cancers can be cured if found early and treated well is retained as a health-system statement [3,4]. It is not transferred to CONFLUENCE as a product claim [56]. Prior GLOBOCAN editions (2020, 2022) are cited so that the 2024/2026 series is not treated as a one-off number [1,54,55,64].

Disease profiles and the thinking lab are scaffolds for questions. They are not protocols and are not the Scholar landing page [56,59]. The Scholar article URL is `/thesis` with the same-directory PDF `/thesis.pdf`.

Do not expand the state vector merely because OnCo lists more cell types [10,23,61]. Cache has no TTL yet [58]. Those are engineering limits, not clinical limits [56].

---

## Limitations

- No new wet-lab measurement [56].
- PR #9 does not identify `p_lactate` or any other Θ [11,52].
- The 3-state ROS notebook remains unaudited [62].
- The three-arm protocol can fail; that would still be a result [12].
- OnCo counts move with `buildDate` [10].
- The optional Grok auditor may miss sources; “insufficient” is preferred to invented citations [59].
- Nigeria 2022 estimates are setting context only [5].
- This document has no DOI.
- Identifiability citations define Gate 4; they are not a new structural-identifiability proof for `confluence_v2_15d` [45-52].

---

## Future work

1. Iterate the adapter without RHS edits [11,58].
2. Admit-list a hypothesis library separate from OnCo Ideas [10,61].
3. Identify `p_lactate` only after gates 1–4 stay green and an estimator with uncertainty exists [47,52,58].
4. External / held-out validation (Gate 6) before any experimental falsifier (Gate 7) [12].
5. Optional later: deposit this PDF on Zenodo or a preprint server and only then add `citation_doi` to the HTML landing page.

Translation (Gate 8) is out of scope until biological and clinical validation that this repository does not claim [3,56].

---

## References

<!-- BIBSTART -->
1. Sung H, Filho AM, Laversanne M, Ferlay J, Siegel RL, Soerjomataram I, et al. Global cancer statistics 2024: GLOBOCAN estimates of incidence and mortality worldwide for 34 cancers in 186 countries. CA Cancer J Clin. 2026;76(4):e70090. doi:10.3322/caac.70090. PMID: 42417444.
2. International Agency for Research on Cancer. Global cancer statistics 2024: GLOBOCAN estimates of incidence and mortality worldwide for 34 cancers in 186 countries [Internet]. Lyon: IARC; 2026 Jul 8 [cited 2026 Sep 20]. Available from: https://www.iarc.who.int/news-events/global-cancer-statistics-2024-globocan-estimates-of-incidence-and-mortality-worldwide-for-34-cancers-in-186-countries/
3. World Health Organization. Cancer [Internet]. Geneva: World Health Organization; 2026 [cited 2026 Sep 20]. Available from: https://www.who.int/news-room/fact-sheets/detail/cancer
4. World Health Organization. Global status report on cancer 2026: the future we choose together [Internet]. Geneva: World Health Organization; 2026 Jul 8 [cited 2026 Sep 20]. Available from: https://www.who.int/publications/i/item/9789240123977
5. International Agency for Research on Cancer. Nigeria fact sheet (GLOBOCAN 2022 estimates) [Internet]. Lyon: IARC Global Cancer Observatory; 2022 [cited 2026 Sep 20]. Available from: https://gco.iarc.who.int/media/globocan/factsheets/populations/566-nigeria-fact-sheet.pdf
6. National Cancer Institute. Triple-negative breast cancer [Internet]. Bethesda (MD): National Cancer Institute; 2026 [cited 2026 Sep 20]. Available from: https://www.cancer.gov/types/breast/patient/triple-negative-brochure
7. National Cancer Institute. Tumor heterogeneity [Internet]. Bethesda (MD): National Cancer Institute; 2026 [cited 2026 Sep 20]. Available from: https://www.cancer.gov/about-nci/organization/dcb/research-programs/tumor-heterogeneity
8. Altrock PM, Liu LL, Michor F. The mathematics of cancer: integrating quantitative models. Nat Rev Cancer. 2015;15(12):730-745. doi:10.1038/nrc4029. PMID: 26597528.
9. Gatenby RA, Silva AS, Gillies RJ, Frieden BR. Adaptive therapy. Cancer Res. 2009;69(11):4894-4903. doi:10.1158/0008-5472.CAN-08-3658. PMID: 19487300.
10. Gomila J, OnCo contributors. OnCo: a public, cited knowledge graph of oncology [Internet]. Version 1.0.0. OnCo; 2026 Sep 16 [cited 2026 Sep 20]. Available from: https://onco.cc
11. Ogbonna KE. feat/onco-adapter-p0 (pull request #9) [Internet]. GitHub; 2026 Sep 18 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/pull/9
12. Ogbonna KE. Validation protocol [Internet]. Project Confluence / GitHub; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/validation/validation_protocol.md
13. Hanahan D, Weinberg RA. The hallmarks of cancer. Cell. 2000;100(1):57-70. doi:10.1016/S0092-8674(00)81683-9. PMID: 10647931.
14. Hanahan D, Weinberg RA. Hallmarks of cancer: the next generation. Cell. 2011;144(5):646-674. doi:10.1016/j.cell.2011.02.013. PMID: 21376230.
15. Hanahan D. Hallmarks of cancer: new dimensions. Cancer Discov. 2022;12(1):31-46. doi:10.1158/2159-8290.CD-21-1059. PMID: 35022204.
16. Nowell PC. The clonal evolution of tumor cell populations. Science. 1976;194(4260):23-28. doi:10.1126/science.959840. PMID: 959840.
17. Merlo LMF, Pepper JW, Reid BJ, Maley CC. Cancer as an evolutionary and ecological process. Nat Rev Cancer. 2006;6(12):924-935. doi:10.1038/nrc2013. PMID: 17109012.
18. Marusyk A, Almendro V, Polyak K. Intra-tumour heterogeneity: a looking glass for cancer? Nat Rev Cancer. 2012;12(5):323-334. doi:10.1038/nrc3261. PMID: 22513401.
19. Gerlinger M, Rowan AJ, Horswell S, Larkin J, Endesfelder D, Gronroos E, et al. Intratumor heterogeneity and branched evolution revealed by multiregion sequencing. N Engl J Med. 2012;366(10):883-892. doi:10.1056/NEJMoa1113205. PMID: 22397650.
20. McGranahan N, Swanton C. Clonal heterogeneity and tumor evolution: past, present, and the future. Cell. 2017;168(4):613-628. doi:10.1016/j.cell.2017.01.018. PMID: 28187284.
21. Dagogo-Jack I, Shaw AT. Tumour heterogeneity and resistance to cancer therapies. Nat Rev Clin Oncol. 2018;15(2):81-94. doi:10.1038/nrclinonc.2017.166. PMID: 29115304.
22. Quail DF, Joyce JA. Microenvironmental regulation of tumor progression and metastasis. Nat Med. 2013;19(11):1423-1437. doi:10.1038/nm.3394. PMID: 24202395.
23. Hanahan D, Coussens LM. Accessories to the crime: functions of cells recruited to the tumor microenvironment. Cancer Cell. 2012;21(3):309-322. doi:10.1016/j.ccr.2012.02.022. PMID: 22439926.
24. Anderson NM, Simon MC. The tumor microenvironment. Curr Biol. 2020;30(16):R921-R925. doi:10.1016/j.cub.2020.06.081. PMID: 32810447.
25. Joyce JA, Fearon DT. T cell exclusion, immune privilege, and the tumor microenvironment. Science. 2015;348(6230):74-80. doi:10.1126/science.aaa6204. PMID: 25838376.
26. Spranger S, Gajewski TF. Impact of oncogenic pathways on evasion of antitumour immune responses. Nat Rev Cancer. 2018;18(3):139-147. doi:10.1038/nrc.2017.117. PMID: 29326431.
27. Wherry EJ, Kurachi M. Molecular and cellular insights into T cell exhaustion. Nat Rev Immunol. 2015;15(8):486-499. doi:10.1038/nri3862. PMID: 26205583.
28. Foulkes WD, Smith IE, Reis-Filho JS. Triple-negative breast cancer. N Engl J Med. 2010;363(20):1938-1948. doi:10.1056/NEJMra1001389. PMID: 21067385.
29. Bianchini G, Balko JM, Mayer IA, Sanders ME, Gianni L. Triple-negative breast cancer: challenges and opportunities of a heterogeneous disease. Nat Rev Clin Oncol. 2016;13(11):674-690. doi:10.1038/nrclinonc.2016.66. PMID: 27184417.
30. Dent R, Trudeau M, Pritchard KI, Hanna WM, Kahn HK, Sawka CA, et al. Triple-negative breast cancer: clinical features and patterns of recurrence. Clin Cancer Res. 2007;13(15):4429-4434. doi:10.1158/1078-0432.CCR-06-3045. PMID: 17671126.
31. Lehmann BD, Bauer JA, Chen X, Sanders ME, Chakravarthy AB, Shyr Y, et al. Identification of human triple-negative breast cancer subtypes and preclinical models for selection of targeted therapies. J Clin Invest. 2011;121(7):2750-2767. doi:10.1172/JCI45014. PMID: 21633166.
32. Warburg O. On the origin of cancer cells. Science. 1956;123(3191):309-314. doi:10.1126/science.123.3191.309. PMID: 13298683.
33. Vander Heiden MG, Cantley LC, Thompson CB. Understanding the Warburg effect: the metabolic requirements of cell proliferation. Science. 2009;324(5930):1029-1033. doi:10.1126/science.1160809. PMID: 19460998.
34. Pavlova NN, Thompson CB. The emerging hallmarks of cancer metabolism. Cell Metab. 2016;23(1):27-47. doi:10.1016/j.cmet.2015.12.006. PMID: 26771115.
35. Gatenby RA, Gillies RJ. Why do cancers have high aerobic glycolysis? Nat Rev Cancer. 2004;4(11):891-899. doi:10.1038/nrc1478. PMID: 15516961.
36. Brand A, Singer K, Koehl GE, Kolitzus M, Schoenhammer G, Thiel A, et al. LDHA-associated lactic acid production blunts tumor immunosurveillance by T and NK cells. Cell Metab. 2016;24(5):657-671. doi:10.1016/j.cmet.2016.08.011. PMID: 27641098.
37. Colegio OR, Chu NQ, Szabo AL, Chu T, Rhebergen AM, Jairam V, et al. Functional polarization of tumour-associated macrophages by tumour-derived lactic acid. Nature. 2014;513(7519):559-563. doi:10.1038/nature13490. PMID: 25043024.
38. Fischer K, Hoffmann P, Voelkl S, Meidenbauer N, Ammer J, Edinger M, et al. Inhibitory effect of tumor cell-derived lactic acid on human T cells. Blood. 2007;109(9):3812-3819. doi:10.1182/blood-2006-07-035972. PMID: 17255361.
39. Anderson ARA, Quaranta V. Integrative mathematical oncology. Nat Rev Cancer. 2008;8(3):227-234. doi:10.1038/nrc2329. PMID: 18273038.
40. Gatenby RA, Brown JS. Integrating evolutionary dynamics into cancer therapy. Nat Rev Clin Oncol. 2020;17(11):675-686. doi:10.1038/s41571-020-0411-1. PMID: 32699310.
41. Zhang J, Cunningham JJ, Brown JS, Gatenby RA. Integrating evolutionary dynamics into treatment of metastatic castrate-resistant prostate cancer. Nat Commun. 2017;8(1):1816. doi:10.1038/s41467-017-01968-5. PMID: 29180633.
42. Gatenby RA, Gillies RJ. A microenvironmental model of carcinogenesis. Nat Rev Cancer. 2008;8(1):56-61. doi:10.1038/nrc2255. PMID: 18059462.
43. Kitano H. Systems biology: a brief overview. Science. 2002;295(5560):1662-1664. doi:10.1126/science.1069492. PMID: 11872829.
44. Kitano H. Computational systems biology. Nature. 2002;420(6912):206-210. doi:10.1038/nature01254. PMID: 12432404.
45. Bellman R, Åström K. On structural identifiability. Math Biosci. 1970;7(3-4):329-339. doi:10.1016/0025-5564(70)90132-X.
46. Ljung L, Glad T. On global identifiability for arbitrary model parametrizations. Automatica. 1994;30(2):265-276. doi:10.1016/0005-1098(94)90029-9.
47. Raue A, Kreutz C, Maiwald T, Bachmann J, Schilling M, Klingmüller U, et al. Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. Bioinformatics. 2009;25(15):1923-1929. doi:10.1093/bioinformatics/btp358. PMID: 19505944.
48. Chis OT, Banga JR, Balsa-Canto E. Structural identifiability of systems biology models: a critical comparison of methods. PLoS One. 2011;6(11):e27755. doi:10.1371/journal.pone.0027755. PMID: 22132135.
49. Villaverde AF, Banga JR. Reverse engineering and identification in systems biology: strategies, perspectives and challenges. J R Soc Interface. 2014;11(91):20130505. doi:10.1098/rsif.2013.0505. PMID: 24307566.
50. Villaverde AF, Barreiro A, Papachristodoulou A. Structural identifiability of dynamic systems biology models. PLoS Comput Biol. 2016;12(10):e1005153. doi:10.1371/journal.pcbi.1005153. PMID: 27792726.
51. Villaverde AF. Observability and structural identifiability of nonlinear biological systems. Complexity. 2019;2019:8497093. doi:10.1155/2019/8497093.
52. Wieland FG, Hauber AL, Rosenblatt M, Tönsing C, Timmer J. On structural and practical identifiability. Curr Opin Syst Biol. 2021;25:60-69. doi:10.1016/j.coisb.2021.03.005.
53. Himmelstein DS, Lizee A, Hessler C, Brueggeman L, Chen SL, Hadley D, et al. Systematic integration of biomedical knowledge prioritizes drugs for repurposing. Elife. 2017;6:e26726. doi:10.7554/eLife.26726. PMID: 28936969.
54. Bray F, Laversanne M, Sung H, Ferlay J, Siegel RL, Soerjomataram I, et al. Global cancer statistics 2022: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA Cancer J Clin. 2024;74(3):229-263. doi:10.3322/caac.21834. PMID: 38572751.
55. Sung H, Ferlay J, Siegel RL, Laversanne M, Soerjomataram I, Jemal A, et al. Global cancer statistics 2020: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA Cancer J Clin. 2021;71(3):209-249. doi:10.3322/caac.21660. PMID: 33538338.
56. Ogbonna KE. DISCLAIMER.md [Internet]. Project Confluence / GitHub; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/DISCLAIMER.md
57. Ogbonna KE. OnCo–CONFLUENCE ontology and evidence-ingestion specification, v0.3 [Internet]. Project Confluence / GitHub; 2026 Sep 18 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/docs/ONCO_CONFLUENCE_ONTOLOGY_SPEC.md
58. Ogbonna KE. OnCo adapter (P0) [Internet]. Project Confluence / GitHub; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/docs/ONCO_ADAPTER.md
59. Ogbonna KE. Citation policy (CONFLUENCE public surfaces) [Internet]. Project Confluence / GitHub; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/docs/CITATION_POLICY.md
60. Ogbonna KE. Project Confluence [Internet]. GitHub; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence
61. Gomila J, OnCo contributors. OnCo source repository [Internet]. Version 1.0.0. GitHub; 2026 Sep 16 [cited 2026 Sep 20]. Available from: https://github.com/judegomila/OnCo
62. Ogbonna KE. TNBC-Metabolic-Strain-MOD [Internet]. GitHub; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/TNBC-Metabolic-Strain-MOD
63. Ogbonna KE. validation/gene_to_parameter_map.json [Internet]. Project Confluence / GitHub; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/validation/gene_to_parameter_map.json
64. International Agency for Research on Cancer. Global Cancer Observatory [Internet]. Lyon: IARC; 2026 [cited 2026 Sep 20]. Available from: https://gco.iarc.who.int/
<!-- BIBEND -->

DOI fields appear only for Crossref- or PubMed-verified journal records. WHO, IARC, NCI and GitHub items are complete Internet citations without a `doi:` field. No DOI is invented. Style: `docs/CITATION_STYLE.md`.

---

## Disclaimer

**Research technical report.** Project Confluence is a computational research framework. It is not a medical device, not a clinical decision-support system, not a diagnostic or therapeutic product, and not a protocol [56]. This document makes no cure claim, no dosing recommendation, and no claim of patient benefit [3,56]. Simulated trajectories are not patient outcomes. Current results are computational unless an external experiment is cited.

OnCo data (onco.cc) is cited under CC BY-NC 4.0; commercial use needs a licence [10,61]. CONFLUENCE software is MIT [60].

Public HTML: https://confluence-research.vercel.app/thesis  
Citeable PDF: https://confluence-research.vercel.app/thesis.pdf
