# Knowledge Gates for Dynamical Oncology Models: Findings from an OnCo x CONFLUENCE Integration

**Document type:** Research findings chapter (working thesis format)
**Status:** Specification and architectural findings. Not a clinical result.
**Date:** 18 September 2026
**Citation style:** numbered Vancouver-style in-text citations [1] matching section 8.

Word export with superscript citations is the local deliverable `ONCO_CONFLUENCE_THESIS_FINDINGS.docx`.

## ABSTRACT

Oncology now has a public, cited knowledge graph (OnCo) [1-3,6] and a separate computational research simulator (CONFLUENCE v2) [7]. The working hypothesis is that those artefacts are complementary provided they are not collapsed into each other.

The hard problem is converting facts into causal, testable, uncertainty-aware dynamical relationships without treating a web page or a legacy numeric map [9] as an identified parameter. This chapter records the P0 adapter [8] and refuses a universal cure as the success criterion [12,14].

Frozen lineages: tnbc_mod_3s [11], confluence_report_6s, confluence_v2_15d [7], confluence_v1_calibrator [9]. OnCo Ideas [4,5] are an external hypothesis shelf.

## 1. INTRODUCTION

OnCo answers what is named, linked, dated and cited [1,6]. CONFLUENCE v2 answers how a frozen 15-D state moves under control [7]. Mixing them treats LDHA as p_lactate or a mutation effect as identified Theta [9]. Data licence CC BY-NC 4.0; software MIT [2,3].

## 2. SPECIFIC AIMS

1. Forbid skip-level promotion from OnCo knowledge to parameters.
2. Verify artefacts including gene_to_parameter_map.json [9].
3. Ship a read-only P0 adapter without editing CancerODE [8].
4. Separate the OnCo idea shelf [5] from any CONFLUENCE hypothesis library.
5. Scope: control and disease-specific clearance, not a universal cure [12,14].

## 3. MATERIALS

OnCo site and API [1,5,6]; IdeaSchema [4]; project-confluence main [7]; PR #9 [8]; TNBC-Metabolic-Strain-MOD [11]; GLOBOCAN/IARC/WHO/ACS [12-16] as context only. No patient data. No ODE refit.

## 4. FINDINGS

Knowledge is not Evidence is not Mechanism is not Parameter is not Prediction. OnCo confidence.probability is not P(H) [17]. Idea maturity is not evidence_level [4]. refuse_knowledge_as_parameter returns forbidden [8].

Legacy map LDHA -> pyruvate_to_lactate (+0.10 / +0.30) [9,10] is assumed and unidentified. Alias trap: pyruvate_to_lactate is not v2 p_lactate (0.22).

OnCo Ideas carry hypothesis, rationale, test, maturity [4,5]. Bulk ingest forbidden.

WHO: many cancers can be cured if found early and treated well; that is not one cure [14]. Childhood ALL 5-year survival ~90% [16]. Global 2024 burden 20.6 million cases and 9.8 million deaths [12,13]. CONFLUENCE and OnCo do not cure patients [1,7].

## 5-7. DISCUSSION, LIMITATIONS, NEXT WORK

Do not expand X because OnCo lists more cell types. Cache has no TTL yet [8]. ROS audit outstanding [11]. Next: iterate PR #9 without RHS edits; admit-list hypothesis library; identify p_lactate only after gates stay green.

## 8. REFERENCES

1. Gomila J, OnCo contributors. OnCo: a public, cited knowledge graph of oncology. 2026. https://onco.cc
2. Gomila J, OnCo contributors. OnCo source repository. GitHub. 2026. https://github.com/judegomila/OnCo (code MIT; data CC BY-NC 4.0).
3. Gomila J. CITATION.cff. OnCo v1.0.0. 16 September 2026. https://github.com/judegomila/OnCo/blob/main/CITATION.cff
4. OnCo. IdeaSchema. src/lib/schema.ts. https://github.com/judegomila/OnCo/blob/main/src/lib/schema.ts
5. OnCo. Ideas. https://onco.cc/ideas/ and https://onco.cc/api/v1/ideas.json
6. OnCo. llms.txt. 18 September 2026. https://onco.cc/llms.txt
7. Ogbonna K. Project Confluence. https://github.com/cloudynirvana/project-confluence
8. Ogbonna K. feat/onco-adapter-p0 (pull request #9). 18 September 2026. https://github.com/cloudynirvana/project-confluence/pull/9
9. Project Confluence. validation/gene_to_parameter_map.json. https://github.com/cloudynirvana/project-confluence/blob/main/validation/gene_to_parameter_map.json
10. Project Confluence. models/complexity_calibrator.py and agents/bioinformatics_miner.py.
11. Ogbonna K. TNBC-Metabolic-Strain-MOD. https://github.com/cloudynirvana/TNBC-Metabolic-Strain-MOD
12. Sung H, Filho AM, Laversanne M, et al. Global cancer statistics 2024. CA Cancer J Clin. 2026. doi:10.3322/caac.70090. https://doi.org/10.3322/caac.70090
13. IARC. Global cancer statistics 2024 news release. 8 July 2026. https://www.iarc.who.int/news-events/global-cancer-statistics-2024-globocan-estimates-of-incidence-and-mortality-worldwide-for-34-cancers-in-186-countries/
14. World Health Organization. Cancer fact sheet. 2026. https://www.who.int/news-room/fact-sheets/detail/cancer
15. World Health Organization. Global status report on cancer 2026. https://www.who.int/publications/i/item/9789240123977
16. American Cancer Society. Prognostic factors and survival rates for childhood leukemia. 22 July 2025. https://www.cancer.org/cancer/types/leukemia-in-children/detection-diagnosis-staging/survival-rates.html
17. OnCo. CONTRIBUTING.md and src/data/confidence.ts. https://github.com/judegomila/OnCo/blob/main/CONTRIBUTING.md

Attribution: Data from OnCo (onco.cc), CC BY-NC 4.0 [1-3]. Disclaimer: not a medical device; not a cure claim.
