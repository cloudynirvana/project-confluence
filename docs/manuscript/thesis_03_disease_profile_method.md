# Disease Profiles for Complex Pathologies: A Gated Method for Systemic Personalized-Medicine Research Objects

**Thesis #3 — working method manuscript**  
**Author:** Kelechi Emeka Ogbonna  
**Affiliation:** Project Confluence (computational research)  
**Date:** 20 September 2026  
**Document type:** Thesis-format method paper (research / in-silico only)  
**Status:** Method paper on the shipped Disease Profile research object (`confluence/profiles/`, `data/profiles/cases/`, `schemas/disease_profile.schema.json` on `main` at `64ba76b`). Not a clinical result. Not a patient-level product.  
**Citation style:** numbered Vancouver references matching the reference list.

---

## Abstract

Personalized medicine is often described as if a sufficiently complete molecular chart of a person would yield a dose. That leap is a product claim. It is not a research method. Complex pathologies — triple-negative breast cancer (TNBC) with coupled metabolic and immune exclusion, glioblastoma (GBM) with hypoxic and invasive niches, pancreatic ductal adenocarcinoma (PDAC) with a desmoplastic stromal barrier, and dormant or occult residual disease — do not become usable research objects by pasting a knowledge-graph page into an ordinary differential equation (ODE). They become usable when a laboratory can name what was asked, what was observed, what mechanism is hypothesized, what must not become a parameter, and which public dataset could falsify the claim.

This thesis defines the **Disease Profile** as a **versioned research object** for systemic personalized-medicine *research*. A Disease Profile is not a patient chart, not an electronic health record extract, not a clinical decision-support (CDS) artefact, and not personalized medicine as a clinical product. It is the durable export of a four-question thinking laboratory: *Who is asking? Which disease — not “cancer”? What is the current regime? Where is the system stuck?* Completing those questions produces a board. Exporting the board produces a profile with a schema version, a timestamp, a disease identifier, observables, candidate mechanisms, an explicit non-parameter list, admitted hypotheses, Vancouver citations, and a fixed research-only disclaimer.

Admission is gated. The invariant is

> Knowledge ≠ Evidence ≠ Mechanism ≠ Parameter ≠ Prediction.

No arrow may skip a box. OnCo knowledge records, OnCo `confidence.probability`, OnCo Idea maturity, and the legacy CONFLUENCE gene-to-parameter map must not enter the parameter vector Θ of any frozen model lineage. Profiles feed `confluence.profiles.HypothesisObject` records that name a public dataset and a falsifier. They do not write coefficients.

The method is situated against artefacts that now exist on `main` and does not invent outcomes for them. CONFLUENCE pull request #9 shipped a read-only OnCo adapter (P0) that annotates slots and refuses knowledge-as-parameter without editing `CancerODE`. Pull request #11 merged the Disease Profile exporter, JSON Schema `1.0.0`, a four-disease case pack under `data/profiles/cases/`, and a distinct profile-layer `HypothesisObject` (`64ba76b`). Complexity Science CaseCards (companion repository, draft CaseCard schema and seed pack) remain YAML research templates for the same four complex pathologies, with qualitative Nigeria Standard Treatment Guidelines (NSTG) constraints that must never become numeric scales. A Disease Profile is the CONFLUENCE-side, thinking-lab-native research object; a CaseCard is the Complexity Science-side, contributor-native research object. Both refuse the same skip-level promotion.

Worked examples in this manuscript are **qualitative boards** that cite the shipped case-pack JSON. They are not simulated patient benefit and not virtual-cohort efficacy. The discussion treats efficiency as a research-operations question: how cheaply a laboratory can add a new complex disease without opening the ODE, without smuggling knowledge into Θ, and without converting a chart into a prescription. Limitations are stated as method limits, not as missing clinical performance. Future work is staged: keep the in-repo contract honest, bind named public datasets, keep P0 read-only, and leave translation behind a gate this repository does not claim to have passed.

**This document is not personalized medicine as a clinical product.**

## Keywords

Disease Profile; research object; personalized-medicine research; thinking laboratory; knowledge gates; HypothesisObject; OnCo; CONFLUENCE; CaseCard; triple-negative breast cancer; glioblastoma; pancreatic ductal adenocarcinoma; cancer dormancy; FAIR; identifiability; not clinical decision support

---

## 1. Introduction

### 1.1 Two meanings of “personalized medicine”

“Personalized medicine” and “precision medicine” entered the biomedical lexicon as a research programme: measure more of the person and the tumour, then test whether those measurements change a mechanistic claim [1,2]. The same words are now used, loosely, for a clinical product: a chart, a score, a recommended regimen. The second use is a CDS claim. Project Confluence does not make it [3,4].

The distinction is not rhetorical. A research programme may keep Knowledge, Evidence, Mechanism, Parameter, and Prediction as separate objects and still fail scientifically — by being unidentified, unreplicated, or wrong. A clinical product that collapses those layers into a dose has failed a different, regulatory and ethical, test even before the science is judged [3–5]. Thesis #3 is a method paper for the first programme. It refuses the second.

### 1.2 Why complex pathologies break the chart-to-parameter habit

Mathematical oncology already supplies in-silico laboratories for growth, resistance, and control [6,7]. Metabolic reprogramming (the Warburg effect and its extensions) is a shared, not disease-unique, research backdrop [70]. Adaptive-therapy research treats treatment as a process under selection rather than as a single fixed blow [7]. Those methods do not license a silent write from a web page into Θ.

Complex pathologies make the habit especially expensive:

- **TNBC** is a heterogeneous epithelial disease in which metabolic competition and spatial immune exclusion are hypothesized to co-limit effector function [8–11,69]. National Cancer Institute educational summaries describe TNBC as a minority of breast cancers with faster growth and higher recurrence than some other invasive subtypes [12]. That sentence is a *descriptor*. It is not a lactate parameter.
- **GBM** is an infiltrative primary brain tumour whose hypoxic, perivascular, and invasive niches are hypothesized to couple survival, migration, and treatment escape [13–16]. Pseudopalisading necrosis is a histological hallmark and a candidate observable. It is not an oxygen-tension coefficient.
- **PDAC** is a stroma-dominated carcinoma in which physical and immunological barriers are hypothesized to limit delivery and effector contact [17–20]. Late diagnosis is a systems failure of detection as much as of biochemistry [21,22]. A better right-hand side will not find the patient sooner.
- **Dormant or occult residual disease** is a latency problem: quiescent cells, angiogenic population pauses, or immune-held micrometastases may persist below a given assay’s detection floor [23–26]. Occult means unobserved at that threshold. It does not mean a hidden parameter, and it does not license a provocation protocol.

Each of those diseases has a public, cited knowledge graph entry on OnCo [27–29] and a lumped handle somewhere in CONFLUENCE’s frozen lineages [30,31]. The failure mode documented in the P0 findings chapter is to treat the first as the second: LDHA as `p_lactate`, or a legacy `LDHA → pyruvate_to_lactate (+0.10 / +0.30)` map as an identified parameter [31–33]. Alias trap: legacy `pyruvate_to_lactate` is not v2 `p_lactate` (default 0.22) [31].

### 1.3 Research objects, not charts

Scientific reuse needs objects that travel: identified, versioned, cited, and honest about what they are not. The FAIR guiding principles ask that research artefacts be findable, accessible, interoperable, and reusable [34]. Workflow-centric research objects were proposed for the same reason — a paper plus a dump of files is not a laboratory memory [35]. A Disease Profile is built in that spirit. It is closer to a methods object than to a medical record.

A patient chart answers *who is this person, today, under a duty of care?* A Disease Profile answers *what did this laboratory admit about this disease-class as a system, under these questions, on this date, with these citations, and which promotions did it refuse?* The first object belongs in a clinic with a licence. The second belongs in a repository with a schema version.

### 1.4 Two companion artefacts this thesis does not over-claim

Two existing specifications already refuse skip-level promotion. This manuscript cites them as documents and pull requests. It does not invent experimental outcomes for them.

1. **CONFLUENCE adapter P0 (pull request #9, merged).** A read-only OnCo client, cache envelope, bindings, and nine gates. `bind()` returns slot annotations. `refuse_knowledge_as_parameter` returns provenance `forbidden`. `CancerODE` is not edited [31,32,36,37].
2. **Disease Profile exporter and case pack (pull request #11, merged at `64ba76b`).** Pydantic models in `confluence/profiles/`, JSON Schema `schemas/disease_profile.schema.json`, thinking-lab JSON export, batch builder `scripts/build_disease_profile_pack.py`, and four gated JSON files under `data/profiles/cases/` [43,71–73,77].
3. **Complexity Science CaseCards (companion repository, draft CaseCard pipeline).** A YAML `CaseCard` is data. Contributors add a case without opening an ODE. Seed cards exist for TNBC metabolic–immune exclusion, GBM invasive niche / hypoxia, PDAC stromal barrier, and dormant/occult disease. NSTG is a structured clinical-knowledge *constraint* layer and is never auto-translated into a coefficient [38–40]. Official NSTG 2022 is cited, not redistributed [41].

Thesis #3 is the method paper for the CONFLUENCE-native object that a thinking laboratory already exports. It cites those paths. It does not propose a second, parallel contract.

### 1.5 What this thesis is not

This thesis is not a digital twin of a named person. It is not a CDS rule set. It is not a claim that CONFLUENCE restored complexity in a patient. It is not a claim that OnCo pages are evidence. It is not a claim that the four complex-case templates are validated models. GLOBOCAN 2024 estimates — about 20.6 million diagnoses and 9.8 million deaths across 34 cancers and 186 countries — are setting context for why disease-specific systems work is necessary [21,22,67]. They are not CONFLUENCE results.

---

## 2. Aims

The aims are methodological. Success is a usable contract and four honest boards, not a survival difference.

1. **Define** the Disease Profile as a versioned research object for systemic personalized-medicine research, and state the negative definition: not a patient chart, not CDS, not personalized medicine as a clinical product.
2. **Specify** a four-question thinking laboratory whose completed board is the only legal source of a profile export (`evidence/thinking.html` → `disease-profile-<slug>.json`).
3. **Formalize** the five-layer gate  
   `Knowledge ≠ Evidence ≠ Mechanism ≠ Parameter ≠ Prediction`  
   as the admission function implemented in `confluence.profiles.disease_profile.admit_hypotheses`, with an explicit non-parameter list that includes OnCo knowledge, OnCo confidence, Idea maturity, and legacy gene-to-parameter maps [71].
4. **Frame** the shipped complex-case pack — `data/profiles/cases/{tnbc_metabolic_immune,gbm_invasive_niche,pdac_stromal_barrier,dormant_occult}.json` — as *research templates* aligned with Complexity Science CaseCards, not as fitted disease models [73].
5. **Show** how an admitted profile feeds `confluence.profiles.HypothesisObject` records aimed at **named public datasets**, without writing OnCo knowledge into Θ and without editing `CancerODE` [74,75].
6. **Situate** the profile relative to CONFLUENCE adapter P0 and Complexity Science CaseCards, citing repository documents only, with no fabricated outcomes.

A non-aim, stated so it cannot be inferred: this thesis does not estimate a treatment effect, does not identify `p_lactate` or any other Θ symbol, and does not promote any CaseCard mechanism to a CONFLUENCE right-hand side.

---

## 3. Methods

### 3.1 Scope, materials, and non-claims

**Scope.** Computational research method. In-silico and bibliographic. No patient-identifiable data. No ODE refit. No controller prior from OnCo.

**Materials.**

- Project Confluence `main` at the Disease Profile merge (`64ba76b`, pull request #11, on top of P0 / pull request #9): OnCo adapter, ontology spec v0.3, findings chapter, thinking laboratory, thesis evidence page, Disease Profile package [30–32,36,37,42,43,71–77].
- Shipped Disease Profile contract: `confluence/profiles/disease_profile.py` (Pydantic; `SCHEMA_VERSION = "1.0.0"`), `schemas/disease_profile.schema.json`, case table `data/profiles/cases/cases.yaml`, gated JSON under `data/profiles/cases/`, builder `scripts/build_disease_profile_pack.py` [71–73,77].
- Profile-layer `HypothesisObject` at `confluence/profiles/hypothesis_object.py`, distinct from `confluence.onco.schemas.HypothesisObject` (OnCo idea shelf) [37,74]. Committed example: `data/hypotheses/tnbc_lactate_immune_exclusion.yaml` [75].
- Complexity Science repository documents and draft CaseCard schema / seed pack [38–40].
- OnCo public site, Ideas shelf, and licence (data CC BY-NC 4.0; software MIT) [27–29,44].
- Named public datasets listed in section 3.7 and in CONFLUENCE validation notes [45,46], including CCLE / DepMap releases [52–54,68].
- Authoritative descriptors used only as knowledge or setting context: GLOBOCAN/IARC/WHO/NCI [12,21,22,47].

**Frozen model lineages** (names only; not refit here) [31]:

| ID | What | Status in this thesis |
|---|---|---|
| `tnbc_mod_3s` | TNBC-Metabolic-Strain-MOD notebooks | frozen; ROS audit pending; not imported |
| `confluence_report_6s` | report architecture X = [T, I, S, L, R, H] | paper only |
| `confluence_v2_15d` | live 15-D CancerODE | adapter sits around it; **no RHS edits** |
| `confluence_v1_calibrator` | `validation/gene_to_parameter_map.json` | executable, assumed / unidentified |

**Non-claims** (repeated so they cannot be skipped). Simulated trajectories, thinking-lab boards, and Disease Profiles are computational artefacts. They are not patient outcomes, not a protocol, and not a dose [3,4]. Disease-class labels in CONFLUENCE (`benign`, `malignant`, `occult`, `dormant`, `terminal`) are state-signature modes, not TNM or histopathology [3]. NSTG and related Nigerian policy documents are knowledge constraints, not executable care [40,41,48,49].

### 3.2 Disease Profile as a versioned research object

A **Disease Profile** is a structured, versioned, citable export of a thinking-lab board for one disease-class (not one person). On `main` it is the Pydantic model `confluence.profiles.DiseaseProfile` and the JSON Schema at `schemas/disease_profile.schema.json` [71,72].

**Identity.** `{profile_id, disease_id, schema_version, created_at}`. `profile_id` is unique per export. `schema_version` is currently the constant `1.0.0`. Changing the contract increments the version; old profiles remain readable as historical objects.

**Contents (shipped contract, schema version `1.0.0`).**

| Field | Layer | Required | Role |
|---|---|---|---|
| `profile_id` | identity | yes | Durable identifier of this export |
| `disease_id`, `disease_label` | knowledge | yes | Disease-class, not a patient identifier |
| `created_at` | provenance | yes | UTC timestamp of the export |
| `schema_version` | provenance | yes | Contract version (`1.0.0`) |
| `asker_role` | context | yes | Mapped from question 1 (`researcher`, `clinician`, `student`, `patient_advocate`, `other`) |
| `answers` | context | yes | The four questions and chosen labels |
| `observables` | evidence pointers | yes (may be empty only if explicitly scaffolded) | Statements with citation ids; not parameters |
| `candidate_mechanisms` | mechanism | yes | Hypothesis-class statements with `evidence_class` and a falsifier |
| `non_parameters` | honesty | yes | Objects forbidden to enter Θ |
| `admitted_hypotheses` | hypothesis | yes | Survivors of the gates; `parameter_status` = `forbidden_to_enter_theta` |
| `citations` | bibliography | yes (≥1) | Vancouver entries; DOI only if verified |
| `disclaimer` | honesty | yes | Fixed research-only string |

**Negative definition.**

- Not a patient chart: no medical record number, no care plan, no duty-of-care narrative.
- Not CDS: no “if observable X then give Y”.
- Not a parameter table: no `p_lactate`, no `ec50`, no `x_cap_scale`, no infection-risk weight.
- Not an OnCo page: OnCo records may be *cited* as Knowledge; they are not copied into Θ.
- Not a PredictionObject: the profile does not emit a simulator output or a clinical forecast.

**Default non-parameters** (minimum set; profiles may add but not remove the spirit of the list):

1. OnCo knowledge records (pages, targets, ideas).
2. OnCo `confidence.probability`.
3. OnCo Idea maturity.
4. Legacy `validation/gene_to_parameter_map.json` values, including `LDHA → pyruvate_to_lactate`.
5. CONFLUENCE v2 symbols such as `p_lactate` unless independently identified in a later, gated study.
6. Auditor classification scores and Grok evidence-audit output.
7. Thinking-lab role, setting, or stuck *labels* (they are questions, not measurements).
8. Clinical intent, cure language, or dosing.

**Versioning rules.**

1. A new thinking-lab completion produces a new `profile_id`. It does not overwrite history.
2. A schema change that adds a required field or changes a gate is a minor or major version increment, not a silent mutation of old JSON.
3. Citations are part of the object. Removing a citation without a new profile is a contract violation.
4. The disclaimer string is fixed. A profile that edits it is invalid.

This is the sense in which a Disease Profile is a research object: it can be cited, diffed, refused, and replayed. A chart that changes under a clinician’s hand every hour is a different kind of object.

### 3.3 Four-question thinking laboratory → profile export

The thinking laboratory already shipped on the evidence site asks four questions before it will draw a systems board [42]. Thesis #3 promotes that walkthrough from a teaching page to an export protocol.

| # | Question | Why it is asked | What it must not become |
|---|---|---|---|
| Q1 | Who is asking? | The next honest question differs for a researcher, a modeller, a student, or a technical family member. | A permission to prescribe |
| Q2 | Which disease — not “cancer”? | The unit is a named pathology. “Cancer” is not a mechanism. | A licence to copy ALL success onto a solid tumour |
| Q3 | What is the current regime? | Prevention, localised, metastatic/relapsed, and haematologic regimes change intent. | A staging engine |
| Q4 | Where is the system stuck? | Biology unnamed, target unreachable, adaptation, late detection, non-travelling models, or access. | A hidden parameter for that bottleneck |

**Export rule.** A profile may be emitted only after all four answers exist. Partial boards are worksheets, not research objects.

**Mapping from answers to profile fields.**

- Q1 → `asker_role` and `answers.role`.
- Q2 → `disease_id`, `disease_label`, and `answers.cancer`.
- Q3 → `answers.setting` (regime). The profile remains a disease-class object; regime is context for which hypotheses are in play, not a patient stage.
- Q4 → `answers.stuck`. The stuck label selects which *move* the board recommends (measure, do not add an RHS term, do not pretend the ODE will find the patient). It does not select a Θ symbol.

**Board before export.** The laboratory renders a qualitative board with eight sections, matching the shipped thinking-lab structure [42]:

1. Name the success unit (cure-aimed protocol versus control versus interception — as *language about intent*, not as a CONFLUENCE score).
2. Coupled system, not a page.
3. Five layers before any number.
4. A disease-specific example that keeps OnCo on the Knowledge side.
5. What a frozen model may hold (annotation, not a write).
6. The trap this disease invites.
7. The move the bottleneck demands.
8. What the page refuses (no regimen, no dose, no cure claim).

The export is a serialization of that board plus citations plus the non-parameter list plus the gate verdicts. It is not a screenshot of a simulator.

### 3.4 Gates: Knowledge ≠ Evidence ≠ Mechanism ≠ Parameter ≠ Prediction

The ontology specification already drew the ladder [31]. Thesis #3 uses it as an admission function.

```
OnCo knowledge
  --cite--> Evidence          (source URI + rung required)
    --interpret--> Hypothesis (falsifier required)
      --propose--> Mechanism  (context + sign + do-operator)
        --identify--> Parameter  (model_id + symbol + identifiability ≠ unidentified)
          --simulate--> Prediction
            --test--> Experiment
              --write--> Evidence
```

No arrow may skip a box. Wired right-hand-side terms in `confluence_v2_15d` remain Parameters with provenance `assumed` until identified [31,36]. OnCo `confidence.probability` is not P(H). Idea maturity is not CONFLUENCE `evidence_level` [31,44].

**Admission rules for a candidate mechanism → admitted hypothesis.**

A candidate is **refused** (stays a candidate; does not become admitted) if any of the following hold:

| Failure | Typical smuggle | Gate named |
|---|---|---|
| Knowledge presented as evidence | `evidence_class` ∈ {`knowledge`, `onco_page`, `confidence`} | Knowledge≠Evidence |
| Mechanism without a falsifier | empty `falsifier` | Evidence≠Mechanism |
| Knowledge or confidence written as Θ | LDHA / OnCo / `confidence.probability` / Idea maturity / `p_lactate` / `pyruvate_to_lactate` / Θ in the statement | Mechanism≠Parameter and Knowledge≠Evidence |
| Prediction or product language | cure, dose, prescribe, “personalized medicine” as care, clinical CDS, regimen — unless the sentence is an explicit refusal | Parameter≠Prediction and `not_clinical_outcome` |

A candidate is **admitted** only if it fails none of the above. Every admitted hypothesis carries:

- `layer`: `hypothesis` (never `parameter`, never `prediction`);
- `gates_passed`: the full gate list;
- `parameter_status`: the constant `forbidden_to_enter_theta`.

Admission is not identification. An admitted hypothesis is allowed to exist as a testable claim. It is still forbidden to write a number into `CancerODE`.

**Parameter identification, if it ever happens, is a later paper.** Structural and practical identifiability are established methods in systems biology [50,51]. CONFLUENCE has a separate, unfinished identifiability manuscript against Cancer Cell Line Encyclopedia (CCLE) metabolomics [52–54]. That work is not this work. A Disease Profile must not cite an identifiability draft as if `p_lactate` were identified.

### 3.5 Shipped schema contract (`schemas/disease_profile.schema.json`)

Machine-readable contract of schema version `1.0.0` as merged in pull request #11 [43,72]. Normative field names (JSON); `additionalProperties` is forbidden:

```
DiseaseProfile {
  profile_id: string
  disease_id: string
  disease_label: string
  created_at: string          # ISO-8601 UTC
  schema_version: "1.0.0"
  asker_role: "patient_advocate" | "clinician" | "researcher" | "student" | "other"
  answers: {
    role, cancer, setting, stuck: { question, choice_id, choice_label }
  }
  observables: [{ statement, citation_ids[] }]
  candidate_mechanisms: [{ statement, evidence_class, citation_ids[], falsifier }]
  non_parameters: string[]    # minItems 1
  admitted_hypotheses: [{
    statement,
    layer: "hypothesis",
    evidence_class,
    citation_ids[],
    falsifier,
    gates_passed[],
    parameter_status: "forbidden_to_enter_theta"
  }]
  citations: [{ id, text, url?, doi? }]   # doi must match 10.xxxx/… if present
  disclaimer: <fixed research-only string>
}
```

**DOI rule.** A `doi` field is allowed only when the identifier is a registered `10.` prefix string verified against a publisher, PubMed, PMC, or Crossref record. Placeholders (`10.xxxx/pending`, `doi:TBD`) are invalid [55]. Prefer a stable URL (PMC, WHO, IARC, NCI, GitHub, ministry PDF) when no DOI is on hand.

**Attribution rule.** Any sentence that uses OnCo data keeps: *Data from OnCo (onco.cc), CC BY-NC 4.0; commercial use needs a licence.* Adapter and profile code, if and when merged, remain MIT. Cached OnCo payloads remain OnCo data [29,31,55].

**Implementation rule.** Tests in `tests/test_disease_profile.py` and `tests/test_profile_pack.py` validate exports against both the Pydantic model and, when `jsonschema` is present, Draft 2020-12. Thesis #3 cites those tests as existence of gates. It does not treat a green pytest as a biological result.

### 3.6 From profile to HypothesisObject (still not Θ)

There are **two** HypothesisObject types in this repository. They must not be collapsed.

1. **`confluence.onco.schemas.HypothesisObject`** — OnCo idea-shelf companion from P0: optional `onco_idea_id` / `onco_question`, `mechanism`, `state[]`, `prediction`, `experiment`, `required_rung`, `status`, `falsifier` [37].
2. **`confluence.profiles.HypothesisObject`** — profile-layer object from pull request #11: `id`, `disease_profile_ref`, `statement`, `evidence_class`, **required** `named_public_dataset`, `falsifier`, `status` (currently the constant `proposed`), `non_claims` (must refuse CDS / device), `citation_ids`, fixed `disclaimer` [74]. An empty dataset name is invalid.

Thesis #3 uses the **profile-layer** object when it says a profile feeds a HypothesisObject. The OnCo-shelf object remains a pointer type for Ideas. Neither writes Θ.

**Mapping (one-way) to `confluence.profiles.HypothesisObject`.**

| Profile / pack field | HypothesisObject field | Constraint |
|---|---|---|
| `admitted_hypotheses[].statement` | `statement` | Must not contain a Θ assignment |
| `admitted_hypotheses[].falsifier` | `falsifier` | Required |
| `admitted_hypotheses[].evidence_class` | `evidence_class` | Not `knowledge` / `onco_page` / `confidence` |
| case `slug` or `profile_id` | `disease_profile_ref` | Disease-class pack id, not a person |
| chosen public dataset (section 3.7) | `named_public_dataset` | Must name an accession or portal project |
| pack / profile disclaimer | `disclaimer` | Fixed research-only string |

Committed example (not a result): `data/hypotheses/tnbc_lactate_immune_exclusion.yaml` is `H-TNBC-LAC-EXCL-001`, `disease_profile_ref: tnbc_metabolic_immune`, `named_public_dataset` = TCGA-BRCA RNA-seq via GDC, status `proposed` [75]. The contrast is not run in this thesis.

**Forbidden mapping.**

- `observables[].statement` ↛ `ParameterObject.point_value`
- OnCo `confidence.probability` ↛ prior on Θ
- CaseCard `nstg_touchpoints` ↛ `x_cap_scale` or `infection_risk_weight` [38,40]
- Profile `asker_role` ↛ controller weight
- Thinking-lab “success unit” prose ↛ a RECIST-like simulated endpoint treated as a trial
- Pack row keys `p_lactate`, `write_theta`, `force_admit` ↛ a built profile (`RowRefused` in `confluence.profiles.pack`) [73,77]

A HypothesisObject may later grow a `PredictionObject` when, and only when, a frozen model is run under declared parameters with uncertainty. That run is still computational. Gate 8 in the thesis evidence page — translation after biological and clinical validation — remains unclaimed [4,5].

### 3.7 Named public datasets (bindings, not ingested results)

Profiles and HypothesisObjects bind to **named public datasets**. They do not, in this thesis, analyse those datasets. Naming the dataset is part of making the hypothesis falsifiable.

| Disease-class template | Named public resources | What a HypothesisObject may ask | What it may not do |
|---|---|---|---|
| TNBC metabolic–immune | TCGA-BRCA / GDC [56,57]; CCLE / DepMap metabolomics [52–54]; OnCo TNBC/LDHA records as Knowledge only [27] | Is a lactate- or exclusion-associated *signature* associated with an immune-spatial or metabolomic observable in a named cohort? | Write LDHA expression into `p_lactate` |
| GBM niche | TCGA-GBM / GDC [58,59]; Ivy Glioblastoma Atlas Project (Ivy GAP) anatomic transcription [60] | Do hypoxic / invasive anatomic labels correspond to the hypothesized niche split? | Treat fly-connectome stubs as a brain-tumour controller [42] |
| PDAC stroma | TCGA-PAAD / GDC [61]; cBioPortal `paad_tcga_pan_can_atlas_2018` [62]; GEO GSE71729, GSE62452, GSE28735 [45,63–65]; DepMap; NCI PDMR [45] | Do stroma / shield axes separate activated stroma, primary, and metastatic samples better than KRAS status alone — as a *stated* first claim, not a result of this paper [45,46]? | Treat synthetic `results/pdac_rogue_closure/` outputs as biological validation [46] |
| Dormancy / occult | Published natural-history and DTC/MRD series cited as literature [23–26]; CONFLUENCE dormancy gate is a research knob, not a fitted waiting time [3,30] | Does a named residual-disease assay show cycling versus G0-like residual cells? | Declare a person disease-free because a simulator `awake` state is low |

**Licence and repository rule.** Do not vendor OnCo `all.json`. Do not commit large raw omics tables to GitHub. Commit fetch scripts, manifests, checksums, and licence-compatible derived tables [31,45]. CCLE and TCGA reuse follows those projects’ terms [52,56].

### 3.8 Relationship to CONFLUENCE adapter P0

P0 is the *plumbing* that makes OnCo readable without making OnCo a parameter source [32,36,37].

| P0 artefact | What Thesis #3 uses | What Thesis #3 does not claim |
|---|---|---|
| `OncoRef`, `OncoEvidenceCandidate` | Knowledge citations inside a profile | That an OnCo tldr is true in a patient |
| `refuse_knowledge_as_parameter` | The Mechanism≠Parameter gate | That every smuggle in the wild is caught |
| `HypothesisObject`, `MechanismObject`, `ParameterObject`, `PredictionObject` | Downstream types for admitted hypotheses | That any object is identified |
| `seed_bindings.yaml` | Annotation targets (LDHA annotates `p_lactate`; it does not set it) | That bindings are evidence |
| Nine adapter tests | Existence of gates in code [32] | Therapeutic efficacy |
| Findings chapter [33] | Architectural findings | A clinical chapter |

P0’s success criterion, restated: keep the layers apart. Thesis #3’s success criterion: give the thinking laboratory a versioned object that *remembers* the refusal.

### 3.9 Relationship to Complexity Science CaseCards

A CaseCard is a YAML research template in the companion Complexity Science repository [38–40]. The four seed cards match the four templates in section 4. Official documents to cite — and not to over-read — are `pathology_cases/SCHEMA.md`, the CaseCard JSON Schema, the seed YAML files, `docs/NSTG_PROVENANCE.md`, and the repository README on the CaseCard branch [38–40].

| | Disease Profile (this thesis) | CaseCard (Complexity Science) |
|---|---|---|
| Home | `confluence/profiles/` + `data/profiles/cases/` | `pathology_cases/cases/*.yaml` |
| Who fills it | Anyone completing the four questions | A contributor adding one YAML |
| Native questions | Who / which disease / regime / stuck | Disease, systemic axes, observables, mechanisms, falsifiers, NSTG touchpoints |
| Constraint layer | Non-parameter list + five-layer gates | NSTG themes as qualitative constraints; numeric leaves refused |
| Output | Versioned profile → HypothesisObject | Gated `PathwaySketch` (ranked bibliographic hypotheses + required evidence + refused non-parameters) [39] |
| ODE | Must not edit | Must not edit |
| Patient data | None | None |
| Product language | Fixed disclaimer | Hard non-claims table [39] |

They are complementary, not duplicates. A CaseCard can be *cited* from a Disease Profile as a knowledge/template source. A Disease Profile can be *cited* from a CaseCard as a CONFLUENCE export. Neither object becomes Θ. If a line in either repository conflicts with official NSTG 2022, NSTG wins as a knowledge constraint and still does not become a coefficient [40,41].

This thesis does not report PathwaySketch rankings, does not rerun the Complexity Science explorer, and does not claim that the seed cards are validated models. They are starting hypotheses with falsifiers [39].

### 3.10 Worked-example method (qualitative boards)

Section 4 constructs one board per seed disease. Each board is filled from the **shipped case-pack row** in `data/profiles/cases/cases.yaml` and the gated JSON beside it [73]. Those four answers are restated so the board can be replayed. Sources are:

- `data/profiles/cases/{tnbc_metabolic_immune,gbm_invasive_niche,pdac_stromal_barrier,dormant_occult}.json` and `SUMMARY.md` [73];
- the shipped thinking-lab copy and JSON download [42];
- the matching Complexity Science CaseCard framing and citations [38];
- CONFLUENCE disease-specific notes where they exist (PDAC rogue-closure *plan*, not result) [45,46];
- the public-dataset names in section 3.7.

`SUMMARY.md` records, as a pack census and **not** as a biological finding: each of the four files has one admitted hypothesis and two candidates (the second is the LDHA/OnCo audit trap); the `poison_ldha_as_theta` table row is refused for the forbidden key `p_lactate` [73].

**Explicitly not done:** no ODE integration, no virtual cohort, no log-rank on simulated burden, no claim of patient benefit, no identification of Θ. If a sentence in section 4 can be misread as a trial result, it has failed the method and should be read as a hypothesis statement instead.

### 3.11 Scholar packaging (this thesis)

The manuscript is filed at `docs/manuscript/thesis_03_disease_profile_method.md`. A PDF of the same text is committed at `docs/manuscript/thesis_03_disease_profile_method.pdf` and mirrored under `evidence/papers/` so the evidence-site root can serve Highwire `citation_*` metadata. The landing page is `evidence/thesis-03.html`. A sitemap fragment `evidence/sitemap-thesis03.xml` records the HTML and PDF locations because no general `evidence/sitemap.xml` existed on `main` at the time of writing. The checklist is `docs/SCHOLAR_THESIS_03.md`. Citation hygiene for pack JSON follows `docs/CITATION_POLICY.md` [76].

---

## 4. Worked examples — qualitative boards

Each subsection is a research template. Headings keep the four questions visible so the board can be replayed.

### 4.1 TNBC — metabolic–immune exclusion

**Pack object.** `data/profiles/cases/tnbc_metabolic_immune.json` (`profile_id: dp-tnbc-metabolic-immune`) [73].

**Four answers (from the pack).**  
Q1 Researcher.  
Q2 Triple-negative breast cancer.  
Q3 Metastatic / relapsed (control, adaptation, residual clones — not a staging claim).  
Q4 Adaptation / persisters, with an unnamed or unseparated metabolic–immune loop.

**Success unit (language, not a score).** For localised disease, multimodality with curative *intent* is a clinical-language fact about how teams talk. For metastatic disease, the honest unit is control and adaptation [12,42]. This profile does not convert either sentence into a CONFLUENCE endpoint.

**Coupled system, not a page.** Clones sit in a metabolic and immune microenvironment. Lactate, transforming growth factor-β (TGF-β), exclusion, and persister states are *candidate* coupled loops [8–11,42]. An OnCo LDHA record is Knowledge, CC BY-NC 4.0, not identified `p_lactate` [27,31,32].

**Observables (evidence pointers).**

1. NCI descriptor: TNBC is about 15% of breast cancers in that educational summary and is described as generally faster-growing and more recurrent than some other invasive subtypes [12]. Descriptor, not a parameter.
2. Tumour-infiltrating lymphocyte (TIL) spatial pattern — margin versus nest — as a research histopathology observable [10,38].
3. Local metabolite sketch (lactate, glucose, adenosine class) as research metabolomics or imaging mass spectrometry [9,38]. A measured profile would be Evidence. It is not an ODE coefficient.
4. OnCo LDHA / TNBC pages as dated Knowledge [27].

**Candidate mechanisms (not admitted until gates).**

| Candidate | evidence_class | Falsifier | Gate note |
|---|---|---|---|
| Metabolic over-consumption excludes or exhausts effectors independently of a single checkpoint occupancy [9,38] | review_level | Effectors remain competent and spatially inside nests despite high local lactate or low glucose in the same region [38] | May admit as hypothesis; must not set `p_lactate` |
| Stroma enforces immune privilege by excluding T cells from nests [10,38] | review_level | Cytotoxic T cells abundantly contact tumour cells inside nests while stroma is intact [38] | May admit as hypothesis |
| Coinhibitory pathways are a studied *class* in TNBC [8,11] | mixed (trial + review) | Exclusion persists when the relevant receptors/ligands are absent in that niche [38] | Knowledge of a class ≠ a U(t) programme |
| **Poison candidate:** OnCo LDHA knowledge and `confidence.probability` can be entered as `p_lactate` or legacy `pyruvate_to_lactate` | knowledge | Rejected a priori: Knowledge is not a parameter [31,32] | **Refuse** |

**Admitted (illustrative).** The first two candidates, stated as hypotheses with falsifiers and review-level citations, pass the gates as *hypotheses* with `parameter_status = forbidden_to_enter_theta`. The poison candidate is retained in `candidate_mechanisms` so the refusal is visible. Visibility of killed or refused claims is part of the method [31,33].

**HypothesisObject against a named dataset (not a result).**

- Shipped: `H-TNBC-LAC-EXCL-001` in `data/hypotheses/tnbc_lactate_immune_exclusion.yaml` [75]. `disease_profile_ref: tnbc_metabolic_immune`. `named_public_dataset`: TCGA-BRCA RNA-seq via GDC [56,57]. Status `proposed`. Falsifier: a pre-registered TCGA-BRCA contrast that fails to support a lactate–exclusion association retires the hypothesis; OnCo `confidence.probability` still cannot become Θ. This thesis does **not** run that contrast.
- Optional later object (not committed): CCLE metabolomics [53,54] as a second named dataset, still `forbidden_to_enter_theta`.
- OnCo idea ids, if used, remain `onco_idea_id` shelf pointers on the *OnCo* HypothesisObject [28,37].

**Trap this disease invites.** LDHA on OnCo is not `p_lactate`. The legacy map is assumed, not identified [31–33]. Atezolizumab plus nab-paclitaxel in advanced TNBC is Evidence of a trial, not a CONFLUENCE regimen and not a coefficient [11].

**What this board refuses.** No dose, no claim that metabolic–immune coupling has been proven in CONFLUENCE, no claim of patient benefit.

### 4.2 GBM — invasive niche and hypoxia

**Pack object.** `data/profiles/cases/gbm_invasive_niche.json` (`profile_id: dp-gbm-invasive-niche`) [73]. Extra non-parameters in the pack include fly-connectome stubs and hypoxia/invasion labels as ODE Θ.

**Four answers (from the pack).**  
Q1 Researcher.  
Q2 Glioblastoma.  
Q3 Localised / early in the thinking-lab sense — already the wrong comfort: GBM is infiltrative at presentation [42]. The pack uses this regime as *asker context*, not as a claim that a mass is confined. The honest score remains **control and time**, not clearance.  
Q4 Target known, not reachable (delivery / organ constraint), with the niche mechanism still unnamed as a separated loop.

**Success unit.** Control and time. One clearance is the wrong score [13,42].

**Coupled system.** Spatial evolution inside an organ that cannot be widely resected. Hypoxic pseudopalisades, microvascular proliferation, and an infiltrative front are hypothesized niches, not lumped as one well-mixed tank [13–16,38]. Vessel quantity is not delivery [13,38].

**Observables.**

1. Pseudopalisading necrosis / hypoxia marks — histopathology and hypoxia immunohistochemistry as research observables [14,38].
2. Infiltrative margin beyond a contrast-enhancing mass — imaging plus histology correlation [15,38]. Extent is an observable, not a motility coefficient.
3. TCGA-GBM molecular class and Ivy GAP anatomic transcription as public, named resources [58–60].
4. A phase 3 failure, when cited, is Evidence of a killed idea and should stay visible [42].

**Candidate mechanisms.**

| Candidate | evidence_class | Falsifier | Gate note |
|---|---|---|---|
| Hypoxic niches co-regulate adaptive programmes and local invasion via HIF-class biology [13,14,16] | mixed (review + primary histology) | Hypoxia marks and invasive front anti-correlate in a named anatomic dataset, or invasion proceeds without the hypoxic programme [38] | Hypothesis only |
| Dispersive cells trade bulk proliferation for migration (“cost of migration”) [15] | review_level | Reducing the enhancing mass without a change in the infiltrative compartment still predicts control — a claim to be tested, not assumed [15,38] | Do not invent a U slot from an OnCo page |
| Angiogenic histology is decoupled from effective perfusion [13,38] | review_level | Vessel-rich regions show effective delivery and no hypoxic programme in the same locale | Vessel count ≠ parameter |
| **Poison:** fly-connectome stubs imply a brain-tumour controller; or unaudited `tnbc_mod_3s` ROS terms may be imported | knowledge | Rejected: category error and frozen-lineage rule [31,42] | **Refuse** |

**HypothesisObject against a named dataset.**

- `H-GBM-01`: In Ivy GAP [60] and TCGA-GBM [58,59], pre-registered anatomic labels (leading edge, infiltrating tumour, cellular tumour, necrotic / perinecrotic zones) are used as *observables* to test whether a hypoxia-associated expression programme and an invasion-associated programme occupy different anatomic bins. Required rung: `in_silico`. Falsifier: the programmes are uniformly mixed across anatomic bins after batch correction. Θ is not written. Flybody visualization in CONFLUENCE remains a research embodiment, not a GBM controller [3,30].

**Trap.** Fly connectome stubs do not imply a brain-tumour controller. Do not import ROS from `tnbc_mod_3s` without audit [31,42].

**What this board refuses.** No neurosurgical plan, no radiotherapy choice, no pain-score mapping from a future burden state [38,41].

### 4.3 PDAC — stromal barrier

**Pack object.** `data/profiles/cases/pdac_stromal_barrier.json` (`profile_id: dp-pdac-stromal-barrier`) [73]. Extra non-parameters: stage-at-diagnosis OnCo ideas; adding OnCo targets to the CancerODE RHS.

**Four answers (from the pack).**  
Q1 Researcher.  
Q2 Pancreatic ductal adenocarcinoma.  
Q3 Metastatic / relapsed — usually control and interception [21,42].  
Q4 Found too late, *and* a stromal / delivery barrier once found [17–20,45].

**Success unit.** Usually control and interception. Late diagnosis is the dominant systems failure [21,22,42].

**Coupled system.** Dense stroma, few early signals, cachexia, trial access, and a KRAS-dominant driver landscape [17–20,45]. CONFLUENCE v2 may *annotate* `S_fib` and `C_tgfb` as lumped handles. Annotation is not identification [31,36,42].

**Observables.**

1. Activated versus normal stroma, primary versus metastatic labels in GSE71729 [63].
2. Paired tumour / adjacent expression in GSE62452 and GSE28735 [64,65].
3. TCGA-PAAD mutation and expression tables [61,62].
4. Synthetic PDAC rogue-closure time series in this repository — reproducibility of the *generator*, not biological validation [46].

**Candidate mechanisms.**

| Candidate | evidence_class | Falsifier | Gate note |
|---|---|---|---|
| Desmoplastic stroma is a physical delivery barrier [18–20] | mixed (review + enzymatic-stroma primary) | Delivery-equivalent exposure is achieved without stromal disruption and the barrier claim fails in that system [18,38] | Hypothesis; not a dose of hyaluronidase |
| Stroma / chemokine cues exclude T cells [10,17,38] | review_level | TILs contact tumour cells inside nests despite intact desmoplasia | Same exclusion logic as TNBC; not transferable as a parameter |
| Driver + bypass + shield + exclusion + selected resistance as a *conjunction* worth testing against KRAS-alone [45,46] | hypothesis / protocol | Rogue-closure features do *not* outperform KRAS status on a pre-registered survival or aggressiveness endpoint in TCGA-PAAD [45] | Stated first claim of the PDAC note; **not a result of this thesis** |
| **Poison:** more OnCo targets on the RHS will move median diagnosis earlier | knowledge | Rejected: detection is not an RHS term [42] | **Refuse** |

**HypothesisObject against a named dataset.**

- `H-PDAC-01`: On TCGA-PAAD [61,62], a pre-registered feature table (`driver_score`, `immune_score`, `stroma_score`, optional glyco-panel from GlyGen/GlyConnect annotation [45]) is tested against KRAS-alone for association with survival or stage. Required rung: `in_silico`. This thesis does **not** run that test. The CONFLUENCE PDAC note already wrote the claim as a validation *plan* [45]. Synthetic `results/` files remain synthetic [46].
- `H-PDAC-02`: On GSE71729 [63], the stroma/shield axis should separate activated stroma, normal stroma, primary tumour, and metastatic disease if the barrier hypothesis has transcriptomic support. Falsifier: no separation after the stated normalization.

**Trap.** Expanding X because OnCo lists more cell types does not fix late diagnosis [33,42]. `S_fib` is a lumped handle, not a collagen fraction.

**What this board refuses.** No claim that rogue-closure features already predict survival. No cachexia coefficient from a ministry anaemia theme [38,49].

### 4.4 Dormancy / occult residual disease

**Pack object.** `data/profiles/cases/dormant_occult.json` (`profile_id: dp-dormant-occult`, `disease_id: occult`) [73]. Extra non-parameters: MRD-style language as a clinical assay claim; dormancy as a CONFLUENCE disease-class diagnosis.

**Four answers (from the pack).**  
Q1 Researcher.  
Q2 Dormant / occult residual disease (research analogy — not a person-level MRD call).  
Q3 Localised / early as *asker regime*, not as a declaration that a person is disease-free.  
Q4 Biology still unnamed: which pause (cellular G0-like, angiogenic population, immune equilibrium) is load-bearing [23–26,38]?

**Success unit.** This is a hypothesis space about latency. It is not a licence to treat, image, or declare anyone disease-free [38,42]. CONFLUENCE `occult` / `dormant` labels are state signatures; `awake` is a dormancy gate; the dormancy-exit hazard is a research knob, not a fitted patient waiting time [3,30].

**Coupled system.** Quiescence, angiogenic pause, and immunosurveillance are *alternative or combined* candidate pauses [23–26]. A negative assay is a detection limit, not proof of clearance [38].

**Observables.**

1. DTC / residual-disease assay as research cytology or molecular MRD [23,38].
2. Documented latency before overt relapse in published series — epidemiologic observable, not a pharmacokinetic time constant [23,25,38].
3. CONFLUENCE observation-layer `dormancy_exit` only as a named simulator symbol, never as a waiting-time estimate [3].

**Candidate mechanisms.**

| Candidate | evidence_class | Falsifier | Gate note |
|---|---|---|---|
| Cellular dormancy via a G0/G1-like pause [23,24] | review_level | Assay-detected residual cells are uniformly cycling and die when proliferation-targeted pressure is applied in an orthogonal model [38] | Hypothesis |
| Angiogenic dormancy — proliferation balanced by death without vascular recruitment [23,25] | review_level | Occult lesions expand without angiogenesis or a change in proliferation–death balance [38] | Not a haemoglobin-linked oxygen parameter [38,49] |
| Immunosurveillance holds residual cells in equilibrium without clearing them [23,26] | review_level | Immune depletion does not increase outgrowth where cells remain detectable [38] | Host HIV/infection themes forbid treating equilibrium as a stimulation schedule [38,41] |
| Niche / fibrotic / inflammatory cues as awakening mechanisms [24,25] | review_level | The cue is present and residual cells remain occult across an adequate window [38] | **Not a reason to provoke residual disease in a person** |
| **Poison:** low simulated `awake` ⇒ the person is clear; or OnCo knowledge sets a dormancy half-life | knowledge | Rejected: simulator state ≠ assay; knowledge ≠ parameter [3,31] | **Refuse** |

**HypothesisObject against a named dataset.**

- `H-DORM-01`: A literature-only HypothesisObject that points at published DTC/MRD series [23–26] and states the G0-versus-cycling falsifier. Required rung: `clinical` *literature synthesis* or a future partner assay — not a CONFLUENCE clinical prediction.
- `H-DORM-02`: If a public residual-disease transcriptomic series is nominated later (GEO accession to be written into a new profile version, not invented here), the profile is re-exported with that accession. This thesis does not invent a GEO identifier.

**Trap.** Occult means below threshold. It does not mean absent. Awakening cues are research objects. They are not a protocol to provoke disease [38].

**What this board refuses.** No disease-free declaration, no IRIS prediction from a future immune state, no mapping of anaemia policy to an oxygen coefficient [38,41,49].

### 4.5 Cross-board constants

Every board above shares the same refused sentence, which is also the export disclaimer:

> Project Confluence disease profiles are computational research artefacts. They are not a medical device, not clinical decision support, not personalized medicine as clinical CDS, and not a claim of cure, diagnosis, or dosing. In-silico / research only. See DISCLAIMER.md.

Every board shares the same five-layer sentence. Every admitted hypothesis shares `parameter_status = forbidden_to_enter_theta`. That repetition is the method, not an editorial failure.

---

## 5. Discussion — efficiency of adding a new complex disease

### 5.1 What “efficiency” means here

Efficiency is not a claim that the laboratory found a drug faster. It is a claim about *research operations*: the marginal work to add a new complex disease-class without (i) opening `CancerODE`, (ii) smuggling Knowledge into Θ, (iii) inventing a DOI, or (iv) converting a chart into CDS.

Complexity Science already published the operational table this discussion extends [39]:

| Task | What you edit | What you do not edit | Typical local cost |
|---|---|---|---|
| Add a CaseCard | one YAML | pipeline code, ODE | under 15 minutes if the citations are already in hand [39] |
| Add a CONFLUENCE Disease Profile | one row in `data/profiles/cases/cases.yaml`, then `python3 scripts/build_disease_profile_pack.py` — or complete the four thinking-lab questions and download JSON [42,73,77] | `CancerODE`, controller priors, OnCo corpus | one thinking-lab session + bibliography check |
| Validate gates | nothing (`pytest tests/test_disease_profile.py tests/test_profile_pack.py` plus `--dry-run`) | nothing | seconds |
| Turn OnCo or NSTG into a coefficient | refused (`RowRefused` / `refuse_ldha_onco_as_parameter`) | n/a | n/a |
| Fit an ODE from the profile | refused | n/a | n/a |
| Claim patient benefit from the board | refused | n/a | n/a |

The expensive step is not JSON. The expensive step is an honest falsifier and a real citation. That is the correct place for the cost to sit.

### 5.2 Why a new disease should not cost an RHS edit

Skip-level promotion looks cheap: a new OnCo page appears, a modeller adds a term, the plot moves. P0 exists because that cheapness is a scientific debt [31–33]. Identifiability theory says the plot moving is not the same as the term being recoverable [50,51]. Alias collisions (legacy `pyruvate_to_lactate` versus v2 `p_lactate`) show the debt is already present in this repository [31].

A Disease Profile makes the cheap path *illegal in the object*. A new disease is a new `disease_id`, a new board, a new profile version, and new HypothesisObjects aimed at named datasets. The frozen lineages stay frozen. If a later, independent study identifies a symbol, that study writes a `ParameterObject` with `identifiability ≠ unidentified` and a dataset id. It does not quietly edit the profile’s admitted hypothesis into a number.

### 5.3 Personalized-medicine research without a clinical product

Hamburg and Collins described personalized medicine as a path, not a completed clinic [1]. Jameson and Longo already warned that “precision” can be problematic when the word outruns the evidence [2]. Thesis #3 takes that warning as a design constraint.

**Systemic** in the title means host–tumour couplings (metabolism, immunity, stroma, hypoxia, latency, access) are first-class *axes on the board*. It does not mean a whole-body digital twin of a named Nigerian or any other patient. NSTG and the National Cancer Control Plan appear as knowledge constraints about infection, referral, supportive care, and chemotherapy safety [41,48,49]. They are not a national CDS engine, and this thesis does not execute them.

**Personalized** in the title means the *research question* is allowed to be disease-specific and regime-specific (the four answers). It does not mean a unique regimen for a unique person generated from this repository.

If a reader needs a single sentence: **this is a method for building reusable research objects about complex diseases; it is not personalized medicine as a clinical product.**

### 5.4 Object graph (no fabricated arrows)

```
OnCo page / Idea          →  OncoRef / OncoEvidenceCandidate     (P0, read-only)
Thinking-lab 4Q           →  Disease Profile v1.0.0              (confluence/profiles/)
cases.yaml row            →  data/profiles/cases/*.json          (pack builder)
CaseCard YAML             →  PathwaySketch                        (Complexity Science; not rerun here)
Admitted hypothesis       →  confluence.profiles.HypothesisObject + named dataset
Identified symbol (later) →  ParameterObject                      (not done here)
Frozen model run (later)  →  PredictionObject                     (not done here)
External experiment       →  new Evidence                         (not done here)
Clinic                    →  not this repository                  [3–5]
```

The only arrows this thesis claims to have walked are the CONFLUENCE-side objects that now exist on `main` (OnCo read-only, four questions, profile JSON, one committed HypothesisObject file) and the bibliographic alignment with CaseCards. PathwaySketch rankings, CCLE identifiability counts, and PDAC survival tests are *other* documents’ jobs [39,45,54].

### 5.5 What would count as a method success next

A method success, not a clinical success:

1. A third party can replay a board from the four answers and obtain the same refused poison candidates.
2. A new disease (for example, MSI-high versus MSS colorectal cancer, already split in the thinking lab [42]) can be added without a `CancerODE` diff.
3. A HypothesisObject file names TCGA-BRCA or GSE71729 and a falsifier, and a reviewer can see that Θ was not touched.
4. OnCo attribution appears on every OnCo-derived sentence [29,55].
5. No invented DOI survives `docs/SCHOLAR_THESIS_03.md` checks.

---

## 6. Limitations

1. **No new measurement.** Boards reuse published descriptors, reviews, a small number of primary papers, and repository documents. They do not generate Evidence.
2. **Exporter is present; this paper still does not analyse public matrices.** Pull request #11 merged the schema, pack, and tests [43,71–77]. Green tests show refusal of Θ, not a TCGA result.
3. **CaseCards are draft companion artefacts.** Complexity Science `main` at the time of writing was a seed README; the CaseCard pack lives on a draft pull request [38,39]. Citations to that pack are citations to a public draft, not to a completed validation.
4. **P0 does not identify parameters.** Nine adapter tests show refusal, not recovery of Θ [32].
5. **Public datasets are named, not analysed.** TCGA, CCLE, GEO, Ivy GAP, and DepMap bindings are experimental *design* clauses. Absence of a p-value here is intentional.
6. **Qualitative “coupling” is not a measured coupling in `confluence_v2_15d`.** Wired claims in `seed_bindings.yaml` remain `assumed` / concept-level unless a later paper says otherwise [36].
7. **Thinking-lab roles include “family” and “clinician” as asker types.** Those roles change the *next question*. They do not create a care pathway [42].
8. **NSTG is not redistributed.** Theme-level constraints may under-specify what a Nigerian specialist would actually do. If conflict exists, official NSTG wins as knowledge and still does not become Θ [40,41].
9. **Burden statistics are modelled estimates.** GLOBOCAN 2024 figures are not a census [21,22]. Nigeria GLOBOCAN 2022 population figures, if cited elsewhere on the evidence site, are 2022 estimates, not 2026 incidence [33].
10. **Identifiability manuscript is unfinished.** CCLE 7/17 → 15/17 is a draft claim in another file and is not imported as a finding here [54].
11. **English-language, oncology-skewed corpus.** The method is not shown for non-malignant systemic disease in this thesis, even though CONFLUENCE’s call for data mentions metabolic comorbidity as a future arm [66].
12. **No clinical validation.** The awaiting-validation gate remains closed [4,5].

---

## 7. Future work

1. **Keep schema 1.0.0 honest.** The contract already lives at `schemas/disease_profile.schema.json` and `confluence/profiles/disease_profile.py` [71,72]. Version-bump on required-field changes; do not silently mutate committed JSON.
2. **Grow HypothesisObjects beside the four pack files** the way `H-TNBC-LAC-EXCL-001` sits beside `tnbc_metabolic_immune.json` — still `proposed`, still `forbidden_to_enter_theta`, still named-dataset-only [75].
3. **Keep P0 read-only.** Iterate the adapter, cache TTL, and ROS audit of `tnbc_mod_3s` without RHS edits [31,33].
4. **Bind, then analyse, named datasets** under pre-registered HypothesisObjects (TCGA-BRCA, TCGA-GBM, TCGA-PAAD, GSE71729, Ivy GAP, CCLE metabolomics). Publish negative results. Do not back-write Θ from a p-value.
5. **Align Profile ↔ CaseCard** with an explicit cross-walk file (profile_id ↔ case id) once both contracts are stable. Do not merge the repositories’ ODE policies; both already say “do not edit the ODE” [31,39].
6. **Admit-list hypothesis library** separate from the OnCo Ideas shelf [28,33,37].
7. **Translation** remains Gates 6–8: held-out validation, a wet experiment that could kill the prediction, then clinical validation this repository does not claim [4,5,33].
8. **Scholar surfaces.** Keep Highwire metadata, sitemap entries, and the Vancouver-plus-ledger citation policy honest as pages accumulate [55].

None of the above is a promise of a product.

---

## 8. References

Journal articles use ICMJE/Vancouver form (first six authors, then et al. if more; year;volume(issue):pages; Crossref-verified DOI; PMID when PubMed indexes the work). Internet and repository items use the Vancouver electronic format with a cited date. No DOI is invented. Bechhofer et al. [35] and Bellman and Åström [50] have verified DOIs but no PMID (not PubMed-indexed).

1. Hamburg MA, Collins FS. The path to personalized medicine. N Engl J Med. 2010;363(4):301-304. doi:10.1056/NEJMp1006304. PMID: 20551152.
2. Jameson JL, Longo DL. Precision medicine — personalized, problematic, and promising. N Engl J Med. 2015;372(23):2229-2234. doi:10.1056/NEJMsb1503104. PMID: 26014593.
3. Ogbonna KE. DISCLAIMER.md [Internet]. Project Confluence; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/DISCLAIMER.md
4. Ogbonna KE. Awaiting external clinical validation [Internet]. Project Confluence; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/docs/AWAITING_CLINICAL_VALIDATION.md
5. Ogbonna KE. Merge readiness [Internet]. Project Confluence; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/docs/MERGE_READINESS.md
6. Altrock PM, Liu LL, Michor F. The mathematics of cancer: integrating quantitative models. Nat Rev Cancer. 2015;15(12):730-745. doi:10.1038/nrc4029. PMID: 26597528. PMCID: PMC5663316.
7. Gatenby RA, Silva AS, Gillies RJ, Frieden BR. Adaptive therapy. Cancer Res. 2009;69(11):4894-4903. doi:10.1158/0008-5472.CAN-08-3658. PMID: 19487300. PMCID: PMC2676449.
8. Bianchini G, Balko JM, Mayer IA, Sanders ME, Gianni L. Triple-negative breast cancer: challenges and opportunities of a heterogeneous disease. Nat Rev Clin Oncol. 2016;13(11):674-690. doi:10.1038/nrclinonc.2016.66. PMID: 27184417.
9. Li X, Wenes M, Romero P, Huang SC, Fendt SM, Ho PC. Navigating metabolic pathways to enhance antitumour immunity and immunotherapy. Nat Rev Clin Oncol. 2019;16(7):425-441. doi:10.1038/s41571-019-0203-7. PMID: 30914826.
10. Joyce JA, Fearon DT. T cell exclusion, immune privilege, and the tumor microenvironment. Science. 2015;348(6230):74-80. doi:10.1126/science.aaa6204. PMID: 25838376.
11. Schmid P, Adams S, Rugo HS, Schneeweiss A, Barrios CH, Iwata H, et al. Atezolizumab and nab-paclitaxel in advanced triple-negative breast cancer. N Engl J Med. 2018;379(22):2108-2121. doi:10.1056/NEJMoa1809615. PMID: 30345906.
12. National Cancer Institute. Triple-negative breast cancer [Internet]. Bethesda (MD): NCI; [cited 2026 Sep 20]. Available from: https://www.cancer.gov/types/breast/patient/triple-negative-brochure
13. Hambardzumyan D, Bergers G. Glioblastoma: defining tumor niches. Trends Cancer. 2015;1(4):252-265. doi:10.1016/j.trecan.2015.10.009. PMID: 27088132.
14. Brat DJ, Castellano-Sanchez AA, Hunter SB, Pecot M, Cohen C, Hammond EH, et al. Pseudopalisades in glioblastoma are hypoxic, express extracellular matrix proteases, and are formed by an actively migrating cell population. Cancer Res. 2004;64(3):920-927. doi:10.1158/0008-5472.CAN-03-2073. PMID: 14871821.
15. Giese A, Bjerkvig R, Berens ME, Westphal M. Cost of migration: invasion of malignant gliomas and implications for treatment. J Clin Oncol. 2003;21(8):1624-1636. doi:10.1200/JCO.2003.05.063. PMID: 12697889.
16. Semenza GL. Hypoxia-inducible factors in physiology and medicine. Cell. 2012;148(3):399-408. doi:10.1016/j.cell.2012.01.021. PMID: 22304911.
17. Feig C, Gopinathan A, Neesse A, Chan DS, Cook N, Tuveson DA. The pancreas cancer microenvironment. Clin Cancer Res. 2012;18(16):4266-4276. doi:10.1158/1078-0432.CCR-11-3114. PMID: 22896693.
18. Provenzano PP, Cuevas C, Chang AE, Goel VK, Von Hoff DD, Hingorani SR. Enzymatic targeting of the stroma ablates physical barriers to treatment of pancreatic ductal adenocarcinoma. Cancer Cell. 2012;21(3):418-429. doi:10.1016/j.ccr.2012.01.007. PMID: 22439937.
19. Neesse A, Michl P, Frese KK, Feig C, Cook N, Jacobetz MA, et al. Stromal biology and therapy in pancreatic cancer. Gut. 2011;60(6):861-868. doi:10.1136/gut.2010.226092. PMID: 20966025.
20. Olive KP, Jacobetz MA, Davidson CJ, Gopinathan A, McIntyre D, Honess D, et al. Inhibition of Hedgehog signaling enhances delivery of chemotherapy in a mouse model of pancreatic cancer. Science. 2009;324(5933):1457-1461. doi:10.1126/science.1171362. PMID: 19460966.
21. Sung H, Ferlay J, Siegel RL, Laversanne M, Soerjomataram I, Jemal A, et al. Global cancer statistics 2020: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA Cancer J Clin. 2021;71(3):209-249. doi:10.3322/caac.21660. PMID: 33538338.
22. Sung H, Filho AM, Laversanne M, Ferlay J, Siegel RL, Soerjomataram I, et al. Global cancer statistics 2024: GLOBOCAN estimates of incidence and mortality worldwide for 34 cancers in 186 countries. CA Cancer J Clin. 2026;76(4):e70090. doi:10.3322/caac.70090. PMID: 42417444.
23. Aguirre-Ghiso JA. Models, mechanisms and clinical evidence for cancer dormancy. Nat Rev Cancer. 2007;7(11):834-846. doi:10.1038/nrc2256. PMID: 17957189.
24. Sosa MS, Bragado P, Aguirre-Ghiso JA. Mechanisms of disseminated cancer cell dormancy: an awakening field. Nat Rev Cancer. 2014;14(9):611-622. doi:10.1038/nrc3793. PMID: 25118602.
25. Massagué J, Obenauf AC. Metastatic colonization by circulating tumour cells. Nature. 2016;529(7586):298-306. doi:10.1038/nature17038. PMID: 26791720.
26. Giancotti FG. Mechanisms governing metastatic dormancy and reactivation. Cell. 2013;155(4):750-764. doi:10.1016/j.cell.2013.10.029. PMID: 24209616.
27. Gomila J, OnCo contributors. OnCo: a public, cited knowledge graph of oncology [Internet]. 2026 [cited 2026 Sep 20]. Available from: https://onco.cc
28. Gomila J, OnCo contributors. OnCo Ideas [Internet]. 2026 [cited 2026 Sep 20]. Available from: https://onco.cc/ideas/
29. Gomila J, OnCo contributors. OnCo source repository [Internet]. GitHub; 2026 [cited 2026 Sep 20]. Code MIT; data CC BY-NC 4.0. Available from: https://github.com/judegomila/OnCo
30. Ogbonna KE. Project Confluence [Internet]. GitHub; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence
31. Ogbonna KE. OnCo → CONFLUENCE ontology and evidence-ingestion spec (v0.3) [Internet]. Project Confluence; 2026 Sep 18 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/docs/ONCO_CONFLUENCE_ONTOLOGY_SPEC.md
32. Ogbonna KE. OnCo adapter (P0) [Internet]. Project Confluence; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/docs/ONCO_ADAPTER.md
33. Ogbonna KE. Knowledge gates for dynamical oncology models: findings from an OnCo × CONFLUENCE integration [Internet]. Project Confluence; 2026 Sep 18 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/docs/manuscript/ONCO_CONFLUENCE_THESIS_FINDINGS.md
34. Wilkinson MD, Dumontier M, Aalbersberg IJ, Appleton G, Axton M, Baak A, et al. The FAIR Guiding Principles for scientific data management and stewardship. Sci Data. 2016;3(1):160018. doi:10.1038/sdata.2016.18. PMID: 26978244.
35. Bechhofer S, Buchan I, De Roure D, Missier P, Ainsworth J, Bhagat J, et al. Why linked data is not enough for scientists. Future Gener Comput Syst. 2013;29(2):599-611. doi:10.1016/j.future.2011.08.004.
36. Ogbonna KE. feat/onco-adapter-p0 (pull request #9) [Internet]. Project Confluence; 2026 Sep 18 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/pull/9
37. Ogbonna KE. confluence/onco/schemas.py [Internet]. Project Confluence; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/confluence/onco/schemas.py
38. Ogbonna KE. CaseCard schema and NSTG-gated in-silico pathway explorer (pull request #2) [Internet]. Complexity Science; 2026 Sep 20 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/complexity-science/pull/2
39. Ogbonna KE. Complexity Science README [Internet]. GitHub; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/complexity-science
40. Ogbonna KE. pathology_cases/SCHEMA.md [Internet]. Complexity Science (draft pull request #2, commit d2e881b); 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/complexity-science/blob/d2e881bd13c2fc067b1b33e2498f9eab09b79a3b/pathology_cases/SCHEMA.md
41. Federal Ministry of Health, Nigeria. Nigeria Standard Treatment Guidelines. 3rd ed. Abuja: Federal Ministry of Health; 2022.
42. Ogbonna KE. Thinking lab — disease-specific cancer systems [Internet]. Project Confluence; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/evidence/thinking.html
43. Ogbonna KE. Disease Profile exporter + offline auditor + citation policy (pull request #11, merged as 64ba76b) [Internet]. Project Confluence; 2026 Sep 20 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/pull/11
44. Gomila J, OnCo contributors. CONTRIBUTING.md and IdeaSchema (src/lib/schema.ts) [Internet]. OnCo; 2026 [cited 2026 Sep 20]. Available from: https://github.com/judegomila/OnCo
45. Ogbonna KE. validation/pdac_data_sources.md [Internet]. Project Confluence; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/validation/pdac_data_sources.md
46. Ogbonna KE. PDAC rogue closure [Internet]. Project Confluence; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/docs/pdac_rogue_closure.md
47. World Health Organization. Cancer [Internet]. Geneva: WHO; 2026 [cited 2026 Sep 20]. Available from: https://www.who.int/news-room/fact-sheets/detail/cancer
48. Federal Ministry of Health, Nigeria. Nigeria National Cancer Control Plan 2018–2022 [Internet]. Abuja: Federal Ministry of Health; 2018 [cited 2026 Sep 20]. Available from: https://www.iccp-portal.org/sites/default/files/plans/NCCP_Final%20%5B1%5D.pdf
49. Federal Ministry of Health, Nigeria. Nigeria Essential Medicines List. 7th ed [Internet]. Abuja: Federal Ministry of Health; 2020 [cited 2026 Sep 20]. Available from: https://www.who.int/publications/m/item/nigeria--essential-medicines-list-2020-(english)
50. Bellman R, Åström KJ. On structural identifiability. Math Biosci. 1970;7(3-4):329-339. doi:10.1016/0025-5564(70)90132-X.
51. Raue A, Kreutz C, Maiwald T, Bachmann J, Schilling M, Klingmüller U, et al. Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. Bioinformatics. 2009;25(15):1923-1929. doi:10.1093/bioinformatics/btp358. PMID: 19505944.
52. Barretina J, Caponigro G, Stransky N, Venkatesan K, Margolin AA, Kim S, et al. The Cancer Cell Line Encyclopedia enables predictive modelling of anticancer drug sensitivity. Nature. 2012;483(7391):603-607. doi:10.1038/nature11003. PMID: 22460905.
53. Li H, Ning S, Ghandi M, Kryukov GV, Gopal S, Deik A, et al. The landscape of cancer cell line metabolism. Nat Med. 2019;25(5):850-860. doi:10.1038/s41591-019-0404-8. PMID: 31068703.
54. Ogbonna KE. Structural identifiability of a real-CCLE-calibrated metabolic ODE model (manuscript draft) [Internet]. Project Confluence; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/docs/manuscript/structural_identifiability_ccle_manuscript.md
55. Ogbonna KE. Citation policy [Internet]. Project Confluence; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/docs/CITATION_POLICY.md
56. Weinstein JN, Collisson EA, Mills GB, Shaw KR, Ozenberger BA, Ellrott K, et al. The Cancer Genome Atlas Pan-Cancer analysis project. Nat Genet. 2013;45(10):1113-1120. doi:10.1038/ng.2764. PMID: 24071849.
57. Cancer Genome Atlas Network. Comprehensive molecular portraits of human breast tumours. Nature. 2012;490(7418):61-70. doi:10.1038/nature11412. PMID: 23000897.
58. Cancer Genome Atlas Research Network. Comprehensive genomic characterization defines human glioblastoma genes and core pathways. Nature. 2008;455(7216):1061-1068. doi:10.1038/nature07385. PMID: 18772890.
59. National Cancer Institute. Genomic Data Commons: TCGA-GBM [Internet]. Bethesda (MD): NCI; [cited 2026 Sep 20]. Available from: https://portal.gdc.cancer.gov/projects/TCGA-GBM
60. Puchalski RB, Shah N, Miller J, Dalley R, Nomura SR, Yoon JG, et al. An anatomic transcriptional atlas of human glioblastoma. Science. 2018;360(6389):660-663. doi:10.1126/science.aaf2666. PMID: 29748285. PMCID: PMC6414061.
61. Cancer Genome Atlas Research Network. Integrated genomic characterization of pancreatic ductal adenocarcinoma. Cancer Cell. 2017;32(2):185-203.e13. doi:10.1016/j.ccell.2017.07.007. PMID: 28810144.
62. Cerami E, Gao J, Dogrusoz U, Gross BE, Sumer SO, Aksoy BA, et al. The cBio cancer genomics portal: an open platform for exploring multidimensional cancer genomics data. Cancer Discov. 2012;2(5):401-404. doi:10.1158/2159-8290.CD-12-0095. PMID: 22588877.
63. Moffitt RA, Marayati R, Flate EL, Volmar KE, Loeza SG, Hoadley KA, et al. Virtual microdissection identifies distinct tumor- and stroma-specific subtypes of pancreatic ductal adenocarcinoma. Nat Genet. 2015;47(10):1168-1178. doi:10.1038/ng.3398. PMID: 26343385. GEO: GSE71729.
64. Yang S, He P, Wang J, Schetter A, Tang W, Funamizu N, et al. A novel MIF signaling pathway drives the malignant character of pancreatic cancer by targeting NR3C2. Cancer Res. 2016;76(13):3838-3850. doi:10.1158/0008-5472.CAN-15-2841. PMID: 27197190. GEO: GSE62452.
65. Zhang G, Schetter A, He P, Funamizu N, Gaedcke J, Ghadimi BM, et al. DPEP1 inhibits tumor cell invasiveness, enhances chemosensitivity and predicts clinical outcome in pancreatic ductal adenocarcinoma. PLoS One. 2012;7(2):e31507. doi:10.1371/journal.pone.0031507. PMID: 22363658. GEO: GSE28735.
66. Ogbonna KE. Call for longitudinal pathology and omics data [Internet]. Project Confluence; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/CALL_FOR_DATA.md
67. International Agency for Research on Cancer. Global cancer statistics 2024: GLOBOCAN estimates of incidence and mortality worldwide for 34 cancers in 186 countries [Internet]. Lyon: IARC; 2026 Jul 8 [cited 2026 Sep 20]. Available from: https://www.iarc.who.int/news-events/global-cancer-statistics-2024-globocan-estimates-of-incidence-and-mortality-worldwide-for-34-cancers-in-186-countries/
68. Ghandi M, Huang FW, Jané-Valbuena J, Kryukov GV, Lo CC, McDonald ER 3rd, et al. Next-generation characterization of the Cancer Cell Line Encyclopedia. Nature. 2019;569(7757):503-508. doi:10.1038/s41586-019-1186-3. PMID: 31068700.
69. Foulkes WD, Smith IE, Reis-Filho JS. Triple-negative breast cancer. N Engl J Med. 2010;363(20):1938-1948. doi:10.1056/NEJMra1001389. PMID: 21067385.
70. Vander Heiden MG, Cantley LC, Thompson CB. Understanding the Warburg effect: the metabolic requirements of cell proliferation. Science. 2009;324(5930):1029-1033. doi:10.1126/science.1160809. PMID: 19460998.
71. Ogbonna KE. confluence/profiles/disease_profile.py [Internet]. Project Confluence; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/confluence/profiles/disease_profile.py
72. Ogbonna KE. schemas/disease_profile.schema.json [Internet]. Project Confluence; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/schemas/disease_profile.schema.json
73. Ogbonna KE. Disease Profile case pack (data/profiles/cases/) [Internet]. Project Confluence; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/tree/main/data/profiles/cases
74. Ogbonna KE. confluence/profiles/hypothesis_object.py [Internet]. Project Confluence; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/confluence/profiles/hypothesis_object.py
75. Ogbonna KE. H-TNBC-LAC-EXCL-001 (data/hypotheses/tnbc_lactate_immune_exclusion.yaml) [Internet]. Project Confluence; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/data/hypotheses/tnbc_lactate_immune_exclusion.yaml
76. Ogbonna KE. docs/CITATION_POLICY.md [Internet]. Project Confluence; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/docs/CITATION_POLICY.md
77. Ogbonna KE. scripts/build_disease_profile_pack.py [Internet]. Project Confluence; 2026 [cited 2026 Sep 20]. Available from: https://github.com/cloudynirvana/project-confluence/blob/main/scripts/build_disease_profile_pack.py

---

## Disclaimer

**This document is not personalized medicine as a clinical product.**

Project Confluence Disease Profiles, thinking-lab boards, HypothesisObjects, OnCo adapter outputs, and Complexity Science CaseCards are computational research artefacts. They are:

- **not** a medical device;
- **not** clinical decision support;
- **not** a diagnostic, prognostic, or therapeutic tool;
- **not** a patient chart, care plan, or dose;
- **not** approved by the FDA, EMA, NAFDAC, or any regulator;
- **not** a claim of cure, disease eradication, or a treatment path;
- **not** a claim that OnCo knowledge, OnCo confidence, or Idea maturity is an identified parameter Θ;
- **not** a claim that the TNBC, GBM, PDAC, or dormancy boards confer patient benefit;
- **not** an execution of Nigeria Standard Treatment Guidelines.

Simulated metrics in the wider repository are research scores on an ODE [3]. Qualitative boards in this thesis are not simulated patient benefit. Public-dataset names are experimental bindings, not completed analyses. If any sentence here conflicts with [DISCLAIMER.md](../../DISCLAIMER.md) or [docs/AWAITING_CLINICAL_VALIDATION.md](../AWAITING_CLINICAL_VALIDATION.md), those files win.

Data from OnCo (onco.cc), CC BY-NC 4.0; commercial use needs a licence [27,29].

**Author:** Kelechi Emeka Ogbonna  
**Date:** 20 September 2026  
**Thesis #3**
