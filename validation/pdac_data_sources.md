# PDAC Validation Data Sources

This file separates simulation outputs from real validation data.

## What We Have Now

The files in `results/` are synthetic model outputs. They are useful for:

- checking equations
- comparing therapy schedules
- debugging the workflow
- generating expected qualitative behavior

They are not real biological validation.

## Tier 1: First Validation Datasets

### TCGA-PAAD / GDC

Link: https://portal.gdc.cancer.gov/projects/TCGA-PAAD

Use for:

- KRAS, TP53, SMAD4, CDKN2A mutation status
- RNA-seq expression
- clinical survival
- tumor stage and grade
- first-pass PDAC cohort validation

Primary validation question:

```text
Do rogue-closure features derived from driver genes, immune genes, stromal genes,
and glyco-shield genes predict survival or aggressive phenotype better than
KRAS status alone?
```

### cBioPortal TCGA PanCancer Atlas PAAD

Link: https://www.cbioportal.org/study/summary?id=paad_tcga_pan_can_atlas_2018

Use for:

- fast exploratory mutation/expression/survival analysis
- downloadable clinical tables
- comparing model-derived feature sets against standard oncogenic features

### GEO GSE71729

Link: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE71729

Use for:

- primary PDAC
- metastatic PDAC
- normal samples
- tumor/stroma subtype validation

Key validation question:

```text
Does the model's stroma/shield axis separate activated stroma, normal stroma,
primary tumor, and metastatic disease?
```

### GEO GSE62452

Link: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE62452

Use for:

- 69 pancreatic tumors
- 61 adjacent non-tumor samples
- survival-associated expression validation

### GEO GSE28735

Link: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE28735

Use for:

- 45 paired tumor/non-tumor PDAC samples
- early differential-expression validation
- survival-linked gene screen replication

## Tier 2: Perturbation And Drug-Response Validation

### DepMap / Cancer Dependency Map

Link: https://depmap.org/portal/download/

Use for:

- PDAC cell-line dependencies
- KRAS, EGFR, STAT3, glycosylation, stromal-proxy vulnerabilities
- CRISPR/RNAi dependency support
- drug sensitivity context through linked DepMap resources

Validation question:

```text
Are model-selected intervention targets dependency-supported in PDAC-lineage
systems?
```

### NCI Patient-Derived Models Repository

Link: https://pdmr.cancer.gov/

Use for:

- PDX / organoid / cell-line model discovery
- PDAC model selection for future perturbation tests
- bridge from patient omics to model systems

### PDX Finder

Link: https://www.pdxfinder.org/

Use for:

- finding PDAC PDX models
- checking available molecular and treatment metadata
- external model-system validation

## Tier 3: Glyco-Shield Evidence

### GlyGen

Link: https://www.glygen.org/

Use for:

- glycoprotein annotation
- glycosylation sites
- building the initial glyco-shield gene/protein panel

Candidate glyco-shield panel:

```text
MUC1, MUC4, MUC16, ST6GAL1, ST3GAL1, ST3GAL4, FUT3, FUT8,
B3GNT3, B4GALT1, MGAT5, GALNT3, GALNT6, SDC1, SDC4, GPC1,
HAS2, LGALS1, LGALS3
```

### GlyConnect

Link: https://glyconnect.expasy.org/

Use for:

- glycoprotein and glycan relationship lookup
- manual curation of glyco-shield features

## Tier 4: Imaging And Spatial Validation

### The Cancer Imaging Archive

Link: https://www.cancerimagingarchive.net/

Use for:

- PDAC imaging cohorts when available
- tumor/stroma/radiomics proxy validation
- future spatial closure validation

## Validation Milestones

1. Build real-data feature table from TCGA-PAAD:

```text
sample_id, survival, stage, KRAS, TP53, SMAD4, CDKN2A,
driver_score, immune_score, stroma_score, glyco_score, rogue_closure_score
```

2. Test whether `rogue_closure_score` predicts survival or aggressive phenotype.

3. Repeat in GSE71729 for primary/metastatic/stroma subtype separation.

4. Use DepMap to check whether proposed intervention axes are dependency-supported.

5. Publish synthetic output separately from real-data validation results.

## Repository Rule

Do not commit large raw datasets directly to GitHub. Commit:

- fetch scripts
- small manifests
- checksums
- derived feature tables when license permits
- validation reports

Use releases, Zenodo, DVC, or external object storage for larger artifacts.

