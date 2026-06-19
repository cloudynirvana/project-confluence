# PDAC Rogue Closure

This executable module turns the Confluence synthesis into a disease-specific validation target for pancreatic ductal adenocarcinoma (PDAC).

```text
PDAC persistence = KRAS/RAS driver closure
                 + EGFR/STAT3 bypass recovery
                 + stromal/glycocalyx shielding
                 + immune exclusion
                 + therapy-selected resistance
```

The model is a hypothesis generator, not a clinical predictor. Its purpose is to define variables that can be tested against public PDAC cohorts.

## Run

```bash
python scripts/run_pdac_rogue_closure.py --all-scenarios
```

Outputs are written to:

```text
results/pdac_rogue_closure/pdac_summary.json
results/pdac_rogue_closure/pdac_timeseries.csv
results/pdac_rogue_closure/pdac_closure_report.svg
```

## Validation Data

The synthetic time series is committed for reproducibility only. Real validation should use linked public repositories:

- TCGA-PAAD / GDC: https://portal.gdc.cancer.gov/projects/TCGA-PAAD
- cBioPortal PAAD: https://www.cbioportal.org/study/summary?id=paad_tcga_pan_can_atlas_2018
- GEO GSE71729: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE71729
- GEO GSE62452: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE62452
- GEO GSE28735: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE28735
- DepMap: https://depmap.org/portal/download/
- NCI PDMR: https://pdmr.cancer.gov/
- PDX Finder: https://www.pdxfinder.org/
- GlyGen: https://www.glygen.org/
- GlyConnect: https://glyconnect.expasy.org/

See `validation/pdac_data_sources.md` for the validation plan.

## First Falsifiable Claim

```text
Rogue-closure features derived from driver, immune, stromal, and glyco-shield
signals should predict PDAC survival or aggressive phenotype better than KRAS
status alone.
```

