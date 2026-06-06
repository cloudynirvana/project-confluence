# External Validation Pipeline Preparation

Status: preparation stage.

This document converts the current synthetic-validation workflow into a staged external-data validation plan. It is deliberately conservative: external datasets are used to test whether the framework produces reproducible signals, not to make clinical claims.

## Operating Rules

- Canonical sources first: use PhysioNet, NCI GDC, and cBioPortal before secondary mirrors.
- Hugging Face is useful for discovery and prototyping, but every dataset must pass dataset-card, license, provenance, and split-policy review before use.
- Do not commit raw biomedical data, restricted data, access tokens, or patient identifiers.
- Use patient-wise or subject-wise splits for every validation task.
- Report failed or null results; no cherry-picking.
- Keep claims narrow: "metric separates states in this dataset" is acceptable; "clinical biomarker validated" is not.

## Validation Ladder

### Tier 0: Synthetic Reproducibility

Current status:

- `individuality-dynamics` runs end-to-end and reproduces lambda* values.
- `project-confluence` adaptive-controller validation now has explicit failure gates.
- Monte Carlo runs end-to-end and saves a repo-local figure.

Gate to pass:

- All synthetic scripts run from a clean clone.
- Results are deterministic with fixed seeds.
- README claims match actual output.

### Tier 1: Model-Free Physiology

Primary datasets:

- PhysioNet Sleep-EDF Expanded for EEG and sleep-stage annotations.
- PhysioNet MIT-BIH Normal Sinus Rhythm Database for ECG/RR healthy-complexity references.

Questions:

- Does the individuality metric track sleep-stage organization?
- Do Phi temporal/informational metrics produce stable healthy reference ranges from real ECG/RR data?

First target figure:

```text
x-axis: time or epoch index
y-axis: lambda*, closure score, or PCI-like response
overlay: Wake / REM / N1 / N2 / N3 labels
split: subject-wise
```

### Tier 2: Oncology Retrospective Cohorts

Primary sources:

- NCI GDC/TCGA harmonized clinical and omics data.
- cBioPortal public cancer-genomics studies.

Questions:

- Can gene-to-parameter mappings be populated from real cohort data?
- Do disease archetypes remain stable across public studies?
- Is Phi_coupling still a weak point because it is correlation-based?

Required caveat:

These analyses are retrospective and exploratory. They do not validate treatment protocols.

### Tier 3: Credentialed ICU Waveforms

Primary source:

- MIMIC-IV Waveform Database through PhysioNet credentialing.

Questions:

- Do Phi metrics track deterioration, recovery, or physiologic regime shifts in multimodal ICU waveforms?
- Does nonlinear coupling add value over Pearson correlation?

Gate:

- Credentialing, data-use agreement, and local secure storage are complete.
- No restricted files are committed to GitHub.

### Tier 4: Expert Validation

Invite reviewers from:

- oncology
- computational biology
- dynamical systems
- control theory
- sleep neuroscience
- clinical physiology
- research software engineering
- data governance / biomedical ethics

Expected reviewer output:

- reproducibility report
- mathematical assumptions audit
- data leakage audit
- clinical-claim boundary audit
- license/provenance audit

## Immediate Work Packages

### WP1: Sleep-EDF Individuality Validation

Repository: `individuality-dynamics`

Inputs:

- PSG EDF file
- Hypnogram EDF annotations

Loader candidate:

```python
mne.datasets.sleep_physionet.age.fetch_data()
mne.io.read_raw_edf()
mne.read_annotations()
```

Outputs:

- per-epoch closure score
- per-epoch lambda* or proxy metric
- stage-wise boxplots
- subject-wise cross-validation table

Acceptance criteria:

- finite metrics for at least 95 percent of artifact-screened epochs
- no subject leakage between train/test splits
- sleep-stage separation tested with confidence intervals

### WP2: ECG/RR Healthy Reference Ranges

Repository: `project-confluence`

Inputs:

- MIT-BIH Normal Sinus Rhythm records and annotations

Loader candidate:

```python
wfdb.rdrecord()
wfdb.rdann()
wfdb.dl_database()
```

Outputs:

- external reference range for `Phi_temporal`
- external reference range for `Phi_informational`
- sensitivity of MSE/spectral-slope metrics to window size

Acceptance criteria:

- record-wise stable estimates
- documented window sizes
- results written to `results/external_validation/physionet_nsrdb/`

### WP3: TCGA/cBioPortal Cohort Grounding

Repository: `project-confluence`

Inputs:

- open TCGA harmonized clinical and genomic files
- cBioPortal public study queries

Outputs:

- study metadata cache
- gene-to-parameter coverage table
- cohort-level archetype summaries

Acceptance criteria:

- every query logs source URL, study ID, timestamp, and filters
- no controlled-access data without explicit authorization
- no patient-level claims

### WP4: Phi_coupling Upgrade Benchmark

Current implementation:

`Phi_coupling` uses mean absolute Pearson correlation.

Validation issue:

The theory needs nonlinear and directional coupling for complex systems.

Candidate upgrades:

- convergent cross mapping
- transfer entropy
- time-lagged mutual information

Acceptance criteria:

- benchmark Pearson against at least one nonlinear/directional measure
- retain Pearson as a baseline, not as final causal evidence
- report computational cost and data-length requirements

## Preparation Command

Run:

```powershell
python scripts/prepare_external_validation.py
```

This creates local validation directories, checks optional dependencies, and prints the first-stage tasks without downloading data.

Optional validation dependencies:

```powershell
pip install -r requirements-validation.txt
```
