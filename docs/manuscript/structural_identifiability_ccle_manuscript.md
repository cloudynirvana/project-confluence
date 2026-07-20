# Structural Identifiability of a Real-CCLE-Calibrated Metabolic ODE Model Across Diverse Cancer Types

**Kelechi Ogbonna**  
Department of Biotechnology, Ahmadu Bello University, Zaria, Nigeria

## Abstract

Computational models of cancer metabolism are often evaluated using synthetic data or narrow single-context calibration, which can limit confidence in their generalizability and obscure parameter-identifiability constraints. This manuscript reports an initial structural identifiability analysis of an ODE-based metabolic state-space model for shared cancer metabolic dynamics. The model is organized around a five-component state representation covering temporal, informational, functional, spatial, and coupling dynamics, with biologically interpretable enzyme-associated channels linked to glycolytic, oxidative, glutaminolytic, and redox metabolism. Using six real metabolomics channels from the Cancer Cell Line Encyclopedia metabolomics dataset (`CCLE_metabolomics_20190502.csv`; 225 metabolites across 928 cancer cell lines), the analysis increased the number of identifiable model parameters from 7 of 17 under the previous synthetic/single-summary validation setting to 15 of 17 under the real multi-metabolite calibration setting. The result was evaluated across three biologically distinct cancer contexts: acute myeloid leukemia, osteosarcoma, and non-small-cell lung cancer. These findings suggest that multi-channel real metabolomics measurements can materially improve the identifiability of mechanistic metabolic ODE models and may provide a stronger basis for downstream model-based inference than synthetic validation alone. Additional parameter-by-parameter tables, reproducible scripts, and sensitivity diagnostics are required before journal submission.

**Keywords:** cancer metabolism; structural identifiability; ordinary differential equations; CCLE; metabolomics; systems biology; mathematical oncology

## 1. Introduction

Metabolic reprogramming is a central feature of cancer biology. Diverse cancer types repeatedly converge on altered glycolysis, mitochondrial metabolism, glutamine utilization, lactate production, and redox regulation, even when their tissue origins and driver mutations differ. This convergence motivates mechanistic mathematical models that attempt to describe shared metabolic-regulatory dynamics across cancer contexts.

Ordinary differential equation (ODE) models are attractive for this purpose because they express mechanistic hypotheses as explicit dynamical systems. However, a persistent limitation is identifiability: even if a model reproduces observed behavior, its internal parameters may not be uniquely recoverable from the available outputs. Structural identifiability asks whether model parameters can in principle be determined from perfect observations of specified outputs, while practical identifiability additionally considers noise, sampling frequency, and finite data.

Many computational cancer models are validated against synthetic outputs or highly compressed summary targets. Such validation can be useful for debugging and proof-of-concept work, but it can also create an overly favorable view of model constraint. This work asks whether replacing synthetic or single-summary validation with real multi-channel metabolomics data improves structural identifiability, and whether the improvement persists across biologically distinct cancer types.

## 2. Methods

### 2.1 Model Overview

Project Confluence implements an ODE-based metabolic state-space model intended to capture shared cancer metabolic dynamics. The current manuscript focuses on the metabolic portion of the model and its parameter identifiability under different output/calibration settings.

The conceptual state representation is organized around five dynamical components:

- Temporal dynamics: time-dependent variability and recovery behavior.
- Informational dynamics: signal complexity, entropy-like behavior, and spectral structure.
- Functional dynamics: adaptive response and restoration capacity.
- Spatial dynamics: heterogeneity and state-space dispersion.
- Coupling dynamics: cross-channel coordination among metabolic and regulatory subsystems.

The full ODE equations, parameter definitions, and output map should be inserted from the analysis commit before submission.

### 2.2 Enzyme-Associated Metabolic Channels

| Enzyme or enzyme family | Biological role | Candidate model/metabolite channel |
| --- | --- | --- |
| HK2 | Glucose phosphorylation and glycolytic entry | Glucose/glycolytic flux channel |
| PKM2 | Pyruvate kinase isoform associated with proliferative glycolysis | Pyruvate/glycolytic outflow channel |
| LDHA | Lactate production and NAD+/NADH cycling | Lactate channel |
| IDH1/2 | TCA-cycle-associated isocitrate metabolism and mutant 2-HG context | alpha-ketoglutarate/citrate-linked channel |
| PDK1 | Regulation of pyruvate dehydrogenase and pyruvate entry into mitochondria | Pyruvate-to-TCA/OXPHOS coupling channel |
| G6PD | Pentose phosphate pathway and redox balance | ROS/glutathione redox proxy channel |

Exact channel names should be verified against `models/ode_system.py` and `models/ccle_metabolite_targets.py` before submission.

### 2.3 Data Source

The real-data calibration uses the Cancer Cell Line Encyclopedia metabolomics dataset, `CCLE_metabolomics_20190502.csv`, which contains 225 metabolite measurements across 928 cancer cell lines. The dataset was generated as part of the CCLE metabolomics resource and is available through Broad/DepMap CCLE resources.

The six metabolomics channels used in the current analysis are lactate, glutamine, glutamate, alpha-ketoglutarate, citrate, and a ROS proxy derived from oxidized/reduced glutathione behavior. Before submission, this section should specify exact column names, preprocessing steps, normalization choices, missing-data handling, and how cell lines were grouped by cancer type.

### 2.4 Cancer-Type Selection

Three cancer types were selected to test whether identifiability improvements were limited to one biological context or persisted across distinct lineages: acute myeloid leukemia (AML), osteosarcoma, and non-small-cell lung cancer (NSCLC).

### 2.5 Structural Identifiability Analysis

The analysis compared two output/calibration settings:

1. A previous synthetic or single-summary validation setting.
2. A real multi-metabolite CCLE calibration setting using six observed metabolomics channels.

The reported summary result is that identifiable parameters increased from 7 of 17 to 15 of 17 after replacing the synthetic/single-summary target with real multi-channel CCLE metabolomics outputs. Before submission, this section must document the exact identifiability method used, including whether the analysis was symbolic, differential-geometric, profile-likelihood-based, Fisher-information-based, or numerical.

## 3. Results

### 3.1 Real Multi-Channel Metabolomics Increased Identifiable Parameters

| Calibration/output setting | Identifiable parameters | Total parameters | Fraction identifiable |
| --- | ---: | ---: | ---: |
| Synthetic/single-summary setting | 7 | 17 | 41.2% |
| Real CCLE six-channel setting | 15 | 17 | 88.2% |

### 3.2 Required Tables and Figures Before Submission

The following items should be generated before journal submission:

- Parameter-by-parameter identifiability table comparing both settings.
- Per-cancer-type summaries for AML, osteosarcoma, and NSCLC.
- Bar chart showing 7/17 versus 15/17 identifiable parameters.
- Diagnostic plots for the two parameters that remain non-identifiable.
- Reproducibility table listing commit hash, dataset version, script entry points, and package versions.

## 4. Discussion

This draft supports the hypothesis that identifiability in mechanistic cancer metabolism models is strongly conditioned by the biological information content of the observed outputs. Adding real multi-channel metabolomics measurements appears to constrain the model more effectively than a synthetic or single-summary validation target.

This distinction matters because model validation and model identifiability are not the same. A model may reproduce selected outputs while still leaving many parameters unconstrained. Conversely, a richer output map can make previously ambiguous parameters recoverable by exposing more independent constraints on the model dynamics.

## 5. Limitations

This manuscript remains a draft and should not yet be submitted without additional verification. Key limitations include:

- The exact identifiability method and reproducibility commands must be documented.
- Structural identifiability does not guarantee practical identifiability under noisy finite data.
- CCLE cell lines are useful models but do not fully represent patient tumors, tissue microenvironment, immune context, or treatment history.
- The ROS proxy based on glutathione behavior requires careful biochemical justification.
- The six-channel selection should be justified against alternative metabolite panels.
- Per-parameter and per-cancer-type results must be reported, not only the aggregate 7/17 to 15/17 summary.

## 6. Future Work

The immediate next step is to generate a full reproducibility package: parameter tables, analysis scripts, exact commit hash, figures, and an executable notebook or command-line workflow. Once the identifiability result is fully documented, the model can support downstream work on virtual screening and natural compound modulation of cancer metabolic channels.

## Data and Code Availability

Code repository: <https://github.com/cloudynirvana/project-confluence>  
Zenodo DOI: <https://doi.org/10.5281/zenodo.21446803>  
Dataset: `CCLE_metabolomics_20190502.csv`, available through Broad/DepMap CCLE resources.  
Analysis commit hash: **to be inserted after the reproducibility scripts and final result tables are committed.**

## Conflict of Interest

The author declares no competing interests.

## Ethics Statement

This manuscript uses publicly available cancer cell-line data and computational modeling. No new human participants, animal experiments, or clinical interventions were performed for this draft.

## Author Affiliation

Kelechi Ogbonna, Department of Biotechnology, Ahmadu Bello University, Zaria, Nigeria.

## References To Add Before Submission

1. Li H, Ning S, Ghandi M, et al. The landscape of cancer cell line metabolism. *Nature Medicine*. 2019;25:850-860. doi:10.1038/s41591-019-0404-8.
2. Ghandi M, Huang FW, Jane-Valbuena J, et al. Next-generation characterization of the Cancer Cell Line Encyclopedia. *Nature*. 2019;569:503-508. doi:10.1038/s41586-019-1186-3.
3. Villaverde AF. Observability and structural identifiability of nonlinear biological systems. *Complexity*. 2019;2019:8497093. doi:10.1155/2019/8497093.
4. Villaverde AF, Banga JR. Reverse engineering and identification in systems biology: strategies, perspectives and challenges. *Journal of the Royal Society Interface*. 2014;11:20130505. doi:10.1098/rsif.2013.0505.
