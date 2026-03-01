# Results Directory

This directory contains all outputs from running the Project Confluence pipeline.

## Key Reports (Start Here)

| File | What It Contains |
|------|-----------------|
| [executive_summary.md](executive_summary.md) | One-page overview of the entire framework and results |
| [confluence_report.md](confluence_report.md) | Full pan-cancer results from the v2.0 Confluence runner |
| [universal_cure_proof.md](universal_cure_proof.md) | Detailed per-cancer protocol optimization results |
| [gaps_and_limitations.md](gaps_and_limitations.md) | Honest risk assessment and known limitations |
| [latiff.md](latiff.md) | Lattice-Informed Formalism (LATIFF) — mathematical foundation |
| [pan_cancer_analysis.md](pan_cancer_analysis.md) | Cross-cancer portability analysis |

## Subdirectories

### `protocols/`
Lab-ready protocol documents for each cancer type. Each cancer has:
- `*_lab_protocol.md` — Human-readable protocol (drug concentrations, cell lines, endpoints)
- `*_lab_protocol.json` — Machine-readable protocol data

### `data/`
Raw JSON outputs from simulation runs:
- `confluence_results.json` — Full results from `confluence_runner.py`
- `calibrated_corrections.json` — Optimized generator corrections per cancer
- `resistance_model.json` — Resistance dynamics data
- `robustness_analysis.json` — Parameter sensitivity data
- `combination_sweep.json` — Drug synergy/antagonism sweep results
