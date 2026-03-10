# Setup Guide — Project Confluence

## Prerequisites

- **Python 3.9+** (tested with 3.11)
- **pip** (package manager)

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/cloudynirvana/project-confluence.git
cd project-confluence

# 2. Create virtual environment (recommended)
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Verify Installation

```bash
# Run the test suite
python -m pytest tests/ -v

# Quick import check
python -c "from models.complexity_profiler import ComplexityProfiler; print('✓ Module 1')"
python -c "from models.patient_fitter import PatientFitter; print('✓ Module 2')"
python -c "from models.drug_optimization_engine import RADOEngine; print('✓ Module 3')"
python -c "from agents.bioinformatics_miner import BioinformaticsMiner; print('✓ Module 4')"
```

## Optional Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `emcee` | MCMC sampling (Module 2) | `pip install emcee>=3.1.0` |
| `optuna` | Bayesian optimization (Module 3) | `pip install optuna>=3.0.0` |
| `requests` | cBioPortal API (Module 4) | `pip install requests>=2.28.0` |
| `corner` | Posterior visualization | `pip install corner>=2.2.0` |

## API Configuration

For the Bioinformatics Miner (Module 4), public cBioPortal access requires no API key. For TCGA bulk downloads, you may need:

1. **cBioPortal**: No key needed (public REST API)
2. **GDC (TCGA)**: Register at [portal.gdc.cancer.gov](https://portal.gdc.cancer.gov)

## Hardware Notes

This framework has been designed and tested on:
- **CPU**: Intel i5-3380M (2.9 GHz, 2 cores)
- **RAM**: 8 GB DDR3
- **OS**: Windows 10+

All algorithms are O(N²) or better — no GPU required.
