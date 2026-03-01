# Real Data Integration — Calibration Results

> Generator matrices refined using literature-derived metabolomics profiles
> from 30 cell lines across 10 cancer types.

## Calibration Summary

| Cancer | R² Before | R² After | RMSE Before | RMSE After | Entries Changed | Frobenius Dist |
|--------|-----------|----------|-------------|------------|-----------------|----------------|
| TNBC | -3.7386 | -3.4084 | 0.3980 | 0.3838 | 11 | 0.0848 |
| PDAC | -2.9123 | -2.2939 | 0.4186 | 0.3841 | 11 | 0.2832 |
| NSCLC | -7.3423 | -3.1121 | 0.3544 | 0.2488 | 14 | 0.3062 |
| Melanoma | -9.5406 | -6.7144 | 0.3075 | 0.2631 | 11 | 0.1604 |
| GBM | -3.0055 | -2.2597 | 0.3504 | 0.3161 | 7 | 0.2039 |
| CRC | -4.1479 | -2.5504 | 0.3804 | 0.3159 | 7 | 0.2224 |
| HGSOC | -4.5680 | -3.4237 | 0.3440 | 0.3066 | 7 | 0.2155 |
| mCRPC | -11.5409 | -9.6801 | 0.3066 | 0.2829 | 7 | 0.2266 |
| AML | -4.4108 | -2.8645 | 0.3591 | 0.3035 | 11 | 0.2064 |
| HCC | -2.5625 | -1.9650 | 0.3639 | 0.3319 | 7 | 0.2287 |

## Basin Curvature Changes

| Cancer | Original Curvature | Calibrated Curvature | Delta |
|--------|-------------------|---------------------|-------|
| TNBC | 0.2045 | 0.2095 | +0.0049 |
| PDAC | 0.1898 | 0.2142 | +0.0244 |
| NSCLC | 0.2148 | 0.2183 | +0.0035 |
| Melanoma | 0.2573 | 0.2875 | +0.0301 |
| GBM | 0.1840 | 0.1623 | -0.0217 |
| CRC | 0.2267 | 0.2122 | -0.0145 |
| HGSOC | 0.1988 | 0.1750 | -0.0238 |
| mCRPC | 0.2052 | 0.1828 | -0.0224 |
| AML | 0.2087 | 0.2554 | +0.0467 |
| HCC | 0.1624 | 0.1468 | -0.0156 |

## Coherence Changes

| Cancer | Original | Calibrated | Delta |
|--------|----------|------------|-------|
| TNBC | 0.5853 | 0.5907 | +0.0054 |
| PDAC | 0.5623 | 0.5876 | +0.0253 |
| NSCLC | 0.5891 | 0.6018 | +0.0127 |
| Melanoma | 0.6205 | 0.6338 | +0.0133 |
| GBM | 0.5777 | 0.5632 | -0.0145 |
| CRC | 0.6016 | 0.5987 | -0.0029 |
| HGSOC | 0.5866 | 0.5718 | -0.0148 |
| mCRPC | 0.5830 | 0.5658 | -0.0172 |
| AML | 0.5918 | 0.6155 | +0.0236 |
| HCC | 0.5547 | 0.5443 | -0.0104 |
## Drug Validation Against IC50 Data

| Drug | Cell Line | IC50 | SAEM Effect Magnitude | In Library? |
|------|-----------|------|----------------------|-------------|
| DCA | PANC-1 | 39.0 mM | 0.3606 | ✅ |
| DCA | HCT116 | 20.0 mM | 0.3606 | ✅ |
| DCA | MDA-MB-231 | 25.0 mM | 0.3606 | ✅ |
| CB-839 | enzymatic_GLS1 | 24.0 nM | 0.5000 | ✅ |
| CB-839 | HG-3_CLL | 410.0 nM | 0.5000 | ✅ |
| CB-839 | MEC-1_CLL | 66.2 µM | 0.5000 | ✅ |
| Metformin | MDA-MB-231 | 10.0 mM | 0.2500 | ✅ |
| Metformin | A549 | 15.0 mM | 0.2500 | ✅ |
| Metformin | PANC-1 | 20.0 mM | 0.2500 | ✅ |
| 2-DG | MDA-MB-231 | 3.0 mM | 0.3082 | ✅ |
| 2-DG | A549 | 5.0 mM | 0.3082 | ✅ |
| 2-DG | PANC-1 | 4.0 mM | 0.3082 | ✅ |
| Olaparib | MDA-MB-231 | 10.0 µM | - | ❌ |
| Olaparib | HCC1937 | 2.0 µM | - | ❌ |
| Olaparib | OVCAR3 | 5.0 µM | - | ❌ |
| Venetoclax | HL-60 | 50.0 nM | - | ❌ |
| Venetoclax | MOLM-13 | 20.0 nM | - | ❌ |
| Venetoclax | OCI-AML3 | 100.0 nM | - | ❌ |
## Simulation Comparison (Original vs Calibrated)

| Cancer | Orig Cure Rate | Cal Cure Rate | Orig Escape | Cal Escape | Δ Cure Rate |
|--------|---------------|---------------|-------------|------------|-------------|
| TNBC | 93.3% | 96.7% | 0.609 | 0.589 | +3.3% |
| PDAC | 76.7% | 36.7% | 0.711 | 1.184 | -40.0% |
| NSCLC | 100.0% | 76.7% | 0.521 | 0.736 | -23.3% |
| Melanoma | 100.0% | 100.0% | 0.525 | 0.517 | +0.0% |
| GBM | 100.0% | 96.7% | 0.597 | 0.647 | -3.3% |
| CRC | 100.0% | 100.0% | 0.502 | 0.507 | +0.0% |
| HGSOC | 100.0% | 96.7% | 0.564 | 0.611 | -3.3% |
| mCRPC | 96.7% | 90.0% | 0.532 | 0.607 | -6.7% |
| AML | 93.3% | 96.7% | 0.583 | 0.531 | +3.3% |
| HCC | 83.3% | 60.0% | 0.682 | 0.956 | -23.3% |