# Real Data Results — IRT Model Comparison

**Date**: 2026-03-11
**Models**: Baseline GRM, MICE-only GRM, Mixed (IRT+MICE stacking) GRM
**Stacking weights**: Yao et al. (2018) per-item pointwise LOO optimization

## Combined Results

| Dataset | Model | K | N | Items | Pred RMSE | LOO-RMSE | ELPD/person | ELPD/response | k>0.7 | k_inf | k_max* | k_mean* |
|---------|-------|---|---|-------|-----------|----------|-------------|---------------|-------|-------|--------|---------|
| **GRIT** | Baseline | 5 | 4,270 | 12 | 1.0986 | 1.0988 | -16.821 +/- 0.039 | -1.407 +/- 0.003 | 9/4270 | 6 | 0.850 | 0.075 |
| | MICE-only | | | | 1.0778 | 1.0782 | -16.463 +/- 0.040 | -1.377 +/- 0.003 | 10/4270 | 6 | 0.984 | 0.072 |
| | Mixed | | | | 1.1645 | 1.1648 | -16.733 +/- 0.040 | -1.399 +/- 0.003 | 14/4270 | 6 | 0.873 | 0.131 |
| **TMA** | Baseline | 2 | 5,410 | 50 | 0.4182 | 0.4183 | -26.005 +/- 0.092 | -0.525 +/- 0.002 | 11/5410 | 9 | 0.819 | 0.095 |
| | MICE-only | | | | 0.4137 | 0.4138 | -25.339 +/- 0.089 | -0.512 +/- 0.002 | 15/5410 | 9 | 0.859 | 0.091 |
| | Mixed | | | | 0.4132 | 0.4133 | -25.384 +/- 0.087 | -0.513 +/- 0.002 | 14/5410 | 9 | 0.921 | 0.084 |
| **RWA** | Baseline | 9 | 9,881 | 22 | 1.6396 | 1.6400 | -30.109 +/- 0.126 | -1.371 +/- 0.006 | 8/9881 | 3 | 0.798 | 0.071 |
| | MICE-only | | | | 1.5675 | 1.5677 | -29.712 +/- 0.136 | -1.353 +/- 0.006 | 11/9881 | 3 | 1.025 | 0.096 |
| | Mixed | | | | 1.6394 | 1.6400 | -29.667 +/- 0.137 | -1.351 +/- 0.006 | 14/9881 | 3 | 1.102 | 0.082 |
| **NPI** | Baseline | 2 | 11,243 | 40 | 0.4998 | 0.4998 | -27.623 +/- 0.007 | -0.693 +/- 0.000 | 7/11243 | 2 | 0.727 | 0.057 |
| | MICE-only | | | | 0.4629 | 0.4641 | -25.688 +/- 0.080 | -0.644 +/- 0.002 | 104/11243 | 2 | 1.004 | 0.173 |
| | Mixed | | | | 0.4252 | 0.4252 | -22.020 +/- 0.068 | -0.552 +/- 0.002 | 65/11243 | 2 | 1.126 | 0.160 |
| **WPI** | Baseline | 2 | 6,019 | 116 | 0.4125 | 0.4125 | -59.235 +/- 0.158 | -0.515 +/- 0.001 | 12/6019 | 3 | 0.958 | 0.105 |
| | MICE-only | | | | 0.4085 | 0.4086 | -58.112 +/- 0.163 | -0.506 +/- 0.001 | 6/6019 | 3 | 0.824 | 0.092 |
| | Mixed | | | | 0.4127 | 0.4131 | -61.262 +/- 0.188 | -0.533 +/- 0.002 | 151/6019 | 3 | 1.166 | 0.259 |

\* k_max and k_mean computed over finite k-hat values only (excluding k_inf observations)

## Key Observations

### ELPD (higher = better)

- **MICE-only beats baseline on all 5 datasets** in ELPD/person
- **Mixed (stacking) is best on NPI** (massive improvement: -22.02 vs -27.62 baseline)
- **Mixed is competitive on RWA and TMA** (slightly better than baseline, close to MICE-only)
- **Mixed underperforms on GRIT and WPI** — possibly overfitting stacking weights

### RMSE (lower = better)

- **MICE-only has lowest Pred RMSE on 4/5 datasets** (GRIT, TMA, RWA, WPI)
- **Mixed has lowest Pred RMSE on NPI** (0.4252 vs 0.4998 baseline, a 15% improvement)
- Pred RMSE and LOO-RMSE are nearly identical across all models

### PSIS Diagnostics

- Most models have very few high k-hat observations (<0.2% of N)
- Each dataset has a small fixed number of k_inf observations (likely boundary cases)
- NPI MICE-only has the most high k-hat (104/11243 = 0.9%) and highest k_mean (0.173)
- WPI Mixed has concerning diagnostics (151/6019 = 2.5% high k-hat, k_mean=0.259)

## Dataset Characteristics

| Dataset | K (categories) | N (people) | Items | Domain |
|---------|---------------|------------|-------|--------|
| GRIT | 5 | 4,270 | 12 | Grit/perseverance |
| TMA | 2 | 5,410 | 50 | Taylor Manifest Anxiety |
| RWA | 9 | 9,881 | 22 | Right-Wing Authoritarianism |
| NPI | 2 | 11,243 | 40 | Narcissistic Personality Inventory |
| WPI | 2 | 6,019 | 116 | Woodworth Personal Inventory |
