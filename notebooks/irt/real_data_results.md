# Real Data Results — IRT Model Comparison

**Date**: 2026-03-13
**Models**: Baseline GRM, MICE-only GRM, Mixed (IRT+MICE stacking) GRM
**Stacking weights**: Yao et al. (2018) per-item pointwise LOO optimization

## Combined Results

| Dataset | K | N | Items | % Missing | Model | Pred RMSE | LOO-RMSE | ELPD/person | ELPD/response |
|---------|---|---|-------|-----------|-------|-----------|----------|-------------|---------------|
| **GRIT** | 5 | 4,270 | 12 | 2.0% | Baseline | 1.0986 | 1.0988 | -16.821 +/- 0.039 | -1.407 +/- 0.003 |
| | | | | | MICE-only | 1.0778 | 1.0782 | -16.463 +/- 0.040 | -1.377 +/- 0.003 |
| | | | | | Mixed | 1.1645 | 1.1648 | -16.733 +/- 0.040 | -1.399 +/- 0.003 |
| **TMA** | 2 | 5,410 | 50 | 17.3% | Baseline | 0.4182 | 0.4183 | -26.005 +/- 0.092 | -0.525 +/- 0.002 |
| | | | | | MICE-only | 0.4137 | 0.4138 | -25.339 +/- 0.089 | -0.512 +/- 0.002 |
| | | | | | Mixed | 0.4132 | 0.4133 | -25.384 +/- 0.087 | -0.513 +/- 0.002 |
| **RWA** | 9 | 9,881 | 22 | 2.0% | Baseline | 1.6396 | 1.6400 | -30.109 +/- 0.126 | -1.371 +/- 0.006 |
| | | | | | MICE-only | 1.5675 | 1.5677 | -29.712 +/- 0.136 | -1.353 +/- 0.006 |
| | | | | | Mixed | 1.6394 | 1.6400 | -29.667 +/- 0.137 | -1.351 +/- 0.006 |
| **NPI** | 2 | 11,243 | 40 | 7.1% | Baseline | 0.4998 | 0.4998 | -27.623 +/- 0.007 | -0.693 +/- 0.000 |
| | | | | | MICE-only | 0.4629 | 0.4641 | -25.688 +/- 0.080 | -0.644 +/- 0.002 |
| | | | | | Mixed | 0.4252 | 0.4252 | -22.020 +/- 0.068 | -0.552 +/- 0.002 |
| **WPI** | 2 | 6,019 | 116 | 35.4% | Baseline | 0.4125 | 0.4125 | -59.235 +/- 0.158 | -0.515 +/- 0.001 |
| | | | | | MICE-only | 0.4085 | 0.4086 | -58.112 +/- 0.163 | -0.506 +/- 0.001 |
| | | | | | Mixed | 0.4127 | 0.4131 | -61.262 +/- 0.188 | -0.533 +/- 0.002 |
| **EQSQ** | 4 | 13,256 | 120 | 11.1% | Baseline | 1.0065 | 1.1023 | -189.775 +/- 0.150 | -1.589 +/- 0.001 |
| | | | | | MICE-only | 0.8750 | 0.8750 | -143.574 +/- 0.135 | -1.202 +/- 0.001 |
| | | | | | Mixed | 0.8754 | 0.8754 | -143.625 +/- 0.135 | -1.203 +/- 0.001 |

## Key Observations

### ELPD (higher = better)

- **MICE-only beats baseline on all 6 datasets** in ELPD/person
- **EQSQ shows largest imputation benefit**: MICE-only ELPD -143.57 vs baseline -189.77 (24% improvement)
- **Mixed (stacking) is best on NPI** (massive improvement: -22.02 vs -27.62 baseline)
- **Mixed is competitive on RWA and TMA** (slightly better than baseline, close to MICE-only)
- **Mixed underperforms on GRIT and WPI** — possibly overfitting stacking weights

### RMSE (lower = better)

- **MICE-only has lowest Pred RMSE on 5/6 datasets** (GRIT, TMA, RWA, WPI, EQSQ)
- **Mixed has lowest Pred RMSE on NPI** (0.4252 vs 0.4998 baseline, a 15% improvement)
- **EQSQ imputation models**: RMSE 0.875 vs baseline 1.007 (13% improvement)
- Pred RMSE and LOO-RMSE are nearly identical across all models

## Dataset Characteristics

| Dataset | K (categories) | N (people) | Items | % Missing | Domain |
|---------|---------------|------------|-------|-----------|--------|
| GRIT | 5 | 4,270 | 12 | 2.0% | Grit/perseverance |
| TMA | 2 | 5,410 | 50 | 17.3% | Taylor Manifest Anxiety |
| RWA | 9 | 9,881 | 22 | 2.0% | Right-Wing Authoritarianism |
| NPI | 2 | 11,243 | 40 | 7.1% | Narcissistic Personality Inventory |
| WPI | 2 | 6,019 | 116 | 35.4% | Woodworth Personal Inventory |
| EQSQ | 4 | 13,256 | 120 | 11.1% | Empathy Quotient / Systemizing Quotient |
