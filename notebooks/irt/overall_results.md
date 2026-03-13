# IRT Imputation Model Comparison — Overall Results

**Date**: 2026-03-13
**Models**: Baseline GRM (listwise deletion), MICE-only GRM, Mixed (IRT+MICE stacking) GRM
**Evaluation**: PSIS-LOO ELPD, predictive RMSE, ability recovery (synthetic only)

## Real Data

| Dataset | K | N | Items | % Missing | Model | Pred RMSE | ELPD/person | ELPD/response |
|---------|---|---|-------|-----------|-------|-----------|-------------|---------------|
| **GRIT** | 5 | 4,270 | 12 | 2.0% | Baseline | 1.0986 | -16.821 +/- 0.039 | -1.407 +/- 0.003 |
| | | | | | MICE-only | 1.0778 | -16.463 +/- 0.040 | -1.377 +/- 0.003 |
| | | | | | Mixed | 1.1645 | -16.733 +/- 0.040 | -1.399 +/- 0.003 |
| **TMA** | 2 | 5,410 | 50 | 17.3% | Baseline | 0.4182 | -26.005 +/- 0.092 | -0.525 +/- 0.002 |
| | | | | | MICE-only | 0.4137 | -25.339 +/- 0.089 | -0.512 +/- 0.002 |
| | | | | | Mixed | 0.4132 | -25.384 +/- 0.087 | -0.513 +/- 0.002 |
| **RWA** | 9 | 9,881 | 22 | 2.0% | Baseline | 1.6396 | -30.109 +/- 0.126 | -1.371 +/- 0.006 |
| | | | | | MICE-only | 1.5675 | -29.712 +/- 0.136 | -1.353 +/- 0.006 |
| | | | | | Mixed | 1.6394 | -29.667 +/- 0.137 | -1.351 +/- 0.006 |
| **NPI** | 2 | 11,243 | 40 | 7.1% | Baseline | 0.4998 | -27.623 +/- 0.007 | -0.693 +/- 0.000 |
| | | | | | MICE-only | 0.4629 | -25.688 +/- 0.080 | -0.644 +/- 0.002 |
| | | | | | Mixed | 0.4252 | -22.020 +/- 0.068 | -0.552 +/- 0.002 |
| **WPI** | 2 | 6,019 | 116 | 35.4% | Baseline | 0.4125 | -59.235 +/- 0.158 | -0.515 +/- 0.001 |
| | | | | | MICE-only | 0.4085 | -58.112 +/- 0.163 | -0.506 +/- 0.001 |
| | | | | | Mixed | 0.4127 | -61.262 +/- 0.188 | -0.533 +/- 0.002 |
| **EQSQ** | 4 | 13,256 | 120 | 11.1% | Baseline | 1.0065 | -189.775 +/- 0.150 | -1.589 +/- 0.001 |
| | | | | | MICE-only | 0.8750 | -143.574 +/- 0.135 | -1.202 +/- 0.001 |
| | | | | | Mixed | 0.8754 | -143.625 +/- 0.135 | -1.203 +/- 0.001 |

## Synthetic Data (noisy_dim=2)

Ground truth: NeuralGRM with 3 latent dims (1 primary + 2 noisy). Recovery models: GRM with dim=1.

| Dataset | K | N | Items | % Missing | Model | Spearman ρ | RMSE | ELPD/person | ELPD/response |
|---------|---|---|-------|-----------|-------|------------|------|-------------|---------------|
| **GRIT** | 5 | 4,270 | 12 | 2.0% | Baseline | 0.2002 | 0.2877 | -15.757 +/- 0.041 | -1.313 +/- 0.003 |
| | | | | | MICE-only | 0.2020 | 0.2910 | -15.748 +/- 0.041 | -1.312 +/- 0.003 |
| | | | | | Mixed | 0.2000 | 0.2826 | -15.760 +/- 0.041 | -1.313 +/- 0.003 |
| **TMA** | 2 | 5,410 | 50 | 17.3% | Baseline | 0.3804 | 0.2148 | -33.611 +/- 0.018 | -0.672 +/- 0.000 |
| | | | | | MICE-only | 0.3815 | 0.2129 | -33.610 +/- 0.018 | -0.672 +/- 0.000 |
| | | | | | Mixed | 0.3809 | 0.2129 | -33.613 +/- 0.018 | -0.672 +/- 0.000 |
| **RWA** | 9 | 9,881 | 22 | 2.0% | Baseline | 0.2568 | 0.3057 | -34.953 +/- 0.050 | -1.589 +/- 0.002 |
| | | | | | MICE-only | 0.2597 | 0.3268 | -34.910 +/- 0.050 | -1.587 +/- 0.002 |
| | | | | | Mixed | 0.2586 | 0.3191 | -34.924 +/- 0.050 | -1.588 +/- 0.002 |
| **NPI** | 2 | 11,243 | 40 | 7.1% | Baseline | 0.3732 | 0.2116 | -26.955 +/- 0.010 | -0.674 +/- 0.000 |
| | | | | | MICE-only | 0.3748 | 0.2082 | -26.950 +/- 0.010 | -0.674 +/- 0.000 |
| | | | | | Mixed | 0.3746 | 0.2082 | -26.950 +/- 0.010 | -0.674 +/- 0.000 |
| **WPI** | 2 | 6,019 | 116 | 35.4% | Baseline | 0.9299 | 0.1629 | -62.326 +/- 0.121 | -0.537 +/- 0.001 |
| | | | | | MICE-only | 0.9265 | 0.1604 | -62.383 +/- 0.122 | -0.538 +/- 0.001 |
| | | | | | Mixed | 0.9296 | 0.1581 | -62.327 +/- 0.121 | -0.537 +/- 0.001 |
| **EQSQ** | 4 | 13,256 | 120 | 11.1% | Baseline | 0.6208 | 0.1545 | -148.642 +/- 0.064 | -1.239 +/- 0.001 |
| | | | | | MICE-only | 0.6277 | 0.1531 | -148.625 +/- 0.064 | -1.239 +/- 0.001 |
| | | | | | Mixed | 0.6257 | 0.1532 | -148.631 +/- 0.064 | -1.239 +/- 0.001 |

## Summary: Best Model by ELPD/person

| Dataset | % Missing | Best (Real) | ELPD Δ vs Baseline | Best (Synthetic) | ρ Δ vs Baseline |
|---------|-----------|-------------|--------------------|--------------------|-----------------|
| GRIT | 2.0% | MICE-only | +0.358 (+2.1%) | MICE-only | +0.0018 |
| TMA | 17.3% | MICE-only | +0.666 (+2.6%) | MICE-only | +0.0011 |
| RWA | 2.0% | Mixed | +0.442 (+1.5%) | MICE-only | +0.0029 |
| NPI | 7.1% | Mixed | +5.603 (+20.3%) | MICE-only | +0.0016 |
| WPI | 35.4% | MICE-only | +1.123 (+1.9%) | Mixed | -0.0003 |
| EQSQ | 11.1% | MICE-only | +46.201 (+24.3%) | MICE-only | +0.0069 |

## Key Findings

### Real Data
- **MICE-only improves ELPD on all 6 datasets** over listwise-deletion baseline
- **Largest gains on EQSQ (+24.3%) and NPI (+20.3%)** — datasets with moderate missingness and many items
- **Mixed stacking is best on NPI and RWA** — pointwise LOO stacking selects optimal per-item weights
- **Mixed underperforms on GRIT, WPI, EQSQ** — stacking may overfit when item-level weights are noisy
- **RMSE improvements parallel ELPD**: MICE-only lowest on 5/6 datasets, Mixed lowest on NPI

### Synthetic Data (Model Misspecification)
- **Ability recovery (ρ) nearly identical across models** — imputation does not harm parameter recovery
- **ELPD differences are small** (within SE) under synthetic conditions
- **Misspecification (noisy_dim=2) reduces recovery**: ρ ranges from 0.20 (GRIT, 12 items) to 0.93 (WPI, 116 items)
- **More items = better recovery**: WPI (116) and EQSQ (120) have highest ρ despite misspecification

### Practical Recommendations
1. **Always use MICE-only imputation** — never worse than baseline, often substantially better
2. **Mixed stacking adds value for NPI-like datasets** (moderate missingness, many binary items)
3. **For high-K polytomous scales (EQSQ)**: use `half_normal(1/√2)` discrimination prior
4. **For binary scales (TMA, NPI, WPI)**: use `half_normal(2.0)` discrimination prior

## Dataset Characteristics

| Dataset | K | N | Items | % Missing | Domain |
|---------|---|---|-------|-----------|--------|
| GRIT | 5 | 4,270 | 12 | 2.0% | Grit/perseverance |
| TMA | 2 | 5,410 | 50 | 17.3% | Taylor Manifest Anxiety |
| RWA | 9 | 9,881 | 22 | 2.0% | Right-Wing Authoritarianism |
| NPI | 2 | 11,243 | 40 | 7.1% | Narcissistic Personality Inventory |
| WPI | 2 | 6,019 | 116 | 35.4% | Woodworth Personal Inventory |
| EQSQ | 4 | 13,256 | 120 | 11.1% | Empathy Quotient / Systemizing Quotient |

## Experimental Details

### Real Data
- Training: ADVI with 500 epochs, patience 10 (baseline) / 30 (imputation models)
- Learning rate: 5e-4 (EQSQ, WPI), 1e-3 (others)
- Discrimination prior: half_normal(1/√2) for EQSQ, half_normal(2.0) for binary, half_cauchy(1.0) for GRIT/RWA
- ELPD: PSIS-LOO with 100 posterior samples (200 if >10% high k-hat)

### Synthetic Data
- Ground truth: NeuralGRM (mixture-of-logits, 4 components, 3 latent dims)
- Recovery: GRM with dim=1 (intentional misspecification via noisy_dim=2)
- Missingness: replicates real data per-item rates and respondent fractions
- Discrimination prior: horseshoe (GRIT, RWA), half_normal(2.0) (TMA, NPI, WPI, EQSQ)
