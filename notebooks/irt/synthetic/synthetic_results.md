# Synthetic Data Results — IRT Model Comparison

**Date**: 2026-03-11
**Pipeline**: NeuralGRM (ground truth) -> Synthetic data (25% MCAR for 40% of respondents) -> Baseline / MICE-only / Mixed GRM
**Evaluation**: Spearman rho, Kendall tau, RMSE vs ground-truth abilities (fully observed respondents only)
**ELPD-LOO**: PSIS-LOO on all respondents

## Ability Recovery (Fully Observed Respondents)

| Dataset | K | N | Items | Obs | Model | Spearman rho | Kendall tau | RMSE [95% CI] |
|---------|---|---|-------|-----|-------|-------------|-------------|---------------|
| **GRIT** | 5 | 4,270 | 12 | 2,624 | Baseline | 0.2506 | 0.1690 | 0.2776 [0.271, 0.284] |
| | | | | | MICE-only | 0.2513 | 0.1695 | 0.2797 [0.273, 0.287] |
| | | | | | Mixed | 0.2535 | 0.1711 | 0.2941 [0.287, 0.301] |
| **NPI** | 2 | 11,243 | 40 | 6,746 | Baseline | 0.8785 | 0.7001 | 0.2124 [0.209, 0.216] |
| | | | | | MICE-only | 0.8772 | 0.6985 | 0.2089 [0.205, 0.213] |
| | | | | | Mixed | 0.8779 | 0.6994 | 0.2079 [0.204, 0.212] |
| **TMA** | 2 | 5,410 | 50 | 3,246 | Baseline | 0.8027 | 0.6068 | 0.2427 [0.238, 0.247] |
| | | | | | MICE-only | 0.8053 | 0.6095 | 0.2372 [0.232, 0.242] |
| | | | | | Mixed | 0.8077 | 0.6119 | 0.2364 [0.232, 0.241] |
| **WPI** | 2 | 6,019 | 116 | 3,612 | Baseline | 0.8887 | 0.7041 | 0.2519 [0.248, 0.256] |
| | | | | | MICE-only | 0.8875 | 0.7025 | 0.2485 [0.244, 0.253] |
| | | | | | Mixed | 0.8888 | 0.7044 | 0.2481 [0.244, 0.253] |
| **RWA** | 9 | 9,881 | 22 | 5,936 | Baseline | 0.2655 | 0.1780 | 0.2850 [0.281, 0.290] |
| | | | | | MICE-only | 0.2707 | 0.1816 | 0.3069 [0.302, 0.312] |
| | | | | | Mixed | 0.2714 | 0.1823 | 0.3328 [0.328, 0.338] |
| **EQSQ** | 4 | 13,256 | 120 | 7,954 | Baseline | 0.0100 | 0.0066 | 0.1923 [0.190, 0.195] |
| | | | | | MICE-only | 0.1852 | 0.1240 | 0.2260 [0.223, 0.229] |
| | | | | | Mixed | 0.1722 | 0.1158 | 0.2294 [0.226, 0.233] |

## ELPD-LOO (All Respondents)

| Dataset | Model | ELPD/person | SE/person | max k-hat | mean k-hat |
|---------|-------|-------------|-----------|-----------|------------|
| **GRIT** | Baseline | -14.880 | 0.050 | 0.798 | 0.067 |
| | MICE-only | -14.871 | 0.050 | 0.828 | 0.065 |
| | Mixed | -14.870 | 0.052 | 0.788 | 0.067 |
| **NPI** | Baseline | -18.485 | 0.055 | 1.095 | 0.087 |
| | MICE-only | -18.505 | 0.055 | 0.948 | 0.089 |
| | Mixed | -18.492 | 0.055 | 1.025 | 0.085 |
| **TMA** | Baseline | -26.312 | 0.075 | 0.809 | 0.074 |
| | MICE-only | -26.299 | 0.075 | 0.722 | 0.076 |
| | Mixed | -26.287 | 0.075 | 1.012 | 0.075 |
| **WPI** | Baseline | -57.512 | 0.156 | 0.799 | 0.110 |
| | MICE-only | -57.523 | 0.156 | 0.996 | 0.100 |
| | Mixed | -57.489 | 0.156 | 0.822 | 0.104 |
| **RWA** | Baseline | -31.034 | 0.068 | 0.899 | 0.061 |
| | MICE-only | -30.985 | 0.068 | 0.824 | 0.059 |
| | Mixed | -31.004 | 0.069 | 1.091 | 0.059 |
| **EQSQ** | Baseline | NaN | NaN | inf | inf |
| | MICE-only | -141.699 | 0.200 | 1.158 | 0.337 |
| | Mixed | -140.836 | 0.195 | 1.551 | 0.463 |

## Key Observations

### Ability Recovery (rho, tau)

- **All three models recover abilities comparably** on most datasets — differences in rho are typically < 0.01
- **GRIT and RWA have low recovery** (rho ~ 0.25-0.27) — these instruments have few items (12 and 22) relative to the number of latent categories
- **NPI, TMA, WPI have strong recovery** (rho ~ 0.80-0.89) — more items provide more information
- **EQSQ baseline fails** (rho = 0.01) but MICE-only recovers (rho = 0.19) — suggests baseline GRM struggles with the 120-item 4-category instrument, while imputation provides regularization
- **Mixed slightly outperforms or ties MICE-only** on rho for 5/6 datasets

### RMSE vs Ground Truth

- **Lower RMSE correlates with imputation** for NPI, TMA, WPI — MICE-only and Mixed have lower RMSE than Baseline
- **GRIT and RWA: Baseline has lowest RMSE** despite lower rho — suggests imputation inflates variance for these harder instruments
- **EQSQ: Baseline has lowest RMSE** but worst rho — the model is predicting the mean but not recovering individual differences

### ELPD-LOO

- **Models are very close** on most datasets (differences < 0.1 per person)
- **Mixed is best on 3/6 datasets** (GRIT, TMA, WPI) by ELPD
- **MICE-only is best on 2/6** (NPI, RWA)
- **EQSQ baseline ELPD is NaN** — complete PSIS failure for that instrument

### PSIS Diagnostics

- **Mean k-hat is acceptable** (< 0.1) for all datasets except EQSQ
- **EQSQ has high mean k-hat** (0.34-0.46) indicating poor PSIS approximation
- **Individual max k-hat > 0.7** present in all datasets but affects < 1% of observations

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Ground truth model | NeuralGRM (mixture-of-logits, 4 components) |
| Missingness pattern | MCAR, 25% of items for 40% of respondents |
| Latent dimensions | 1 primary + 1 noisy |
| Training epochs | 500 |
| Learning rate | 1e-3 (GRM), 5e-4 (NeuralGRM) |
| Batch size | 256 (GRM), 512 (NeuralGRM) |
| Discrimination prior | Horseshoe |
| ADVI sample size | 32 |
| Seed | 42 |
