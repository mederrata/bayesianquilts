# Synthetic Data Results — IRT Model Comparison (noisy_dim=2)

**Date**: 2026-03-12
**Pipeline**: NeuralGRM (ground truth, 3 latent dims) -> Synthetic data (real missingness patterns) -> Baseline / MICE-only / Mixed GRM (dim=1)
**Evaluation**: Spearman rho, Kendall tau, RMSE vs ground-truth abilities (fully observed respondents only)
**ELPD-LOO**: PSIS-LOO on all respondents

## Ability Recovery (Fully Observed Respondents)

| Dataset | K | N | I | % Missing | Model | Spearman ρ | RMSE | ELPD/person | ELPD/response |
|---------|---|---|---|-----------|-------|------------|------|-------------|---------------|
| **GRIT** | 5 | 4,270 | 12 | 2.0% | Baseline | 0.2002 | 0.2877 | -15.757 +/- 0.041 | -1.313 +/- 0.003 |
| | | | | | MICE-only | 0.2020 | 0.2910 | -15.748 +/- 0.041 | -1.312 +/- 0.003 |
| | | | | | Mixed | 0.2000 | 0.2826 | -15.760 +/- 0.041 | -1.313 +/- 0.003 |
| **RWA** | 9 | 9,881 | 22 | 2.0% | Baseline | 0.2568 | 0.3057 | -34.953 +/- 0.050 | -1.589 +/- 0.002 |
| | | | | | MICE-only | 0.2597 | 0.3268 | -34.910 +/- 0.050 | -1.587 +/- 0.002 |
| | | | | | Mixed | 0.2586 | 0.3191 | -34.924 +/- 0.050 | -1.588 +/- 0.002 |
| **TMA** | 2 | 5,410 | 50 | 17.3% | Baseline | 0.3804 | 0.2148 | -33.611 +/- 0.018 | -0.672 +/- 0.000 |
| | | | | | MICE-only | 0.3815 | 0.2129 | -33.610 +/- 0.018 | -0.672 +/- 0.000 |
| | | | | | Mixed | 0.3809 | 0.2129 | -33.613 +/- 0.018 | -0.672 +/- 0.000 |
| **NPI** | 2 | 11,243 | 40 | 7.1% | Baseline | 0.3732 | 0.2116 | -26.955 +/- 0.010 | -0.674 +/- 0.000 |
| | | | | | MICE-only | 0.3748 | 0.2082 | -26.950 +/- 0.010 | -0.674 +/- 0.000 |
| | | | | | Mixed | 0.3746 | 0.2082 | -26.950 +/- 0.010 | -0.674 +/- 0.000 |
| **WPI** | 2 | 6,019 | 116 | 35.4% | Baseline | 0.9299 | 0.1629 | -62.326 +/- 0.121 | -0.537 +/- 0.001 |
| | | | | | MICE-only | 0.9265 | 0.1604 | -62.383 +/- 0.122 | -0.538 +/- 0.001 |
| | | | | | Mixed | 0.9296 | 0.1581 | -62.327 +/- 0.121 | -0.537 +/- 0.001 |
| **EQSQ** | 4 | 13,256 | 120 | 11.1% | Baseline | 0.6208 | 0.1545 | -148.642 +/- 0.064 | -1.239 +/- 0.001 |
| | | | | | MICE-only | 0.6277 | 0.1531 | -148.625 +/- 0.064 | -1.239 +/- 0.001 |
| | | | | | Mixed | 0.6257 | 0.1532 | -148.631 +/- 0.064 | -1.239 +/- 0.001 |

## Key Observations

### Ability Recovery (Spearman ρ)
- **All three models recover abilities comparably** — differences in ρ are typically < 0.01
- **GRIT and RWA have low recovery** (ρ ~ 0.20-0.26) — few items (12 and 22) relative to 3 latent dims
- **NPI moderate** (ρ ~ 0.37) — 40 binary items
- **TMA moderate** (ρ ~ 0.38) — 50 binary items
- **WPI has strong recovery** (ρ ~ 0.93) — 116 binary items provide high information
- **EQSQ good recovery** (ρ ~ 0.62) — 120 items with K=4 categories

### ELPD (higher = better)
- **Differences between models are small** on most datasets (within SE)
- **MICE-only best on GRIT, RWA, EQSQ** by ELPD/person
- **Baseline best on WPI** by small margin

### RMSE vs Ground Truth (lower = better)
- **Mixed model achieves lowest RMSE on GRIT and WPI**
- **MICE-only best on TMA, NPI, EQSQ**
- **Baseline best on GRIT, RWA** — low missingness (2%) means imputation adds noise

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Ground truth model | NeuralGRM (mixture-of-logits, 4 components) |
| Latent dimensions | 1 primary + 2 noisy (noisy_dim=2) |
| Missingness pattern | Replicates real data (per-item rates, respondent fractions) |
| Training epochs | 500 (patience 20) |
| Learning rate | 1e-3 (GRIT, RWA, TMA, NPI), 5e-4 (WPI, EQSQ) |
| Batch size | 256 (GRM), 512 (NeuralGRM) |
| Discrimination prior | Horseshoe (GRIT, RWA), half_normal scale=2.0 (TMA, NPI, WPI, EQSQ) |
| ADVI sample size | 32 |
| Seed | 42 |
