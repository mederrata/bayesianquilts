# IRT Example Scripts

## Capability Matrix

The bayesianquilts IRT framework supports the following inference modes:

| Inference | Abilities | Weights | Imputation | Example Script |
|-----------|-----------|---------|------------|----------------|
| Joint ADVI | estimated jointly | none | none | `fit_weighted_irt.py` |
| Joint ADVI | estimated jointly | IPW | none | `fit_weighted_irt.py` |
| Joint ADVI | estimated jointly | none | pairwise stacking | `fit_weighted_irt.py` |
| Joint ADVI | estimated jointly | IPW | pairwise stacking | `fit_weighted_irt.py` |
| Marginal ADVI | integrated out | none | none | `fit_marginal_irt.py` |
| Marginal ADVI | integrated out | IPW | none | `fit_marginal_irt.py` |
| Marginal ADVI | integrated out | none | pairwise stacking | `fit_marginal_irt.py` |
| Marginal ADVI | integrated out | IPW | pairwise stacking | `fit_marginal_irt.py` |
| Marginal MCMC | integrated out | none | none | `fit_marginal_irt.py` |
| Marginal MCMC | integrated out | IPW | none | `fit_marginal_irt.py` |
| Marginal MCMC | integrated out | none | pairwise stacking | `fit_marginal_irt.py` |
| Marginal MCMC | integrated out | IPW | pairwise stacking | `fit_marginal_irt.py` |
| Joint ADVI (factorized) | estimated jointly | IPW | pairwise stacking | `fit_weighted_factorized_irt.py` |
| Marginal MCMC (factorized) | integrated out per-scale | IPW | pairwise stacking | `fit_marginal_factorized_irt.py` |

### Inference modes

- **Joint ADVI**: Standard variational inference estimating item parameters and abilities simultaneously. Fast but biased for small samples or complex posteriors.
- **Marginal ADVI**: Variational inference on item parameters only, with abilities integrated out on a Gauss-Hermite quadrature grid. Reduces the parameter space dramatically. Supports mean-field (`rank=0`) and low-rank (`rank>0`) surrogates.
- **Marginal MCMC**: BlackJAX NUTS on item parameters with abilities integrated out. Gold standard for item parameter estimation.

### After fitting item parameters

All marginal inference modes support **EAP ability recovery** via `model.compute_eap_abilities(data)`, which computes posterior mean abilities by numerical integration given fixed item parameters.

### Weights and imputation

- **IPW weights**: Pass `sample_weights` in the data dict. Used in the likelihood for pseudo-posterior inference under biased sampling.
- **Pairwise stacking imputation**: Attach imputation PMFs via `_imputation_pmfs` in the data dict. Missing items contribute Rao-Blackwellized likelihood terms instead of being dropped.

## Scripts

| Script | Purpose |
|--------|---------|
| `fit_weighted_irt.py` | Full joint ADVI pipeline with survey weights |
| `fit_weighted_stacking.py` | Fit PairwiseOrdinalStackingModel with optional survey weights |
| `fit_weighted_factorized_irt.py` | Multi-scale joint ADVI with survey weights |
| `fit_marginal_irt.py` | Marginal ADVI + MCMC for unidimensional GRM |
| `fit_marginal_factorized_irt.py` | Per-scale marginal MCMC for factorized GRM |
| `example_ipw_groups.py` | Creating IPW group weights from stratified data |
