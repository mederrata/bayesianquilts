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
| Marginal MCMC + IS | integrated out | none | pairwise IS-reweight | `fit_is_irt.py` |
| Marginal MCMC + IS | integrated out | none | mixed IS-reweight | `fit_is_irt.py` |
| Marginal MCMC + IS | integrated out | IPW | pairwise IS-reweight | `fit_is_irt.py` |
| Marginal MCMC + IS | integrated out | IPW | mixed IS-reweight | `fit_is_irt.py` |
| Marginal MCMC + IS (factorized) | integrated out per-scale | IPW | pairwise/mixed IS-reweight | `fit_is_factorized_irt.py` |
| Marginal ADVI + post-hoc BCM | integrated out | none | mixed (pairwise + IRT blend) | `fit_bcm_with_imputation.py` |

### Inference modes

- **Joint ADVI**: Standard variational inference estimating item parameters and abilities simultaneously. Fast but biased for small samples or complex posteriors.
- **Marginal ADVI**: Variational inference on item parameters only, with abilities integrated out on a Gauss-Hermite quadrature grid. Reduces the parameter space dramatically. Supports mean-field (`rank=0`) and low-rank (`rank>0`) surrogates.
- **Marginal MCMC**: BlackJAX NUTS on item parameters with abilities integrated out. Gold standard for item parameter estimation.
- **Marginal MCMC + IS**: Run MCMC once on the baseline model, then importance-sample-reweight those draws toward the imputed posteriors (pairwise, mixed). Much cheaper than running separate MCMC for each variant.

### After fitting item parameters

All marginal inference modes support **EAP ability recovery** via `model.compute_eap_abilities(data)`, which computes posterior mean abilities by numerical integration given fixed item parameters.

### Post-hoc bias correction (BCM)

- **Bias-Correction Map (BCM)**: A scoring-time post-hoc correction that
  maps a naive (or imputed) subset-IRT score to the score the same
  respondent would have received from the full item bank, given which
  items happened to be administered. Implemented as `BCMConditional` in
  `libfabulouscatpy.biascorrection`. See `fit_bcm_with_imputation.py` for
  an end-to-end pipeline that fits the pairwise stacking imputation
  model, the baseline GRM, the `IrtMixedImputationModel`, and the BCM
  using imputation-blended scoring for both the subset and gold scores.
  This regime is required when no respondent has a complete response
  vector (so a non-imputed "gold" cannot be defined).
  `fit_marginal_irt.py` adds the same BCM step (Step 6) on top of its
  marginal-MCMC `mixed` variant; skip with `--skip-bcm`.

### Converged artifact (libfab + gofluttercat consumable)

Both `fit_bcm_with_imputation.py` and `fit_marginal_irt.py` write a
`converged/` subdirectory after fitting, produced by
`bayesianquilts.io.converged.export_artifact`. The layout is the one
both `libfabulouscatpy.irt.converged.load_artifact` and gofluttercat's
Go-side loaders read:

```
<output_dir>/converged/
  items/<item_key>.json       # per-item GRM params (libfab + gofluttercat factorized shape)
  scales.json                 # per-scale metadata
  imputation/                 # gofluttercat-consumable imputation bundle
  manifest.yaml               # provenance: timestamps, git SHA, settings
  bcm_<scale>.json            # per-J isotonic BCMSet (Go-readable; written by the BCM-aware examples)
```

The IRT model is **standardised to N(0,1) abilities** before extraction
(`model.standardize_abilities()` for joint-ADVI;
`model.standardize_marginal(data)` for marginal-MCMC), so the
discriminations and cumulative cutpoints in `items/` are on the scale
gofluttercat's prior assumes. The `BCMConditional` joblib (richer
per-item-indicator corrector for Python-side use) is saved at the
example's `<output_dir>/` root; the isotonic `BCMSet` JSON inside
`converged/` is the gofluttercat-compatible companion fit on the same
`(subset, gold)` triples.

The per-item JSON uses default integer response labels and the item key
as placeholder question text. For datasets with curated metadata, copy
the corresponding `<item>.json` files from
`gofluttercat/backend-golang/<scale>/factorized/` into `converged/items/`
and replace only the `scales: {<scale>: {...}}` payload.

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
| `fit_is_irt.py` | ADVI → MCMC baseline → IS reweight for pairwise/mixed |
| `fit_is_factorized_irt.py` | Per-scale ADVI → MCMC → IS reweight pipeline |
| `example_ipw_groups.py` | Creating IPW group weights from stratified data |
| `fit_bcm_with_imputation.py` | End-to-end: pairwise imputation + baseline GRM (ADVI) + `IrtMixedImputationModel` + `BCMConditional` trained with imputation-blended subset/gold scoring (requires `libfabulouscatpy`) |

## Default dataset

The marginal and IS scripts default to the EQSQ dataset (120 items, K=4).
Factorized scripts use the natural Empathy/Systemizing 2-scale split.

## Convergence monitoring

All MCMC scripts print max R-hat after sampling and automatically extend
chains (up to 3 rounds) if max R-hat > 1.05, using the `resume=True`
parameter on `fit_marginal_mcmc`.

## Outputs

Each script produces:
- **Forest plots**: Item discriminations and difficulties (with model comparisons for imputation scripts)
- **Ability histograms**: EAP ability distributions per variant
- **Ability scatter plots**: Baseline vs imputation variant abilities
- **Summary table**: LOO-RMSE, LOO-ELPD, IS diagnostics (k-hat, ESS)
