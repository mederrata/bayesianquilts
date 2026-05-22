# bayesianquilts (development version)

# bayesianquilts 0.9.0

## CRAN submission

Initial CRAN release.

## New features (since 0.2.0)

### Adaptive Importance Sampling

* New transformations: `NaturalLikelihoodDescent`, `Variance`, `PMM3`,
  `MM3`, and `MixIS` (Silva & Zanella 2024).
* Analytical divergence (`compute_divergence_Q`) for `LikelihoodDescent`,
  `KLDivergence`, `NaturalKLDivergence`, `PMM1`, `PMM2`, `PMM3`, and
  `Variance`. This fixes a zero-Jacobian collapse that affected the
  small-`h` regime under the previous numerical-divergence path.
* `AdaptiveImportanceSampler$adaptive_is_loo` accepts `f_fn` (target for
  the Variance transform), `n_mix_samples` (for MixIS), and `seed`; it
  reports per-transformation wall-clock timing in `results$timings`.
* `DEFAULT_TRANSFORMATION_ORDER` mirrors the Python reference order
  (identity → mm1 → mm2 → mm3 → mixis → pmm1 → pmm2 → pmm3 → ll → nll →
  kl → nkl → var).

### Quilted-model decomposition

* New module: `Dimension`, `Interactions`, `Decomposed`, and
  `MultiwayContingencyTable` for the additive interaction decomposition.
* `Decomposed$generalization_preserving_scales(...)` implements the
  generalization-preserving prior scales of Chang (2026); the
  one-shot wrapper `quilt_prior_scales()` is provided for convenience.

### brms pairing

* `quilt_brms_formula()` / `quilt_brms_priors()` translate a `Decomposed`
  object into a brms formula plus prior list with scales coming from
  `generalization_preserving_scales`.
* `fit_quilt_brms()` end-to-end wrapper: derive scales, assemble
  formula/priors, call `brms::brm()`.

### Parameter-level ensembling

* `component_predict()`, `component_loo_elpd()`, and
  `ensemble_components()` extract per-component contributions from a
  fitted brmsfit and combine them with uniform, AIC, or LOO-stacking
  weights.
