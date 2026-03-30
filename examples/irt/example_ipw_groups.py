#!/usr/bin/env python
"""Example: IRT calibration with IPW survey weights for biased sampling.

Demonstrates the full pipeline when the calibration dataset contains an
oversampled subgroup (e.g., a clinical cohort deliberately enriched to
improve discrimination at the tails).  The same IPW group_weights flow
into both the pairwise imputation model and the IRT likelihood so that
both stages target the population-level conditionals rather than the
biased sample.

The script synthesizes a small dataset with two strata:
  - "general"  (95% of target population, 60% of calibration sample)
  - "clinical" ( 5% of target population, 40% of calibration sample)

Then runs:
  1. Pairwise stacking imputation  (IPW-weighted)
  2. Baseline GRM                  (IPW-weighted likelihood)
  3. Mixed imputation model
  4. Adaptive ignorability thresholds
  5. Pairwise-only GRM             (IPW-weighted likelihood + thresholds)
  6. Mixed GRM                     (IPW-weighted likelihood + thresholds)

Usage:
    python example_ipw_groups.py --output-dir ipw_example_results/
"""

import argparse
import gc
import json
import os

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import pandas as pd
import jax.numpy as jnp

from pathlib import Path


def generate_stratified_data(
    n_general=600, n_clinical=400, n_items=10, K=5, seed=42,
    missing_rate=0.15,
):
    """Synthesize ordinal responses from two strata with different ability distributions.

    General stratum: abilities ~ N(0, 1)
    Clinical stratum: abilities ~ N(-1.5, 0.8)  (lower ability, tighter spread)

    This mimics a calibration design where a clinical cohort is deliberately
    oversampled to improve measurement precision at the low end.
    """
    rng = np.random.default_rng(seed)
    n = n_general + n_clinical

    # Abilities
    theta_gen = rng.normal(0.0, 1.0, n_general)
    theta_clin = rng.normal(-1.5, 0.8, n_clinical)
    theta = np.concatenate([theta_gen, theta_clin])

    # Item parameters: discriminations and difficulties
    discriminations = rng.uniform(0.5, 2.0, n_items)
    # K-1 ordered thresholds per item
    difficulties = np.sort(
        rng.normal(0, 1.5, (n_items, K - 1)), axis=1
    )

    # Generate GRM responses
    responses = np.zeros((n, n_items), dtype=int)
    for j in range(n_items):
        a_j = discriminations[j]
        d_j = difficulties[j]  # (K-1,)
        # P(Y >= k) = logistic(a * (theta - d_k))
        logits = a_j * (theta[:, None] - d_j[None, :])  # (n, K-1)
        cum_probs = 1.0 / (1.0 + np.exp(-logits))

        u = rng.uniform(size=n)
        for k in range(K - 1):
            responses[:, j] += (u < cum_probs[:, k]).astype(int)
            u = rng.uniform(size=n)

    # Introduce MCAR missingness
    mask = rng.random((n, n_items)) < missing_rate
    responses = responses.astype(float)
    responses[mask] = np.nan

    # Build DataFrame
    item_keys = [f'item_{j}' for j in range(n_items)]
    df = pd.DataFrame(responses, columns=item_keys)
    df['group'] = ['general'] * n_general + ['clinical'] * n_clinical

    return df, item_keys


def compute_survey_weights(groups, group_weights):
    """IPW weights that re-weight the biased sample to match population proportions."""
    n = len(groups)
    w = np.ones(n, dtype=np.float64)
    for g, W_g in group_weights.items():
        g_mask = groups == g
        n_g = g_mask.sum()
        if n_g > 0:
            w[g_mask] = W_g * n / n_g
    n_eff = w.sum()**2 / (w**2).sum()
    return (w * n_eff / w.sum()).astype(np.float32), n_eff


def make_data_dict(df, item_keys, groups, group_weights):
    data = {}
    for col in item_keys:
        vals = df[col].to_numpy(dtype=np.float64)
        data[col] = np.where(np.isnan(vals), np.nan, vals).astype(np.float32)
    n = len(df)
    data['person'] = np.arange(n, dtype=np.float32)

    w, n_eff = compute_survey_weights(groups, group_weights)
    data['sample_weights'] = w
    print(f"  Survey weights: n_eff={n_eff:.0f}/{n}, "
          f"range=[{w.min():.3f}, {w.max():.3f}]")
    return data


def make_data_factory(batch, batch_size, n_people):
    steps_per_epoch = int(np.ceil(n_people / batch_size))

    def factory():
        indices = np.arange(n_people)
        np.random.shuffle(indices)
        n_needed = steps_per_epoch * batch_size
        if n_needed > n_people:
            indices = np.concatenate([
                indices,
                np.random.choice(n_people, n_needed - n_people, replace=True),
            ])
        for start in range(0, n_needed, batch_size):
            idx = indices[start:start + batch_size]
            yield {k: v[idx] for k, v in batch.items()}

    return factory, steps_per_epoch


def calibrate_manually(model, n_samples=32, seed=42):
    """Sample from surrogate posterior and compute calibrated expectations."""
    import jax
    rng = jax.random.PRNGKey(seed)
    samples = model.sample_surrogate_posterior(rng, n_samples)
    model.surrogate_sample = samples
    expectations = {}
    for key, val in samples.items():
        expectations[key] = val.mean(axis=0) if val.ndim > 1 else val
    model.calibrated_expectations = expectations


def run(output_dir='ipw_example_results'):
    from bayesianquilts.irt.grm import GRModel
    from bayesianquilts.imputation.pairwise_stacking import PairwiseOrdinalStackingModel
    from bayesianquilts.imputation.mixed import IrtMixedImputationModel

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    K = 5
    sample_size = 16
    seed = 42
    num_epochs = 50
    batch_size = 128
    learning_rate = 1e-3
    lr_decay_factor = 0.975

    # ------------------------------------------------------------------
    # Generate synthetic stratified data
    # ------------------------------------------------------------------
    print("=== Generating stratified synthetic data ===")
    df, item_keys = generate_stratified_data(
        n_general=600, n_clinical=400, n_items=10, K=K, seed=seed,
    )
    n_people = len(df)
    n_items = len(item_keys)
    groups = df['group'].to_numpy()

    # Population proportions — the target we want to recover.
    # The calibration sample has 40% clinical, but the population is 5%.
    group_weights = {'general': 0.95, 'clinical': 0.05}

    print(f"  Respondents: {n_people}")
    print(f"  Items: {n_items}, Categories: {K}")
    for g in ['general', 'clinical']:
        n_g = (groups == g).sum()
        print(f"  {g}: n={n_g} (sample {100*n_g/n_people:.0f}%, "
              f"population {100*group_weights[g]:.0f}%)")

    # ------------------------------------------------------------------
    # Build data dict with IPW sample_weights in the batch.
    # These weights enter the GRM log-likelihood (grm.py:940-942)
    # so that the ELBO targets population-level item parameters.
    # ------------------------------------------------------------------
    print("\n=== Building IPW-weighted data dict ===")
    batch = make_data_dict(df, item_keys, groups, group_weights)
    factory, steps_per_epoch = make_data_factory(batch, batch_size, n_people)

    # ------------------------------------------------------------------
    # Stage 1: Pairwise stacking imputation (IPW-weighted)
    #
    # The same group_weights are passed to the MICE model so that the
    # pairwise regressions and Dirichlet-multinomial tables learn the
    # population-level conditionals P(z_i | x_l) rather than the
    # sample-biased P(z_i | x_l, sampled).
    # ------------------------------------------------------------------
    print("\n=== Stage 1: Fitting pairwise stacking imputation (IPW-weighted) ===")
    responses_df = df[item_keys].copy()

    pairwise_model = PairwiseOrdinalStackingModel(
        pathfinder_num_samples=100, pathfinder_maxiter=50,
        batch_size=512, verbose=True,
    )
    pairwise_model.fit(
        responses_df,
        n_top_features=min(8, n_items - 1),
        n_jobs=1,
        seed=seed,
        groups=groups,                # <-- stratum labels per respondent
        group_weights=group_weights,  # <-- population proportions
    )
    pairwise_model.compute_optimal_stacking_weights()
    pairwise_model.save_to_disk(str(out / 'pairwise_stacking_model'))
    gc.collect()

    # ------------------------------------------------------------------
    # Stage 2: Baseline GRM (no imputation, IPW-weighted likelihood)
    #
    # sample_weights in the data dict causes the GRM to weight each
    # respondent's contribution by their IPW weight, correcting for the
    # oversampled clinical cohort.
    # ------------------------------------------------------------------
    print("\n=== Stage 2: Fitting baseline GRM (IPW-weighted) ===")
    model_baseline = GRModel(
        item_keys=item_keys, num_people=n_people, dim=1,
        response_cardinality=K, dtype=jnp.float32,
    )
    model_baseline.fit(
        factory, batch_size=batch_size, dataset_size=n_people,
        num_epochs=num_epochs, steps_per_epoch=steps_per_epoch,
        learning_rate=learning_rate, lr_decay_factor=lr_decay_factor,
        patience=10, zero_nan_grads=True,
        sample_size=sample_size, seed=seed, max_nan_recoveries=50,
    )
    model_baseline.save_to_disk(str(out / 'grm_baseline'))
    calibrate_manually(model_baseline, n_samples=sample_size, seed=101)
    gc.collect()

    # ------------------------------------------------------------------
    # Stage 3: Mixed imputation model (pairwise + IRT stacking)
    # ------------------------------------------------------------------
    print("\n=== Stage 3: Building mixed imputation model ===")
    mixed_imp = IrtMixedImputationModel(
        irt_model=model_baseline,
        mice_model=pairwise_model,
        data_factory=factory,
        irt_elpd_batch_size=4,
    )
    print(mixed_imp.summary())
    with open(out / 'mixed_weights.json', 'w') as f:
        json.dump(mixed_imp.weights, f, indent=2)
    gc.collect()

    # ------------------------------------------------------------------
    # Stage 3b: Adaptive ignorability thresholds
    #
    # These thresholds determine which items can safely skip imputation
    # (where the variance cost exceeds the ELPD gain).  The threshold
    # computation uses the IPW-weighted model since both the ELPD
    # estimates and the Var[log a_i] come from the weighted likelihood.
    # ------------------------------------------------------------------
    print("\n=== Stage 3b: Computing adaptive thresholds ===")
    model_baseline.imputation_model = mixed_imp
    calibrate_manually(model_baseline, n_samples=sample_size, seed=101)
    model_baseline.compute_adaptive_thresholds(
        factory, baseline_model=model_baseline,
        sample_size=sample_size, seed=seed,
    )
    thresholds = model_baseline._adaptive_thresholds
    model_baseline.imputation_model = None

    with open(out / 'adaptive_thresholds.json', 'w') as f:
        json.dump(thresholds, f, indent=2)

    # Report which items are ignored
    ignored_pw = [k for k, t in thresholds.items() if t >= 1.0]
    ignored_mix = [k for k in item_keys
                   if mixed_imp.get_item_weight(k) <= thresholds.get(k, 0.0)]
    print(f"  Pairwise-only: imputing {n_items - len(ignored_pw)}/{n_items}")
    print(f"  Mixed: imputing {n_items - len(ignored_mix)}/{n_items}")
    for k in item_keys:
        t = thresholds[k]
        w = mixed_imp.get_item_weight(k)
        status_pw = "IGNORED" if t >= 1.0 else "imputed"
        status_mix = "IGNORED" if w <= t else "imputed"
        print(f"    {k}: threshold={t:.4f}, w_pw={w:.3f}  "
              f"[pw: {status_pw}, mix: {status_mix}]")
    gc.collect()

    # ------------------------------------------------------------------
    # Stage 4: Pairwise-only GRM (IPW-weighted + thresholds)
    # ------------------------------------------------------------------
    print("\n=== Stage 4: Fitting pairwise-only GRM (IPW-weighted) ===")
    from bayesianquilts.imputation.mixed import PairwiseOnlyImputationModel
    pw_only_imp = PairwiseOnlyImputationModel(pairwise_model)

    model_pairwise = GRModel(
        item_keys=item_keys, num_people=n_people, dim=1,
        response_cardinality=K, dtype=jnp.float32,
        imputation_model=pw_only_imp,
    )
    model_pairwise._adaptive_thresholds = thresholds
    model_pairwise.fit(
        factory, batch_size=batch_size, dataset_size=n_people,
        num_epochs=num_epochs, steps_per_epoch=steps_per_epoch,
        learning_rate=learning_rate, lr_decay_factor=lr_decay_factor,
        patience=30, zero_nan_grads=True,
        sample_size=sample_size, seed=seed + 1, max_nan_recoveries=50,
    )
    model_pairwise.save_to_disk(str(out / 'grm_pairwise'))
    calibrate_manually(model_pairwise, n_samples=sample_size, seed=103)
    gc.collect()

    # ------------------------------------------------------------------
    # Stage 5: Mixed GRM (IPW-weighted + thresholds)
    # ------------------------------------------------------------------
    print("\n=== Stage 5: Fitting mixed GRM (IPW-weighted) ===")
    model_mixed = GRModel(
        item_keys=item_keys, num_people=n_people, dim=1,
        response_cardinality=K, dtype=jnp.float32,
        imputation_model=mixed_imp,
    )
    model_mixed._adaptive_thresholds = thresholds
    model_mixed.fit(
        factory, batch_size=batch_size, dataset_size=n_people,
        num_epochs=num_epochs, steps_per_epoch=steps_per_epoch,
        learning_rate=learning_rate, lr_decay_factor=lr_decay_factor,
        patience=30, zero_nan_grads=True,
        sample_size=sample_size, seed=seed + 2, max_nan_recoveries=50,
    )
    model_mixed.save_to_disk(str(out / 'grm_mixed'))
    calibrate_manually(model_mixed, n_samples=sample_size, seed=102)
    gc.collect()

    # ------------------------------------------------------------------
    # Compare abilities across models
    # ------------------------------------------------------------------
    print("\n=== Results ===")
    models = {
        'Baseline': model_baseline,
        'Pairwise': model_pairwise,
        'Mixed': model_mixed,
    }
    for label, mdl in models.items():
        ab = np.array(mdl.calibrated_expectations['abilities']).flatten()
        print(f"  {label}: ability mean={ab.mean():.3f}, std={ab.std():.3f}")

    print(f"\nAll artifacts saved to {out}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Example: IRT with IPW survey weights for group bias correction')
    parser.add_argument('--output-dir', default='ipw_example_results')
    args = parser.parse_args()
    run(output_dir=args.output_dir)
