#!/usr/bin/env python
"""Run a single IRT GRM notebook as a Python script.

Executes the same logic as grm_single_scale.ipynb but as a script,
with explicit gc.collect() between stages to avoid OOM.

Usage:
    python run_single_notebook.py --dataset rwa
    python run_single_notebook.py --dataset npi --skip-baseline
"""

import argparse
import gc
import os
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import jax
import jax.numpy as jnp

# float32 with log_scale parameterization + STL for stability


DATASET_CONFIGS = {
    'grit': {
        'module': 'bayesianquilts.data.grit',
        'n_top_features': 12,
    },
    'rwa': {
        'module': 'bayesianquilts.data.rwa',
        'n_top_features': 22,
    },
    'npi': {
        'module': 'bayesianquilts.data.npi',
        'n_top_features': 40,
    },
    'tma': {
        'module': 'bayesianquilts.data.tma',
        'n_top_features': 14,
    },
    'wpi': {
        'module': 'bayesianquilts.data.wpi',
        'n_top_features': 20,
    },
    'eqsq': {
        'module': 'bayesianquilts.data.eqsq',
        'n_top_features': 30,
    },
    'bouldering': {
        'module': 'bayesianquilts.data.bouldering',
        'n_top_features': 40,
    },
}


def make_data_dict(dataframe):
    data = {}
    for col in dataframe.columns:
        arr = dataframe[col].to_numpy().astype(np.float32)
        data[col] = arr
    data['person'] = np.arange(len(dataframe), dtype=np.float32)
    return data


def calibrate_manually(model, n_samples=32, seed=42):
    try:
        surrogate = model.surrogate_distribution_generator(model.params)
        key = jax.random.PRNGKey(seed)
        samples = surrogate.sample(n_samples, seed=key)
        expectations = {k: jnp.mean(v, axis=0) for k, v in samples.items()}
        model.calibrated_expectations = expectations
        model.surrogate_sample = samples
    except KeyError as e:
        print(f"  Warning: surrogate sampling failed ({e}), using point estimates")
        point_estimates = {}
        for key_name, value in model.params.items():
            parts = key_name.split('\\')
            if len(parts) >= 4:
                param_name = parts[0]
                if parts[-2] == 'normal' and parts[-1] == 'loc':
                    point_estimates[param_name] = value
        model.calibrated_expectations = point_estimates


def predictive_rmse(model, data_dict, item_keys, response_cardinality, loo=False, n_samples=100, seed=42):
    """Compute RMSE of E[Y] vs observed responses.

    If loo=True, computes LOO-RMSE using PSIS-weighted posterior samples:
    for each person, the prediction uses all other persons' data (via PSIS).
    """
    K = response_cardinality
    categories = jnp.arange(K, dtype=jnp.float64)

    if loo:
        # LOO-RMSE via PSIS reweighting
        from bayesianquilts.metrics.nppsis import psisloo
        surrogate = model.surrogate_distribution_generator(model.params)
        key = jax.random.PRNGKey(seed)
        samples = surrogate.sample(n_samples, seed=key)
        S = n_samples
        N = model.num_people
        I = len(item_keys)

        # Compute per-person log-likelihoods and response probs per sample
        # Use full dataset as one batch
        people = np.arange(N, dtype=np.int32)
        data_dict_full = dict(data_dict)
        data_dict_full['person'] = people.astype(np.float32)

        pred = model.predictive_distribution(data_dict_full, **samples)
        log_lik = np.array(pred['log_likelihood'])  # (S, N)

        # PSIS weights per person
        _, loos_pw, ks = psisloo(log_lik)

        # Compute expected responses per sample: need probs per sample
        # Use grm_model_prob_d with each sample's params
        disc = samples['discriminations']
        diff0 = samples['difficulties0']
        ddiff = samples.get('ddifficulties')
        abilities = samples['abilities']
        probs_all = model.grm_model_prob_d(abilities, disc, diff0, ddiff)
        # probs_all: (S, N, I, K)
        expected_per_sample = jnp.sum(probs_all * categories[None, None, None, :], axis=-1)  # (S, N, I)

        # PSIS LOO weights: for person n, downweight sample s proportional
        # to exp(-log_lik[s,n])
        log_ratios = -log_lik  # (S, N) — negative because we leave out person n
        # Normalize per person
        log_ratios = log_ratios - jnp.max(log_ratios, axis=0, keepdims=True)
        weights = jnp.exp(log_ratios)
        weights = weights / jnp.sum(weights, axis=0, keepdims=True)  # (S, N)

        # LOO expected response: weighted average over samples
        loo_expected = jnp.sum(weights[:, :, None] * expected_per_sample, axis=0)  # (N, I)

        se_sum = 0.0
        count = 0
        for i, key_name in enumerate(item_keys):
            obs = np.array(data_dict[key_name], dtype=np.float64)
            pred_i = np.array(loo_expected[:, i])
            valid = ~np.isnan(obs) & (obs >= 0) & (obs < K)
            se_sum += np.sum((obs[valid] - pred_i[valid]) ** 2)
            count += int(np.sum(valid))
        return float(np.sqrt(se_sum / count))
    else:
        # In-sample RMSE using calibrated expectations
        ce = model.calibrated_expectations
        probs = model.grm_model_prob_d(
            ce['abilities'], ce['discriminations'],
            ce['difficulties0'], ce.get('ddifficulties'))
        expected = jnp.sum(probs * categories[None, :], axis=-1)  # (N, I)

        se_sum = 0.0
        count = 0
        for i, key_name in enumerate(item_keys):
            obs = np.array(data_dict[key_name], dtype=np.float64)
            pred_i = np.array(expected[:, i])
            valid = ~np.isnan(obs) & (obs >= 0) & (obs < K)
            se_sum += np.sum((obs[valid] - pred_i[valid]) ** 2)
            count += int(np.sum(valid))
        return float(np.sqrt(se_sum / count))


def run_dataset(dataset_name, work_dir, skip_baseline=False, skip_mice=False,
                num_epochs=200, batch_size=256, learning_rate=1e-3,
                lr_decay_factor=0.975, sample_size=32, seed=42,
                parameterization="log_scale",
                discrimination_prior_scale=None,
                expected_sparsity=None,
                slab_scale=2.0,
                slab_df=4):
    import importlib
    from pathlib import Path

    config = DATASET_CONFIGS[dataset_name]
    mod = importlib.import_module(config['module'])
    item_keys = mod.item_keys
    response_cardinality = mod.response_cardinality

    work_dir = Path(work_dir)
    os.chdir(work_dir)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Working dir: {work_dir}")
    print(f"{'='*60}")

    # Load data (with reorientation if supported)
    import inspect
    get_data_kwargs = {'polars_out': True}
    if 'reorient' in inspect.signature(mod.get_data).parameters:
        get_data_kwargs['reorient'] = True
    df, num_people = mod.get_data(**get_data_kwargs)
    print(f"People: {num_people}, Items: {len(item_keys)}, K: {response_cardinality}")

    batch = make_data_dict(df)
    n_bad = sum(
        np.sum(np.isnan(batch[k]) | (batch[k] < 0) | (batch[k] >= response_cardinality))
        for k in item_keys
    )
    print(f"Bad/missing values: {n_bad}")

    SUBSAMPLE_N = num_people
    steps_per_epoch = int(np.ceil(SUBSAMPLE_N / batch_size))
    SNAPSHOT_EPOCH = 50

    def data_factory():
        indices = np.arange(SUBSAMPLE_N)
        np.random.shuffle(indices)
        n_needed = steps_per_epoch * batch_size
        if n_needed > SUBSAMPLE_N:
            indices = np.concatenate([
                indices,
                np.random.choice(SUBSAMPLE_N, n_needed - SUBSAMPLE_N, replace=True),
            ])
        for start in range(0, n_needed, batch_size):
            idx_batch = indices[start:start + batch_size]
            yield {k: v[idx_batch] for k, v in batch.items()}

    # ---- Stage 1: Pairwise Ordinal Stacking imputation ----
    # Fit imputation model first (independent of GRM, lets us fail fast)
    from bayesianquilts.imputation.pairwise_stacking import PairwiseOrdinalStackingModel

    mice_path = work_dir / 'pairwise_stacking_model.yaml'
    if skip_mice and mice_path.exists():
        print("\n--- Loading existing pairwise stacking model ---")
        pairwise_model = PairwiseOrdinalStackingModel.load(str(mice_path))
        print("Pairwise stacking model loaded.")
    else:
        print("\n--- Fitting pairwise stacking imputation model ---")
        pandas_df = df.select(item_keys).to_pandas()
        pandas_df = pandas_df.replace(-1, np.nan)
        print(f"Missing values per item:\n{pandas_df.isna().sum()}")

        pairwise_model = PairwiseOrdinalStackingModel(
            prior_scale=1.0,
            pathfinder_num_samples=100,
            pathfinder_maxiter=50,
            batch_size=512,
            verbose=True,
        )
        pairwise_model.fit(
            pandas_df,
            n_top_features=config['n_top_features'],
            n_jobs=1,
            seed=42,
        )
        pairwise_model.save(str(mice_path))
        print(f"Pairwise stacking model saved to {mice_path}")

    gc.collect()

    # ---- Stage 2: Baseline GRM ----
    from bayesianquilts.irt.grm import GRModel

    snapshot_params = None
    if skip_baseline and (work_dir / 'grm_baseline' / 'params.h5').exists():
        print("\n--- Loading existing baseline GRM ---")
        model_baseline = GRModel.load_from_disk(work_dir / 'grm_baseline')
        calibrate_manually(model_baseline, n_samples=32, seed=101)
        print("Baseline loaded.")
    else:
        print("\n--- Fitting baseline GRM ---")
        model_baseline = GRModel(
            item_keys=item_keys,
            num_people=SUBSAMPLE_N,
            dim=1,
            response_cardinality=response_cardinality,
            dtype=jnp.float32,
            parameterization=parameterization,
            discrimination_prior_scale=discrimination_prior_scale,
            expected_sparsity=expected_sparsity,
            slab_scale=slab_scale,
            slab_df=slab_df,
        )
        res_baseline = model_baseline.fit(
            data_factory,
            batch_size=batch_size,
            dataset_size=SUBSAMPLE_N,
            num_epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            learning_rate=learning_rate,
            lr_decay_factor=lr_decay_factor,
            patience=10,
            zero_nan_grads=True,
            snapshot_epoch=SNAPSHOT_EPOCH,
            sample_size=sample_size,
            seed=seed,
            max_nan_recoveries=50,
        )
        losses_baseline = res_baseline[0]
        snapshot_params = res_baseline[2] if len(res_baseline) > 2 else None
        if losses_baseline:
            print(f"Baseline final loss: {losses_baseline[-1]:.2f} ({len(losses_baseline)} epochs)")
        else:
            print("WARNING: Baseline training returned no epochs")
        model_baseline.save_to_disk('grm_baseline')
        calibrate_manually(model_baseline, n_samples=32, seed=101)

    # Count observed responses for per-response ELPD
    n_items = len(item_keys)
    n_observed_responses = sum(
        np.sum((batch[k] >= 0) & (batch[k] < response_cardinality) & ~np.isnan(batch[k]))
        for k in item_keys
    )
    print(f"  Total observed responses: {n_observed_responses} "
          f"({n_observed_responses / SUBSAMPLE_N:.1f} per person, {n_items} items)")

    gc.collect()

    # ---- Stage 3: Build mixed imputation model ----
    from bayesianquilts.imputation.mixed import IrtMixedImputationModel
    import json

    print("\n--- Building mixed imputation model ---")
    mixed_imputation = IrtMixedImputationModel(
        irt_model=model_baseline,
        mice_model=pairwise_model,
        data_factory=data_factory,
        irt_elpd_batch_size=4,
    )
    print(mixed_imputation.summary())

    weights_path = work_dir / 'mixed_weights.json'
    with open(weights_path, 'w') as f:
        json.dump(mixed_imputation.weights, f, indent=2)
    print(f"Saved mixed weights to {weights_path}")

    gc.collect()

    # ---- Stage 3b: Compute adaptive ignorability thresholds ----
    # Temporarily set mixed model on baseline to compute thresholds.
    # Items where w_pairwise < threshold are treated as ignorable during
    # subsequent GRM training (Rao-Blackwell variance reduction).
    print("\n--- Computing adaptive ignorability thresholds ---")
    model_baseline.imputation_model = mixed_imputation
    calibrate_manually(model_baseline, n_samples=sample_size, seed=101)
    model_baseline.compute_adaptive_thresholds(
        data_factory, sample_size=sample_size, seed=seed)
    adaptive_thresholds = model_baseline._adaptive_thresholds
    model_baseline.imputation_model = None  # restore baseline to no-imputation

    # Save thresholds
    thresholds_path = work_dir / 'adaptive_thresholds.json'
    with open(thresholds_path, 'w') as f:
        json.dump(adaptive_thresholds, f, indent=2)
    print(f"Saved adaptive thresholds to {thresholds_path}")

    gc.collect()

    # ---- Stage 4: Pairwise-only GRM (with adaptive thresholds) ----
    print("\n--- Fitting pairwise-only GRM ---")
    model_mice_only = GRModel(
        item_keys=item_keys,
        num_people=SUBSAMPLE_N,
        dim=1,
        response_cardinality=response_cardinality,
        dtype=jnp.float32,
        imputation_model=pairwise_model,
        parameterization=parameterization,
        discrimination_prior_scale=discrimination_prior_scale,
        expected_sparsity=expected_sparsity,
        slab_scale=slab_scale,
        slab_df=slab_df,
    )
    model_mice_only._adaptive_thresholds = adaptive_thresholds
    ignored_pw = [k for k, t in adaptive_thresholds.items() if t >= 1.0]
    imputed_pw = [k for k in item_keys if k not in ignored_pw]
    print(f"  Pairwise-only: imputing {len(imputed_pw)}/{n_items}, "
          f"ignoring {len(ignored_pw)}/{n_items} items' missing values")
    if ignored_pw:
        print(f"    Ignored: {', '.join(ignored_pw)}")
    if imputed_pw:
        print(f"    Imputed: {', '.join(imputed_pw)}")
    res_mice = model_mice_only.fit(
        data_factory,
        batch_size=batch_size,
        dataset_size=SUBSAMPLE_N,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        learning_rate=learning_rate,
        lr_decay_factor=lr_decay_factor,
        patience=30,
        zero_nan_grads=True,
        sample_size=sample_size,
        seed=seed + 1,
        max_nan_recoveries=50,
    )
    losses_mice = res_mice[0]
    if losses_mice:
        print(f"Pairwise-only final loss: {losses_mice[-1]:.2f} ({len(losses_mice)} epochs)")
    else:
        print("WARNING: Pairwise-only training returned no epochs")
    model_mice_only.save_to_disk('grm_mice_only')
    calibrate_manually(model_mice_only, n_samples=32, seed=103)

    gc.collect()

    # ---- Stage 5: Mixed-imputed GRM (with adaptive thresholds) ----
    print("\n--- Fitting mixed-imputed GRM ---")
    model_imputed = GRModel(
        item_keys=item_keys,
        num_people=SUBSAMPLE_N,
        dim=1,
        response_cardinality=response_cardinality,
        dtype=jnp.float32,
        imputation_model=mixed_imputation,
        parameterization=parameterization,
        discrimination_prior_scale=discrimination_prior_scale,
        expected_sparsity=expected_sparsity,
        slab_scale=slab_scale,
        slab_df=slab_df,
    )
    model_imputed._adaptive_thresholds = adaptive_thresholds
    ignored_mixed = [k for k in item_keys
                     if mixed_imputation.get_item_weight(k) <= adaptive_thresholds.get(k, 0.0)]
    imputed_mixed = [k for k in item_keys if k not in ignored_mixed]
    print(f"  Mixed: imputing {len(imputed_mixed)}/{n_items}, "
          f"ignoring {len(ignored_mixed)}/{n_items} items' missing values")
    if ignored_mixed:
        print(f"    Ignored: {', '.join(ignored_mixed)}")
    if imputed_mixed:
        print(f"    Imputed: {', '.join(imputed_mixed)}")
    res_imputed = model_imputed.fit(
        data_factory,
        batch_size=batch_size,
        dataset_size=SUBSAMPLE_N,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        learning_rate=learning_rate,
        lr_decay_factor=lr_decay_factor,
        patience=30,
        zero_nan_grads=True,
        sample_size=sample_size,
        seed=seed + 1,
        max_nan_recoveries=50,
    )
    losses_imputed = res_imputed[0]
    if losses_imputed:
        print(f"Imputed final loss: {losses_imputed[-1]:.2f} ({len(losses_imputed)} epochs)")
    else:
        print("WARNING: Imputed training returned no epochs")
    model_imputed.save_to_disk('grm_imputed')
    calibrate_manually(model_imputed, n_samples=32, seed=102)

    gc.collect()

    # ---- ELPD-LOO for all models (after all fitting is done) ----
    for label, mdl, seed_val in [('Baseline', model_baseline, 101),
                                  ('Pairwise-only', model_mice_only, 103),
                                  ('Mixed', model_imputed, 102)]:
        print(f"\n--- Computing ELPD-LOO for {label} ---")
        try:
            mdl._compute_elpd_loo(data_factory, n_samples=100, seed=seed_val, use_ais=True)
            elpd_val = mdl.elpd_loo
            elpd_se = mdl.elpd_loo_se
            n_obs = getattr(mdl, 'elpd_loo_n_obs', SUBSAMPLE_N)
            print(f"    ELPD/person: {elpd_val/n_obs:.4f} ± {elpd_se/n_obs:.4f}")
            print(f"    ELPD/response: {elpd_val/n_observed_responses:.4f} ± {elpd_se/n_observed_responses:.4f}")
        except Exception as e:
            print(f"  ELPD-LOO failed for {label}: {e}")
        gc.collect()

    # ---- Summary ----
    import json as _json

    print(f"\n{'='*100}")
    print(f"SUMMARY: {dataset_name.upper()}")
    print(f"{'='*100}")
    print(f"{'Model':<15} {'Pred RMSE':>10} {'LOO-RMSE':>10} {'ELPD/person':>20} {'ELPD/response':>20}")
    print(f"{'-'*100}")

    summary_results = {
        'dataset': dataset_name,
        'num_people': SUBSAMPLE_N,
        'num_items': n_items,
        'response_cardinality': response_cardinality,
        'n_observed_responses': int(n_observed_responses),
        'adaptive_thresholds': adaptive_thresholds,
        'ignorability': {
            'pairwise_only': {
                'ignored': ignored_pw,
                'imputed': imputed_pw,
            },
            'mixed': {
                'ignored': ignored_mixed,
                'imputed': imputed_mixed,
            },
        },
        'models': {},
    }

    for label, mdl in [('Baseline', model_baseline),
                        ('Pairwise-only', model_mice_only),
                        ('Mixed', model_imputed)]:
        entry = {}
        elpd = getattr(mdl, 'elpd_loo', None)
        se = getattr(mdl, 'elpd_loo_se', None)
        n = getattr(mdl, 'elpd_loo_n_obs', SUBSAMPLE_N)
        if elpd is not None and not np.isnan(elpd):
            pp = f"{elpd/n:.4f} ± {se/n:.4f}"
            pr = f"{elpd/n_observed_responses:.4f} ± {se/n_observed_responses:.4f}"
            entry['elpd_loo'] = float(elpd)
            entry['elpd_loo_se'] = float(se)
            entry['elpd_per_person'] = float(elpd / n)
            entry['elpd_per_response'] = float(elpd / n_observed_responses)
            entry['n_obs'] = int(n)
            khat = getattr(mdl, 'elpd_loo_khat', None)
            if khat is not None:
                entry['max_khat'] = float(np.max(khat))
                entry['mean_khat'] = float(np.mean(khat))
                entry['n_high_khat'] = int(np.sum(khat > 0.7))
        else:
            pp = "nan"
            pr = "nan"
        try:
            rmse = predictive_rmse(mdl, batch, item_keys, response_cardinality)
            rmse_str = f"{rmse:.4f}"
            entry['pred_rmse'] = float(rmse)
        except Exception:
            rmse_str = "nan"
        try:
            loo_rmse = predictive_rmse(mdl, batch, item_keys, response_cardinality, loo=True, n_samples=100)
            loo_rmse_str = f"{loo_rmse:.4f}"
            entry['loo_rmse'] = float(loo_rmse)
        except Exception as e:
            print(f"  LOO-RMSE failed for {label}: {e}")
            loo_rmse_str = "nan"
        print(f"{label:<15} {rmse_str:>10} {loo_rmse_str:>10} {pp:>20} {pr:>20}")
        summary_results['models'][label] = entry

    print(f"{'='*100}")
    print(f"Artifacts: grm_baseline/, grm_mice_only/, grm_imputed/")
    print(f"{'='*100}")

    # Save results JSON
    results_path = work_dir / 'results.json'
    with open(results_path, 'w') as f:
        _json.dump(summary_results, f, indent=2)
    print(f"Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Load existing baseline instead of re-fitting')
    parser.add_argument('--skip-mice', action='store_true',
                        help='Load existing MICE model instead of re-fitting')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr-decay-factor', type=float, default=0.975)
    parser.add_argument('--sample-size', type=int, default=32,
                        help='MC samples per ADVI gradient step (default 32)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible ADVI training')
    parser.add_argument('--parameterization', default='log_scale',
                        choices=['softplus', 'log_scale', 'natural'],
                        help='ADVI scale parameterization (default log_scale)')
    parser.add_argument('--discrimination-prior-scale', type=float, default=1.0,
                        help='Scale for half_normal/half_cauchy discrimination prior (default 1.0)')
    parser.add_argument('--expected-sparsity', type=float, default=None,
                        help='Expected fraction of relevant items (e.g. 0.1 = 1/10)')
    parser.add_argument('--slab-scale', type=float, default=2.0,
                        help='Slab scale for regularized horseshoe (default 2.0)')
    parser.add_argument('--slab-df', type=int, default=4,
                        help='Slab degrees of freedom for regularized horseshoe (default 4)')
    args = parser.parse_args()

    work_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        args.dataset,
    )

    run_dataset(
        args.dataset,
        work_dir,
        skip_baseline=args.skip_baseline,
        skip_mice=args.skip_mice,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_decay_factor=args.lr_decay_factor,
        sample_size=args.sample_size,
        seed=args.seed,
        parameterization=args.parameterization,
        discrimination_prior_scale=args.discrimination_prior_scale,
        expected_sparsity=args.expected_sparsity,
        slab_scale=args.slab_scale,
        slab_df=args.slab_df,
    )


if __name__ == '__main__':
    main()
