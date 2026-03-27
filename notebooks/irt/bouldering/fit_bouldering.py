#!/usr/bin/env python
"""Fit GRM pipeline on IFSC bouldering data.

Runs the same four-stage pipeline as the psychometric notebooks:
  1. Baseline GRM (ignorable missingness)
  2. PairwiseOrdinalStackingModel imputation model
  3. IrtMixedImputationModel (blends MICE + IRT baseline)
  4. Imputed GRM (warm-started from baseline)

Usage:
    python fit_bouldering.py [--skip-baseline] [--skip-mice]
    python fit_bouldering.py --gender women
"""

import argparse
import gc
import json
import os
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def make_data_dict(dataframe):
    data = {}
    for col in dataframe.columns:
        arr = dataframe[col].to_numpy().astype(np.float64)
        data[col] = arr
    data['person'] = np.arange(len(dataframe), dtype=np.float64)
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


def run_bouldering(work_dir, gender='men', skip_baseline=False, skip_mice=False,
                   num_epochs=200, batch_size=256, learning_rate=2e-4,
                   lr_decay_factor=0.975):
    import importlib
    from pathlib import Path

    # Load bouldering data
    from bayesianquilts.data.bouldering import get_data, item_keys, response_cardinality, item_labels

    work_dir = Path(work_dir)
    os.chdir(work_dir)

    print(f"\n{'='*60}")
    print(f"IFSC Bouldering ({gender})")
    print(f"Working dir: {work_dir}")
    print(f"{'='*60}")

    df, num_people = get_data(polars_out=True, gender=gender)
    n_items = len(item_keys)
    print(f"People: {num_people}, Items: {n_items}, K: {response_cardinality}")

    batch = make_data_dict(df)
    n_bad = sum(
        np.sum(np.isnan(batch[k]) | (batch[k] < 0) | (batch[k] >= response_cardinality))
        for k in item_keys
    )
    print(f"Missing values: {n_bad}")

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

    # ---- Stage 1: Baseline GRM ----
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
            dtype=jnp.float64,
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
        )
        losses_baseline = res_baseline[0]
        snapshot_params = res_baseline[2] if len(res_baseline) > 2 else None
        print(f"Baseline final loss: {losses_baseline[-1]:.2f}")
        model_baseline.save_to_disk('grm_baseline')
        calibrate_manually(model_baseline, n_samples=32, seed=101)

    gc.collect()

    # ---- Stage 2: Pairwise Ordinal Stacking ----
    from bayesianquilts.imputation.pairwise_stacking import PairwiseOrdinalStackingModel

    mice_path = work_dir / 'pairwise_stacking_model.yaml'
    # Limit top features given the large item count
    n_top_features = min(n_items, 40)

    if skip_mice and mice_path.exists():
        print("\n--- Loading existing pairwise stacking model ---")
        pairwise_model = PairwiseOrdinalStackingModel.load(str(mice_path))
        print("Pairwise stacking model loaded.")
    else:
        print(f"\n--- Fitting pairwise stacking model (n_top_features={n_top_features}) ---")
        pandas_df = df.select(item_keys).to_pandas()
        pandas_df = pandas_df.replace(-1, np.nan)
        n_missing = pandas_df.isna().sum().sum()
        print(f"Total missing values: {n_missing}")

        pairwise_model = PairwiseOrdinalStackingModel(
            prior_scale=1.0,
            pathfinder_num_samples=100,
            pathfinder_maxiter=50,
            batch_size=512,
            verbose=True,
        )
        pairwise_model.fit(
            pandas_df,
            n_top_features=n_top_features,
            n_jobs=1,
            seed=42,
        )
        pairwise_model.save(str(mice_path))
        print(f"Pairwise stacking model saved to {mice_path}")

    gc.collect()

    # ---- Stage 3: Mixed imputation model ----
    from bayesianquilts.imputation.mixed import IrtMixedImputationModel

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

    # ---- Stage 4: Imputed GRM ----
    print("\n--- Fitting imputed GRM ---")
    model_imputed = GRModel(
        item_keys=item_keys,
        num_people=SUBSAMPLE_N,
        dim=1,
        response_cardinality=response_cardinality,
        dtype=jnp.float64,
        imputation_model=mixed_imputation,
    )

    if snapshot_params is not None:
        print(f"Warm-starting from baseline epoch-{SNAPSHOT_EPOCH} snapshot")

    res_imputed = model_imputed.fit(
        data_factory,
        batch_size=batch_size,
        dataset_size=SUBSAMPLE_N,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        learning_rate=learning_rate,
        lr_decay_factor=lr_decay_factor,
        patience=10,
        zero_nan_grads=True,
        initial_values=snapshot_params,
    )
    losses_imputed = res_imputed[0]
    print(f"Imputed final loss: {losses_imputed[-1]:.2f}")
    model_imputed.save_to_disk('grm_imputed')
    calibrate_manually(model_imputed, n_samples=32, seed=102)

    # ---- Save item label mapping ----
    with open(work_dir / 'item_labels.json', 'w') as f:
        json.dump(item_labels, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE: Bouldering ({gender})")
    print(f"Artifacts: grm_baseline/, pairwise_stacking_model.yaml, grm_imputed/")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Fit GRM pipeline on IFSC bouldering data."
    )
    parser.add_argument('--gender', default='men', choices=['men', 'women'])
    parser.add_argument('--skip-baseline', action='store_true')
    parser.add_argument('--skip-mice', action='store_true')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr-decay-factor', type=float, default=0.975)
    args = parser.parse_args()

    work_dir = os.path.dirname(os.path.abspath(__file__))

    run_bouldering(
        work_dir,
        gender=args.gender,
        skip_baseline=args.skip_baseline,
        skip_mice=args.skip_mice,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_decay_factor=args.lr_decay_factor,
    )


if __name__ == '__main__':
    main()
