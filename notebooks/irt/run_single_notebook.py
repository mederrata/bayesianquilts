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

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


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
}


def make_data_dict(dataframe):
    data = {}
    for col in dataframe.columns:
        arr = dataframe[col].to_numpy().astype(np.float64)
        data[col] = arr
    data['person'] = np.arange(len(dataframe), dtype=np.float64)
    return data


def calibrate_manually(model, n_samples=32, seed=42):
    surrogate = model.surrogate_distribution_generator(model.params)
    key = jax.random.PRNGKey(seed)
    samples = surrogate.sample(n_samples, seed=key)
    expectations = {k: jnp.mean(v, axis=0) for k, v in samples.items()}
    model.calibrated_expectations = expectations
    model.surrogate_sample = samples


def run_dataset(dataset_name, work_dir, skip_baseline=False, skip_mice=False,
                num_epochs=200, batch_size=256, learning_rate=2e-4):
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

    # Load data
    df, num_people = mod.get_data(polars_out=True)
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
        for start in range(0, SUBSAMPLE_N, batch_size):
            end = min(start + batch_size, SUBSAMPLE_N)
            idx_batch = indices[start:end]
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
            kappa_scale=0.1,
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

    # ---- Stage 2: MICE LOO ----
    from bayesianquilts.imputation.mice_loo import MICEBayesianLOO

    mice_path = work_dir / 'mice_loo_model.yaml'
    if skip_mice and mice_path.exists():
        print("\n--- Loading existing MICE LOO model ---")
        mice_loo = MICEBayesianLOO.load(str(mice_path))
        print("MICE LOO loaded.")
    else:
        print("\n--- Fitting MICE LOO ---")
        pandas_df = df.select(item_keys).to_pandas()
        pandas_df = pandas_df.replace(-1, np.nan)
        print(f"Missing values per item:\n{pandas_df.isna().sum()}")

        mice_loo = MICEBayesianLOO(
            random_state=42,
            prior_scale=1.0,
            pathfinder_num_samples=100,
            pathfinder_maxiter=50,
            batch_size=512,
            verbose=True,
        )
        mice_loo.fit_loo_models(
            pandas_df,
            n_top_features=config['n_top_features'],
            n_jobs=1,
            fit_zero_predictors=True,
            seed=42,
        )
        mice_loo.save(str(mice_path))
        print(f"MICE LOO saved to {mice_path}")

    gc.collect()

    # ---- Stage 3: Mixed imputation model ----
    from bayesianquilts.imputation.mixed import IrtMixedImputationModel

    print("\n--- Building mixed imputation model ---")
    mixed_imputation = IrtMixedImputationModel(
        irt_model=model_baseline,
        mice_model=mice_loo,
        data_factory=data_factory,
        irt_elpd_batch_size=4,
    )
    print(mixed_imputation.summary())

    gc.collect()

    # ---- Stage 4: Imputed GRM ----
    print("\n--- Fitting imputed GRM ---")
    model_imputed = GRModel(
        item_keys=item_keys,
        num_people=SUBSAMPLE_N,
        dim=1,
        kappa_scale=0.1,
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
        patience=10,
        zero_nan_grads=True,
        initial_values=snapshot_params,
    )
    losses_imputed = res_imputed[0]
    print(f"Imputed final loss: {losses_imputed[-1]:.2f}")
    model_imputed.save_to_disk('grm_imputed')
    calibrate_manually(model_imputed, n_samples=32, seed=102)

    print(f"\n{'='*60}")
    print(f"DONE: {dataset_name.upper()}")
    print(f"Artifacts: grm_baseline/, mice_loo_model.yaml, grm_imputed/")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Load existing baseline instead of re-fitting')
    parser.add_argument('--skip-mice', action='store_true',
                        help='Load existing MICE model instead of re-fitting')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-4)
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
    )


if __name__ == '__main__':
    main()
