#!/usr/bin/env python
"""Run only the MICE LOO + imputed GRM stages for a dataset.

Assumes the baseline GRM is already saved to grm_baseline/.
Fits MICE LOO, builds mixed imputation, fits imputed GRM.
Uses explicit gc.collect() between stages to minimize memory.

Usage:
    python run_mice_imputed.py --dataset rwa
    python run_mice_imputed.py --dataset rwa --skip-mice
"""

import argparse
import gc
import os
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import jax
import jax.numpy as jnp

# float32 with log_scale parameterization + STL for stability

DATASET_CONFIGS = {
    'grit': {'module': 'bayesianquilts.data.grit', 'n_top_features': 12},
    'rwa': {'module': 'bayesianquilts.data.rwa', 'n_top_features': 22},
    'npi': {'module': 'bayesianquilts.data.npi', 'n_top_features': 40},
    'tma': {'module': 'bayesianquilts.data.tma', 'n_top_features': 14},
    'wpi': {'module': 'bayesianquilts.data.wpi', 'n_top_features': 20},
    'eqsq': {'module': 'bayesianquilts.data.eqsq', 'n_top_features': 30},
}


def make_data_dict(dataframe):
    data = {}
    for col in dataframe.columns:
        data[col] = dataframe[col].to_numpy().astype(np.float32)
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
        # K=2 models may not have ddifficulties surrogate params
        print(f"  Warning: surrogate sampling failed ({e}), using point estimates")
        point_estimates = {}
        for key_name, value in model.params.items():
            parts = key_name.split('\\')
            if len(parts) >= 4:
                param_name = parts[0]
                if parts[-2] == 'normal' and parts[-1] == 'loc':
                    point_estimates[param_name] = value
        model.calibrated_expectations = point_estimates


def run_mice_imputed(dataset_name, work_dir, skip_mice=False,
                     num_epochs=200, batch_size=256, learning_rate=1e-3,
                     lr_decay_factor=0.975, sample_size=32, seed=42,
                     parameterization="log_scale"):
    import importlib
    from pathlib import Path
    from bayesianquilts.irt.grm import GRModel

    config = DATASET_CONFIGS[dataset_name]
    mod = importlib.import_module(config['module'])
    item_keys = mod.item_keys
    response_cardinality = mod.response_cardinality

    work_dir = Path(work_dir)
    os.chdir(work_dir)

    print(f"\n{'='*60}")
    print(f"MICE+Imputed for: {dataset_name.upper()}")
    print(f"{'='*60}")

    # Load data
    df, num_people = mod.get_data(polars_out=True)
    print(f"People: {num_people}, Items: {len(item_keys)}, K: {response_cardinality}")

    batch = make_data_dict(df)
    SUBSAMPLE_N = num_people
    steps_per_epoch = int(np.ceil(SUBSAMPLE_N / batch_size))

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

    # ---- Re-create baseline model from saved params ----
    print("\n--- Loading baseline GRM from disk ---")
    model_baseline = GRModel(
        item_keys=item_keys,
        num_people=SUBSAMPLE_N,
        dim=1,
        kappa_scale=0.1,
        response_cardinality=response_cardinality,
        dtype=jnp.float32,
        parameterization=parameterization,
    )
    array_data = GRModel._load_arrays_hdf5(work_dir / 'grm_baseline')
    for name, val in array_data.items():
        if hasattr(model_baseline, name):
            setattr(model_baseline, name, val)
    calibrate_manually(model_baseline, n_samples=32, seed=101)
    print("Baseline loaded and calibrated.")
    gc.collect()

    # ---- Stage: MICE LOO ----
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

    # ---- Stage: Mixed imputation ----
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

    # ---- Stage: Imputed GRM ----
    print("\n--- Fitting imputed GRM ---")
    model_imputed = GRModel(
        item_keys=item_keys,
        num_people=SUBSAMPLE_N,
        dim=1,
        kappa_scale=0.1,
        response_cardinality=response_cardinality,
        dtype=jnp.float32,
        imputation_model=mixed_imputation,
        parameterization=parameterization,
    )

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
        sample_size=sample_size,
        seed=seed,
    )
    losses_imputed = res_imputed[0]
    print(f"Imputed final loss: {losses_imputed[-1]:.2f}")
    model_imputed.save_to_disk('grm_imputed')

    print(f"\n{'='*60}")
    print(f"DONE: {dataset_name.upper()}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--skip-mice', action='store_true')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-decay-factor', type=float, default=0.975)
    parser.add_argument('--sample-size', type=int, default=32,
                        help='MC samples per ADVI gradient step (default 32)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible ADVI training')
    parser.add_argument('--parameterization', default='log_scale',
                        choices=['softplus', 'log_scale', 'natural'],
                        help='ADVI scale parameterization (default log_scale)')
    args = parser.parse_args()

    work_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        args.dataset,
    )
    run_mice_imputed(
        args.dataset, work_dir,
        skip_mice=args.skip_mice,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_decay_factor=args.lr_decay_factor,
        sample_size=args.sample_size,
        seed=args.seed,
        parameterization=args.parameterization,
    )


if __name__ == '__main__':
    main()
