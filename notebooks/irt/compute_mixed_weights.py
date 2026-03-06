#!/usr/bin/env python
"""Compute mixed imputation weights from existing fitted models.

Loads a fitted baseline GRM and MICE model, constructs the
IrtMixedImputationModel to compute per-item weights, and saves
them as mixed_weights.json.

Does NOT refit any models — only loads from disk and computes weights.

Usage:
    python compute_mixed_weights.py --dataset grit
    python compute_mixed_weights.py --all
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

DATASET_CONFIGS = {
    'grit': {'module': 'bayesianquilts.data.grit'},
    'rwa': {'module': 'bayesianquilts.data.rwa'},
    'npi': {'module': 'bayesianquilts.data.npi'},
    'tma': {'module': 'bayesianquilts.data.tma'},
    'wpi': {'module': 'bayesianquilts.data.wpi'},
}


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


def compute_weights(dataset_name, batch_size=256):
    import importlib
    from pathlib import Path

    config = DATASET_CONFIGS[dataset_name]
    mod = importlib.import_module(config['module'])
    item_keys = mod.item_keys
    response_cardinality = mod.response_cardinality

    work_dir = Path(__file__).parent / dataset_name
    baseline_dir = work_dir / 'grm_baseline'
    mice_path = work_dir / 'mice_loo_model.yaml'

    if not baseline_dir.exists():
        print(f"SKIP {dataset_name}: {baseline_dir} not found")
        return None
    if not mice_path.exists():
        print(f"SKIP {dataset_name}: {mice_path} not found")
        return None

    # Load data
    df, num_people = mod.get_data(polars_out=True)
    batch = make_data_dict(df)

    def data_factory():
        indices = np.arange(num_people)
        np.random.shuffle(indices)
        for start in range(0, num_people, batch_size):
            end = min(start + batch_size, num_people)
            idx_batch = indices[start:end]
            yield {k: v[idx_batch] for k, v in batch.items()}

    # Load baseline GRM
    from bayesianquilts.irt.grm import GRModel
    import yaml as _yaml
    print(f"Loading baseline GRM for {dataset_name}...")
    # Fix kappa_scale serialization: flatten nested list to scalar before loading
    cfg_path = baseline_dir / 'config.yaml'
    with open(cfg_path) as _f:
        _cfg = _yaml.safe_load(_f)
    ks = _cfg.get('kappa_scale', 0.1)
    while isinstance(ks, list):
        ks = ks[0]
    if ks != _cfg.get('kappa_scale'):
        _cfg['kappa_scale'] = ks
        with open(cfg_path, 'w') as _f:
            _yaml.dump(_cfg, _f)
    model_baseline = GRModel.load_from_disk(baseline_dir)
    calibrate_manually(model_baseline, n_samples=32, seed=101)

    # Load MICE model
    from bayesianquilts.imputation.mice_loo import MICEBayesianLOO
    print(f"Loading MICE model for {dataset_name}...")
    mice_loo = MICEBayesianLOO.load(str(mice_path))

    # Compute mixed weights
    from bayesianquilts.imputation.mixed import IrtMixedImputationModel
    print(f"Computing mixed weights for {dataset_name}...")
    mixed = IrtMixedImputationModel(
        irt_model=model_baseline,
        mice_model=mice_loo,
        data_factory=data_factory,
        irt_elpd_batch_size=4,
    )
    print(mixed.summary())

    weights = mixed.weights
    out_path = work_dir / 'mixed_weights.json'
    with open(out_path, 'w') as f:
        json.dump(weights, f, indent=2)
    print(f"Saved {len(weights)} weights to {out_path}")

    gc.collect()
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        datasets = list(DATASET_CONFIGS.keys())
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.error("Specify --dataset or --all")

    for ds in datasets:
        compute_weights(ds)
        gc.collect()


if __name__ == "__main__":
    main()
