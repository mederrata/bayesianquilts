#!/usr/bin/env python
"""Reweight baseline MCMC samples to approximate imputed posteriors via IS.

For each dataset with baseline MCMC samples, loads the pairwise stacking
model and uses importance_reweight() to approximate the imputed posterior
without refitting. Reports diagnostics (ESS, k-hat) and compares
reweighted discriminations/difficulties against directly fitted variants.

Supports both GRModel (single-scale) and FactorizedGRModel (multi-domain).

Usage:
    cd notebooks/irt/
    uv run python reweight_baseline_mcmc.py --dataset scs
    uv run python reweight_baseline_mcmc.py --dataset promis_neuropathic_pain
    uv run python reweight_baseline_mcmc.py --dataset all
"""

import os
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

import argparse
import importlib
import inspect
import numpy as np
import jax.numpy as jnp
from pathlib import Path

from run_single_notebook import DATASET_CONFIGS, make_data_dict


# Datasets that use FactorizedGRModel with get_multidomain_data
FACTORIZED_DATASETS = {
    'promis_neuropathic_pain': {
        'module': 'bayesianquilts.data.promis_neuropathic_pain',
        'loader': 'get_multidomain_data',
        'loader_kwargs': {'min_items': 10},
    },
    'promis_copd': {
        'module': 'bayesianquilts.data.promis_copd',
        'loader': 'get_multidomain_data',
        'loader_kwargs': {'min_items': 10},
    },
}


def _load_model(work_dir):
    """Load the baseline MCMC model, auto-detecting GRM vs FactorizedGRM."""
    from bayesianquilts.irt.grm import GRModel
    from bayesianquilts.irt.factorizedgrm import FactorizedGRModel

    baseline_path = work_dir / 'grm_mcmc_baseline'
    if not (baseline_path / 'params.h5').exists():
        baseline_path = work_dir / 'grm_mcmc'
        if not (baseline_path / 'params.h5').exists():
            return None, None

    # Check config to determine model type
    import yaml
    config_path = baseline_path / 'config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    class_name = config.get('_class_name', 'GRModel')
    if class_name == 'FactorizedGRModel':
        model = FactorizedGRModel.load_from_disk(baseline_path)
    else:
        model = GRModel.load_from_disk(baseline_path)

    return model, baseline_path


def _load_data(dataset_name):
    """Load data, handling both standard and factorized datasets."""
    if dataset_name in FACTORIZED_DATASETS:
        fconfig = FACTORIZED_DATASETS[dataset_name]
        mod = importlib.import_module(fconfig['module'])
        loader = getattr(mod, fconfig['loader'])
        kwargs = {'polars_out': True, **fconfig.get('loader_kwargs', {})}
        result = loader(**kwargs)
        df, num_people, scale_indices = result
        item_keys = mod.item_keys
        response_cardinality = mod.response_cardinality
        return df, num_people, item_keys, response_cardinality, scale_indices
    elif dataset_name in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_name]
        mod = importlib.import_module(config['module'])
        get_data_kwargs = {'polars_out': True}
        if 'reorient' in inspect.signature(mod.get_data).parameters:
            get_data_kwargs['reorient'] = True
        df, num_people = mod.get_data(**get_data_kwargs)
        item_keys = mod.item_keys
        response_cardinality = mod.response_cardinality
        return df, num_people, item_keys, response_cardinality, None
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _make_disc_fn(model):
    """Return a function that extracts all discriminations from a param dict."""
    from bayesianquilts.irt.factorizedgrm import FactorizedGRModel
    if isinstance(model, FactorizedGRModel):
        def fn(p):
            parts = []
            for d in range(len(model.scale_indices)):
                key = f'discriminations_{d}'
                if key in p:
                    parts.append(np.array(p[key]).flatten())
            return np.concatenate(parts)
        return fn
    else:
        return lambda p: np.array(p['discriminations']).flatten()


def run_dataset(dataset_name):
    from bayesianquilts.imputation.pairwise_stacking import PairwiseOrdinalStackingModel

    work_dir = Path(__file__).parent / dataset_name
    os.chdir(work_dir)

    # Load data
    df, num_people, item_keys, response_cardinality, scale_indices = \
        _load_data(dataset_name)
    data = make_data_dict(df)

    is_factorized = scale_indices is not None
    model_type = "FactorizedGRM" if is_factorized else "GRM"

    print(f"\n{'='*70}")
    print(f"  {dataset_name.upper()}: N={num_people}, I={len(item_keys)}, "
          f"K={response_cardinality}, {model_type}")
    if is_factorized:
        for dname, idx in scale_indices.items():
            print(f"    {dname}: {len(idx)} items")
    print(f"{'='*70}")

    # Load baseline MCMC model
    model, baseline_path = _load_model(work_dir)
    if model is None:
        print(f"  No baseline MCMC model found, skipping")
        return None

    # Load baseline MCMC samples
    npz_path = work_dir / 'mcmc_samples' / 'mcmc_baseline.npz'
    if not npz_path.exists():
        npz_path = work_dir / 'mcmc_samples' / 'mcmc_item_params.npz'
    if not npz_path.exists():
        print(f"  No baseline MCMC samples found, skipping")
        return None

    mcmc_data = np.load(str(npz_path))
    mcmc_samples = {}
    item_vars = model._item_var_list()
    for key in item_vars:
        if key in mcmc_data:
            mcmc_samples[key] = mcmc_data[key]
    print(f"  Loaded {len(mcmc_samples)} param arrays from {npz_path.name}")
    for k, v in mcmc_samples.items():
        print(f"    {k}: {v.shape}")

    if not mcmc_samples:
        print(f"  No matching param arrays found!")
        print(f"  Model expects: {item_vars}")
        print(f"  NPZ has: {list(mcmc_data.keys())}")
        return None

    # Load pairwise stacking model
    stacking_path = work_dir / 'pairwise_stacking_model.yaml'
    if not stacking_path.exists():
        print(f"  No pairwise stacking model, skipping")
        return None

    pairwise_model = PairwiseOrdinalStackingModel.load(str(stacking_path))
    print(f"  Loaded pairwise stacking model")

    # --- IS Reweighting: discriminations ---
    disc_fn = _make_disc_fn(model)
    print(f"\n  --- Reweighting for discriminations ---")
    disc_result = model.importance_reweight(
        data=data,
        mcmc_samples=mcmc_samples,
        imputation_model=pairwise_model,
        fn=disc_fn,
        max_samples=200,
        seed=42,
        verbose=True,
    )

    print(f"\n  IS Discriminations (reweighted):")
    if disc_result['expectation'] is not None:
        disc_is = disc_result['expectation']
        print(f"    Mean: {np.mean(disc_is):.4f}")
        if len(disc_is) <= 50:
            print(f"    Per-item: {np.round(disc_is, 3)}")

    # Compare against direct pairwise MCMC if available
    pw_path = work_dir / 'grm_mcmc_pairwise'
    if (pw_path / 'params.h5').exists():
        from bayesianquilts.irt.factorizedgrm import FactorizedGRModel
        if isinstance(model, FactorizedGRModel):
            pw_model = FactorizedGRModel.load_from_disk(pw_path)
        else:
            from bayesianquilts.irt.grm import GRModel
            pw_model = GRModel.load_from_disk(pw_path)

        from run_single_notebook import calibrate_manually
        calibrate_manually(pw_model, n_samples=32, seed=101)
        disc_direct = _make_disc_fn(pw_model)(pw_model.calibrated_expectations)

        print(f"\n  Direct MCMC Pairwise discriminations:")
        print(f"    Mean: {np.mean(disc_direct):.4f}")
        if len(disc_direct) <= 50:
            print(f"    Per-item: {np.round(disc_direct, 3)}")

        if disc_result['expectation'] is not None:
            diff = disc_is - disc_direct
            print(f"\n  IS vs Direct difference:")
            print(f"    Mean abs diff: {np.mean(np.abs(diff)):.4f}")
            print(f"    Max abs diff:  {np.max(np.abs(diff)):.4f}")
            corr = np.corrcoef(disc_is, disc_direct)[0, 1]
            print(f"    Correlation:   {corr:.6f}")

    # --- Summary ---
    print(f"\n  {'='*50}")
    print(f"  Summary for {dataset_name.upper()}")
    print(f"  {'='*50}")
    print(f"  k-hat:    {disc_result['khat']:.3f} "
          f"({'OK' if disc_result['khat'] < 0.7 else 'WARNING'})")
    print(f"  ESS:      {disc_result['ess']:.1f}/{disc_result['n_samples']} "
          f"({100*disc_result['ess']/disc_result['n_samples']:.1f}%)")
    print(f"  Tempered: {disc_result['tempered']}")

    return disc_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    all_datasets = (
        list(DATASET_CONFIGS.keys()) + list(FACTORIZED_DATASETS.keys())
    )
    datasets = all_datasets if args.dataset == 'all' else [args.dataset]

    for ds in datasets:
        if ds not in DATASET_CONFIGS and ds not in FACTORIZED_DATASETS:
            print(f"Unknown dataset: {ds}")
            continue
        try:
            run_dataset(ds)
        except Exception as e:
            import traceback
            print(f"ERROR on {ds}: {e}")
            traceback.print_exc()


if __name__ == '__main__':
    main()
