#!/usr/bin/env python
"""Reweight baseline MCMC samples to approximate imputed posteriors via IS.

For each dataset with baseline MCMC samples, loads the pairwise stacking
model and uses importance_reweight() to approximate the imputed posterior
without refitting. Reports diagnostics (ESS, k-hat) and compares
reweighted discriminations/difficulties against directly fitted variants.

Usage:
    cd notebooks/irt/
    uv run python reweight_baseline_mcmc.py --dataset scs
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


def run_dataset(dataset_name):
    from bayesianquilts.irt.grm import GRModel
    from bayesianquilts.imputation.pairwise_stacking import PairwiseOrdinalStackingModel

    config = DATASET_CONFIGS[dataset_name]
    mod = importlib.import_module(config['module'])
    item_keys = mod.item_keys
    response_cardinality = mod.response_cardinality

    work_dir = Path(__file__).parent / dataset_name
    os.chdir(work_dir)

    # Load data
    get_data_kwargs = {'polars_out': True}
    if 'reorient' in inspect.signature(mod.get_data).parameters:
        get_data_kwargs['reorient'] = True
    df, num_people = mod.get_data(**get_data_kwargs)
    data = make_data_dict(df)

    print(f"\n{'='*70}")
    print(f"  {dataset_name.upper()}: N={num_people}, I={len(item_keys)}, K={response_cardinality}")
    print(f"{'='*70}")

    # Load baseline MCMC model
    baseline_path = work_dir / 'grm_mcmc_baseline'
    if not (baseline_path / 'params.h5').exists():
        # Fall back to grm_mcmc (old naming)
        baseline_path = work_dir / 'grm_mcmc'
        if not (baseline_path / 'params.h5').exists():
            print(f"  No baseline MCMC model found, skipping")
            return None

    model = GRModel.load_from_disk(baseline_path)

    # Load baseline MCMC samples
    npz_path = work_dir / 'mcmc_samples' / 'mcmc_baseline.npz'
    if not npz_path.exists():
        npz_path = work_dir / 'mcmc_samples' / 'mcmc_item_params.npz'
    if not npz_path.exists():
        print(f"  No baseline MCMC samples found, skipping")
        return None

    mcmc_data = np.load(str(npz_path))
    mcmc_samples = {}
    for key in model._item_var_list():
        if key in mcmc_data:
            mcmc_samples[key] = mcmc_data[key]
    print(f"  Loaded {len(mcmc_samples)} param arrays from {npz_path.name}")
    for k, v in mcmc_samples.items():
        print(f"    {k}: {v.shape}")

    # Load pairwise stacking model
    stacking_path = work_dir / 'pairwise_stacking_model.yaml'
    if not stacking_path.exists():
        print(f"  No pairwise stacking model, skipping")
        return None

    pairwise_model = PairwiseOrdinalStackingModel.load(str(stacking_path))
    print(f"  Loaded pairwise stacking model")

    # --- IS Reweighting: discriminations ---
    print(f"\n  --- Reweighting for discriminations ---")
    disc_result = model.importance_reweight(
        data=data,
        mcmc_samples=mcmc_samples,
        imputation_model=pairwise_model,
        fn=lambda p: np.array(p['discriminations']).flatten(),
        max_samples=200,
        seed=42,
        verbose=True,
    )

    print(f"\n  IS Discriminations (reweighted):")
    if disc_result['expectation'] is not None:
        disc_is = disc_result['expectation']
        print(f"    Mean: {np.mean(disc_is):.4f}")
        print(f"    Per-item: {np.round(disc_is, 3)}")

    # Compare against direct pairwise MCMC if available
    pw_path = work_dir / 'grm_mcmc_pairwise'
    if (pw_path / 'params.h5').exists():
        pw_model = GRModel.load_from_disk(pw_path)
        from run_single_notebook import calibrate_manually
        calibrate_manually(pw_model, n_samples=32, seed=101)
        disc_direct = np.array(
            pw_model.calibrated_expectations['discriminations']).flatten()
        print(f"\n  Direct MCMC Pairwise discriminations:")
        print(f"    Mean: {np.mean(disc_direct):.4f}")
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

    datasets = (['scs', 'gcbs', 'grit', 'rwa', 'tma']
                if args.dataset == 'all' else [args.dataset])

    for ds in datasets:
        if ds not in DATASET_CONFIGS:
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
