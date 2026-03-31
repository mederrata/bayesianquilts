#!/usr/bin/env python
"""Reprocess saved MCMC samples with fixed EAP computation.

Loads MCMC samples from mcmc_item_params.npz, recomputes EAP abilities
with the two-pass grid refinement, re-standardizes, refits surrogate,
and re-saves everything.

Usage:
    uv run python reprocess_mcmc.py --dataset scs
    uv run python reprocess_mcmc.py --all
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


DATASET_CONFIGS = {
    'scs': {'module': 'bayesianquilts.data.scs'},
    'gcbs': {'module': 'bayesianquilts.data.gcbs'},
    'grit': {'module': 'bayesianquilts.data.grit'},
    'rwa': {'module': 'bayesianquilts.data.rwa'},
    'npi': {'module': 'bayesianquilts.data.npi'},
    'tma': {'module': 'bayesianquilts.data.tma'},
    'wpi': {'module': 'bayesianquilts.data.wpi'},
    'eqsq': {'module': 'bayesianquilts.data.eqsq'},
}


def reprocess(dataset_name):
    import importlib
    import inspect
    from pathlib import Path
    from bayesianquilts.irt.grm import GRModel

    config = DATASET_CONFIGS[dataset_name]
    mod = importlib.import_module(config['module'])
    item_keys = mod.item_keys
    response_cardinality = mod.response_cardinality

    model_dir = Path(os.path.expanduser(
        f'~/workspace/bayesianquilts/notebooks/irt/{dataset_name}/grm_baseline'))
    mcmc_path = model_dir.parent / 'mcmc_samples' / 'mcmc_item_params.npz'

    if not mcmc_path.exists():
        print(f"  {dataset_name}: no MCMC samples found, skipping")
        return

    print(f"\n{'='*60}")
    print(f"Reprocessing: {dataset_name.upper()}")
    print(f"{'='*60}")

    # Load data
    get_data_kwargs = {'polars_out': True}
    if 'reorient' in inspect.signature(mod.get_data).parameters:
        get_data_kwargs['reorient'] = True
    df, num_people = mod.get_data(**get_data_kwargs)
    data = {}
    for col in df.columns:
        data[col] = df[col].to_numpy().astype(np.float32)
    data['person'] = np.arange(num_people, dtype=np.float32)

    # Load model
    model = GRModel.load_from_disk(str(model_dir))

    # Load imputation model
    stacking_path = model_dir.parent / 'pairwise_stacking_model.yaml'
    if stacking_path.exists():
        from bayesianquilts.imputation.pairwise_stacking import PairwiseOrdinalStackingModel
        pairwise_model = PairwiseOrdinalStackingModel.load(str(stacking_path))
        model.imputation_model = pairwise_model
        pmfs, weights = model._compute_batch_pmfs(data)
        if pmfs is not None:
            data['_imputation_pmfs'] = pmfs
            if weights is not None:
                data['_imputation_weights'] = weights
        print(f"  Imputation PMFs attached")

    # Load MCMC samples
    saved = dict(np.load(str(mcmc_path)))
    mcmc_samples = {}
    item_var_list = [v for v in model.var_list if not v.startswith('abilities')]
    for var in item_var_list:
        if var in saved:
            mcmc_samples[var] = jnp.array(saved[var])
    model.mcmc_samples = mcmc_samples

    print(f"  Loaded MCMC: {list(mcmc_samples.keys())}")
    for var, s in mcmc_samples.items():
        print(f"    {var}: {s.shape}")

    # Recompute EAP with fixed two-pass grid
    print(f"\n  Computing EAP (two-pass grid)...")
    eap_result = model.compute_eap_abilities(data)
    print(f"  EAP mean: {float(jnp.mean(eap_result['eap'])):.4f}")
    print(f"  EAP std:  {float(jnp.std(eap_result['eap'])):.4f}")
    print(f"  Mean PSD: {float(jnp.mean(eap_result['psd'])):.4f}")

    # Standardize
    print(f"\n  Standardizing...")
    stats = model.standardize_marginal(data)

    # Recompute EAP after standardization
    eap_result = model.compute_eap_abilities(data)
    print(f"  Post-std EAP mean: {float(jnp.mean(eap_result['eap'])):.4f}")
    print(f"  Post-std EAP std:  {float(jnp.std(eap_result['eap'])):.4f}")
    print(f"  Post-std Mean PSD: {float(jnp.mean(eap_result['psd'])):.4f}")

    # Fit surrogate
    print(f"\n  Fitting surrogate to MCMC...")
    model.fit_surrogate_to_mcmc()

    # Inject EAP abilities into surrogate_sample
    eap_arr = np.array(eap_result['eap'])
    model.surrogate_sample['abilities'] = jnp.array(
        eap_arr[:, np.newaxis, np.newaxis, np.newaxis]
    )[np.newaxis, ...]

    # Save model
    mcmc_model_dir = model_dir.parent / 'grm_mcmc'
    print(f"\n  Saving model to {mcmc_model_dir}...")
    model.save_to_disk(str(mcmc_model_dir))

    # Update NPZ
    save_dict = {}
    for var_name, samples in model.mcmc_samples.items():
        save_dict[var_name] = np.array(samples)
    save_dict['eap'] = np.array(eap_result['eap'])
    save_dict['psd'] = np.array(eap_result['psd'])
    save_dict['standardize_mu'] = stats['mu']
    save_dict['standardize_sigma'] = stats['sigma']
    np.savez(str(mcmc_path), **save_dict)
    print(f"  NPZ updated: {mcmc_path}")

    del model
    gc.collect()
    print(f"\nDone: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description='Reprocess MCMC samples with fixed EAP')
    parser.add_argument('--dataset', default=None,
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        datasets = list(DATASET_CONFIGS.keys())
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.error("Specify --dataset or --all")

    for ds in datasets:
        try:
            reprocess(ds)
        except Exception as e:
            print(f"ERROR on {ds}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
