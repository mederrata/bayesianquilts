#!/usr/bin/env python
"""Run marginal MCMC on a fitted baseline GRM.

Loads a previously fitted GRM (ADVI) from disk, optionally attaches an
imputation model for non-ignorable missing data, then runs NUTS on item
parameters with abilities Rao-Blackwellized out on a quadrature grid.

Usage:
    python run_marginal_mcmc.py --dataset rwa
    python run_marginal_mcmc.py --dataset tma --num-chains 2 --num-warmup 200 --num-samples 500
    python run_marginal_mcmc.py --dataset npi --no-imputation
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


def make_data_dict(dataframe, num_people):
    data = {}
    for col in dataframe.columns:
        arr = dataframe[col].to_numpy().astype(np.float32)
        data[col] = arr
    data['person'] = np.arange(num_people, dtype=np.float32)
    return data


def attach_imputation_pmfs(model, data, pairwise_model):
    """Compute imputation PMFs for missing cells and attach to data dict."""
    pmfs, weights = model._compute_batch_pmfs(data)
    if pmfs is not None:
        data['_imputation_pmfs'] = pmfs
        if weights is not None:
            data['_imputation_weights'] = weights
    return data


def run_mcmc(dataset_name, model_dir, num_chains, num_warmup, num_samples,
             use_imputation, seed):
    import importlib
    import inspect
    from pathlib import Path
    from bayesianquilts.irt.grm import GRModel

    config = DATASET_CONFIGS[dataset_name]
    mod = importlib.import_module(config['module'])
    item_keys = mod.item_keys
    response_cardinality = mod.response_cardinality

    print(f"\n{'='*60}")
    print(f"Marginal MCMC: {dataset_name.upper()}")
    print(f"  Items: {len(item_keys)}, K: {response_cardinality}")
    print(f"  Chains: {num_chains}, Warmup: {num_warmup}, Samples: {num_samples}")
    print(f"  Imputation: {'yes' if use_imputation else 'no'}")
    print(f"{'='*60}")

    # Load data
    get_data_kwargs = {'polars_out': True}
    if 'reorient' in inspect.signature(mod.get_data).parameters:
        get_data_kwargs['reorient'] = True
    df, num_people = mod.get_data(**get_data_kwargs)
    data = make_data_dict(df, num_people)
    print(f"  People: {num_people}")

    # Load fitted baseline GRM
    model_path = Path(model_dir)
    print(f"  Loading ADVI model from {model_path}")
    model = GRModel.load_from_disk(str(model_path))

    # Optionally load and attach imputation model
    if use_imputation:
        stacking_path = model_path.parent / 'pairwise_stacking_model.yaml'
        if stacking_path.exists():
            from bayesianquilts.imputation.pairwise_stacking import PairwiseOrdinalStackingModel
            print(f"  Loading imputation model from {stacking_path}")
            pairwise_model = PairwiseOrdinalStackingModel.load(str(stacking_path))
            model.imputation_model = pairwise_model

            # Compute and attach imputation PMFs to data dict
            print("  Computing imputation PMFs for missing cells...")
            data = attach_imputation_pmfs(model, data, pairwise_model)
            if '_imputation_pmfs' in data:
                n_missing = np.sum(
                    np.any(np.isnan(data.get('_imputation_pmfs', np.array([]))), axis=-1)
                    if '_imputation_pmfs' in data else 0
                )
                print(f"  Imputation PMFs attached")
            del pairwise_model
        else:
            print(f"  WARNING: No pairwise stacking model found at {stacking_path}")
            print(f"  Running without imputation")

    # Run marginal MCMC (uses Gauss-Hermite quadrature by default)
    print(f"\nStarting NUTS...")
    mcmc_samples = model.fit_marginal_mcmc(
        data,
        theta_grid=None,
        num_chains=num_chains,
        num_warmup=num_warmup,
        num_samples=num_samples,
        target_accept_prob=0.85,
        step_size=0.01,
        seed=seed,
        verbose=True,
    )

    # Compute EAP abilities from MCMC item params
    print(f"\nComputing EAP abilities...")
    eap_result = model.compute_eap_abilities(data)
    print(f"  EAP mean: {np.mean(np.array(eap_result['eap'])):.4f}")
    print(f"  EAP std:  {np.std(np.array(eap_result['eap'])):.4f}")
    print(f"  Mean PSD: {np.mean(np.array(eap_result['psd'])):.4f}")

    # Save MCMC samples
    output_dir = model_path.parent / 'mcmc_samples'
    os.makedirs(output_dir, exist_ok=True)

    save_dict = {}
    for var_name, samples in mcmc_samples.items():
        save_dict[var_name] = np.array(samples)
    save_dict['eap'] = np.array(eap_result['eap'])
    save_dict['psd'] = np.array(eap_result['psd'])
    save_dict['num_chains'] = num_chains
    save_dict['num_warmup'] = num_warmup
    save_dict['num_samples'] = num_samples

    out_path = output_dir / 'mcmc_item_params.npz'
    np.savez(out_path, **save_dict)
    print(f"\nSaved MCMC samples to {out_path}")

    # Compute and print summary diagnostics
    print(f"\n--- MCMC Summary ---")
    for var_name, samples in mcmc_samples.items():
        # samples: (num_chains, num_samples, ...)
        flat = np.array(samples).reshape(-1, *samples.shape[2:])
        print(f"  {var_name}:")
        print(f"    shape: {samples.shape}")
        print(f"    mean:  {np.mean(flat):.4f}")
        print(f"    std:   {np.std(flat):.4f}")
        print(f"    range: [{np.min(flat):.4f}, {np.max(flat):.4f}]")

        # Effective sample size (simple batch means estimator)
        if samples.shape[0] > 1:
            chain_means = np.mean(np.array(samples), axis=1)
            between_var = np.var(chain_means, axis=0, ddof=1)
            within_var = np.mean(np.var(np.array(samples), axis=1, ddof=1), axis=0)
            n = samples.shape[1]
            r_hat = np.sqrt(
                ((n - 1) / n * within_var + between_var) /
                np.maximum(within_var, 1e-30)
            )
            print(f"    R-hat: mean={np.mean(r_hat):.4f}, "
                  f"max={np.max(r_hat):.4f}")

    # Compare MCMC posterior means to ADVI means
    if model.params is not None:
        print(f"\n--- ADVI vs MCMC comparison ---")
        for var_name in mcmc_samples:
            mcmc_mean = np.mean(
                np.array(mcmc_samples[var_name]).reshape(
                    -1, *mcmc_samples[var_name].shape[2:]),
                axis=0
            )
            # Find ADVI loc
            loc_key = None
            for pk in model.params:
                if pk.startswith(var_name) and pk.endswith('loc'):
                    loc_key = pk
                    break
            if loc_key is not None:
                advi_mean = np.array(model.params[loc_key])
                diff = np.abs(mcmc_mean.squeeze() - advi_mean.squeeze())
                print(f"  {var_name}:")
                print(f"    |MCMC - ADVI| mean: {np.mean(diff):.4f}, "
                      f"max: {np.max(diff):.4f}")

    del model, mcmc_samples
    gc.collect()
    print(f"\nDone: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description='Run marginal MCMC on fitted baseline GRM')
    parser.add_argument('--dataset', required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset name')
    parser.add_argument('--model-dir', default=None,
                        help='Path to grm_baseline/ directory (default: auto)')
    parser.add_argument('--num-chains', type=int, default=4)
    parser.add_argument('--num-warmup', type=int, default=500)
    parser.add_argument('--num-samples', type=int, default=500)
    parser.add_argument('--no-imputation', action='store_true',
                        help='Skip imputation model (treat missing as ignorable)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    model_dir = args.model_dir or os.path.expanduser(
        f'~/workspace/bayesianquilts/notebooks/irt/{args.dataset}/grm_baseline')

    if not os.path.exists(os.path.join(model_dir, 'params.h5')):
        print(f"ERROR: No params.h5 found in {model_dir}")
        print(f"Fit the baseline model first with: "
              f"python run_single_notebook.py --dataset {args.dataset}")
        sys.exit(1)

    run_mcmc(
        args.dataset,
        model_dir,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        use_imputation=not args.no_imputation,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
