#!/usr/bin/env python
"""Run marginal MCMC for all 3 model variants: baseline, pairwise, mixed.

For each variant, loads the fitted ADVI baseline GRM, optionally attaches
an imputation model, then runs BlackJAX NUTS on item parameters with
abilities Rao-Blackwellized out on a Gauss-Hermite quadrature grid.

Results are saved per-variant:
  - grm_mcmc_baseline/    (no imputation)
  - grm_mcmc_pairwise/    (pairwise stacking imputation)
  - grm_mcmc_mixed/       (mixed: pairwise + IRT baseline)

Usage:
    python run_marginal_mcmc.py --dataset rwa
    python run_marginal_mcmc.py --dataset tma --num-chains 2 --num-warmup 200
    python run_marginal_mcmc.py --dataset npi --variants baseline pairwise
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
    'promis_sleep': {'module': 'bayesianquilts.data.promis_sleep'},
    'promis_substance_use': {'module': 'bayesianquilts.data.promis_substance_use'},
    'promis_neuropathic_pain': {'module': 'bayesianquilts.data.promis_neuropathic_pain', 'factorized': True},
    'promis_copd': {'module': 'bayesianquilts.data.promis_copd', 'factorized': True},
}


def make_data_dict(dataframe, num_people):
    data = {}
    for col in dataframe.columns:
        arr = dataframe[col].to_numpy().astype(np.float32)
        data[col] = arr
    data['person'] = np.arange(num_people, dtype=np.float32)
    return data


def run_single_variant(model, data, variant_name, output_dir,
                       num_chains, num_warmup, num_samples, step_size, seed):
    """Run MCMC for one variant and save results."""
    print(f"\n  --- Variant: {variant_name} ---")
    sys.stdout.flush()

    mcmc_samples = model.fit_marginal_mcmc(
        data,
        theta_grid=None,
        num_chains=num_chains,
        num_warmup=num_warmup,
        num_samples=num_samples,
        target_accept_prob=0.85,
        step_size=step_size,
        seed=seed,
        verbose=True,
    )

    # EAP
    print(f"\n  Computing EAP abilities...")
    eap_result = model.compute_eap_abilities(data)
    print(f"    EAP mean: {np.mean(np.array(eap_result['eap'])):.4f}")
    print(f"    EAP std:  {np.std(np.array(eap_result['eap'])):.4f}")
    print(f"    Mean PSD: {np.mean(np.array(eap_result['psd'])):.4f}")

    # Standardize
    print(f"\n  Standardizing...")
    stats = model.standardize_marginal(data)
    eap_result = model.compute_eap_abilities(data)
    print(f"    Post-std EAP std: {np.std(np.array(eap_result['eap'])):.4f}")

    # Fit surrogate
    model.fit_surrogate_to_mcmc()

    # Inject EAP abilities
    eap_arr = np.array(eap_result['eap'])
    model.surrogate_sample['abilities'] = jnp.array(
        eap_arr[:, np.newaxis, np.newaxis, np.newaxis]
    )[np.newaxis, ...]

    # Save model
    model_dir = output_dir / f'grm_mcmc_{variant_name}'
    print(f"  Saving to {model_dir}")
    model.save_to_disk(str(model_dir))

    # Save NPZ
    npz_dir = output_dir / 'mcmc_samples'
    os.makedirs(npz_dir, exist_ok=True)
    save_dict = {}
    for var_name, samples in mcmc_samples.items():
        save_dict[var_name] = np.array(samples)
    save_dict['eap'] = np.array(eap_result['eap'])
    save_dict['psd'] = np.array(eap_result['psd'])
    save_dict['standardize_mu'] = stats['mu']
    save_dict['standardize_sigma'] = stats['sigma']
    npz_path = npz_dir / f'mcmc_{variant_name}.npz'
    np.savez(str(npz_path), **save_dict)
    print(f"  NPZ: {npz_path}")

    # Summary
    print(f"\n  --- {variant_name} Summary ---")
    for var_name, samples in mcmc_samples.items():
        flat = np.array(samples).reshape(-1, *samples.shape[2:])
        print(f"    {var_name}: mean={np.mean(flat):.4f}, std={np.std(flat):.4f}")
        if samples.shape[0] > 1:
            chain_means = np.mean(np.array(samples), axis=1)
            between_var = np.var(chain_means, axis=0, ddof=1)
            within_var = np.mean(np.var(np.array(samples), axis=1, ddof=1), axis=0)
            n = samples.shape[1]
            r_hat = np.sqrt(
                ((n - 1) / n * within_var + between_var) /
                np.maximum(within_var, 1e-30)
            )
            print(f"      R-hat: mean={np.mean(r_hat):.4f}, max={np.max(r_hat):.4f}")
    sys.stdout.flush()


def run_dataset(dataset_name, model_dir, num_chains, num_warmup, num_samples,
                step_size, seed, variants):
    import importlib
    import inspect
    from pathlib import Path
    from bayesianquilts.irt.grm import GRModel

    config = DATASET_CONFIGS[dataset_name]
    is_factorized = config.get('factorized', False)
    mod = importlib.import_module(config['module'])
    item_keys = mod.item_keys
    response_cardinality = mod.response_cardinality

    if is_factorized:
        from bayesianquilts.irt.factorizedgrm import FactorizedGRModel as ModelClass
    else:
        ModelClass = GRModel

    model_path = Path(model_dir)
    output_dir = model_path.parent

    print(f"\n{'='*60}")
    print(f"Marginal MCMC: {dataset_name.upper()}")
    print(f"  Items: {len(item_keys)}, K: {response_cardinality}")
    print(f"  Chains: {num_chains}, Warmup: {num_warmup}, Samples: {num_samples}")
    print(f"  Step size: {step_size}")
    print(f"  Variants: {variants}")
    print(f"{'='*60}")

    # Load data
    if is_factorized and hasattr(mod, 'get_multidomain_data'):
        df, num_people, scale_indices = mod.get_multidomain_data(
            polars_out=True, min_items=10)
    else:
        get_data_kwargs = {'polars_out': True}
        if 'reorient' in inspect.signature(mod.get_data).parameters:
            get_data_kwargs['reorient'] = True
        df, num_people = mod.get_data(**get_data_kwargs)
    base_data = make_data_dict(df, num_people)
    print(f"  People: {num_people}")

    # Load pairwise stacking model (needed for pairwise and mixed)
    pairwise_model = None
    stacking_path = output_dir / 'pairwise_stacking_model.yaml'
    if stacking_path.exists() and ('pairwise' in variants or 'mixed' in variants):
        from bayesianquilts.imputation.pairwise_stacking import PairwiseOrdinalStackingModel
        print(f"  Loading pairwise stacking model...")
        pairwise_model = PairwiseOrdinalStackingModel.load(str(stacking_path))

    # ---- Baseline: no imputation ----
    if 'baseline' in variants:
        model = ModelClass.load_from_disk(str(model_path))
        data = dict(base_data)  # no imputation PMFs
        run_single_variant(model, data, 'baseline', output_dir,
                           num_chains, num_warmup, num_samples, step_size, seed)
        del model
        gc.collect()

    # ---- Pairwise: pairwise stacking only ----
    if 'pairwise' in variants and pairwise_model is not None:
        model = ModelClass.load_from_disk(str(model_path))
        model.imputation_model = pairwise_model
        data = dict(base_data)
        pmfs, weights = model._compute_batch_pmfs(data)
        if pmfs is not None:
            data['_imputation_pmfs'] = pmfs
            # For pairwise-only, don't pass weights (full imputation, no IS blend)
        print(f"  Pairwise imputation PMFs attached")
        run_single_variant(model, data, 'pairwise', output_dir,
                           num_chains, num_warmup, num_samples, step_size, seed + 1)
        del model
        gc.collect()

    # ---- Mixed: pairwise + IRT baseline blend ----
    if 'mixed' in variants and pairwise_model is not None:
        from bayesianquilts.imputation.mixed import IrtMixedImputationModel
        model = ModelClass.load_from_disk(str(model_path))

        # Build mixed imputation model from pairwise + baseline
        # Need calibrated expectations for the IRT component
        surrogate = model.surrogate_distribution_generator(model.params)
        key = jax.random.PRNGKey(seed + 100)
        samples = surrogate.sample(32, seed=key)
        model.surrogate_sample = samples
        model.calibrated_expectations = {
            k: jnp.mean(v, axis=0) for k, v in samples.items()
        }

        def _data_factory():
            yield base_data

        mixed_model = IrtMixedImputationModel(
            irt_model=model,
            mice_model=pairwise_model,
            data_factory=_data_factory,
        )
        model.imputation_model = mixed_model

        data = dict(base_data)
        pmfs, weights = model._compute_batch_pmfs(data)
        if pmfs is not None:
            data['_imputation_pmfs'] = pmfs
            if weights is not None:
                data['_imputation_weights'] = weights
        print(f"  Mixed imputation PMFs attached (with IS weights)")
        run_single_variant(model, data, 'mixed', output_dir,
                           num_chains, num_warmup, num_samples, step_size, seed + 2)
        del model, mixed_model
        gc.collect()

    del pairwise_model
    gc.collect()
    print(f"\nDone: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description='Run marginal MCMC for all model variants')
    parser.add_argument('--dataset', required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset name')
    parser.add_argument('--model-dir', default=None,
                        help='Path to grm_baseline/ directory (default: auto)')
    parser.add_argument('--num-chains', type=int, default=2)
    parser.add_argument('--num-warmup', type=int, default=200)
    parser.add_argument('--num-samples', type=int, default=300)
    parser.add_argument('--step-size', type=float, default=0.01,
                        help='Initial NUTS step size')
    parser.add_argument('--variants', nargs='+',
                        default=['baseline', 'pairwise', 'mixed'],
                        choices=['baseline', 'pairwise', 'mixed'],
                        help='Which model variants to run')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    model_dir = args.model_dir or os.path.expanduser(
        f'~/workspace/bayesianquilts/notebooks/irt/{args.dataset}/grm_baseline')

    if not os.path.exists(os.path.join(model_dir, 'params.h5')):
        print(f"ERROR: No params.h5 found in {model_dir}")
        sys.exit(1)

    run_dataset(
        args.dataset,
        model_dir,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        step_size=args.step_size,
        seed=args.seed,
        variants=args.variants,
    )


if __name__ == '__main__':
    main()
