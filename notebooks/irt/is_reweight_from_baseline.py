#!/usr/bin/env python
"""Importance-sample pairwise and mixed posteriors from baseline MCMC samples.

Instead of running separate MCMC for each imputation variant, reweights
the baseline samples using the likelihood ratio. Much faster when the
baseline has good convergence.

Usage:
    uv run python is_reweight_from_baseline.py --dataset scs
    uv run python is_reweight_from_baseline.py --all
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


def reweight_dataset(dataset_name, max_samples=None):
    import importlib
    import inspect
    from pathlib import Path
    from bayesianquilts.irt.grm import GRModel
    from bayesianquilts.imputation.pairwise_stacking import PairwiseOrdinalStackingModel
    from bayesianquilts.imputation.mixed import IrtMixedImputationModel

    config = DATASET_CONFIGS[dataset_name]
    mod = importlib.import_module(config['module'])
    item_keys = mod.item_keys
    response_cardinality = mod.response_cardinality

    base_dir = Path(os.path.expanduser(
        f'~/workspace/bayesianquilts/notebooks/irt/{dataset_name}'))
    baseline_npz = base_dir / 'mcmc_samples' / 'mcmc_baseline.npz'

    if not baseline_npz.exists():
        print(f"  {dataset_name}: no baseline MCMC, skipping")
        return

    print(f"\n{'='*60}")
    print(f"IS Reweighting: {dataset_name.upper()}")
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
    model = GRModel.load_from_disk(str(base_dir / 'grm_baseline'))

    # Load baseline MCMC samples
    saved = dict(np.load(str(baseline_npz)))
    skip_keys = {'eap', 'psd', 'standardize_mu', 'standardize_sigma',
                 'eap_standardized', 'psd_standardized'}
    mcmc_samples = {}
    for k, v in saved.items():
        if k in skip_keys:
            continue
        if hasattr(v, 'shape') and len(v.shape) >= 3:
            mcmc_samples[k] = jnp.array(v)
    first_key = list(mcmc_samples.keys())[0]
    print(f"  Baseline samples: {mcmc_samples[first_key].shape} "
          f"({list(mcmc_samples.keys())})")

    # Check baseline R-hat first
    max_rhat = 0
    for k, s in mcmc_samples.items():
        if s.shape[0] <= 1:
            continue
        cm = np.mean(np.array(s), axis=1)
        bv = np.var(cm, axis=0, ddof=1)
        wv = np.mean(np.var(np.array(s), axis=1, ddof=1), axis=0)
        n = s.shape[1]
        rh = np.sqrt(((n-1)/n*wv + bv) / np.maximum(wv, 1e-30))
        max_rhat = max(max_rhat, float(np.max(rh)))
    print(f"  Baseline max R-hat: {max_rhat:.4f}")
    if max_rhat > 1.1:
        print(f"  WARNING: Baseline R-hat > 1.1, IS results may be unreliable")

    # Load pairwise stacking model
    stacking_path = base_dir / 'pairwise_stacking_model.yaml'
    if not stacking_path.exists():
        print(f"  No stacking model, skipping")
        return
    pairwise_model = PairwiseOrdinalStackingModel.load(str(stacking_path))

    # ---- Pairwise IS reweighting ----
    print(f"\n  --- Pairwise (IS from baseline) ---")
    sys.stdout.flush()
    pairwise_result = model.importance_reweight(
        data, mcmc_samples, pairwise_model,
        max_samples=max_samples, verbose=True,
    )
    print(f"  Pairwise: k-hat={pairwise_result['khat']:.3f}, "
          f"ESS={pairwise_result['ess']:.0f}/{pairwise_result['n_samples']}")

    # ---- Mixed IS reweighting ----
    print(f"\n  --- Mixed (IS from baseline) ---")
    sys.stdout.flush()

    # Build mixed imputation model
    surrogate = model.surrogate_distribution_generator(model.params)
    key = jax.random.PRNGKey(101)
    samples = surrogate.sample(32, seed=key)
    model.surrogate_sample = samples
    model.calibrated_expectations = {
        k: jnp.mean(v, axis=0) for k, v in samples.items()
    }

    def _data_factory():
        yield data

    mixed_model = IrtMixedImputationModel(
        irt_model=model,
        mice_model=pairwise_model,
        data_factory=_data_factory,
    )

    mixed_result = model.importance_reweight(
        data, mcmc_samples, mixed_model,
        max_samples=max_samples, verbose=True,
    )
    print(f"  Mixed: k-hat={mixed_result['khat']:.3f}, "
          f"ESS={mixed_result['ess']:.0f}/{mixed_result['n_samples']}")

    # ---- Compute IS-weighted LOO-ELPD for each variant ----
    print(f"\n  --- Computing IS-weighted metrics ---")

    # Save IS results
    is_dir = base_dir / 'mcmc_samples'
    os.makedirs(is_dir, exist_ok=True)

    np.savez(
        str(is_dir / 'is_pairwise.npz'),
        log_weights=pairwise_result['log_weights'],
        psis_weights=pairwise_result['psis_weights'],
        khat=pairwise_result['khat'],
        ess=pairwise_result['ess'],
    )
    np.savez(
        str(is_dir / 'is_mixed.npz'),
        log_weights=mixed_result['log_weights'],
        psis_weights=mixed_result['psis_weights'],
        khat=mixed_result['khat'],
        ess=mixed_result['ess'],
    )

    print(f"\n  Saved IS results to {is_dir}")

    del model, pairwise_model
    gc.collect()
    print(f"\nDone: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description='IS reweight baseline MCMC for pairwise/mixed posteriors')
    parser.add_argument('--dataset', default=None,
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max MCMC draws to use (default: all)')
    args = parser.parse_args()

    if args.all:
        datasets = list(DATASET_CONFIGS.keys())
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.error("Specify --dataset or --all")

    for ds in datasets:
        try:
            reweight_dataset(ds, max_samples=args.max_samples)
        except Exception as e:
            print(f"ERROR on {ds}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
