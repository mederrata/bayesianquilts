#!/usr/bin/env python
"""Full marginal IRT pipeline: 3 ADVI variants + 3 MCMC variants with IPW.

Demonstrates the complete marginal inference pipeline for a unidimensional
GRM, matching the analysis used in journal_article.tex:

1. Fit pairwise stacking imputation model (with optional IPW weights)
2. Fit all 3 model variants via marginal ADVI:
   - Baseline (no imputation)
   - Pairwise (stacking imputation only)
   - Mixed (pairwise + IRT baseline blend)
3. Fit all 3 model variants via marginal MCMC (BlackJAX NUTS):
   - Baseline, Pairwise, Mixed (same as above)
4. Standardize abilities and compute EAP for each variant
5. Save all results

Usage:
    uv run python fit_marginal_irt.py --dataset scs
    uv run python fit_marginal_irt.py --dataset rwa --step-size 0.001
    uv run python fit_marginal_irt.py --dataset npi --skip-advi
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
    'scs': {'module': 'bayesianquilts.data.scs', 'n_top_features': 10},
    'gcbs': {'module': 'bayesianquilts.data.gcbs', 'n_top_features': 15},
    'grit': {'module': 'bayesianquilts.data.grit', 'n_top_features': 12},
    'rwa': {'module': 'bayesianquilts.data.rwa', 'n_top_features': 22},
    'npi': {'module': 'bayesianquilts.data.npi', 'n_top_features': 40},
    'tma': {'module': 'bayesianquilts.data.tma', 'n_top_features': 14},
    'wpi': {'module': 'bayesianquilts.data.wpi', 'n_top_features': 20},
    'eqsq': {'module': 'bayesianquilts.data.eqsq', 'n_top_features': 30},
}


def make_data_dict(dataframe, num_people):
    data = {}
    for col in dataframe.columns:
        data[col] = dataframe[col].to_numpy().astype(np.float32)
    data['person'] = np.arange(num_people, dtype=np.float32)
    return data


def compute_ipw_weights(pandas_df, n_groups=3):
    """Create IPW weights from score-based stratification."""
    total_score = pandas_df.sum(axis=1, skipna=True).values
    valid = ~np.isnan(total_score)
    quantiles = np.quantile(total_score[valid],
                            np.linspace(0, 1, n_groups + 1)[1:-1])
    groups = np.digitize(total_score, bins=quantiles)
    group_counts = np.bincount(groups, minlength=n_groups)
    weights = np.array(
        [1.0 / max(group_counts[g], 1) for g in groups], dtype=np.float32
    )
    weights *= len(weights) / weights.sum()
    ess = 1.0 / np.sum((weights / weights.sum()) ** 2)
    return weights, groups, ess


def calibrate_model(model, seed=101, n_samples=32):
    """Calibrate surrogate expectations for mixed imputation model."""
    surrogate = model.surrogate_distribution_generator(model.params)
    key = jax.random.PRNGKey(seed)
    samples = surrogate.sample(n_samples, seed=key)
    model.surrogate_sample = samples
    model.calibrated_expectations = {
        k: jnp.mean(v, axis=0) for k, v in samples.items()
    }


def run_variant_advi(model, data, variant_name, output_dir,
                     num_samples=10, num_epochs=2000, learning_rate=0.01,
                     rank=0, seed=42):
    """Run marginal ADVI for one variant."""
    print(f"\n{'─'*50}")
    print(f"  ADVI: {variant_name}")
    print(f"{'─'*50}")
    sys.stdout.flush()

    losses, params = model.fit_marginal_advi(
        data,
        num_samples=num_samples,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        rank=rank,
        seed=seed,
        verbose=True,
    )

    # EAP
    eap_result = model.compute_eap_abilities(data)
    print(f"  EAP std: {float(jnp.std(eap_result['eap'])):.4f}, "
          f"PSD: {float(jnp.mean(eap_result['psd'])):.4f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, f'advi_{variant_name}.npz'),
        losses=np.array(losses),
        eap=np.array(eap_result['eap']),
        psd=np.array(eap_result['psd']),
    )
    return losses, params


def run_variant_mcmc(model, data, variant_name, output_dir,
                     num_chains=2, num_warmup=500, num_samples=500,
                     step_size=0.01, seed=42):
    """Run marginal MCMC for one variant, standardize, and save."""
    print(f"\n{'─'*50}")
    print(f"  MCMC: {variant_name}")
    print(f"{'─'*50}")
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

    # EAP before standardization
    eap_result = model.compute_eap_abilities(data)
    print(f"  Pre-std EAP std: {float(jnp.std(eap_result['eap'])):.4f}, "
          f"PSD: {float(jnp.mean(eap_result['psd'])):.4f}")

    # Standardize
    stats = model.standardize_marginal(data)

    # EAP after standardization
    eap_result = model.compute_eap_abilities(data)
    print(f"  Post-std EAP std: {float(jnp.std(eap_result['eap'])):.4f}, "
          f"PSD: {float(jnp.mean(eap_result['psd'])):.4f}")

    # Fit surrogate to MCMC
    model.fit_surrogate_to_mcmc()

    # Inject EAP abilities into surrogate_sample
    eap_arr = np.array(eap_result['eap'])
    model.surrogate_sample['abilities'] = jnp.array(
        eap_arr[:, np.newaxis, np.newaxis, np.newaxis]
    )[np.newaxis, ...]

    # Save model
    model_dir = os.path.join(output_dir, f'grm_mcmc_{variant_name}')
    model.save_to_disk(model_dir)

    # Save NPZ
    save_dict = {}
    for var_name, samples in mcmc_samples.items():
        save_dict[var_name] = np.array(samples)
    save_dict['eap'] = np.array(eap_result['eap'])
    save_dict['psd'] = np.array(eap_result['psd'])
    save_dict['standardize_mu'] = stats['mu']
    save_dict['standardize_sigma'] = stats['sigma']
    np.savez(os.path.join(output_dir, f'mcmc_{variant_name}.npz'), **save_dict)

    # R-hat summary
    for var_name, samples in mcmc_samples.items():
        if samples.shape[0] > 1:
            chain_means = np.mean(np.array(samples), axis=1)
            between_var = np.var(chain_means, axis=0, ddof=1)
            within_var = np.mean(
                np.var(np.array(samples), axis=1, ddof=1), axis=0)
            n = samples.shape[1]
            r_hat = np.sqrt(
                ((n - 1) / n * within_var + between_var) /
                np.maximum(within_var, 1e-30)
            )
            print(f"  {var_name} R-hat: "
                  f"mean={np.mean(r_hat):.4f}, max={np.max(r_hat):.4f}")

    return mcmc_samples


def main():
    parser = argparse.ArgumentParser(
        description='Full marginal IRT pipeline: ADVI + MCMC × 3 variants')
    parser.add_argument('--dataset', required=True,
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: dataset name)')
    parser.add_argument('--num-chains', type=int, default=2)
    parser.add_argument('--num-warmup', type=int, default=500)
    parser.add_argument('--num-samples', type=int, default=500)
    parser.add_argument('--step-size', type=float, default=0.01)
    parser.add_argument('--advi-epochs', type=int, default=2000)
    parser.add_argument('--advi-rank', type=int, default=0,
                        help='Low-rank surrogate rank (0=mean-field)')
    parser.add_argument('--skip-advi', action='store_true')
    parser.add_argument('--skip-mcmc', action='store_true')
    parser.add_argument('--use-ipw', action='store_true', default=True,
                        help='Use IPW weights (default: True)')
    parser.add_argument('--no-ipw', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    import importlib
    import inspect
    from bayesianquilts.irt.grm import GRModel
    from bayesianquilts.imputation.pairwise_stacking import (
        PairwiseOrdinalStackingModel
    )
    from bayesianquilts.imputation.mixed import (
        IrtMixedImputationModel, PairwiseOnlyImputationModel
    )

    config = DATASET_CONFIGS[args.dataset]
    mod = importlib.import_module(config['module'])
    item_keys = mod.item_keys
    response_cardinality = mod.response_cardinality

    output_dir = args.output_dir or args.dataset
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Marginal IRT Pipeline: {args.dataset.upper()}")
    print(f"  Items: {len(item_keys)}, K: {response_cardinality}")
    print(f"{'='*60}")

    # ---- Load data ----
    get_data_kwargs = {'polars_out': True}
    if 'reorient' in inspect.signature(mod.get_data).parameters:
        get_data_kwargs['reorient'] = True
    df, num_people = mod.get_data(**get_data_kwargs)
    pandas_df = df.select(item_keys).to_pandas().replace(-1, np.nan)
    base_data = make_data_dict(df, num_people)
    print(f"  People: {num_people}")

    # ---- IPW weights ----
    use_ipw = args.use_ipw and not args.no_ipw
    if use_ipw:
        weights, groups, ess = compute_ipw_weights(pandas_df)
        base_data['sample_weights'] = weights
        print(f"  IPW: {len(set(groups))} groups, ESS: {ess:.0f}")

    # ================================================================
    # Step 1: Fit pairwise stacking imputation model
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Step 1: Pairwise Stacking Imputation")
    print(f"{'='*60}")

    stacking_path = os.path.join(output_dir, 'pairwise_stacking_model.yaml')
    if os.path.exists(stacking_path):
        print(f"  Loading from {stacking_path}")
        pairwise_model = PairwiseOrdinalStackingModel.load(stacking_path)
    else:
        pairwise_model = PairwiseOrdinalStackingModel(
            prior_scale=1.0,
            pathfinder_num_samples=100,
            pathfinder_maxiter=50,
            batch_size=512,
            verbose=True,
        )
        pairwise_model.fit(
            pandas_df,
            n_top_features=config['n_top_features'],
            n_jobs=1,
            seed=args.seed,
        )
        pairwise_model.save(stacking_path)
        print(f"  Saved to {stacking_path}")

    # Precompute imputation PMFs for pairwise variant
    def make_pairwise_data():
        model_tmp = GRModel(
            item_keys=item_keys, num_people=num_people,
            response_cardinality=response_cardinality, dim=1,
            imputation_model=pairwise_model, dtype=jnp.float64,
        )
        data = dict(base_data)
        pmfs, _ = model_tmp._compute_batch_pmfs(data)
        if pmfs is not None:
            data['_imputation_pmfs'] = pmfs
        del model_tmp
        return data

    pairwise_data = make_pairwise_data()

    # ================================================================
    # Step 2: Fit baseline ADVI (for mixed imputation model)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Step 2: Baseline ADVI (needed for mixed model)")
    print(f"{'='*60}")

    baseline_grm_path = os.path.join(output_dir, 'grm_baseline')
    if os.path.exists(os.path.join(baseline_grm_path, 'params.h5')):
        print(f"  Loading from {baseline_grm_path}")
        baseline_model = GRModel.load_from_disk(baseline_grm_path)
    else:
        baseline_model = GRModel(
            item_keys=item_keys, num_people=num_people,
            response_cardinality=response_cardinality, dim=1,
            dtype=jnp.float64,
        )

        def data_factory():
            yield base_data

        baseline_model.fit(
            data_factory,
            dataset_size=num_people,
            batch_size=num_people,
            num_epochs=2000,
            learning_rate=0.01,
        )
        baseline_model.save_to_disk(baseline_grm_path)
        print(f"  Saved to {baseline_grm_path}")

    calibrate_model(baseline_model)

    # Build mixed imputation model
    def make_data_factory():
        def factory():
            yield base_data
        return factory

    mixed_imputation = IrtMixedImputationModel(
        irt_model=baseline_model,
        mice_model=pairwise_model,
        data_factory=make_data_factory(),
    )

    # Precompute mixed imputation PMFs
    def make_mixed_data():
        model_tmp = GRModel(
            item_keys=item_keys, num_people=num_people,
            response_cardinality=response_cardinality, dim=1,
            imputation_model=mixed_imputation, dtype=jnp.float64,
        )
        data = dict(base_data)
        pmfs, weights = model_tmp._compute_batch_pmfs(data)
        if pmfs is not None:
            data['_imputation_pmfs'] = pmfs
            if weights is not None:
                data['_imputation_weights'] = weights
        del model_tmp
        return data

    mixed_data = make_mixed_data()

    # Data dicts for each variant
    variant_data = {
        'baseline': dict(base_data),  # no imputation
        'pairwise': pairwise_data,
        'mixed': mixed_data,
    }

    # ================================================================
    # Step 3: Marginal ADVI for all 3 variants
    # ================================================================
    if not args.skip_advi:
        print(f"\n{'='*60}")
        print(f"Step 3: Marginal ADVI (3 variants)")
        print(f"{'='*60}")

        for variant_name, data in variant_data.items():
            model = GRModel.load_from_disk(baseline_grm_path)
            run_variant_advi(
                model, data, variant_name, output_dir,
                num_epochs=args.advi_epochs,
                rank=args.advi_rank,
                seed=args.seed,
            )
            del model
            gc.collect()

    # ================================================================
    # Step 4: Marginal MCMC for all 3 variants
    # ================================================================
    if not args.skip_mcmc:
        print(f"\n{'='*60}")
        print(f"Step 4: Marginal MCMC (3 variants)")
        print(f"{'='*60}")

        for i, (variant_name, data) in enumerate(variant_data.items()):
            model = GRModel.load_from_disk(baseline_grm_path)
            run_variant_mcmc(
                model, data, variant_name, output_dir,
                num_chains=args.num_chains,
                num_warmup=args.num_warmup,
                num_samples=args.num_samples,
                step_size=args.step_size,
                seed=args.seed + i,
            )
            del model
            gc.collect()

    print(f"\n{'='*60}")
    print(f"Pipeline complete: {args.dataset.upper()}")
    print(f"  Output: {output_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
