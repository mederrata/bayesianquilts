#!/usr/bin/env python
"""IS-based factorized (multiscale) IRT pipeline.

Same approach as fit_is_irt.py but for instruments with multiple scales.
Each scale is fitted independently:

1. Fit shared pairwise stacking imputation model
2. For each scale:
   a. Fit baseline via marginal ADVI
   b. Run marginal MCMC on baseline (warm-started from ADVI)
   c. IS-reweight baseline MCMC samples for pairwise and mixed variants
3. Assemble multi-dimensional ability profiles
4. Produce per-scale forest plots, comparison plots, and stats

Usage:
    uv run python fit_is_factorized_irt.py --dataset eqsq
    uv run python fit_is_factorized_irt.py --dataset rwa --step-size 0.001
"""

import argparse
import gc
import json
import os
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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

COLORS = {'Baseline': '#4477AA', 'Pairwise': '#228833', 'Mixed': '#EE6677'}
MARKERS = {'Baseline': 'o', 'Pairwise': 'D', 'Mixed': 's'}


# ============================================================
# Utilities
# ============================================================

def make_data_dict(dataframe, num_people):
    data = {}
    for col in dataframe.columns:
        data[col] = dataframe[col].to_numpy().astype(np.float32)
    data['person'] = np.arange(num_people, dtype=np.float32)
    return data


def compute_ipw_weights(pandas_df, n_groups=3):
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
    surrogate = model.surrogate_distribution_generator(model.params)
    key = jax.random.PRNGKey(seed)
    samples = surrogate.sample(n_samples, seed=key)
    model.surrogate_sample = samples
    model.calibrated_expectations = {
        k: jnp.mean(v, axis=0) for k, v in samples.items()
    }


def print_rhat(mcmc_samples, prefix="  "):
    max_rhat_overall = 0.0
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
            max_rhat = float(np.max(r_hat))
            max_rhat_overall = max(max_rhat_overall, max_rhat)
            print(f"{prefix}{var_name} R-hat: "
                  f"mean={np.mean(r_hat):.4f}, max={max_rhat:.4f}")
    print(f"{prefix}Max R-hat (overall): {max_rhat_overall:.4f}")
    return max_rhat_overall


def is_weighted_stats(mcmc_samples, psis_weights):
    stats = {}
    for k, v in mcmc_samples.items():
        flat = np.asarray(v).reshape(-1, *v.shape[2:])
        w = psis_weights[:, None] if flat.ndim > 1 else psis_weights
        mean = np.sum(w * flat, axis=0)
        var = np.sum(w * (flat - mean) ** 2, axis=0)
        stats[k] = {'mean': mean, 'std': np.sqrt(np.maximum(var, 0.0))}
    return stats


def predictive_rmse(model, data_dict, scale_item_keys, K):
    ce = model.calibrated_expectations
    categories = jnp.arange(K, dtype=jnp.float64)
    probs = model.grm_model_prob_d(
        ce['abilities'], ce['discriminations'],
        ce['difficulties0'], ce.get('ddifficulties'))
    expected = jnp.sum(probs * categories[None, :], axis=-1)
    se_sum, count = 0.0, 0
    for i, key_name in enumerate(scale_item_keys):
        obs = np.array(data_dict[key_name], dtype=np.float64)
        pred_i = np.array(expected[:, i])
        valid = ~np.isnan(obs) & (obs >= 0) & (obs < K)
        se_sum += np.sum((obs[valid] - pred_i[valid]) ** 2)
        count += int(np.sum(valid))
    return float(np.sqrt(se_sum / count))


# ============================================================
# Plotting
# ============================================================

def _tufte(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=3, width=0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)


def plot_forest(scale_item_keys, variant_stats, param_key, xlabel, out_path):
    """Forest plot comparing IS-reweighted posterior summaries."""
    n_items = len(scale_item_keys)
    fig, ax = plt.subplots(figsize=(6, max(4, n_items * 0.3)))
    y_pos = np.arange(n_items)
    n_models = len(variant_stats)
    for k, (label, stats) in enumerate(variant_stats.items()):
        offset = (k - n_models / 2 + 0.5) * 0.2
        ax.errorbar(stats[param_key]['mean'], y_pos + offset,
                    xerr=stats[param_key]['std'],
                    fmt=MARKERS.get(label, 'o'), capsize=2, markersize=4,
                    elinewidth=1, color=COLORS.get(label, 'gray'),
                    alpha=0.7, label=label)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(scale_item_keys,
                       fontsize=max(5, 9 - n_items // 20))
    ax.set_xlabel(xlabel)
    if param_key == 'discriminations':
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3,
                   linewidth=0.5)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=9)
    _tufte(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ability_histogram(eap_dict, out_path):
    """Ability histograms comparing baseline/pairwise/mixed."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for label, eap in eap_dict.items():
        ax.hist(eap, bins=40, histtype='step', linewidth=1.5,
                label=label, color=COLORS.get(label, 'gray'))
    ax.set_xlabel('EAP ability')
    ax.set_ylabel('Count')
    ax.legend(frameon=False, fontsize=9)
    _tufte(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ability_scatter_2d(abilities, scale_names, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(abilities[:, 0], abilities[:, 1], alpha=0.3, s=5,
               color='#4477AA')
    ax.set_xlabel(f'Ability ({scale_names[0]})')
    ax.set_ylabel(f'Ability ({scale_names[1]})')
    r = np.corrcoef(abilities[:, 0], abilities[:, 1])[0, 1]
    ax.set_title(f'r = {r:.3f}', fontsize=10)
    _tufte(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='IS-based factorized IRT: ADVI → MCMC → IS per scale')
    parser.add_argument('--dataset', default='eqsq',
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--num-chains', type=int, default=2)
    parser.add_argument('--num-warmup', type=int, default=500)
    parser.add_argument('--num-samples', type=int, default=500)
    parser.add_argument('--step-size', type=float, default=0.01)
    parser.add_argument('--advi-epochs', type=int, default=2000)
    parser.add_argument('--advi-rank', type=int, default=0)
    parser.add_argument('--use-ipw', action='store_true', default=True)
    parser.add_argument('--no-ipw', action='store_true')
    parser.add_argument('--n-scales', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    import importlib
    import inspect
    from bayesianquilts.irt.grm import GRModel
    from bayesianquilts.irt.factorizedgrm import FactorizedGRModel  # noqa: F811
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

    n_items = len(item_keys)

    # Split items into subscales
    # For EQSQ: natural split E/S; for others: equal partition
    if args.dataset == 'eqsq':
        e_items = [k for k in item_keys if k.startswith('E')]
        s_items = [k for k in item_keys if k.startswith('S')]
        scale_indices = [
            [item_keys.index(k) for k in e_items],
            [item_keys.index(k) for k in s_items],
        ]
        scale_names = ['Empathy', 'Systemizing']
    else:
        items_per_scale = n_items // args.n_scales
        scale_indices = []
        scale_names = []
        for s in range(args.n_scales):
            start = s * items_per_scale
            end = n_items if s == args.n_scales - 1 else (s + 1) * items_per_scale
            scale_indices.append(list(range(start, end)))
            scale_names.append(f'Scale {s}')

    print(f"\n{'='*60}")
    print(f"IS-based Factorized IRT Pipeline: {args.dataset.upper()}")
    print(f"  Items: {n_items}, K: {response_cardinality}")
    print(f"  Scales: {len(scale_indices)}")
    for s, (idx, name) in enumerate(zip(scale_indices, scale_names)):
        print(f"    {name}: {len(idx)} items")
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
    # Step 1: Shared pairwise stacking
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Step 1: Pairwise Stacking Imputation (shared)")
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

    # ================================================================
    # Step 2: Per-scale pipeline
    # ================================================================
    all_eap = {}
    all_is_results = {}
    all_eap_variants = {}

    for dim_idx, (indices, sname) in enumerate(
            zip(scale_indices, scale_names)):
        scale_item_keys = [item_keys[i] for i in indices]
        scale_dir = os.path.join(output_dir, f'scale_{dim_idx}')
        os.makedirs(scale_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"{sname}: {len(indices)} items")
        print(f"{'='*60}")

        # Scale-specific data
        scale_data = {k: base_data[k] for k in scale_item_keys}
        scale_data['person'] = base_data['person']
        if 'sample_weights' in base_data:
            scale_data['sample_weights'] = base_data['sample_weights']

        # ---- 2a: Baseline ADVI ----
        print(f"\n--- {sname}: Baseline ADVI ---")
        scale_model = GRModel(
            item_keys=scale_item_keys, num_people=num_people,
            response_cardinality=response_cardinality, dim=1,
            dtype=jnp.float64,
        )

        def data_factory(d=scale_data):
            yield d

        scale_model.fit(
            data_factory,
            dataset_size=num_people,
            batch_size=num_people,
            num_epochs=args.advi_epochs,
            learning_rate=0.01,
        )

        calibrate_model(scale_model)

        # Marginal ADVI for MCMC warm-start
        scale_model.fit_marginal_advi(
            scale_data,
            num_samples=10,
            num_epochs=args.advi_epochs,
            learning_rate=0.01,
            rank=args.advi_rank,
            seed=args.seed,
            verbose=True,
        )

        # ---- 2b: Marginal MCMC ----
        print(f"\n--- {sname}: Marginal MCMC (baseline) ---")
        mcmc_samples = scale_model.fit_marginal_mcmc(
            scale_data,
            theta_grid=None,
            num_chains=args.num_chains,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            target_accept_prob=0.85,
            step_size=args.step_size,
            seed=args.seed + dim_idx,
            verbose=True,
        )

        # R-hat check — resume if convergence is poor
        max_rhat = print_rhat(mcmc_samples)
        resume_round = 0
        while max_rhat > 1.05 and resume_round < 3:
            resume_round += 1
            print(f"  Max R-hat {max_rhat:.4f} > 1.05, "
                  f"extending chains (round {resume_round}/3)...")
            mcmc_samples = scale_model.fit_marginal_mcmc(
                scale_data,
                num_samples=args.num_samples,
                seed=args.seed + dim_idx * 10 + resume_round,
                verbose=True,
                resume=True,
            )
            max_rhat = print_rhat(mcmc_samples)

        # Standardize
        scale_model.standardize_marginal(scale_data)
        scale_model.fit_surrogate_to_mcmc()

        # EAP
        eap_baseline = scale_model.compute_eap_abilities(scale_data)
        print(f"  EAP: mean={float(jnp.mean(eap_baseline['eap'])):.4f}, "
              f"std={float(jnp.std(eap_baseline['eap'])):.4f}")

        # Save baseline
        np.savez(os.path.join(scale_dir, 'mcmc_baseline.npz'),
                 eap=np.array(eap_baseline['eap']),
                 psd=np.array(eap_baseline['psd']),
                 **{k: np.array(v) for k, v in mcmc_samples.items()})
        scale_model.save_to_disk(
            os.path.join(scale_dir, 'grm_mcmc_baseline'))

        all_eap[dim_idx] = eap_baseline

        # ---- 2c: Build imputation models ----
        print(f"\n--- {sname}: Build imputation models ---")

        def make_factory(d=scale_data):
            def factory():
                yield d
            return factory

        mixed_imputation = IrtMixedImputationModel(
            irt_model=scale_model,
            mice_model=pairwise_model,
            data_factory=make_factory(),
        )
        print(mixed_imputation.summary())

        with open(os.path.join(scale_dir, 'mixed_weights.json'), 'w') as f:
            json.dump(mixed_imputation.weights, f, indent=2)

        pairwise_only = PairwiseOnlyImputationModel(
            mice_model=pairwise_model,
        )

        # ---- 2d: IS reweight ----
        print(f"\n--- {sname}: IS Reweighting ---")
        scale_is = {}
        scale_eap_variants = {'Baseline': np.array(eap_baseline['eap'])}

        for variant_name, imp_model in [('pairwise', pairwise_only),
                                         ('mixed', mixed_imputation)]:
            print(f"\n  IS Reweight: {variant_name}")
            is_res = scale_model.importance_reweight(
                data=scale_data,
                mcmc_samples=mcmc_samples,
                imputation_model=imp_model,
                fn=None,
                verbose=True,
            )
            print(f"  k-hat: {is_res['khat']:.3f}, "
                  f"ESS: {is_res['ess']:.1f}/{is_res['n_samples']}")

            # EAP under IS variant
            eap_var = scale_model.compute_eap_abilities(scale_data)
            scale_eap_variants[variant_name.capitalize()] = np.array(
                eap_var['eap'])

            np.savez(
                os.path.join(scale_dir, f'is_{variant_name}.npz'),
                log_weights=np.array(is_res['log_weights']),
                psis_weights=np.array(is_res['psis_weights']),
                khat=is_res['khat'], ess=is_res['ess'],
                eap=np.array(eap_var['eap']),
            )
            scale_is[variant_name] = is_res
            gc.collect()

        all_is_results[dim_idx] = scale_is
        all_eap_variants[dim_idx] = scale_eap_variants

        # ---- 2e: Per-scale plots ----
        print(f"\n--- {sname}: Generating plots ---")

        # Posterior summaries for forest plots
        baseline_stats = {}
        for k, v in mcmc_samples.items():
            flat = np.asarray(v).reshape(-1, *v.shape[2:])
            baseline_stats[k] = {
                'mean': np.mean(flat, axis=0),
                'std': np.std(flat, axis=0),
            }
        pairwise_stats = is_weighted_stats(
            mcmc_samples, scale_is['pairwise']['psis_weights'])
        mixed_stats = is_weighted_stats(
            mcmc_samples, scale_is['mixed']['psis_weights'])
        variant_stats = {
            'Baseline': baseline_stats,
            'Pairwise': pairwise_stats,
            'Mixed': mixed_stats,
        }

        plot_forest(scale_item_keys, variant_stats, 'discriminations',
                    'Discrimination',
                    os.path.join(scale_dir, 'forest_discriminations.png'))
        print(f"  Saved forest_discriminations.png")

        plot_forest(scale_item_keys, variant_stats, 'difficulties0',
                    'Difficulty (first threshold)',
                    os.path.join(scale_dir, 'forest_difficulties.png'))
        print(f"  Saved forest_difficulties.png")

        plot_ability_histogram(
            scale_eap_variants,
            os.path.join(scale_dir, 'ability_histograms.png'))
        print(f"  Saved ability_histograms.png")

        # RMSE
        eap_arr = np.array(eap_baseline['eap'])
        scale_model.surrogate_sample['abilities'] = jnp.array(
            eap_arr[:, np.newaxis, np.newaxis, np.newaxis]
        )[np.newaxis, ...]
        scale_model.calibrated_expectations = {
            k: jnp.mean(v, axis=0)
            for k, v in scale_model.surrogate_sample.items()
        }
        try:
            rmse = predictive_rmse(
                scale_model, scale_data, scale_item_keys,
                response_cardinality)
            print(f"  {sname} RMSE: {rmse:.4f}")
        except Exception as e:
            print(f"  {sname} RMSE failed: {e}")

        del scale_model
        gc.collect()

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Summary: {args.dataset.upper()} ({len(scale_indices)} scales)")
    print(f"{'='*60}")

    # Multi-dimensional ability profiles (from baseline)
    abilities = np.column_stack([
        np.array(all_eap[d]['eap']) for d in range(len(scale_indices))
    ])
    print(f"  Ability matrix: {abilities.shape}")
    print(f"  Per-dimension means: {np.mean(abilities, axis=0)}")
    print(f"  Per-dimension SDs:   {np.std(abilities, axis=0)}")
    if abilities.shape[1] >= 2:
        r = np.corrcoef(abilities.T)[0, 1]
        print(f"  Correlation: {r:.4f}")
        plot_ability_scatter_2d(
            abilities, scale_names,
            os.path.join(output_dir, 'ability_scatter_2d.png'))
        print(f"  Saved ability_scatter_2d.png")

    # IS diagnostics table
    print(f"\n{'Scale':<15} {'Variant':<12} {'k-hat':>8} {'ESS':>10} {'ESS%':>8}")
    print(f"{'─'*55}")
    for dim_idx in range(len(scale_indices)):
        for variant, res in all_is_results[dim_idx].items():
            ess_pct = 100.0 * res['ess'] / res['n_samples']
            print(f"{scale_names[dim_idx]:<15} {variant:<12} "
                  f"{res['khat']:>8.3f} "
                  f"{res['ess']:>10.1f} {ess_pct:>7.1f}%")
    print(f"{'='*55}")

    # Save combined
    np.savez(
        os.path.join(output_dir, 'abilities_combined.npz'),
        abilities=abilities,
        **{f'eap_dim{d}': np.array(all_eap[d]['eap'])
           for d in range(len(scale_indices))},
        **{f'psd_dim{d}': np.array(all_eap[d]['psd'])
           for d in range(len(scale_indices))},
    )
    print(f"  Output: {output_dir}/")


if __name__ == '__main__':
    main()
