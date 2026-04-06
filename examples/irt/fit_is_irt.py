#!/usr/bin/env python
"""IS-based marginal IRT pipeline: ADVI → warm-start MCMC → IS reweight.

Demonstrates the importance-sampling approach for imputation-based IRT models.
Instead of running separate MCMC chains for each model variant, we:

1. Fit pairwise stacking imputation model
2. Fit baseline via marginal ADVI (needed for mixed imputation + MCMC init)
3. Run marginal MCMC on the baseline model only (warm-started from ADVI)
4. Reweight baseline MCMC samples via IS for pairwise and mixed variants
5. Produce comparison plots and compute LOO-RMSE / LOO-ELPD

This is much cheaper than running independent MCMC for each variant, since
MCMC is the bottleneck and IS reweighting is O(S × N) per variant.

Usage:
    uv run python fit_is_irt.py --dataset eqsq
    uv run python fit_is_irt.py --dataset rwa --step-size 0.001
    uv run python fit_is_irt.py --dataset npi --num-samples 1000
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


def compute_max_rhat(mcmc_samples):
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
            print(f"  {var_name} R-hat: "
                  f"mean={np.mean(r_hat):.4f}, max={max_rhat:.4f}")
    print(f"  Max R-hat (overall): {max_rhat_overall:.4f}")
    return max_rhat_overall


def calibrate_model(model, seed=101, n_samples=32):
    surrogate = model.surrogate_distribution_generator(model.params)
    key = jax.random.PRNGKey(seed)
    samples = surrogate.sample(n_samples, seed=key)
    model.surrogate_sample = samples
    model.calibrated_expectations = {
        k: jnp.mean(v, axis=0) for k, v in samples.items()
    }


def is_weighted_stats(mcmc_samples, psis_weights):
    """Compute IS-weighted mean and std for each parameter."""
    stats = {}
    for k, v in mcmc_samples.items():
        flat = np.asarray(v).reshape(-1, *v.shape[2:])
        w = psis_weights[:, None] if flat.ndim > 1 else psis_weights
        mean = np.sum(w * flat, axis=0)
        var = np.sum(w * (flat - mean) ** 2, axis=0)
        stats[k] = {'mean': mean, 'std': np.sqrt(np.maximum(var, 0.0))}
    return stats


def predictive_rmse(model, data_dict, item_keys, K):
    ce = model.calibrated_expectations
    categories = jnp.arange(K, dtype=jnp.float64)
    probs = model.grm_model_prob_d(
        ce['abilities'], ce['discriminations'],
        ce['difficulties0'], ce.get('ddifficulties'))
    expected = jnp.sum(probs * categories[None, :], axis=-1)
    se_sum, count = 0.0, 0
    for i, key_name in enumerate(item_keys):
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


def plot_forest(item_keys, variant_stats, param_key, xlabel, out_path):
    """Forest plot comparing posterior summaries across variants.

    variant_stats: dict of {label: {'mean': array, 'std': array}}
    """
    n_items = len(item_keys)
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
    ax.set_yticklabels(item_keys, fontsize=max(5, 9 - n_items // 20))
    ax.set_xlabel(xlabel)
    if param_key == 'discriminations':
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=9)
    _tufte(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ability_histograms(models_ab, out_path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for label, ab in models_ab.items():
        ax.hist(ab, bins=40, histtype='step', linewidth=1.5,
                label=label, color=COLORS.get(label, 'gray'))
    ax.set_xlabel('Standardized ability')
    ax.set_ylabel('Count')
    ax.legend(frameon=False, fontsize=9)
    _tufte(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ability_scatter(ab_base, ab_other, label_other, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ab_base, ab_other, alpha=0.3, s=5, color='#4477AA')
    lims = [min(ab_base.min(), ab_other.min()) - 0.2,
            max(ab_base.max(), ab_other.max()) + 0.2]
    ax.plot(lims, lims, '--', color='gray', linewidth=0.5)
    ax.set_xlabel('Baseline ability')
    ax.set_ylabel(f'{label_other} ability')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    _tufte(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# IS Reweighting
# ============================================================

def run_is_reweight(model, data, mcmc_samples, imputation_model,
                    variant_name, output_dir, verbose=True):
    print(f"\n{'─'*50}")
    print(f"  IS Reweight: {variant_name}")
    print(f"{'─'*50}")
    sys.stdout.flush()

    is_result = model.importance_reweight(
        data=data,
        mcmc_samples=mcmc_samples,
        imputation_model=imputation_model,
        fn=None,
        verbose=verbose,
    )

    print(f"  k-hat: {is_result['khat']:.3f}")
    print(f"  ESS: {is_result['ess']:.1f}/{is_result['n_samples']}")
    print(f"  Tempered: {is_result['tempered']}")

    eap_result = model.compute_eap_abilities(data)
    print(f"  EAP std: {float(jnp.std(eap_result['eap'])):.4f}, "
          f"PSD: {float(jnp.mean(eap_result['psd'])):.4f}")

    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, f'is_{variant_name}.npz'),
        log_weights=np.array(is_result['log_weights']),
        psis_weights=np.array(is_result['psis_weights']),
        khat=is_result['khat'],
        ess=is_result['ess'],
        tempered=is_result['tempered'],
        eap=np.array(eap_result['eap']),
        psd=np.array(eap_result['psd']),
    )

    return is_result, eap_result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='IS-based marginal IRT: ADVI → MCMC baseline → IS reweight')
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
    print(f"IS-based Marginal IRT Pipeline: {args.dataset.upper()}")
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
    # Step 1: Pairwise stacking imputation
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

    # ================================================================
    # Step 2: Baseline ADVI
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Step 2: Baseline ADVI")
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
            num_epochs=args.advi_epochs,
            learning_rate=0.01,
        )
        baseline_model.save_to_disk(baseline_grm_path)
        print(f"  Saved to {baseline_grm_path}")

    calibrate_model(baseline_model)

    # Marginal ADVI for MCMC warm-start
    print(f"  Running marginal ADVI for MCMC warm-start...")
    baseline_model.fit_marginal_advi(
        base_data,
        num_samples=10,
        num_epochs=args.advi_epochs,
        learning_rate=0.01,
        rank=args.advi_rank,
        seed=args.seed,
        verbose=True,
    )

    # ================================================================
    # Step 3: Marginal MCMC (baseline only)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Step 3: Marginal MCMC (baseline only, ADVI warm-start)")
    print(f"{'='*60}")

    mcmc_samples = baseline_model.fit_marginal_mcmc(
        base_data,
        theta_grid=None,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        target_accept_prob=0.85,
        step_size=args.step_size,
        seed=args.seed,
        verbose=True,
    )

    # R-hat check — resume if convergence is poor
    max_rhat = compute_max_rhat(mcmc_samples)
    resume_round = 0
    while max_rhat > 1.05 and resume_round < 3:
        resume_round += 1
        print(f"\n  Max R-hat {max_rhat:.4f} > 1.05, "
              f"extending chains (round {resume_round}/3)...")
        mcmc_samples = baseline_model.fit_marginal_mcmc(
            base_data,
            theta_grid=None,
            num_samples=args.num_samples,
            seed=args.seed + resume_round * 100,
            verbose=True,
            resume=True,
        )
        max_rhat = compute_max_rhat(mcmc_samples)

    # Standardize and compute EAP
    stats = baseline_model.standardize_marginal(base_data)
    baseline_model.fit_surrogate_to_mcmc()
    eap_baseline = baseline_model.compute_eap_abilities(base_data)

    # Inject EAP into surrogate_sample for ELPD-LOO
    eap_arr = np.array(eap_baseline['eap'])
    baseline_model.surrogate_sample['abilities'] = jnp.array(
        eap_arr[:, np.newaxis, np.newaxis, np.newaxis]
    )[np.newaxis, ...]

    # Save baseline
    save_dict = {'eap': np.array(eap_baseline['eap']),
                 'psd': np.array(eap_baseline['psd']),
                 'standardize_mu': stats['mu'],
                 'standardize_sigma': stats['sigma']}
    for var_name, samples in mcmc_samples.items():
        save_dict[var_name] = np.array(samples)
    np.savez(os.path.join(output_dir, 'mcmc_baseline.npz'), **save_dict)
    baseline_model.save_to_disk(
        os.path.join(output_dir, 'grm_mcmc_baseline'))

    # ================================================================
    # Step 4: Build imputation models
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Step 4: Build imputation models")
    print(f"{'='*60}")

    def make_data_factory():
        def factory():
            yield base_data
        return factory

    mixed_imputation = IrtMixedImputationModel(
        irt_model=baseline_model,
        mice_model=pairwise_model,
        data_factory=make_data_factory(),
    )
    print(mixed_imputation.summary())

    with open(os.path.join(output_dir, 'mixed_weights.json'), 'w') as f:
        json.dump(mixed_imputation.weights, f, indent=2)

    pairwise_only = PairwiseOnlyImputationModel(
        mice_model=pairwise_model,
    )

    # ================================================================
    # Step 5: IS reweight for pairwise and mixed
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Step 5: IS Reweighting (pairwise + mixed)")
    print(f"{'='*60}")

    is_results = {}
    eap_results = {}

    is_results['pairwise'], eap_results['pairwise'] = run_is_reweight(
        baseline_model, base_data, mcmc_samples,
        pairwise_only, 'pairwise', output_dir,
    )
    gc.collect()

    is_results['mixed'], eap_results['mixed'] = run_is_reweight(
        baseline_model, base_data, mcmc_samples,
        mixed_imputation, 'mixed', output_dir,
    )
    gc.collect()

    # ================================================================
    # Step 6: Compute stats (LOO-RMSE, LOO-ELPD)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Step 6: Model Evaluation")
    print(f"{'='*60}")

    # Predictive RMSE for baseline
    rmse_baseline = predictive_rmse(
        baseline_model, base_data, item_keys, response_cardinality)
    print(f"  Baseline RMSE: {rmse_baseline:.4f}")

    # ELPD-LOO for baseline
    elpd_baseline = np.nan
    try:
        def data_factory_elpd():
            yield base_data
        baseline_model._compute_elpd_loo(
            data_factory_elpd, n_samples=100, seed=args.seed, use_ais=True)
        elpd_baseline = baseline_model.elpd_loo
        print(f"  Baseline ELPD-LOO: {elpd_baseline:.2f} "
              f"+/- {baseline_model.elpd_loo_se:.2f}")
    except Exception as e:
        print(f"  Baseline ELPD-LOO failed: {e}")

    # ================================================================
    # Step 7: Plots
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Step 7: Generating plots")
    print(f"{'='*60}")

    # Compute posterior summaries for each variant
    # Baseline: unweighted MCMC means/stds
    baseline_stats = {}
    for k, v in mcmc_samples.items():
        flat = np.asarray(v).reshape(-1, *v.shape[2:])
        baseline_stats[k] = {
            'mean': np.mean(flat, axis=0),
            'std': np.std(flat, axis=0),
        }

    # IS-weighted summaries for pairwise and mixed
    pairwise_stats = is_weighted_stats(
        mcmc_samples, is_results['pairwise']['psis_weights'])
    mixed_stats = is_weighted_stats(
        mcmc_samples, is_results['mixed']['psis_weights'])

    variant_stats = {
        'Baseline': baseline_stats,
        'Pairwise': pairwise_stats,
        'Mixed': mixed_stats,
    }

    # Forest plots
    plot_forest(item_keys, variant_stats, 'discriminations',
                'Discrimination', os.path.join(output_dir, 'forest_discriminations.png'))
    print(f"  Saved forest_discriminations.png")

    plot_forest(item_keys, variant_stats, 'difficulties0',
                'Difficulty (first threshold)',
                os.path.join(output_dir, 'forest_difficulties.png'))
    print(f"  Saved forest_difficulties.png")

    # Ability histograms
    models_ab = {
        'Baseline': np.array(eap_baseline['eap']),
        'Pairwise': np.array(eap_results['pairwise']['eap']),
        'Mixed': np.array(eap_results['mixed']['eap']),
    }
    plot_ability_histograms(models_ab,
                            os.path.join(output_dir, 'ability_histograms.png'))
    print(f"  Saved ability_histograms.png")

    # Ability scatter: baseline vs each imputation variant
    for label in ['Pairwise', 'Mixed']:
        plot_ability_scatter(
            models_ab['Baseline'], models_ab[label], label,
            os.path.join(output_dir, f'ability_scatter_{label.lower()}.png'))
        print(f"  Saved ability_scatter_{label.lower()}.png")

    # ================================================================
    # Summary table
    # ================================================================
    n_observed = sum(
        np.sum((base_data[k] >= 0) & (base_data[k] < response_cardinality)
               & ~np.isnan(base_data[k]))
        for k in item_keys
    )

    print(f"\n{'='*70}")
    print(f"{'Variant':<12} {'k-hat':>8} {'ESS':>8} {'ESS%':>7} "
          f"{'RMSE':>8} {'ELPD/resp':>12}")
    print(f"{'─'*70}")
    print(f"{'Baseline':<12} {'—':>8} "
          f"{args.num_chains * args.num_samples:>8} {'100.0':>7} "
          f"{rmse_baseline:>8.4f} "
          f"{elpd_baseline / n_observed:>12.4f}" if np.isfinite(elpd_baseline)
          else f"{'Baseline':<12} {'—':>8} "
          f"{args.num_chains * args.num_samples:>8} {'100.0':>7} "
          f"{rmse_baseline:>8.4f} {'nan':>12}")
    for name, res in is_results.items():
        ess_pct = 100.0 * res['ess'] / res['n_samples']
        print(f"{name:<12} {res['khat']:>8.3f} "
              f"{res['ess']:>8.1f} {ess_pct:>6.1f}% "
              f"{'—':>8} {'—':>12}")
    print(f"{'='*70}")
    print(f"  Output: {output_dir}/")


if __name__ == '__main__':
    main()
