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
5. Produce comparison plots and compute LOO-RMSE / LOO-ELPD
6. Save all results

Usage:
    uv run python fit_marginal_irt.py --dataset eqsq
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


def compute_max_rhat(mcmc_samples, prefix="  "):
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


def calibrate_model(model, seed=101, n_samples=32):
    surrogate = model.surrogate_distribution_generator(model.params)
    key = jax.random.PRNGKey(seed)
    samples = surrogate.sample(n_samples, seed=key)
    model.surrogate_sample = samples
    model.calibrated_expectations = {
        k: jnp.mean(v, axis=0) for k, v in samples.items()
    }


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


def plot_forest(item_keys, models, param_key, xlabel, out_path):
    n_items = len(item_keys)
    fig, ax = plt.subplots(figsize=(6, max(4, n_items * 0.3)))
    y_pos = np.arange(n_items)
    n_models = len(models)
    for k, (label, mdl) in enumerate(models.items()):
        vals = np.array(mdl.surrogate_sample[param_key]).reshape(-1, n_items)
        offset = (k - n_models / 2 + 0.5) * 0.2
        ax.errorbar(vals.mean(0), y_pos + offset, xerr=vals.std(0),
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
# Variant runners
# ============================================================

def run_variant_advi(model, data, variant_name, output_dir,
                     num_samples=10, num_epochs=2000, learning_rate=0.01,
                     rank=0, seed=42):
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

    eap_result = model.compute_eap_abilities(data)
    print(f"  EAP std: {float(jnp.std(eap_result['eap'])):.4f}, "
          f"PSD: {float(jnp.mean(eap_result['psd'])):.4f}")

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
    """Run marginal MCMC for one variant, standardize, and save.

    Returns (model, mcmc_samples, eap_result) so caller can use for plots.
    """
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

    # R-hat check — resume if convergence is poor
    max_rhat = compute_max_rhat(mcmc_samples)
    resume_round = 0
    while max_rhat > 1.05 and resume_round < 3:
        resume_round += 1
        print(f"\n  Max R-hat {max_rhat:.4f} > 1.05, "
              f"extending chains (round {resume_round}/3)...")
        mcmc_samples = model.fit_marginal_mcmc(
            data,
            theta_grid=None,
            num_samples=num_samples,
            seed=seed + resume_round * 100,
            verbose=True,
            resume=True,
        )
        max_rhat = compute_max_rhat(mcmc_samples)

    # Standardize
    stats = model.standardize_marginal(data)

    # EAP
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

    return model, mcmc_samples, eap_result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Full marginal IRT pipeline: ADVI + MCMC x 3 variants')
    parser.add_argument('--dataset', default='eqsq',
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--num-chains', type=int, default=2)
    parser.add_argument('--num-warmup', type=int, default=500)
    parser.add_argument('--num-samples', type=int, default=500)
    parser.add_argument('--step-size', type=float, default=0.01)
    parser.add_argument('--advi-epochs', type=int, default=2000)
    parser.add_argument('--advi-rank', type=int, default=0)
    parser.add_argument('--skip-advi', action='store_true')
    parser.add_argument('--skip-mcmc', action='store_true')
    parser.add_argument('--use-ipw', action='store_true', default=True)
    parser.add_argument('--no-ipw', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-bcm', action='store_true',
                        help='Skip Step 6 (BCM training + gofluttercat bundle export)')
    parser.add_argument('--bcm-subset-sizes', type=int, nargs='+',
                        default=[5, 10, 20, 40],
                        help='Item subset sizes for BCM training (the full '
                             'battery and near-full sizes are always added on '
                             'top so the correction vanishes at completion)')
    parser.add_argument('--bcm-n-subsets', type=int, default=100,
                        help='Random subset draws per size')
    parser.add_argument('--bcm-max-respondents', type=int, default=200,
                        help='Stratified subsample size for BCM training')
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
    from bayesianquilts.io.converged import export_artifact

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
    import pandas as pd
    pandas_df = pd.DataFrame({
        k: np.where(df[k].to_numpy() == -1, np.nan,
                    df[k].to_numpy().astype(float))
        for k in item_keys
    })
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

    # Precompute imputation PMFs for pairwise variant
    def make_pairwise_data():
        model_tmp = GRModel(
            item_keys=item_keys, num_people=num_people,
            response_cardinality=response_cardinality, dim=1,
            imputation_model=pairwise_model, dtype=jnp.float64,
            share_discriminations=True,
        )
        data = dict(base_data)
        pmfs, _ = model_tmp._compute_batch_pmfs(data)
        if pmfs is not None:
            data['_imputation_pmfs'] = pmfs
        del model_tmp
        return data

    pairwise_data = make_pairwise_data()

    # ================================================================
    # Step 2: Baseline ADVI (for mixed imputation model)
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
            share_discriminations=True,
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

    # Standardize baseline_model BEFORE wiring it into the mixed imputation
    # so the IRT-component PMFs are computed on the same N(0,1) ability
    # scale that the downstream MCMC variants will use after
    # standardize_marginal. Without this, the imputation PMFs reference an
    # unscaled theta while the GRM scoring uses the scaled theta, which
    # would inflate residual bias on the BCM training triples.
    base_std = baseline_model.standardize_abilities()
    print(f"  Standardized baseline: "
          f"mu={float(jnp.mean(base_std['mu'])):.4f}, "
          f"sigma={float(jnp.mean(base_std['sigma'])):.4f}")

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
            share_discriminations=True,
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

    variant_data = {
        'baseline': dict(base_data),
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
    mcmc_models = {}
    mcmc_eaps = {}

    if not args.skip_mcmc:
        print(f"\n{'='*60}")
        print(f"Step 4: Marginal MCMC (3 variants)")
        print(f"{'='*60}")

        for i, (variant_name, data) in enumerate(variant_data.items()):
            model = GRModel.load_from_disk(baseline_grm_path)
            model, mcmc_samples, eap_result = run_variant_mcmc(
                model, data, variant_name, output_dir,
                num_chains=args.num_chains,
                num_warmup=args.num_warmup,
                num_samples=args.num_samples,
                step_size=args.step_size,
                seed=args.seed + i,
            )
            mcmc_models[variant_name] = model
            mcmc_eaps[variant_name] = eap_result
            gc.collect()

    # ================================================================
    # Step 5: Model evaluation and plots
    # ================================================================
    if mcmc_models:
        print(f"\n{'='*60}")
        print(f"Step 5: Evaluation and Plots")
        print(f"{'='*60}")

        named_models = {
            'Baseline': mcmc_models.get('baseline'),
            'Pairwise': mcmc_models.get('pairwise'),
            'Mixed': mcmc_models.get('mixed'),
        }
        named_models = {k: v for k, v in named_models.items() if v is not None}

        # Forest plots
        plot_forest(item_keys, named_models, 'discriminations',
                    'Discrimination',
                    os.path.join(output_dir, 'forest_discriminations.png'))
        print(f"  Saved forest_discriminations.png")

        plot_forest(item_keys, named_models, 'difficulties0',
                    'Difficulty (first threshold)',
                    os.path.join(output_dir, 'forest_difficulties.png'))
        print(f"  Saved forest_difficulties.png")

        # Ability histograms
        models_ab = {}
        for vname, eap in mcmc_eaps.items():
            label = vname.capitalize()
            models_ab[label] = np.array(eap['eap'])

        plot_ability_histograms(
            models_ab, os.path.join(output_dir, 'ability_histograms.png'))
        print(f"  Saved ability_histograms.png")

        # Ability scatter: baseline vs imputation variants
        if 'Baseline' in models_ab:
            for label in ['Pairwise', 'Mixed']:
                if label in models_ab:
                    plot_ability_scatter(
                        models_ab['Baseline'], models_ab[label], label,
                        os.path.join(output_dir,
                                     f'ability_scatter_{label.lower()}.png'))
                    print(f"  Saved ability_scatter_{label.lower()}.png")

        # LOO-RMSE and LOO-ELPD
        n_observed = sum(
            np.sum((base_data[k] >= 0)
                   & (base_data[k] < response_cardinality)
                   & ~np.isnan(base_data[k]))
            for k in item_keys
        )

        print(f"\n{'='*70}")
        print(f"{'Model':<12} {'RMSE':>8} {'ELPD/resp':>12} {'ELPD SE':>10}")
        print(f"{'─'*70}")

        for label, mdl in named_models.items():
            # RMSE
            try:
                rmse = predictive_rmse(mdl, base_data, item_keys,
                                       response_cardinality)
                rmse_str = f"{rmse:.4f}"
            except Exception:
                rmse_str = "nan"

            # ELPD-LOO
            elpd_str = "nan"
            se_str = "nan"
            try:
                def factory_fn():
                    yield base_data
                mdl._compute_elpd_loo(
                    factory_fn, n_samples=100, seed=args.seed, use_ais=True)
                elpd_str = f"{mdl.elpd_loo / n_observed:.4f}"
                se_str = f"{mdl.elpd_loo_se / n_observed:.4f}"
            except Exception as e:
                print(f"  {label} ELPD failed: {e}")

            print(f"{label:<12} {rmse_str:>8} {elpd_str:>12} {se_str:>10}")
            gc.collect()

        print(f"{'='*70}")

    # ================================================================
    # Step 6: converged artifact + BCM (mixed variant)
    #
    # Upstream's ``bayesianquilts.io.converged.export_artifact`` writes
    # the canonical ``items/`` + ``scales.json`` + ``imputation/`` +
    # ``manifest.yaml`` bundle that libfab and gofluttercat both read.
    # We then fit BCMConditional (Python-side, richer per-item-indicator
    # corrector) and BCMSet (Go-side, per-J isotonic, drop-in for
    # gofluttercat's biascorrection package) on top of the same triples
    # and save them alongside the converged bundle.
    # ================================================================
    final_irt = mcmc_models.get('mixed') if mcmc_models else None
    if final_irt is not None:
        bundle_dir = os.path.join(output_dir, 'converged')
        print(f"\n=== Exporting converged artifact (libfab + gofluttercat) ===")
        # export_artifact needs an imputation model with .save() / .save_to_disk();
        # IrtMixedImputationModel has neither, so pass the underlying
        # PairwiseOrdinalStackingModel. Mixed-blend weights go into extra_manifest.
        mixed_weights = None
        if hasattr(mixed_imputation, '_weights') and mixed_imputation._weights:
            mixed_weights = {str(k): float(v)
                             for k, v in mixed_imputation._weights.items()}
        export_artifact(
            irt_model=final_irt,
            imputation_model=pairwise_model,
            out_dir=bundle_dir,
            scale_names=[args.dataset],
            fit_method='marginal_mcmc',
            source_script=os.path.basename(__file__),
            extra_manifest={'dataset': args.dataset, 'use_ipw': use_ipw,
                            'standardized': True,
                            'mixed_weights': mixed_weights},
        )
        print(f"  -> {bundle_dir}")

        if not args.skip_bcm:
            print(f"\n=== Step 6: BCM (mixed variant) ===")
            from _bcm_triples import (
                build_bcm_triples, extract_item_params_from_mcmc,
            )
            from libfabulouscatpy.biascorrection import (
                BCMConditional, fit_bcm_set,
            )

            bcm_model = final_irt
            bcm_model.imputation_model = mixed_imputation
            bcm_item_params = extract_item_params_from_mcmc(bcm_model)

            rng_bcm = np.random.default_rng(args.seed + 7)
            subset_scores, indicators_mat, golds, _ = build_bcm_triples(
                model=bcm_model,
                base_data=base_data,
                item_keys=item_keys,
                subset_sizes=args.bcm_subset_sizes,
                n_subsets_per_size=args.bcm_n_subsets,
                max_respondents=args.bcm_max_respondents,
                rng=rng_bcm,
                item_params=bcm_item_params,
            )
            print(f"  n_triples = {subset_scores.size}")

            # Richer Python-side corrector with per-item indicator features.
            bcm_cond = BCMConditional.fit(
                subset_scores, indicators_mat, golds,
                item_keys=item_keys, scale_name=args.dataset,
                n_folds=5, seed=args.seed,
                max_iter=200, learning_rate=0.05, max_depth=4,
            )
            bcm_cond_path = os.path.join(
                bundle_dir, f'bcm_{args.dataset}_conditional.joblib')
            bcm_cond.save(bcm_cond_path)
            l2_naive = float(np.sqrt(np.mean((subset_scores - golds) ** 2)))
            l2_bcm = float(np.sqrt(np.mean(
                (bcm_cond.oof_predictions - golds) ** 2)))
            print(f"  BCMConditional -> {bcm_cond_path}")
            print(f"  L2(subset - gold) = {l2_naive:.4f}; "
                  f"L2(BCMcond - gold) = {l2_bcm:.4f}")

            # Per-J isotonic BCMSet — what gofluttercat's Go-side reader expects.
            js = indicators_mat.sum(axis=1).astype(int)
            cells = {}
            for j in np.unique(js):
                if j < 1:
                    continue
                mask = js == j
                if mask.sum() < 2:
                    continue
                cells[int(j)] = (subset_scores[mask], golds[mask])
            bcm_set = fit_bcm_set(cells, scale=args.dataset)
            bcm_set_path = os.path.join(bundle_dir, f'bcm_{args.dataset}.json')
            bcm_set.save(bcm_set_path)
            print(f"  BCMSet (per-J isotonic) -> {bcm_set_path}")

    print(f"\n{'='*60}")
    print(f"Pipeline complete: {args.dataset.upper()}")
    print(f"  Output: {output_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
