#!/usr/bin/env python
"""Factorized (multiscale) IRT pipeline with survey weights.

Same flow as fit_weighted_irt.py but for instruments with multiple scales.
Each scale is fitted independently, sharing a single pairwise stacking
imputation model across all scales.

Usage:
    python fit_weighted_factorized_irt.py \
        --data responses.csv \
        --scales '{"anxiety": ["anx_1","anx_2","anx_3"], "depression": ["dep_1","dep_2","dep_3"]}' \
        --group-col stratum \
        --group-weights '{"general": 0.95, "clinic": 0.05}' \
        --response-cardinality 5 \
        --output-dir results/
"""

import argparse
import gc
import json
import os

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import spearmanr, kendalltau


# ============================================================
# Shared utilities (same as fit_weighted_irt.py)
# ============================================================

def compute_survey_weights(groups, group_weights):
    n = len(groups)
    w = np.ones(n, dtype=np.float64)
    for g, W_g in group_weights.items():
        g_mask = groups == g
        n_g = g_mask.sum()
        if n_g > 0:
            w[g_mask] = W_g * n / n_g
    n_eff = w.sum()**2 / (w**2).sum()
    return (w * n_eff / w.sum()).astype(np.float32), n_eff


def make_data_dict(df, item_keys, groups=None, group_weights=None):
    data = {}
    for col in item_keys:
        data[col] = df[col].to_numpy().astype(np.float32)
    n = len(df)
    data['person'] = np.arange(n, dtype=np.float32)
    if groups is not None and group_weights is not None:
        w, n_eff = compute_survey_weights(groups, group_weights)
        data['sample_weights'] = w
        print(f"Survey weights: n_eff={n_eff:.0f}, "
              f"min={w.min():.3f}, max={w.max():.3f}")
    return data


def make_data_factory(batch, batch_size, n_people):
    steps_per_epoch = int(np.ceil(n_people / batch_size))

    def factory():
        indices = np.arange(n_people)
        np.random.shuffle(indices)
        n_needed = steps_per_epoch * batch_size
        if n_needed > n_people:
            indices = np.concatenate([
                indices,
                np.random.choice(n_people, n_needed - n_people, replace=True),
            ])
        for start in range(0, n_needed, batch_size):
            idx = indices[start:start + batch_size]
            yield {k: v[idx] for k, v in batch.items()}

    return factory, steps_per_epoch


def calibrate_manually(model, n_samples=32, seed=42):
    try:
        surrogate = model.surrogate_distribution_generator(model.params)
        key = jax.random.PRNGKey(seed)
        samples = surrogate.sample(n_samples, seed=key)
        model.calibrated_expectations = {
            k: jnp.mean(v, axis=0) for k, v in samples.items()
        }
        model.surrogate_sample = samples
    except KeyError as e:
        import sys
        sys.stderr.write(
            f"\033[91mWARNING: Surrogate sampling failed ({e}), "
            f"falling back to point estimates from params. "
            f"Forest plots and predictions will use point estimates "
            f"instead of posterior samples!\033[0m\n"
        )
        sys.stderr.flush()
        point_estimates = {}
        for key_name, value in model.params.items():
            parts = key_name.split('\\')
            if len(parts) >= 4 and parts[-2] == 'normal' and parts[-1] == 'loc':
                point_estimates[parts[0]] = value
        model.calibrated_expectations = point_estimates


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

COLORS = {'Baseline': '#4477AA', 'Pairwise': '#228833', 'Mixed': '#EE6677'}
MARKERS = {'Baseline': 'o', 'Pairwise': 'D', 'Mixed': 's'}


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


def plot_ability_histograms(models_ab, scale_name, out_path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for label, ab in models_ab.items():
        ax.hist(ab, bins=40, histtype='step', linewidth=1.5,
                label=label, color=COLORS.get(label, 'gray'))
    ax.set_xlabel(f'Standardized ability ({scale_name})')
    ax.set_ylabel('Count')
    ax.legend(frameon=False, fontsize=9)
    _tufte(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_w_irt_weights(mixed_model, item_keys, out_path):
    n_items = len(item_keys)
    w_pw = np.array([mixed_model.get_item_weight(k) for k in item_keys])
    fig, ax = plt.subplots(figsize=(max(6, n_items * 0.4), 4))
    x = np.arange(n_items)
    ax.bar(x, 1.0 - w_pw, color='#CCBB44', edgecolor='none', alpha=0.8)
    ax.bar(x, w_pw, bottom=1.0 - w_pw, color='#228833', edgecolor='none', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(item_keys, rotation=90, fontsize=max(5, 8 - n_items // 20))
    ax.set_ylabel('Weight')
    ax.set_ylim(0, 1)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='#CCBB44', label='IRT'),
        Patch(color='#228833', label='Pairwise'),
    ], frameon=False, fontsize=9)
    _tufte(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Main pipeline
# ============================================================

def run(
    df, scales_dict, response_cardinality,
    groups=None, group_weights=None,
    output_dir='results',
    num_epochs=200, batch_size=256, learning_rate=1e-3,
    lr_decay_factor=0.975, sample_size=32,
    n_top_features=20, seed=42,
    reference_group=None,
):
    from bayesianquilts.irt.grm import GRModel
    from bayesianquilts.imputation.pairwise_stacking import PairwiseOrdinalStackingModel
    from bayesianquilts.imputation.mixed import IrtMixedImputationModel

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_people = len(df)

    scale_names = list(scales_dict.keys())
    all_item_keys = []
    for name in scale_names:
        all_item_keys.extend(scales_dict[name])

    print(f"Scales: {scale_names}")
    print(f"Total items: {len(all_item_keys)}")
    for name in scale_names:
        print(f"  {name}: {len(scales_dict[name])} items")

    batch = make_data_dict(df, all_item_keys, groups, group_weights)
    factory, steps_per_epoch = make_data_factory(batch, batch_size, n_people)

    ref_idx = None
    if reference_group is not None and groups is not None:
        ref_idx = np.where(groups == reference_group)[0]
        print(f"Reference group '{reference_group}': {len(ref_idx)} respondents")

    # ---- Stage 1: Pairwise stacking imputation (all items) ----
    print("\n=== Stage 1: Fitting pairwise stacking imputation ===")
    pandas_df = df[all_item_keys].copy().replace(-1, np.nan)

    pairwise_model = PairwiseOrdinalStackingModel(
        pathfinder_num_samples=100, pathfinder_maxiter=50,
        batch_size=512, verbose=True,
    )
    pairwise_model.fit(
        pandas_df, n_top_features=n_top_features, n_jobs=1, seed=seed,
        groups=groups, group_weights=group_weights,
    )
    pairwise_model.compute_optimal_stacking_weights()
    pairwise_model.save(str(out / 'pairwise_stacking_model.yaml'))
    pairwise_model.save_to_disk(str(out / 'pairwise_stacking_model'))
    gc.collect()

    # ---- Per-scale fitting ----
    all_stats = []

    for dim, scale_name in enumerate(scale_names):
        scale_items = scales_dict[scale_name]
        n_items = len(scale_items)
        scale_out = out / scale_name
        scale_out.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Scale: {scale_name} ({n_items} items)")
        print(f"{'='*60}")

        # Stage 2: Baseline
        print(f"\n--- Baseline GRM for {scale_name} ---")
        mdl_base = GRModel(
            item_keys=scale_items, num_people=n_people, dim=1,
            response_cardinality=response_cardinality, dtype=jnp.float32,
        )
        mdl_base.fit(
            factory, batch_size=batch_size, dataset_size=n_people,
            num_epochs=num_epochs, steps_per_epoch=steps_per_epoch,
            learning_rate=learning_rate, lr_decay_factor=lr_decay_factor,
            patience=10, zero_nan_grads=True,
            sample_size=sample_size, seed=seed, max_nan_recoveries=50,
        )
        mdl_base.save_to_disk(str(scale_out / 'grm_baseline'))
        calibrate_manually(mdl_base, n_samples=sample_size, seed=101)
        gc.collect()

        # Stage 3: Mixed imputation model
        print(f"\n--- Mixed imputation for {scale_name} ---")
        mixed_imp = IrtMixedImputationModel(
            irt_model=mdl_base, mice_model=pairwise_model,
            data_factory=factory, irt_elpd_batch_size=4,
        )
        print(mixed_imp.summary())
        with open(scale_out / 'mixed_weights.json', 'w') as f:
            json.dump(mixed_imp.weights, f, indent=2)
        gc.collect()

        # Stage 3b: Adaptive thresholds
        print(f"\n--- Adaptive thresholds for {scale_name} ---")
        mdl_base.imputation_model = mixed_imp
        calibrate_manually(mdl_base, n_samples=sample_size, seed=101)
        mdl_base.compute_adaptive_thresholds(
            factory, baseline_model=mdl_base,
            sample_size=sample_size, seed=seed,
        )
        thresholds = mdl_base._adaptive_thresholds
        mdl_base.imputation_model = None
        with open(scale_out / 'adaptive_thresholds.json', 'w') as f:
            json.dump(thresholds, f, indent=2)
        gc.collect()

        # Stage 4: Pairwise-only
        print(f"\n--- Pairwise-only GRM for {scale_name} ---")
        mdl_pw = GRModel(
            item_keys=scale_items, num_people=n_people, dim=1,
            response_cardinality=response_cardinality, dtype=jnp.float32,
            imputation_model=pairwise_model,
        )
        mdl_pw._adaptive_thresholds = thresholds
        ignored_pw = [k for k, t in thresholds.items() if t >= 1.0]
        print(f"  Imputing {n_items - len(ignored_pw)}/{n_items}")
        mdl_pw.fit(
            factory, batch_size=batch_size, dataset_size=n_people,
            num_epochs=num_epochs, steps_per_epoch=steps_per_epoch,
            learning_rate=learning_rate, lr_decay_factor=lr_decay_factor,
            patience=30, zero_nan_grads=True,
            sample_size=sample_size, seed=seed + 1, max_nan_recoveries=50,
        )
        mdl_pw.save_to_disk(str(scale_out / 'grm_pairwise'))
        calibrate_manually(mdl_pw, n_samples=sample_size, seed=103)
        gc.collect()

        # Stage 5: Mixed
        print(f"\n--- Mixed-imputed GRM for {scale_name} ---")
        mdl_mix = GRModel(
            item_keys=scale_items, num_people=n_people, dim=1,
            response_cardinality=response_cardinality, dtype=jnp.float32,
            imputation_model=mixed_imp,
        )
        mdl_mix._adaptive_thresholds = thresholds
        ignored_mix = [k for k in scale_items
                       if mixed_imp.get_item_weight(k) <= thresholds.get(k, 0.0)]
        print(f"  Imputing {n_items - len(ignored_mix)}/{n_items}")
        mdl_mix.fit(
            factory, batch_size=batch_size, dataset_size=n_people,
            num_epochs=num_epochs, steps_per_epoch=steps_per_epoch,
            learning_rate=learning_rate, lr_decay_factor=lr_decay_factor,
            patience=30, zero_nan_grads=True,
            sample_size=sample_size, seed=seed + 2, max_nan_recoveries=50,
        )
        mdl_mix.save_to_disk(str(scale_out / 'grm_mixed'))
        calibrate_manually(mdl_mix, n_samples=sample_size, seed=102)
        gc.collect()

        # Standardize abilities
        scale_models = {'Baseline': mdl_base, 'Pairwise': mdl_pw, 'Mixed': mdl_mix}
        models_ab = {}
        for label, mdl in scale_models.items():
            stats = mdl.standardize_abilities(reference_idx=ref_idx)
            ab = np.array(mdl.calibrated_expectations['abilities']).flatten()
            models_ab[label] = ab
            mu_d = np.array(stats['mu'])
            print(f"{scale_name}/{label}: ability mean={mu_d[0]:.3f}, std={np.array(stats['sigma'])[0]:.3f}")

        # ELPD-LOO
        for label, mdl in scale_models.items():
            try:
                mdl._compute_elpd_loo(factory, n_samples=100, seed=seed, use_ais=True)
                print(f"  {label} ELPD/person: {mdl.elpd_loo / n_people:.4f}")
            except Exception as e:
                print(f"  {label} ELPD failed: {e}")
            gc.collect()

        # Stats
        rows = []
        ref_ab = models_ab['Mixed']
        for label, mdl in scale_models.items():
            try:
                rmse = predictive_rmse(mdl, batch, scale_items, response_cardinality)
            except Exception:
                rmse = np.nan
            elpd = getattr(mdl, 'elpd_loo', np.nan)
            rho, _ = spearmanr(ref_ab, models_ab[label])
            rows.append({
                'scale': scale_name, 'model': label, 'rmse': rmse,
                'elpd_per_person': elpd / n_people if np.isfinite(elpd) else np.nan,
                'spearman_rho': rho,
            })
        stats_df = pd.DataFrame(rows)
        all_stats.append(stats_df)
        print(f"\n{stats_df.to_string(index=False, float_format='%.4f')}")

        # Plots
        plot_forest(scale_items, scale_models, 'discriminations',
                    'Discrimination', scale_out / 'forest_discriminations.png')
        plot_forest(scale_items, scale_models, 'difficulties0',
                    'Difficulty', scale_out / 'forest_difficulties.png')
        plot_ability_histograms(models_ab, scale_name,
                                scale_out / 'ability_histograms.png')
        plot_w_irt_weights(mixed_imp, scale_items,
                          scale_out / 'w_irt_weights.png')

    # Combined
    combined = pd.concat(all_stats, ignore_index=True)
    print(f"\n{'='*90}")
    print(combined.to_string(index=False, float_format='%.4f'))
    print(f"{'='*90}")
    combined.to_csv(out / 'comparison_stats.csv', index=False)
    print(f"\nAll artifacts saved to {out}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--scales', required=True,
                        help='JSON dict of scale_name -> [item_col, ...]')
    parser.add_argument('--group-col', default=None)
    parser.add_argument('--group-weights', default=None)
    parser.add_argument('--reference-group', default=None)
    parser.add_argument('--response-cardinality', type=int, required=True)
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n-top-features', type=int, default=20)
    parser.add_argument('--sample-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    scales_dict = json.loads(args.scales)
    groups = df[args.group_col].to_numpy() if args.group_col else None
    gw = json.loads(args.group_weights) if args.group_weights else None

    run(
        df, scales_dict, args.response_cardinality,
        groups=groups, group_weights=gw,
        output_dir=args.output_dir,
        num_epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, n_top_features=args.n_top_features,
        sample_size=args.sample_size, seed=args.seed,
        reference_group=args.reference_group,
    )
