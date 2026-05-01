#!/usr/bin/env python
"""Full IRT pipeline with survey weights, optimal stacking, and adaptive thresholds.

Follows the same flow as bayesianquilts/notebooks/irt/run_single_notebook.py:
  1. Fit pairwise stacking imputation model
  2. Fit baseline GRM (no imputation)
  3. Build mixed imputation model (pairwise + IRT stacking)
  4. Compute adaptive ignorability thresholds from baseline model
  5. Fit pairwise-only GRM (with thresholds)
  6. Fit mixed-imputed GRM (with thresholds)
  7. Standardize abilities (anchored to reference group if provided)
  8. Compute ELPD-LOO, RMSE, rank correlations
  9. Produce plots

Usage:
    python fit_weighted_irt.py \
        --data responses.csv \
        --items item_0 item_1 ... item_19 \
        --group-col stratum \
        --group-weights '{"general": 0.90, "oversample": 0.10}' \
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
# Data preparation
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
    w_normalized = (w * n_eff / w.sum()).astype(np.float32)
    return w_normalized, n_eff


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


def plot_ability_histograms(models_abilities, out_path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for label, ab_std in models_abilities.items():
        ax.hist(ab_std, bins=40, histtype='step', linewidth=1.5,
                label=label, color=COLORS.get(label, 'gray'))
    ax.set_xlabel('Standardized ability')
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
    df, item_keys, response_cardinality,
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
    n_items = len(item_keys)

    batch = make_data_dict(df, item_keys, groups, group_weights)
    factory, steps_per_epoch = make_data_factory(batch, batch_size, n_people)

    # Reference group for ability standardization
    ref_idx = None
    if reference_group is not None and groups is not None:
        ref_idx = np.where(groups == reference_group)[0]
        print(f"Reference group '{reference_group}': {len(ref_idx)} respondents")

    # ---- Stage 1: Pairwise stacking imputation ----
    print("\n=== Stage 1: Fitting pairwise stacking imputation ===")
    pandas_df = df[item_keys].copy()
    pandas_df = pandas_df.replace(-1, np.nan)

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

    # ---- Stage 2: Baseline GRM (no imputation) ----
    print("\n=== Stage 2: Fitting baseline GRM ===")
    model_baseline = GRModel(
        item_keys=item_keys, num_people=n_people, dim=1,
        response_cardinality=response_cardinality, dtype=jnp.float32,
    )
    model_baseline.fit(
        factory, batch_size=batch_size, dataset_size=n_people,
        num_epochs=num_epochs, steps_per_epoch=steps_per_epoch,
        learning_rate=learning_rate, lr_decay_factor=lr_decay_factor,
        patience=10, zero_nan_grads=True,
        sample_size=sample_size, seed=seed, max_nan_recoveries=50,
    )
    model_baseline.save_to_disk(str(out / 'grm_baseline'))
    calibrate_manually(model_baseline, n_samples=sample_size, seed=101)
    gc.collect()

    # ---- Stage 3: Build mixed imputation model ----
    print("\n=== Stage 3: Building mixed imputation model ===")
    mixed_imputation = IrtMixedImputationModel(
        irt_model=model_baseline,
        mice_model=pairwise_model,
        data_factory=factory,
        irt_elpd_batch_size=4,
    )
    print(mixed_imputation.summary())
    with open(out / 'mixed_weights.json', 'w') as f:
        json.dump(mixed_imputation.weights, f, indent=2)
    gc.collect()

    # ---- Stage 3b: Compute adaptive ignorability thresholds ----
    print("\n=== Stage 3b: Computing adaptive ignorability thresholds ===")
    # Temporarily set imputation model on baseline to compute thresholds
    model_baseline.imputation_model = mixed_imputation
    calibrate_manually(model_baseline, n_samples=sample_size, seed=101)
    model_baseline.compute_adaptive_thresholds(
        factory, baseline_model=model_baseline,
        sample_size=sample_size, seed=seed,
    )
    adaptive_thresholds = model_baseline._adaptive_thresholds
    model_baseline.imputation_model = None

    with open(out / 'adaptive_thresholds.json', 'w') as f:
        json.dump(adaptive_thresholds, f, indent=2)
    gc.collect()

    # ---- Stage 4: Pairwise-only GRM (with thresholds) ----
    print("\n=== Stage 4: Fitting pairwise-only GRM ===")
    model_pairwise = GRModel(
        item_keys=item_keys, num_people=n_people, dim=1,
        response_cardinality=response_cardinality, dtype=jnp.float32,
        imputation_model=pairwise_model,
    )
    model_pairwise._adaptive_thresholds = adaptive_thresholds
    ignored_pw = [k for k, t in adaptive_thresholds.items() if t >= 1.0]
    print(f"  Imputing {n_items - len(ignored_pw)}/{n_items}, "
          f"ignoring {len(ignored_pw)}/{n_items}")
    model_pairwise.fit(
        factory, batch_size=batch_size, dataset_size=n_people,
        num_epochs=num_epochs, steps_per_epoch=steps_per_epoch,
        learning_rate=learning_rate, lr_decay_factor=lr_decay_factor,
        patience=30, zero_nan_grads=True,
        sample_size=sample_size, seed=seed + 1, max_nan_recoveries=50,
    )
    model_pairwise.save_to_disk(str(out / 'grm_pairwise'))
    calibrate_manually(model_pairwise, n_samples=sample_size, seed=103)
    gc.collect()

    # ---- Stage 5: Mixed-imputed GRM (with thresholds) ----
    print("\n=== Stage 5: Fitting mixed-imputed GRM ===")
    model_mixed = GRModel(
        item_keys=item_keys, num_people=n_people, dim=1,
        response_cardinality=response_cardinality, dtype=jnp.float32,
        imputation_model=mixed_imputation,
    )
    model_mixed._adaptive_thresholds = adaptive_thresholds
    ignored_mix = [k for k in item_keys
                   if mixed_imputation.get_item_weight(k) <= adaptive_thresholds.get(k, 0.0)]
    print(f"  Imputing {n_items - len(ignored_mix)}/{n_items}, "
          f"ignoring {len(ignored_mix)}/{n_items}")
    model_mixed.fit(
        factory, batch_size=batch_size, dataset_size=n_people,
        num_epochs=num_epochs, steps_per_epoch=steps_per_epoch,
        learning_rate=learning_rate, lr_decay_factor=lr_decay_factor,
        patience=30, zero_nan_grads=True,
        sample_size=sample_size, seed=seed + 2, max_nan_recoveries=50,
    )
    model_mixed.save_to_disk(str(out / 'grm_mixed'))
    calibrate_manually(model_mixed, n_samples=sample_size, seed=102)
    gc.collect()

    # ---- Standardize abilities ----
    models = {'Baseline': model_baseline, 'Pairwise': model_pairwise, 'Mixed': model_mixed}
    models_ab = {}
    for label, mdl in models.items():
        stats = mdl.standardize_abilities(reference_idx=ref_idx)
        mu_d, sigma_d = np.array(stats['mu']), np.array(stats['sigma'])
        ab = np.array(mdl.calibrated_expectations['abilities']).flatten()
        models_ab[label] = ab
        suffix = f" (anchored to '{reference_group}')" if ref_idx is not None else ""
        print(f"{label}: ability mean={mu_d[0]:.3f}, std={sigma_d[0]:.3f}{suffix}")

    # ---- ELPD-LOO ----
    n_observed = sum(
        np.sum((batch[k] >= 0) & (batch[k] < response_cardinality) & ~np.isnan(batch[k]))
        for k in item_keys
    )
    for label, mdl in models.items():
        print(f"\n--- ELPD-LOO: {label} ---")
        try:
            mdl._compute_elpd_loo(factory, n_samples=100, seed=seed, use_ais=True)
            print(f"  ELPD/person: {mdl.elpd_loo / n_people:.4f} "
                  f"+/- {mdl.elpd_loo_se / n_people:.4f}")
            print(f"  ELPD/response: {mdl.elpd_loo / n_observed:.4f} "
                  f"+/- {mdl.elpd_loo_se / n_observed:.4f}")
        except Exception as e:
            print(f"  Failed: {e}")
        gc.collect()

    # ---- Summary table ----
    print(f"\n{'='*90}")
    print(f"{'Model':<15} {'RMSE':>10} {'ELPD/person':>20} {'ELPD/response':>20}")
    print(f"{'-'*90}")
    for label, mdl in models.items():
        try:
            rmse = predictive_rmse(mdl, batch, item_keys, response_cardinality)
            rmse_str = f"{rmse:.4f}"
        except Exception:
            rmse_str = "nan"
        elpd = getattr(mdl, 'elpd_loo', np.nan)
        se = getattr(mdl, 'elpd_loo_se', np.nan)
        if np.isfinite(elpd):
            pp = f"{elpd/n_people:.4f} +/- {se/n_people:.4f}"
            pr = f"{elpd/n_observed:.4f} +/- {se/n_observed:.4f}"
        else:
            pp = pr = "nan"
        print(f"{label:<15} {rmse_str:>10} {pp:>20} {pr:>20}")
    print(f"{'='*90}")

    # ---- Plots ----
    plot_forest(item_keys, models, 'discriminations', 'Discrimination',
                out / 'forest_discriminations.png')
    plot_forest(item_keys, models, 'difficulties0', 'Difficulty (first threshold)',
                out / 'forest_difficulties.png')
    plot_ability_histograms(models_ab, out / 'ability_histograms.png')
    plot_w_irt_weights(mixed_imputation, item_keys, out / 'w_irt_weights.png')

    print(f"\nAll artifacts saved to {out}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--items', nargs='+', required=True)
    parser.add_argument('--group-col', default=None)
    parser.add_argument('--group-weights', default=None)
    parser.add_argument('--reference-group', default=None,
                        help='Group label to anchor ability standardization')
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
    groups = df[args.group_col].to_numpy() if args.group_col else None
    gw = json.loads(args.group_weights) if args.group_weights else None

    run(
        df, args.items, args.response_cardinality,
        groups=groups, group_weights=gw,
        output_dir=args.output_dir,
        num_epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, n_top_features=args.n_top_features,
        sample_size=args.sample_size, seed=args.seed,
        reference_group=args.reference_group,
    )
