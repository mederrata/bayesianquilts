#!/usr/bin/env python
"""Run full bouldering experiments for the journal article.

Produces:
  1. Real data evaluation table (Table 1 equivalent): RMSE, ELPD/n, ELPD/resp
     for Baseline, Univariate, Mixed — men and women separately.
  2. Synthetic evaluation table (Table 2 equivalent): Spearman rho, Kendall tau,
     RMSE, ELPD/obs — men and women separately.
  3. Forest plots of top 10 climbers' ability distributions (baseline vs imputed).

Usage:
    python run_bouldering_experiments.py --gender men
    python run_bouldering_experiments.py --gender women
    python run_bouldering_experiments.py --gender both
    python run_bouldering_experiments.py --gender both --skip-synthetic  # real data + plots only
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Add synthetic common to path
SCRIPT_DIR = Path(__file__).resolve().parent
IRT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(IRT_DIR / 'synthetic'))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bouldering_data(gender='men'):
    """Load bouldering data and return (data_dict, item_keys, K, N, climber_names)."""
    from bayesianquilts.data.bouldering import (
        get_data, item_keys, response_cardinality, item_labels, climber_names,
    )
    df, num_people = get_data(polars_out=True, gender=gender)
    # Re-import after get_data populates module globals
    from bayesianquilts.data import bouldering as bmod
    ik = list(bmod.item_keys)
    K = bmod.response_cardinality
    labels = dict(bmod.item_labels)
    names = list(bmod.climber_names)

    data = {}
    for col in df.columns:
        data[col] = df[col].to_numpy().astype(np.float64)
    data['person'] = np.arange(num_people, dtype=np.float64)

    return data, ik, K, num_people, names, labels, df


def make_data_factory(data_dict, batch_size, num_people):
    """Batched data factory with wrap-around for even batch sizes."""
    steps_per_epoch = int(np.ceil(num_people / batch_size))

    def data_factory():
        indices = np.arange(num_people)
        np.random.shuffle(indices)
        n_needed = steps_per_epoch * batch_size
        if n_needed > num_people:
            indices = np.concatenate([
                indices,
                np.random.choice(num_people, n_needed - num_people, replace=True),
            ])
        for start in range(0, n_needed, batch_size):
            idx_batch = indices[start:start + batch_size]
            yield {k: v[idx_batch] for k, v in data_dict.items()}

    return data_factory


def calibrate_manually(model, n_samples=32, seed=42):
    """Generate surrogate samples and calibrated expectations."""
    try:
        surrogate = model.surrogate_distribution_generator(model.params)
        key = jax.random.PRNGKey(seed)
        samples = surrogate.sample(n_samples, seed=key)
        expectations = {k: jnp.mean(v, axis=0) for k, v in samples.items()}
        model.calibrated_expectations = expectations
        model.surrogate_sample = samples
    except KeyError as e:
        print(f"  Warning: surrogate sampling failed ({e}), using point estimates")
        point_estimates = {}
        for key_name, value in model.params.items():
            parts = key_name.split('\\')
            if len(parts) >= 4:
                param_name = parts[0]
                if parts[-2] == 'normal' and parts[-1] == 'loc':
                    point_estimates[param_name] = value
        model.calibrated_expectations = point_estimates


# ---------------------------------------------------------------------------
# Real data pipeline: fit baseline, MICE-only, mixed
# ---------------------------------------------------------------------------

def run_real_data(gender, work_dir, skip_baseline=False, skip_mice=False,
                  num_epochs=200, imputed_epochs=None, batch_size=256,
                  learning_rate=2e-4, lr_decay_factor=0.975, patience=30):
    """Fit the 3-model pipeline on real bouldering data for one gender.

    Args:
        imputed_epochs: Number of epochs for imputed models (default: same as
            num_epochs). Use fewer epochs for sparse data where imputation is slow.

    Returns:
        (models_dict, data_dict, item_keys, K, N, climber_names)
    """
    if imputed_epochs is None:
        imputed_epochs = num_epochs
    from bayesianquilts.irt.grm import GRModel
    from bayesianquilts.imputation.mice_loo import MICEBayesianLOO
    from bayesianquilts.imputation.mixed import IrtMixedImputationModel

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    data_dict, item_keys, K, N, names, labels, df = load_bouldering_data(gender)
    n_items = len(item_keys)
    steps_per_epoch = int(np.ceil(N / batch_size))

    print(f"\n{'='*60}")
    print(f"IFSC Bouldering ({gender}) — Real Data")
    print(f"People: {N}, Items: {n_items}, K: {K}")
    print(f"{'='*60}")

    factory_fn = lambda: make_data_factory(data_dict, batch_size, N)()

    # Count observed responses
    n_observed = sum(
        np.sum((data_dict[k] >= 0) & (data_dict[k] < K) & ~np.isnan(data_dict[k]))
        for k in item_keys
    )
    pct_missing = 1.0 - n_observed / (N * n_items)
    print(f"Observed responses: {n_observed} ({pct_missing:.1%} missing)")

    # --- Stage 1: Baseline ---
    baseline_path = work_dir / 'grm_baseline'
    if skip_baseline and (baseline_path / 'params.h5').exists():
        print("\n--- Loading baseline GRM ---")
        model_baseline = GRModel.load_from_disk(baseline_path)
        calibrate_manually(model_baseline, n_samples=32, seed=101)
    else:
        print("\n--- Fitting baseline GRM ---")
        model_baseline = GRModel(
            item_keys=item_keys, num_people=N, dim=1,
            response_cardinality=K, dtype=jnp.float64,
        )
        # Snapshot at epoch 5 for warm-starting imputed models
        # (early checkpoint before model overfits to ignorable missingness)
        snapshot_ep = min(5, max(1, num_epochs // 4))
        res = model_baseline.fit(
            factory_fn, batch_size=batch_size, dataset_size=N,
            num_epochs=num_epochs, steps_per_epoch=steps_per_epoch,
            learning_rate=learning_rate, lr_decay_factor=lr_decay_factor,
            patience=patience, zero_nan_grads=True,
            snapshot_epoch=snapshot_ep,
            max_nan_recoveries=50,
        )
        snapshot_params = res[2] if len(res) > 2 else None
        print(f"  Final loss: {res[0][-1]:.2f}")
        if snapshot_params is not None:
            print(f"  Saved epoch-{snapshot_ep} snapshot for warm-starting")
            np.savez(str(work_dir / 'baseline_snapshot.npz'),
                     **{k: np.array(v) for k, v in snapshot_params.items()})
        model_baseline.save_to_disk(str(baseline_path))
        calibrate_manually(model_baseline, n_samples=32, seed=101)
    gc.collect()

    # --- Stage 2: MICE LOO ---
    mice_path = work_dir / 'mice_loo_model.yaml'
    n_top = min(n_items, 40)
    if skip_mice and mice_path.exists():
        print("\n--- Loading MICE LOO ---")
        mice_loo = MICEBayesianLOO.load(str(mice_path))
    else:
        print(f"\n--- Fitting MICE LOO (n_top_features={n_top}) ---")
        pandas_df = df.select(item_keys).to_pandas().replace(-1, np.nan)
        mice_loo = MICEBayesianLOO(
            random_state=42, prior_scale=1.0,
            pathfinder_num_samples=100, pathfinder_maxiter=50,
            batch_size=512, verbose=True,
        )
        mice_loo.fit_loo_models(
            pandas_df, n_top_features=n_top, n_jobs=1,
            fit_zero_predictors=True, seed=42,
        )
        mice_loo.save(str(mice_path))
    gc.collect()

    # Warm-start: use early snapshot if available, otherwise final baseline params
    snapshot_path = work_dir / 'baseline_snapshot.npz'
    if snapshot_path.exists():
        loaded = np.load(str(snapshot_path), allow_pickle=True)
        baseline_init = {k: jnp.array(loaded[k]) for k in loaded.files}
        print(f"  Using early-epoch snapshot as warm-start for imputed models")
    else:
        baseline_init = dict(model_baseline.params)
        print(f"  Using final baseline params as warm-start for imputed models")

    # --- Stage 3: MICE-only GRM ---
    mice_only_path = work_dir / 'grm_mice_only'
    if skip_baseline and (mice_only_path / 'params.h5').exists():
        print("\n--- Loading MICE-only GRM ---")
        model_mice_only = GRModel.load_from_disk(mice_only_path)
        calibrate_manually(model_mice_only, n_samples=32, seed=103)
    else:
        print(f"\n--- Fitting MICE-only GRM ({imputed_epochs} epochs, warm-started) ---")
        model_mice_only = GRModel(
            item_keys=item_keys, num_people=N, dim=1,
            response_cardinality=K, dtype=jnp.float64,
            imputation_model=mice_loo,
        )
        res = model_mice_only.fit(
            factory_fn, batch_size=batch_size, dataset_size=N,
            num_epochs=imputed_epochs, steps_per_epoch=steps_per_epoch,
            learning_rate=learning_rate, lr_decay_factor=lr_decay_factor,
            patience=patience, zero_nan_grads=True,
            sample_size=32, seed=43, max_nan_recoveries=50,
            initial_values=baseline_init,
        )
        print(f"  Final loss: {res[0][-1]:.2f}")
        model_mice_only.save_to_disk(str(mice_only_path))
        calibrate_manually(model_mice_only, n_samples=32, seed=103)
    gc.collect()

    # --- Stage 4: Mixed imputation + GRM ---
    print("\n--- Building mixed imputation model ---")
    mixed_imputation = IrtMixedImputationModel(
        irt_model=model_baseline, mice_model=mice_loo,
        data_factory=factory_fn, irt_elpd_batch_size=4,
    )
    print(mixed_imputation.summary())
    with open(work_dir / 'mixed_weights.json', 'w') as f:
        json.dump(mixed_imputation.weights, f, indent=2)
    gc.collect()

    mixed_path = work_dir / 'grm_imputed'
    if skip_baseline and (mixed_path / 'params.h5').exists():
        print("\n--- Loading mixed-imputed GRM ---")
        model_mixed = GRModel.load_from_disk(mixed_path)
        calibrate_manually(model_mixed, n_samples=32, seed=102)
    else:
        print(f"\n--- Fitting mixed-imputed GRM ({imputed_epochs} epochs, warm-started) ---")
        model_mixed = GRModel(
            item_keys=item_keys, num_people=N, dim=1,
            response_cardinality=K, dtype=jnp.float64,
            imputation_model=mixed_imputation,
        )
        res = model_mixed.fit(
            factory_fn, batch_size=batch_size, dataset_size=N,
            num_epochs=imputed_epochs, steps_per_epoch=steps_per_epoch,
            learning_rate=learning_rate, lr_decay_factor=lr_decay_factor,
            patience=patience, zero_nan_grads=True,
            sample_size=32, seed=44, max_nan_recoveries=50,
            initial_values=baseline_init,
        )
        print(f"  Final loss: {res[0][-1]:.2f}")
        model_mixed.save_to_disk(str(mixed_path))
        calibrate_manually(model_mixed, n_samples=32, seed=102)
    gc.collect()

    # Save item labels and climber names
    with open(work_dir / 'item_labels.json', 'w') as f:
        json.dump(labels, f, indent=2)
    with open(work_dir / 'climber_names.json', 'w') as f:
        json.dump(names, f, indent=2)

    models = {
        'baseline': model_baseline,
        'mice_only': model_mice_only,
        'mixed': model_mixed,
    }
    return models, data_dict, item_keys, K, N, names, n_observed


# ---------------------------------------------------------------------------
# Real data evaluation (Table 1 equivalent)
# ---------------------------------------------------------------------------

def compute_predictive_rmse(model, data_dict, item_keys, K):
    """In-sample RMSE of E[Y] vs observed responses."""
    categories = jnp.arange(K, dtype=jnp.float64)
    ce = model.calibrated_expectations
    probs = model.grm_model_prob_d(
        ce['abilities'], ce['discriminations'],
        ce['difficulties0'], ce.get('ddifficulties'))
    expected = jnp.sum(probs * categories[None, :], axis=-1)  # (N, I)

    se_sum = 0.0
    count = 0
    for i, key in enumerate(item_keys):
        obs = np.array(data_dict[key], dtype=np.float64)
        pred_i = np.array(expected[:, i])
        valid = ~np.isnan(obs) & (obs >= 0) & (obs < K)
        se_sum += np.sum((obs[valid] - pred_i[valid]) ** 2)
        count += int(np.sum(valid))
    return float(np.sqrt(se_sum / count))


def eval_real_data(models, data_dict, item_keys, K, N, n_observed, work_dir, gender):
    """Compute Table 1 metrics for all models."""
    batch_size = 256

    def elpd_factory():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            yield {k: data_dict[k][start:end] for k in data_dict}

    results = {}
    for label, model in models.items():
        print(f"\n  Evaluating {label}...")
        try:
            rmse = compute_predictive_rmse(model, data_dict, item_keys, K)
        except Exception as e:
            print(f"    RMSE failed: {e}")
            rmse = None

        elpd = elpd_se = None
        try:
            model._compute_elpd_loo(elpd_factory, n_samples=100, seed=101, use_ais=True)
            elpd = float(model.elpd_loo)
            elpd_se = float(model.elpd_loo_se)
        except Exception as e:
            print(f"    ELPD failed: {e}")

        results[label] = {
            'rmse': rmse,
            'elpd': elpd,
            'elpd_se': elpd_se,
            'n_observed': n_observed,
            'N': N,
        }

    # Print table
    print(f"\n{'='*100}")
    print(f"  BOULDERING ({gender.upper()}) — Real Data Results")
    print(f"  N={N}, K={K}, Items={len(item_keys)}, "
          f"Missing={1.0 - n_observed/(N*len(item_keys)):.1%}")
    print(f"{'='*100}")
    print(f"{'Model':<15} {'RMSE':>10} {'ELPD/person':>20} {'ELPD/resp':>20}")
    print(f"{'-'*100}")
    for label in ['baseline', 'mice_only', 'mixed']:
        r = results[label]
        rmse_s = f"{r['rmse']:.4f}" if r['rmse'] else "--"
        if r['elpd'] is not None:
            ep = f"{r['elpd']/N:.4f} +/- {r['elpd_se']/N:.4f}"
            er = f"{r['elpd']/n_observed:.4f} +/- {r['elpd_se']/n_observed:.4f}"
        else:
            ep = er = "--"
        print(f"{label:<15} {rmse_s:>10} {ep:>20} {er:>20}")
    print(f"{'='*100}")

    with open(Path(work_dir) / 'real_data_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Synthetic evaluation (Table 2 equivalent)
# ---------------------------------------------------------------------------

def run_synthetic(gender, work_dir, epochs=500, lr=1e-3, batch_size=256):
    """Run full synthetic pipeline for bouldering with given gender."""
    from common.pipeline import (
        fit_neural_grm, generate_synthetic_data, compute_missingness_stats,
        fit_grm_baseline, fit_grm_imputed, calibrate_model,
        compare_ability_ordering, make_comparison_plots,
        make_data_factory as make_factory,
    )
    from bayesianquilts.imputation.mice_loo import MICEBayesianLOO
    from bayesianquilts.imputation.mixed import IrtMixedImputationModel

    work_dir = Path(work_dir)
    synth_dir = work_dir / 'synthetic'
    synth_dir.mkdir(parents=True, exist_ok=True)

    data_dict, item_keys, K, N, names, labels, df = load_bouldering_data(gender)
    print(f"\n{'='*60}")
    print(f"BOULDERING ({gender.upper()}) — Synthetic Evaluation")
    print(f"People: {N}, Items: {len(item_keys)}, K: {K}")
    print(f"{'='*60}")

    # 1. Fit NeuralGRM as ground truth
    neural_model = fit_neural_grm(
        data_dict, item_keys, K, N,
        save_dir=synth_dir / 'neural_grm',
        dim=1, batch_size=max(batch_size, 512),
        num_epochs=epochs, learning_rate=lr * 0.5,
        patience=20, lr_decay_factor=0.975, clip_norm=1.0,
        reload=True, noisy_dim=2,
        sample_size=32, seed=42, parameterization='log_scale',
    )

    # 2. Extract true abilities
    cal_abilities = np.array(neural_model.calibrated_expectations['abilities'])
    true_abilities = cal_abilities[:, :1, :, :]
    print(f"  True abilities shape: {true_abilities.shape}")

    # 3. Compute real data missingness pattern
    miss_stats = compute_missingness_stats(data_dict, item_keys, K)
    print(f"  Incomplete respondents: {miss_stats['incomplete_frac']:.1%}")

    # 4. Generate synthetic data
    synth_data = generate_synthetic_data(
        neural_model, item_keys, K,
        abilities=true_abilities,
        missingness_stats=miss_stats, seed=42,
    )

    # 5. Fit MICE on synthetic
    synth_df = pd.DataFrame({k: synth_data[k] for k in item_keys}).replace(-1, np.nan)
    mice_loo = MICEBayesianLOO(
        random_state=42, prior_scale=1.0,
        pathfinder_num_samples=100, pathfinder_maxiter=50,
        batch_size=512, verbose=True,
    )
    mice_loo.fit_loo_models(
        synth_df, n_top_features=min(len(item_keys), 40),
        n_jobs=1, fit_zero_predictors=True, seed=42,
    )
    mice_loo.save(str(synth_dir / 'mice_loo_model.yaml'))
    gc.collect()

    # 6. Fit baseline GRM on synthetic
    baseline_model, snapshot = fit_grm_baseline(
        synth_data, item_keys, K, N,
        save_dir=synth_dir / 'grm_baseline',
        batch_size=batch_size, num_epochs=epochs, learning_rate=lr,
        patience=20, lr_decay_factor=0.975, clip_norm=1.0,
        snapshot_epoch=50, sample_size=32, seed=42,
        compute_elpd_loo=True,
    )
    gc.collect()

    # 7. Fit MICE-only GRM on synthetic
    mice_only_model = fit_grm_imputed(
        synth_data, item_keys, K, N,
        save_dir=synth_dir / 'grm_mice_only',
        imputation_model=mice_loo,
        batch_size=batch_size, num_epochs=epochs, learning_rate=lr,
        patience=20, lr_decay_factor=0.975, clip_norm=1.0,
        initial_values=snapshot, sample_size=32, seed=43,
        compute_elpd_loo=True,
    )
    gc.collect()

    # 8. Build mixed imputation and fit mixed GRM
    factory = make_factory(synth_data, batch_size, N)
    mixed_imp = IrtMixedImputationModel(
        irt_model=baseline_model, mice_model=mice_loo,
        data_factory=factory, irt_elpd_batch_size=4,
    )
    print(mixed_imp.summary())
    gc.collect()

    imputed_model = fit_grm_imputed(
        synth_data, item_keys, K, N,
        save_dir=synth_dir / 'grm_imputed',
        imputation_model=mixed_imp,
        batch_size=batch_size, num_epochs=epochs, learning_rate=lr,
        patience=20, lr_decay_factor=0.975, clip_norm=1.0,
        initial_values=snapshot, sample_size=32, seed=44,
        compute_elpd_loo=True,
    )
    gc.collect()

    # 9. Compare ability orderings
    has_missing = np.zeros(N, dtype=bool)
    for key in item_keys:
        has_missing |= (synth_data[key] < 0)
    obs_mask = ~has_missing

    true_flat = true_abilities[:, 0, 0, 0]

    def _dim0(abil):
        a = np.array(abil)
        if a.ndim == 4:
            return a[:, 0, 0, 0]
        return a.flatten()

    baseline_ab = np.array(baseline_model.calibrated_expectations['abilities'])
    mice_ab = np.array(mice_only_model.calibrated_expectations['abilities'])
    imputed_ab = np.array(imputed_model.calibrated_expectations['abilities'])

    results = {}
    for label, ab in [('baseline', baseline_ab), ('mice_only', mice_ab), ('mixed', imputed_ab)]:
        results[label] = compare_ability_ordering(
            true_flat[obs_mask], _dim0(ab)[obs_mask])

    # ELPD-LOO
    for label, model in [('baseline', baseline_model), ('mice_only', mice_only_model),
                         ('mixed', imputed_model)]:
        if hasattr(model, 'elpd_loo') and model.elpd_loo is not None:
            results[label]['elpd_loo'] = float(model.elpd_loo)
            results[label]['elpd_loo_se'] = float(model.elpd_loo_se)

    # Print Table 2 format
    print(f"\n{'='*100}")
    print(f"  BOULDERING ({gender.upper()}) — Synthetic Evaluation (Table 2)")
    print(f"{'='*100}")
    print(f"{'Metric':<18} {'Baseline':>20} {'Univariate':>20} {'Mixed':>20}")
    print(f"{'-'*100}")
    for metric, key in [('Spearman rho', 'spearman_r'), ('Kendall tau', 'kendall_tau'),
                         ('RMSE', 'rmse')]:
        vals = []
        for label in ['baseline', 'mice_only', 'mixed']:
            ci = results[label].get(f'{key}_ci', [0, 0])
            se = (ci[1] - ci[0]) / (2 * 1.96) if ci else 0
            vals.append(f"{results[label][key]:.4f} ({se:.4f})")
        print(f"{metric:<18} {vals[0]:>20} {vals[1]:>20} {vals[2]:>20}")
    # ELPD/obs
    vals = []
    for label in ['baseline', 'mice_only', 'mixed']:
        e = results[label].get('elpd_loo')
        se = results[label].get('elpd_loo_se')
        if e is not None:
            n_obs = getattr(baseline_model, 'elpd_loo_n_obs', N)
            vals.append(f"{e/n_obs:.4f} ({se/n_obs:.4f})")
        else:
            vals.append("--")
    print(f"{'ELPD/obs':<18} {vals[0]:>20} {vals[1]:>20} {vals[2]:>20}")
    print(f"{'='*100}")

    # Save
    with open(synth_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Generate comparison plots
    make_comparison_plots(
        true_abilities, baseline_ab, mice_ab, imputed_ab,
        f'bouldering_{gender}', synth_dir / 'plots',
    )

    return results


# ---------------------------------------------------------------------------
# Forest plots of top climbers
# ---------------------------------------------------------------------------

def plot_top_climbers(models, data_dict, item_keys, K, N, climber_names,
                      work_dir, gender, n_top=10):
    """Forest plots of top climbers' posterior ability distributions.

    Produces two plots:
    1. Combined forest plot: all 3 strategies side-by-side per climber
    2. Per-strategy forest plots: one panel each for baseline and mixed,
       showing the same climbers ranked by that strategy's estimates.
    """
    import matplotlib.pyplot as plt

    work_dir = Path(work_dir)

    COLORS = {
        'baseline': '#4477AA',
        'mice_only': '#228833',
        'mixed': '#EE6677',
    }
    LABELS = {
        'baseline': 'Baseline',
        'mice_only': 'Univariate',
        'mixed': 'Mixed',
    }
    MARKERS = {
        'baseline': 'o',
        'mice_only': 's',
        'mixed': 'D',
    }

    def _set_tufte(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=3, width=0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

    # Extract ability samples for all models: (S, N)
    abilities = {}
    for label, model in models.items():
        ab = np.array(model.surrogate_sample['abilities'])
        S = ab.shape[0]
        abilities[label] = ab.reshape(S, N)

    # Rank by mixed mean ability (descending) — most informative ranking
    mixed_mean = abilities['mixed'].mean(axis=0)
    top_idx = np.argsort(mixed_mean)[::-1][:n_top]

    top_names = [climber_names[i] if i < len(climber_names) else f"Person {i}"
                 for i in top_idx]

    # --- Plot 1: Combined forest plot (all strategies, same climbers) ---
    fig, ax = plt.subplots(figsize=(7, max(4, n_top * 0.55)))
    y_pos = np.arange(n_top)
    offset = 0.2

    for j, label in enumerate(['baseline', 'mice_only', 'mixed']):
        ab = abilities[label][:, top_idx]
        means = ab.mean(axis=0)
        # Use 90% credible interval
        lo = np.percentile(ab, 5, axis=0)
        hi = np.percentile(ab, 95, axis=0)
        xerr = np.array([means - lo, hi - means])

        ax.errorbar(
            means, y_pos + (j - 1) * offset, xerr=xerr,
            fmt=MARKERS[label], capsize=2, markersize=4, elinewidth=0.8,
            color=COLORS[label], alpha=0.85, label=LABELS[label],
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel('Ability (posterior mean, 90% CI)')
    ax.set_title(f'Top {n_top} Climbers — {gender.title()}', fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.invert_yaxis()
    _set_tufte(ax)
    plt.tight_layout()

    out_path = work_dir / f'forest_top{n_top}_{gender}.pdf'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Forest plot (combined) saved: {out_path}")

    # --- Plot 2: Side-by-side panels ranked independently per strategy ---
    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, n_top * 0.5)),
                              sharey=False)

    for ax_idx, label in enumerate(['baseline', 'mice_only', 'mixed']):
        ax = axes[ax_idx]
        ab_all = abilities[label]  # (S, N)
        means_all = ab_all.mean(axis=0)

        # Rank by this strategy's mean ability
        this_top_idx = np.argsort(means_all)[::-1][:n_top]
        this_names = [climber_names[i] if i < len(climber_names) else f"Person {i}"
                      for i in this_top_idx]

        ab = ab_all[:, this_top_idx]
        means = ab.mean(axis=0)
        lo = np.percentile(ab, 5, axis=0)
        hi = np.percentile(ab, 95, axis=0)
        xerr = np.array([means - lo, hi - means])

        ax.errorbar(
            means, y_pos, xerr=xerr,
            fmt=MARKERS[label], capsize=2, markersize=5, elinewidth=1.0,
            color=COLORS[label], alpha=0.85,
        )

        # Also show the other two strategies for the SAME people (for comparison)
        for j, other_label in enumerate(['baseline', 'mice_only', 'mixed']):
            if other_label == label:
                continue
            other_ab = abilities[other_label][:, this_top_idx]
            other_means = other_ab.mean(axis=0)
            other_lo = np.percentile(other_ab, 5, axis=0)
            other_hi = np.percentile(other_ab, 95, axis=0)
            other_xerr = np.array([other_means - other_lo, other_hi - other_means])
            shift = 0.25 if other_label > label else -0.25
            ax.errorbar(
                other_means, y_pos + shift, xerr=other_xerr,
                fmt=MARKERS[other_label], capsize=1.5, markersize=3,
                elinewidth=0.6, color=COLORS[other_label], alpha=0.4,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(this_names, fontsize=8)
        ax.set_xlabel('Ability')
        ax.set_title(f'Ranked by {LABELS[label]}', fontsize=10)
        ax.invert_yaxis()
        _set_tufte(ax)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker=MARKERS[l], color=COLORS[l], label=LABELS[l],
               markersize=5, linestyle='None', alpha=0.85)
        for l in ['baseline', 'mice_only', 'mixed']
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f'Top {n_top} {gender.title()} Climbers by Strategy',
                 fontsize=12, y=1.06)
    plt.tight_layout()

    out_path2 = work_dir / f'forest_by_strategy_top{n_top}_{gender}.pdf'
    fig.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Forest plot (by strategy) saved: {out_path2}")

    return top_idx, top_names


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def generate_latex_table1(all_results, output_path):
    """Generate LaTeX for Table 1 equivalent (real data)."""
    lines = [
        r'\begin{table}[ht]',
        r'    \centering',
        r'    \caption{\textbf{Predictive performance on IFSC Bouldering data:}',
        r'           Comparing baseline (ignoring missingness) versus two stacked '
        r'imputation ensembles, mean (SE).',
        r'           Lower RMSE and higher (less negative) ELPD are better; '
        r'bold = best per row.}',
        r'    \label{tab:bouldering_actual}',
        r'    \begin{tabular}{@{}l l c c c@{}}',
        r'           \toprule',
        r'           Dataset & Metric & Baseline & Univariate & Mixed \\',
        r'           \midrule',
    ]

    for gender, results in all_results.items():
        label = f"Boulder ({gender.title()[0]})"
        N = results['baseline']['N']
        n_obs = results['baseline']['n_observed']

        def _fmt(lbl, metric):
            r = results[lbl]
            if metric == 'rmse':
                return f"{r['rmse']:.3f}" if r['rmse'] else '---'
            elif metric == 'elpd_n':
                if r['elpd'] is not None:
                    return f"${'-' if r['elpd'] < 0 else ''}${abs(r['elpd']/N):.2f} ({r['elpd_se']/N:.2f})"
                return '---'
            elif metric == 'elpd_r':
                if r['elpd'] is not None:
                    return f"${'-' if r['elpd'] < 0 else ''}${abs(r['elpd']/n_obs):.4f} ({r['elpd_se']/n_obs:.4f})"
                return '---'

        def _best(metric):
            vals = {}
            for lbl in ['baseline', 'mice_only', 'mixed']:
                r = results[lbl]
                if metric == 'rmse' and r['rmse'] is not None:
                    vals[lbl] = r['rmse']
                elif metric.startswith('elpd') and r['elpd'] is not None:
                    vals[lbl] = r['elpd']
            if not vals:
                return None
            if metric == 'rmse':
                return min(vals, key=vals.get)
            return max(vals, key=vals.get)

        lines.append(f'           \\multirow{{3}}{{*}}{{{label}}}')
        for metric_label, metric_key in [('RMSE', 'rmse'), ('ELPD/n', 'elpd_n'),
                                          ('ELPD/resp', 'elpd_r')]:
            best = _best(metric_key if metric_key != 'elpd_n' and metric_key != 'elpd_r' else 'elpd')
            row = f'                   & {metric_label}'
            for lbl in ['baseline', 'mice_only', 'mixed']:
                val = _fmt(lbl, metric_key)
                if lbl == best:
                    row += f' & \\textbf{{{val}}}'
                else:
                    row += f' & {val}'
            row += r' \\'
            lines.append(row)
        lines.append(r'           \addlinespace[3pt]')

    lines.append(r'           \bottomrule')
    lines.append(r'    \end{tabular}')
    lines.append(r'\end{table}')

    tex = '\n'.join(lines)
    with open(output_path, 'w') as f:
        f.write(tex)
    print(f"LaTeX Table 1 saved: {output_path}")
    return tex


def generate_latex_table2(all_results, output_path):
    """Generate LaTeX for Table 2 equivalent (synthetic)."""
    lines = [
        r'\begin{table}[ht]',
        r'    \centering',
        r'    \caption{\textbf{Latent ability recovery on IFSC Bouldering data:}',
        r'           Comparing baseline (naive marginalization) versus two stacked '
        r'imputation ensembles, mean (SE).',
        r'           Higher Spearman $\rho$ and Kendall $\tau$, lower RMSE, and '
        r'higher (less negative) ELPD/obs are better; bold = best per row.}',
        r'    \label{tab:bouldering_synthetic}',
        r'    \begin{tabular}{@{}l l c c c@{}}',
        r'           \toprule',
        r'           Dataset & Metric & Baseline & Univariate & Mixed \\',
        r'           \midrule',
    ]

    for gender, results in all_results.items():
        label = f"Boulder ({gender.title()[0]})"
        metrics = [
            ('Spearman $\\rho$', 'spearman_r', 'higher'),
            ('Kendall $\\tau$', 'kendall_tau', 'higher'),
            ('RMSE', 'rmse', 'lower'),
        ]

        lines.append(f'           \\multirow{{4}}{{*}}{{{label}}}')
        for metric_label, key, direction in metrics:
            vals = {}
            for lbl in ['baseline', 'mice_only', 'mixed']:
                v = results[lbl][key]
                ci = results[lbl].get(f'{key}_ci', [0, 0])
                se = (ci[1] - ci[0]) / (2 * 1.96) if ci else 0
                vals[lbl] = (v, se)
            best = (min if direction == 'lower' else max)(vals, key=lambda x: vals[x][0])
            row = f'                   & {metric_label}'
            for lbl in ['baseline', 'mice_only', 'mixed']:
                v, se = vals[lbl]
                s = f'{v:.3f} ({se:.3f})'
                if lbl == best:
                    row += f' & $\\mathbf{{{v:.3f}}}$ ({se:.3f})'
                else:
                    row += f' & {v:.3f} ({se:.3f})'
            row += r' \\'
            lines.append(row)

        # ELPD/obs row
        elpd_vals = {}
        for lbl in ['baseline', 'mice_only', 'mixed']:
            e = results[lbl].get('elpd_loo')
            se = results[lbl].get('elpd_loo_se')
            if e is not None:
                elpd_vals[lbl] = (e, se)
        if elpd_vals:
            best = max(elpd_vals, key=lambda x: elpd_vals[x][0])
            row = '                   & ELPD/obs'
            for lbl in ['baseline', 'mice_only', 'mixed']:
                if lbl in elpd_vals:
                    e, se = elpd_vals[lbl]
                    s = f'${"-" if e < 0 else ""}${abs(e):.2f} ({se:.2f})'
                    if lbl == best:
                        row += f' & \\textbf{{{s}}}'
                    else:
                        row += f' & {s}'
                else:
                    row += ' & ---'
            row += r' \\'
            lines.append(row)

        lines.append(r'           \addlinespace[3pt]')

    lines.append(r'           \bottomrule')
    lines.append(r'    \end{tabular}')
    lines.append(r'\end{table}')

    tex = '\n'.join(lines)
    with open(output_path, 'w') as f:
        f.write(tex)
    print(f"LaTeX Table 2 saved: {output_path}")
    return tex


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run bouldering experiments for journal article."
    )
    parser.add_argument('--gender', default='both', choices=['men', 'women', 'both'])
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Load existing baseline GRM')
    parser.add_argument('--skip-mice', action='store_true',
                        help='Load existing MICE model')
    parser.add_argument('--skip-synthetic', action='store_true',
                        help='Skip synthetic evaluation (real data + plots only)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs for real data models')
    parser.add_argument('--imputed-epochs', type=int, default=25,
                        help='Training epochs for imputed models (default 25, warm-started from early baseline snapshot)')
    parser.add_argument('--synthetic-epochs', type=int, default=500,
                        help='Training epochs for synthetic evaluation')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience (default 30, higher for sparse data)')
    parser.add_argument('--n-top', type=int, default=10,
                        help='Number of top climbers for forest plots')
    args = parser.parse_args()

    genders = ['men', 'women'] if args.gender == 'both' else [args.gender]
    base_dir = SCRIPT_DIR

    all_real_results = {}
    all_synth_results = {}

    for gender in genders:
        work_dir = base_dir / gender
        work_dir.mkdir(parents=True, exist_ok=True)

        # Real data pipeline
        models, data_dict, item_keys, K, N, names, n_obs = run_real_data(
            gender, work_dir,
            skip_baseline=args.skip_baseline,
            skip_mice=args.skip_mice,
            num_epochs=args.epochs,
            imputed_epochs=args.imputed_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
        )

        # Real data evaluation
        real_results = eval_real_data(
            models, data_dict, item_keys, K, N, n_obs, work_dir, gender)
        all_real_results[gender] = real_results

        # Forest plots
        plot_top_climbers(
            models, data_dict, item_keys, K, N, names,
            work_dir, gender, n_top=args.n_top)

        # Synthetic evaluation
        if not args.skip_synthetic:
            synth_results = run_synthetic(
                gender, work_dir,
                epochs=args.synthetic_epochs,
                lr=1e-3, batch_size=args.batch_size,
            )
            all_synth_results[gender] = synth_results

        gc.collect()

    # Generate LaTeX tables
    tex_dir = base_dir / 'tex'
    tex_dir.mkdir(exist_ok=True)

    generate_latex_table1(all_real_results, tex_dir / 'bouldering_table1.tex')
    if all_synth_results:
        generate_latex_table2(all_synth_results, tex_dir / 'bouldering_table2.tex')

    print(f"\nAll experiments complete. Outputs in {base_dir}/")


if __name__ == '__main__':
    main()
