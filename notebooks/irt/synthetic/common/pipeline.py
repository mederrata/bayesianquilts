"""Shared pipeline utilities for the synthetic data evaluation.

Provides functions to:
- Load psychometric datasets
- Fit NeuralGRModel and standard GRModel
- Generate synthetic data from a fitted model
- Compare ability ordering between true and estimated values
- Produce comparison plots
"""

import os
import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Using float32 for training (with log_scale parameterization + STL for stability)


# -------------------------------------------------------------------------
# Dataset loading
# -------------------------------------------------------------------------

DATASET_MODULES = {
    'grit': 'bayesianquilts.data.grit',
    'rwa': 'bayesianquilts.data.rwa',
    'eqsq': 'bayesianquilts.data.eqsq',
    'npi': 'bayesianquilts.data.npi',
    'wpi': 'bayesianquilts.data.wpi',
    'tma': 'bayesianquilts.data.tma',
    'bouldering': 'bayesianquilts.data.bouldering',
}


def load_dataset(dataset_name: str, cache_dir=None):
    """Load a psychometric dataset by name.

    Args:
        dataset_name: One of 'grit', 'rwa', 'eqsq', 'npi', 'wpi', 'tma'.
        cache_dir: Optional directory for caching downloaded data.

    Returns:
        (data_dict, item_keys, response_cardinality, num_people)
        where data_dict has keys: person, item_key1, item_key2, ...
    """
    if dataset_name not in DATASET_MODULES:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(DATASET_MODULES.keys())}"
        )
    mod = importlib.import_module(DATASET_MODULES[dataset_name])
    item_keys = mod.item_keys
    response_cardinality = mod.response_cardinality
    kwargs = {'polars_out': True}
    if cache_dir is not None:
        kwargs['cache_dir'] = cache_dir
    df, num_people = mod.get_data(**kwargs)

    # Convert to numpy data dict
    data = {}
    for col in df.columns:
        data[col] = df[col].to_numpy().astype(np.float32)
    data['person'] = np.arange(num_people, dtype=np.float32)

    return data, item_keys, response_cardinality, num_people


# -------------------------------------------------------------------------
# Data factory helpers
# -------------------------------------------------------------------------

def make_data_factory(data_dict, batch_size, num_people):
    """Create a batched data factory for GRM fitting."""
    def data_factory():
        indices = np.arange(num_people)
        np.random.shuffle(indices)
        for start in range(0, num_people, batch_size):
            end = min(start + batch_size, num_people)
            idx_batch = indices[start:end]
            yield {k: v[idx_batch] for k, v in data_dict.items()}
    return data_factory


def calibrate_model(model, n_samples=32, seed=42):
    """Set calibration expectations from the surrogate posterior.

    Uses point estimates (mode of the variational posterior) extracted
    directly from the variational parameters, without sampling.

    For Normal surrogates, the stored ``loc`` value is already in the
    transformed (natural) parameter space and equals the posterior mode.
    For InverseGamma surrogates, the mode is ``scale / (concentration + 1)``.

    We also draw samples for downstream use (e.g. uncertainty estimates)
    and store them in ``model.surrogate_sample``.
    """
    # --- Point estimates (no sampling) ---
    point_estimates = {}
    for key, value in model.params.items():
        parts = key.split('\\')
        if len(parts) < 4:
            continue
        param_name = parts[0]
        dist_type = parts[-2]
        param_type = parts[-1]

        if dist_type == 'normal' and param_type == 'loc':
            # For Normal surrogate, loc in transformed space IS the mode
            point_estimates[param_name] = value
        elif dist_type == 'igamma' and param_type == 'scale':
            # For InverseGamma(a, b), mode = b / (a + 1)
            conc_key = key.replace('\\scale', '\\concentration')
            if conc_key in model.params:
                point_estimates[param_name] = value / (model.params[conc_key] + 1)

    model.calibrated_expectations = point_estimates

    # --- Also store posterior samples for uncertainty analysis ---
    try:
        surrogate = model.surrogate_distribution_generator(model.params)
        key = jax.random.PRNGKey(seed)
        samples = surrogate.sample(n_samples, seed=key)
        model.surrogate_sample = samples
    except KeyError as e:
        # Surrogate generator may expect params absent from saved model
        # (e.g. ddifficulties for K=2). Point estimates suffice for synthesis.
        print(f"  Warning: surrogate sampling skipped ({e}); point estimates OK")


# -------------------------------------------------------------------------
# Model fitting
# -------------------------------------------------------------------------

def fit_neural_grm(
    data_dict, item_keys, response_cardinality, num_people, save_dir,
    dim=1, batch_size=256,
    num_epochs=500, learning_rate=1e-3, patience=10,
    kappa_scale=0.5, eta_scale=0.1,
    lr_decay_factor=0.9, clip_norm=1.0,
    reload=False,
    noisy_dim=False,
    noisy_dim_eta_scale=0.01,
    noisy_dim_ability_scale=2.0,
    sample_size=32,
    seed=42,
    parameterization="log_scale",
    pathfinder_init=False,
):
    """Fit a NeuralGRModel on the given data and save to disk.

    If ``reload=True`` and a saved model exists at ``save_dir``, it is loaded
    from disk instead of re-training (saves hours of compute).

    Returns:
        Fitted NeuralGRModel instance.
    """
    from bayesianquilts.irt.neural_grm import NeuralGRModel

    save_dir = Path(save_dir)

    if reload and (save_dir / 'params.h5').exists():
        print(f"  Reloading NeuralGRM from {save_dir}")
        model = NeuralGRModel.load_from_disk(save_dir)
        calibrate_model(model)
        return model

    model = NeuralGRModel(
        item_keys=item_keys,
        num_people=num_people,
        dim=dim,
        kappa_scale=kappa_scale,
        eta_scale=eta_scale,
        response_cardinality=response_cardinality,
        dtype=jnp.float32,
        noisy_dim=noisy_dim,
        noisy_dim_eta_scale=noisy_dim_eta_scale,
        noisy_dim_ability_scale=noisy_dim_ability_scale,
        parameterization=parameterization,
    )

    steps_per_epoch = int(np.ceil(num_people / batch_size))
    factory = make_data_factory(data_dict, batch_size, num_people)

    losses, params = model.fit(
        factory,
        batch_size=batch_size,
        dataset_size=num_people,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        learning_rate=learning_rate,
        patience=patience,
        lr_decay_factor=lr_decay_factor,
        clip_norm=clip_norm,
        zero_nan_grads=True,
        compute_elpd_loo=False,  # NeuralGRM is for data generation, not comparison
        sample_size=sample_size,
        seed=seed,
        pathfinder_init=pathfinder_init,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_to_disk(save_dir)
    calibrate_model(model)
    np.save(save_dir / 'losses.npy', np.array(losses))

    return model


def fit_grm_baseline(
    data_dict, item_keys, response_cardinality, num_people, save_dir,
    dim=1, batch_size=256, num_epochs=500, learning_rate=1e-3,
    patience=10, kappa_scale=0.1,
    lr_decay_factor=0.9, clip_norm=1.0,
    snapshot_epoch=None, sample_size=32,
    seed=42,
    parameterization="log_scale",
    pathfinder_init=False,
):
    """Fit a standard GRM (no imputation) and save to disk.

    Returns:
        (model, snapshot_params) where snapshot_params is the variational
        parameters at ``snapshot_epoch`` (None if not requested).
    """
    from bayesianquilts.irt.grm import GRModel

    model = GRModel(
        item_keys=item_keys,
        num_people=num_people,
        dim=dim,
        kappa_scale=kappa_scale,
        response_cardinality=response_cardinality,
        dtype=jnp.float32,
        parameterization=parameterization,
    )

    steps_per_epoch = int(np.ceil(num_people / batch_size))
    factory = make_data_factory(data_dict, batch_size, num_people)

    res = model.fit(
        factory,
        batch_size=batch_size,
        dataset_size=num_people,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        learning_rate=learning_rate,
        patience=patience,
        lr_decay_factor=lr_decay_factor,
        clip_norm=clip_norm,
        zero_nan_grads=True,
        snapshot_epoch=snapshot_epoch,
        sample_size=sample_size,
        seed=seed,
        pathfinder_init=pathfinder_init,
    )
    losses = res[0]
    snapshot_params = res[2] if len(res) > 2 else None

    calibrate_model(model)

    save_dir = Path(save_dir)
    model.save_to_disk(save_dir)
    np.save(save_dir / 'losses.npy', np.array(losses))

    return model, snapshot_params


def fit_grm_imputed(
    data_dict, item_keys, response_cardinality, num_people, save_dir,
    imputation_model, dim=1, batch_size=256, num_epochs=500,
    learning_rate=1e-3, patience=10, kappa_scale=0.1,
    lr_decay_factor=0.9, clip_norm=1.0,
    initial_values=None, sample_size=32,
    seed=42,
    parameterization="log_scale",
    pathfinder_init=False,
):
    """Fit a GRM with MICEBayesianLOO imputation and save to disk.

    Args:
        initial_values: Optional initial variational parameters (e.g. from
            an early baseline checkpoint) to warm-start training.

    Returns:
        Fitted GRModel instance.
    """
    from bayesianquilts.irt.grm import GRModel

    model = GRModel(
        item_keys=item_keys,
        num_people=num_people,
        dim=dim,
        kappa_scale=kappa_scale,
        response_cardinality=response_cardinality,
        dtype=jnp.float32,
        imputation_model=imputation_model,
        parameterization=parameterization,
    )
    steps_per_epoch = int(np.ceil(num_people / batch_size))
    factory = make_data_factory(data_dict, batch_size, num_people)

    res = model.fit(
        factory,
        batch_size=batch_size,
        dataset_size=num_people,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        learning_rate=learning_rate,
        patience=patience,
        lr_decay_factor=lr_decay_factor,
        clip_norm=clip_norm,
        zero_nan_grads=True,
        initial_values=initial_values,
        sample_size=sample_size,
        seed=seed,
        pathfinder_init=pathfinder_init,
    )
    losses = res[0]

    calibrate_model(model)

    save_dir = Path(save_dir)
    model.save_to_disk(save_dir)
    np.save(save_dir / 'losses.npy', np.array(losses))

    return model


def fit_grm_is(
    baseline_model, data_dict, item_keys, num_people,
    imputation_model, save_dir, batch_size=256,
    n_samples=256, is_batch_size=4,
):
    """Reweight a fitted baseline GRM via importance sampling.

    Uses the baseline (ignorability) posterior as proposal and
    reweights samples by the imputation-adjusted likelihood ratio.
    This avoids refitting and guarantees that when all stacking
    weights are zero, the result is identical to the baseline.

    Args:
        baseline_model: A fitted GRModel (baseline, ignorability).
        data_dict: Synthetic data dict with missingness.
        item_keys: List of item column names.
        num_people: Number of respondents.
        imputation_model: IrtMixedImputationModel with per-item weights.
        save_dir: Directory to save IS diagnostics.
        batch_size: Data batch size for iterating over respondents.
        n_samples: Number of posterior samples for IS.
        is_batch_size: Chunk size over posterior samples (memory).

    Returns:
        dict with IS results (weights, abilities, diagnostics).
    """
    factory = make_data_factory(data_dict, batch_size, num_people)
    is_results = baseline_model.fit_is(
        data_factory=factory,
        imputation_model=imputation_model,
        n_samples=n_samples,
        batch_size=is_batch_size,
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_dir / 'is_diagnostics.npz',
        log_is_weights=is_results['log_is_weights'],
        psis_weights=is_results['psis_weights'],
        khat=is_results['khat'],
        ess=is_results['ess'],
    )

    return is_results


# -------------------------------------------------------------------------
# Synthetic data generation
# -------------------------------------------------------------------------

def sample_abilities(num_people, dim=1, seed=42):
    """Create a grid of abilities using normal quantiles.

    Uses equally-spaced quantiles of the standard normal distribution
    to produce a deterministic, evenly-covered grid over the latent space.
    For dim > 1, each dimension gets an independently shuffled copy of
    the 1D grid (Latin Hypercube design).

    Args:
        num_people: Number of grid points.
        dim: Latent dimension (default 1).
        seed: Random seed (used only for shuffling in dim > 1).

    Returns:
        abilities array with shape (N, dim, 1, 1) matching the IRT convention.
    """
    from scipy.stats import norm

    # Equally-spaced quantiles: avoid 0 and 1 by using midpoints
    quantile_points = (np.arange(num_people) + 0.5) / num_people
    grid_1d = norm.ppf(quantile_points)

    if dim == 1:
        abilities = grid_1d[:, np.newaxis, np.newaxis, np.newaxis]
    else:
        # Latin Hypercube: independently shuffled grids per dimension
        rng = np.random.default_rng(seed)
        abilities = np.zeros((num_people, dim, 1, 1))
        for d in range(dim):
            perm = rng.permutation(num_people)
            abilities[:, d, 0, 0] = grid_1d[perm]

    return abilities


def generate_synthetic_data(model, item_keys, response_cardinality,
                            abilities=None, missingness_rate=0.0,
                            missing_respondent_frac=0.4,
                            seed=0):
    """Generate synthetic responses from a fitted NeuralGRModel or GRModel.

    Missingness is applied only to a fraction of respondents. The remaining
    respondents are fully observed.

    Args:
        model: A fitted IRT model with calibrated_expectations set.
        item_keys: List of item column names.
        response_cardinality: Number of response categories.
        abilities: Optional ability array to use. If None, uses model's calibrated.
        missingness_rate: Fraction of responses to set missing (MCAR) for
            respondents selected to have missing data. Default 0.
        missing_respondent_frac: Fraction of respondents who have any missing
            data. The rest are fully observed. Default 0.4.
        seed: Random seed for response sampling and missingness.

    Returns:
        data_dict with keys: person, item_key1, ..., item_keyN
    """
    responses = np.array(model.simulate_data(abilities=abilities, seed=seed))
    # responses shape: (N, I) or (N, D, I) — flatten to (N, I)
    if responses.ndim > 2:
        responses = responses.reshape(responses.shape[0], -1)

    # Ensure integer responses in valid range
    responses = np.clip(responses, 0, response_cardinality - 1).astype(np.int32)

    N, I = responses.shape
    data_dict = {'person': np.arange(N, dtype=np.float32)}

    for i, key in enumerate(item_keys):
        data_dict[key] = responses[:, i].astype(np.float32)

    # Introduce MCAR missingness for a subset of respondents
    if missingness_rate > 0:
        rng = np.random.default_rng(seed + 1000)
        # Select which respondents have missing data
        n_missing_people = int(N * missing_respondent_frac)
        missing_people = rng.choice(N, size=n_missing_people, replace=False)
        # For selected respondents, each response is missing with prob missingness_rate
        item_mask = rng.random((n_missing_people, I)) < missingness_rate
        for i, key in enumerate(item_keys):
            vals = data_dict[key].copy()
            vals[missing_people[item_mask[:, i]]] = -1.0
            data_dict[key] = vals

    return data_dict


# -------------------------------------------------------------------------
# Comparison metrics
# -------------------------------------------------------------------------

def compare_ability_ordering(true_abilities, estimated_abilities,
                             n_bootstrap=1000, ci_level=0.95, seed=42):
    """Compute rank correlation metrics with bootstrap confidence intervals.

    Args:
        true_abilities: Array of true ability values (N,) or (N, D, 1, 1).
        estimated_abilities: Array of estimated ability values (same shape).
        n_bootstrap: Number of bootstrap resamples for CI estimation.
        ci_level: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict with spearman_r, kendall_tau, rmse, and bootstrap CIs.
    """
    true_flat = np.array(true_abilities).flatten()
    est_flat = np.array(estimated_abilities).flatten()
    N = len(true_flat)

    spearman_r, spearman_p = stats.spearmanr(true_flat, est_flat)
    kendall_tau, kendall_p = stats.kendalltau(true_flat, est_flat)
    rmse = np.sqrt(np.mean((true_flat - est_flat) ** 2))

    # Bootstrap CIs
    rng = np.random.default_rng(seed)
    boot_spearman = np.empty(n_bootstrap)
    boot_kendall = np.empty(n_bootstrap)
    boot_rmse = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        t, e = true_flat[idx], est_flat[idx]
        boot_spearman[b] = stats.spearmanr(t, e).statistic
        boot_kendall[b] = stats.kendalltau(t, e).statistic
        boot_rmse[b] = np.sqrt(np.mean((t - e) ** 2))

    alpha = 1 - ci_level
    lo, hi = alpha / 2, 1 - alpha / 2

    return {
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'spearman_ci': [float(np.percentile(boot_spearman, lo * 100)),
                        float(np.percentile(boot_spearman, hi * 100))],
        'kendall_tau': float(kendall_tau),
        'kendall_p': float(kendall_p),
        'kendall_ci': [float(np.percentile(boot_kendall, lo * 100)),
                       float(np.percentile(boot_kendall, hi * 100))],
        'rmse': float(rmse),
        'rmse_ci': [float(np.percentile(boot_rmse, lo * 100)),
                    float(np.percentile(boot_rmse, hi * 100))],
    }


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def _set_tufte_style(ax):
    """Apply Tufte-style formatting: minimal ink, no chartjunk."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=3, width=0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)


# Consistent colors across all synthetic plots
COLORS = {
    'true': '#222222',
    'baseline': '#4477AA',
    'mice_only': '#228833',
    'mixed': '#EE6677',
}


def make_comparison_plots(true_abilities, baseline_abilities,
                          mice_only_abilities, imputed_abilities,
                          dataset_name, save_dir):
    """Generate Tufte-style comparison plots for 3 conditions.

    Produces:
    - Scatter plots of true vs estimated abilities (3 panels)
    - Dot plot of rank correlation metrics (replaces bar chart)
    - Step histograms of ability distributions

    Args:
        true_abilities: Array of true ability values.
        baseline_abilities: Abilities from baseline GRM.
        mice_only_abilities: Abilities from MICE-only GRM.
        imputed_abilities: Abilities from mixed-imputed GRM.
        dataset_name: Name of the dataset (for titles).
        save_dir: Directory to save plot files.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    true_flat = np.array(true_abilities).flatten()
    base_flat = np.array(baseline_abilities).flatten()
    mice_flat = np.array(mice_only_abilities).flatten()
    imp_flat = np.array(imputed_abilities).flatten()

    # 1. Scatter plots — 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    conditions = [
        ('Baseline (ignorable)', base_flat, COLORS['baseline']),
        ('MICE-only', mice_flat, COLORS['mice_only']),
        ('Mixed (MICE + IRT)', imp_flat, COLORS['mixed']),
    ]
    for ax, (label, est_flat, color) in zip(axes, conditions):
        ax.scatter(true_flat, est_flat, alpha=0.15, s=6, edgecolors='none',
                   color=color)
        lims = [min(true_flat.min(), est_flat.min()),
                max(true_flat.max(), est_flat.max())]
        ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=0.5)
        rho = stats.spearmanr(true_flat, est_flat).statistic
        ax.set_xlabel('True ability')
        ax.set_ylabel('Estimated ability')
        ax.set_title(f'{label} (ρ = {rho:.3f})', fontsize=10)
        ax.set_aspect('equal')
        _set_tufte_style(ax)

    fig.suptitle(f'{dataset_name.upper()} — True vs Estimated Abilities',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(save_dir / 'scatter_abilities.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. Dot plot of metrics (high data-ink ratio, no bar chart)
    base_m = compare_ability_ordering(true_flat, base_flat)
    mice_m = compare_ability_ordering(true_flat, mice_flat)
    imp_m = compare_ability_ordering(true_flat, imp_flat)

    metric_names = ['Spearman ρ', 'Kendall τ', '1 − RMSE']
    y_pos = np.arange(len(metric_names))

    fig, ax = plt.subplots(figsize=(6, 3))
    offset = 0.12
    for i, (label, m, color, marker) in enumerate([
        ('Baseline', base_m, COLORS['baseline'], 'o'),
        ('MICE-only', mice_m, COLORS['mice_only'], 's'),
        ('Mixed', imp_m, COLORS['mixed'], 'D'),
    ]):
        vals = [m['spearman_r'], m['kendall_tau'], max(0, 1 - m['rmse'])]
        ax.scatter(vals, y_pos + (i - 1) * offset, marker=marker, s=40,
                   color=color, zorder=3, label=label, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metric_names)
    ax.set_xlabel('Value')
    ax.set_title(f'{dataset_name.upper()} — Ability Ordering Metrics', fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.invert_yaxis()
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    _set_tufte_style(ax)
    plt.tight_layout()
    fig.savefig(save_dir / 'metric_comparison.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3. Ability distribution step histograms
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(true_flat, bins=40, histtype='step', linewidth=1.5,
            label='True', color=COLORS['true'])
    ax.hist(base_flat, bins=40, histtype='step', linewidth=1.2,
            label='Baseline', color=COLORS['baseline'])
    ax.hist(mice_flat, bins=40, histtype='step', linewidth=1.2,
            label='MICE-only', color=COLORS['mice_only'])
    ax.hist(imp_flat, bins=40, histtype='step', linewidth=1.2,
            label='Mixed', color=COLORS['mixed'])
    ax.set_xlabel('Ability')
    ax.set_ylabel('Count')
    ax.set_title(f'{dataset_name.upper()} — Ability Distributions', fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    _set_tufte_style(ax)
    plt.tight_layout()
    fig.savefig(save_dir / 'ability_distributions.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Plots saved to {save_dir}")
