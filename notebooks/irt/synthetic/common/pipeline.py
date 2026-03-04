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

jax.config.update("jax_enable_x64", True)


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
        data[col] = df[col].to_numpy().astype(np.float64)
    data['person'] = np.arange(num_people, dtype=np.float64)

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


def compute_elpd_loo(model, data_dict, batch_size=512, n_samples=100, seed=42):
    """Compute PSIS-LOO for a fitted IRT model.

    Samples from the surrogate posterior, computes per-person log-likelihoods
    across the full dataset, and runs PSIS-LOO diagnostics.

    For imputed models (with an ``imputation_model``), the imputation PMFs
    are pre-computed for the full dataset and attached to each batch.

    Results are stored on the model as attributes so they persist via
    ``save_to_disk``.

    Args:
        model: A fitted GRModel with params set.
        data_dict: Full dataset dict (person, item columns).
        batch_size: People per batch for log-likelihood computation.
        n_samples: Number of surrogate posterior draws for PSIS.
        seed: Random seed for sampling.

    Returns:
        Dict with elpd_loo, elpd_loo_se, elpd_loo_per_obs,
        elpd_loo_se_per_obs, pointwise_loo, khat, n_obs.
    """
    from bayesianquilts.metrics.nppsis import psisloo

    # Sample from surrogate posterior
    surrogate = model.surrogate_distribution_generator(model.params)
    key = jax.random.PRNGKey(seed)
    samples = surrogate.sample(n_samples, seed=key)

    N = len(data_dict[model.item_keys[0]])

    # For imputed models, pre-compute PMFs for the full dataset
    has_imputation = (
        hasattr(model, 'imputation_model')
        and model.imputation_model is not None
    )
    if has_imputation:
        if model._has_missing_values(data_dict):
            print("  Computing imputation PMFs for full dataset...")
            full_pmfs = model._compute_batch_pmfs(data_dict)
        else:
            full_pmfs = np.zeros(
                (N, model.num_items, model.response_cardinality),
                dtype=np.float64,
            )
    else:
        full_pmfs = None

    # Compute log-likelihoods in batches over people
    all_log_liks = []
    indices = np.arange(N)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        idx = indices[start:end]

        batch = {}
        for k, v in data_dict.items():
            if isinstance(v, (np.ndarray, jnp.ndarray)) and len(v) == N:
                batch[k] = v[idx]
            else:
                batch[k] = v
        # person key must be actual person indices (for abilities indexing)
        batch[model.person_key] = idx.astype(np.float64)

        if full_pmfs is not None:
            batch['_imputation_pmfs'] = full_pmfs[idx]

        pred = model.predictive_distribution(batch, **samples)
        all_log_liks.append(np.array(pred['log_likelihood']))

    log_lik = np.concatenate(all_log_liks, axis=1)  # (S, N)

    # Run PSIS-LOO
    loo, loos, ks = psisloo(log_lik)

    n_obs = N
    elpd_loo = float(loo)
    elpd_loo_se = float(np.std(loos) * np.sqrt(n_obs))
    elpd_loo_per_obs = elpd_loo / n_obs
    elpd_loo_se_per_obs = elpd_loo_se / n_obs

    results = {
        'elpd_loo': elpd_loo,
        'elpd_loo_se': elpd_loo_se,
        'elpd_loo_per_obs': elpd_loo_per_obs,
        'elpd_loo_se_per_obs': elpd_loo_se_per_obs,
        'pointwise_loo': loos,
        'khat': ks,
        'n_obs': n_obs,
    }

    # Store on model for persistence via save_to_disk
    model.elpd_loo = elpd_loo
    model.elpd_loo_se = elpd_loo_se
    model.elpd_loo_per_obs = elpd_loo_per_obs
    model.elpd_loo_se_per_obs = elpd_loo_se_per_obs
    model.elpd_loo_pointwise = loos
    model.elpd_loo_khat = ks
    model.elpd_loo_n_obs = n_obs

    # Report diagnostics
    bad_k = np.sum(ks > 0.7)
    print(f"  ELPD-LOO: {elpd_loo:.2f} (SE: {elpd_loo_se:.2f})")
    print(f"  ELPD-LOO per obs: {elpd_loo_per_obs:.4f} (SE: {elpd_loo_se_per_obs:.4f})")
    print(f"  k-hat: max={np.max(ks):.3f}, "
          f"mean={np.mean(ks):.3f}, "
          f">{0.7}: {bad_k}/{n_obs}")

    return results


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
    nn_hidden_sizes=(4,), per_item_nn=True, dim=1, batch_size=256,
    num_epochs=500, learning_rate=2e-4, patience=10,
    kappa_scale=0.5, eta_scale=0.1,
    lr_decay_factor=0.9, clip_norm=1.0,
    reload=False,
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
        nn_hidden_sizes=nn_hidden_sizes,
        per_item_nn=per_item_nn,
        dtype=jnp.float64,
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
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_to_disk(save_dir)
    calibrate_model(model)
    np.save(save_dir / 'losses.npy', np.array(losses))

    return model


def fit_grm_baseline(
    data_dict, item_keys, response_cardinality, num_people, save_dir,
    dim=1, batch_size=256, num_epochs=500, learning_rate=2e-4,
    patience=10, kappa_scale=0.1,
    lr_decay_factor=0.9, clip_norm=1.0,
):
    """Fit a standard GRM (no imputation) and save to disk.

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
        dtype=jnp.float64,
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
    )

    calibrate_model(model)

    # Compute ELPD-LOO
    print("  Computing ELPD-LOO...")
    compute_elpd_loo(model, data_dict, batch_size=batch_size)

    save_dir = Path(save_dir)
    model.save_to_disk(save_dir)
    np.save(save_dir / 'losses.npy', np.array(losses))

    return model


def fit_grm_imputed(
    data_dict, item_keys, response_cardinality, num_people, save_dir,
    imputation_model, dim=1, batch_size=256, num_epochs=500,
    learning_rate=2e-4, patience=10, kappa_scale=0.1,
    lr_decay_factor=0.9, clip_norm=1.0,
):
    """Fit a GRM with MICEBayesianLOO imputation and save to disk.

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
        dtype=jnp.float64,
        imputation_model=imputation_model,
    )
    model.validate_imputation_model()

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
    )

    calibrate_model(model)

    # Compute ELPD-LOO
    print("  Computing ELPD-LOO...")
    compute_elpd_loo(model, data_dict, batch_size=batch_size)

    save_dir = Path(save_dir)
    model.save_to_disk(save_dir)
    np.save(save_dir / 'losses.npy', np.array(losses))

    return model


# -------------------------------------------------------------------------
# Synthetic data generation
# -------------------------------------------------------------------------

def sample_abilities(num_people, dim=1, seed=42):
    """Sample fresh abilities from a standard normal prior.

    Args:
        num_people: Number of people.
        dim: Latent dimension (default 1).
        seed: Random seed.

    Returns:
        abilities array with shape (N, dim, 1, 1) matching the IRT convention.
    """
    rng = np.random.default_rng(seed)
    abilities = rng.standard_normal((num_people, dim, 1, 1))
    return abilities


def generate_synthetic_data(model, item_keys, response_cardinality,
                            abilities=None, missingness_rate=0.0, seed=0):
    """Generate synthetic responses from a fitted NeuralGRModel or GRModel.

    Args:
        model: A fitted IRT model with calibrated_expectations set.
        item_keys: List of item column names.
        response_cardinality: Number of response categories.
        abilities: Optional ability array to use. If None, uses model's calibrated.
        missingness_rate: Fraction of responses to set missing (MCAR). Default 0.
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
    data_dict = {'person': np.arange(N, dtype=np.float64)}

    for i, key in enumerate(item_keys):
        data_dict[key] = responses[:, i].astype(np.float64)

    # Introduce MCAR missingness
    if missingness_rate > 0:
        rng = np.random.default_rng(seed + 1000)
        mask = rng.random((N, I)) < missingness_rate
        for i, key in enumerate(item_keys):
            data_dict[key] = np.where(mask[:, i], -1.0, data_dict[key])

    return data_dict


# -------------------------------------------------------------------------
# Comparison metrics
# -------------------------------------------------------------------------

def compare_ability_ordering(true_abilities, estimated_abilities):
    """Compute rank correlation metrics between true and estimated abilities.

    Args:
        true_abilities: Array of true ability values (N,) or (N, D, 1, 1).
        estimated_abilities: Array of estimated ability values (same shape).

    Returns:
        Dict with spearman_r, spearman_p, kendall_tau, kendall_p, rmse.
    """
    true_flat = np.array(true_abilities).flatten()
    est_flat = np.array(estimated_abilities).flatten()

    spearman_r, spearman_p = stats.spearmanr(true_flat, est_flat)
    kendall_tau, kendall_p = stats.kendalltau(true_flat, est_flat)
    rmse = np.sqrt(np.mean((true_flat - est_flat) ** 2))

    return {
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'kendall_tau': float(kendall_tau),
        'kendall_p': float(kendall_p),
        'rmse': float(rmse),
    }


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def make_comparison_plots(true_abilities, baseline_abilities, imputed_abilities,
                          dataset_name, save_dir):
    """Generate comparison plots between true and estimated abilities.

    Produces:
    - Scatter plots of true vs estimated abilities
    - Rank correlation bar chart
    - Ability distribution histograms

    Args:
        true_abilities: Array of true ability values.
        baseline_abilities: Abilities from baseline GRM.
        imputed_abilities: Abilities from imputed GRM.
        dataset_name: Name of the dataset (for titles).
        save_dir: Directory to save plot files.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    true_flat = np.array(true_abilities).flatten()
    base_flat = np.array(baseline_abilities).flatten()
    imp_flat = np.array(imputed_abilities).flatten()

    # 1. Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(true_flat, base_flat, alpha=0.2, s=8, edgecolors='none')
    lims = [
        min(true_flat.min(), base_flat.min()),
        max(true_flat.max(), base_flat.max()),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='y = x')
    rho = stats.spearmanr(true_flat, base_flat).statistic
    ax.set_xlabel('True Ability')
    ax.set_ylabel('Estimated Ability')
    ax.set_title(f'Baseline GRM (Spearman r = {rho:.3f})')
    ax.legend()

    ax = axes[1]
    ax.scatter(true_flat, imp_flat, alpha=0.2, s=8, edgecolors='none')
    lims = [
        min(true_flat.min(), imp_flat.min()),
        max(true_flat.max(), imp_flat.max()),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='y = x')
    rho = stats.spearmanr(true_flat, imp_flat).statistic
    ax.set_xlabel('True Ability')
    ax.set_ylabel('Estimated Ability')
    ax.set_title(f'Imputed GRM (Spearman r = {rho:.3f})')
    ax.legend()

    fig.suptitle(f'{dataset_name.upper()} — True vs Estimated Abilities', fontsize=14)
    plt.tight_layout()
    fig.savefig(save_dir / 'scatter_abilities.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. Rank correlation bar chart
    base_metrics = compare_ability_ordering(true_flat, base_flat)
    imp_metrics = compare_ability_ordering(true_flat, imp_flat)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    width = 0.35
    metric_names = ['Spearman r', 'Kendall tau', '1 - RMSE']
    base_vals = [
        base_metrics['spearman_r'],
        base_metrics['kendall_tau'],
        max(0, 1 - base_metrics['rmse']),
    ]
    imp_vals = [
        imp_metrics['spearman_r'],
        imp_metrics['kendall_tau'],
        max(0, 1 - imp_metrics['rmse']),
    ]
    ax.bar(x - width / 2, base_vals, width, label='Baseline', color='tab:blue', alpha=0.8)
    ax.bar(x + width / 2, imp_vals, width, label='Imputed', color='tab:orange', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel('Value')
    ax.set_title(f'{dataset_name.upper()} — Ability Ordering Metrics')
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    fig.savefig(save_dir / 'metric_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3. Ability distribution histograms
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(true_flat, bins=40, histtype='step', linewidth=2,
            label='True', color='black')
    ax.hist(base_flat, bins=40, histtype='step', linewidth=2,
            label='Baseline', color='tab:blue')
    ax.hist(imp_flat, bins=40, histtype='step', linewidth=2,
            label='Imputed', color='tab:orange')
    ax.set_xlabel('Ability')
    ax.set_ylabel('Count')
    ax.set_title(f'{dataset_name.upper()} — Ability Distributions')
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_dir / 'ability_distributions.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Plots saved to {save_dir}")
