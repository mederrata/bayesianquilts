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


def calibrate_model(model, n_samples=32, seed=42):
    """Set calibration expectations from the surrogate posterior."""
    surrogate = model.surrogate_distribution_generator(model.params)
    key = jax.random.PRNGKey(seed)
    samples = surrogate.sample(n_samples, seed=key)
    model.calibrated_expectations = {k: jnp.mean(v, axis=0) for k, v in samples.items()}
    model.surrogate_sample = samples


# -------------------------------------------------------------------------
# Model fitting
# -------------------------------------------------------------------------

def fit_neural_grm(
    data_dict, item_keys, response_cardinality, num_people, save_dir,
    nn_hidden_sizes=(32, 32), dim=1, batch_size=256,
    num_epochs=500, learning_rate=2e-4, patience=10, kappa_scale=0.1,
):
    """Fit a NeuralGRModel on the given data and save to disk.

    Returns:
        Fitted NeuralGRModel instance.
    """
    from bayesianquilts.irt.neural_grm import NeuralGRModel

    model = NeuralGRModel(
        item_keys=item_keys,
        num_people=num_people,
        dim=dim,
        kappa_scale=kappa_scale,
        response_cardinality=response_cardinality,
        nn_hidden_sizes=nn_hidden_sizes,
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
        zero_nan_grads=True,
    )

    save_dir = Path(save_dir)
    model.save_to_disk(save_dir)
    calibrate_model(model)
    np.save(save_dir / 'losses.npy', np.array(losses))

    return model


def fit_grm_baseline(
    data_dict, item_keys, response_cardinality, num_people, save_dir,
    dim=1, batch_size=256, num_epochs=500, learning_rate=2e-4,
    patience=10, kappa_scale=0.1,
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
        zero_nan_grads=True,
    )

    save_dir = Path(save_dir)
    model.save_to_disk(save_dir)
    calibrate_model(model)
    np.save(save_dir / 'losses.npy', np.array(losses))

    return model


def fit_grm_imputed(
    data_dict, item_keys, response_cardinality, num_people, save_dir,
    imputation_model, dim=1, batch_size=256, num_epochs=500,
    learning_rate=2e-4, patience=10, kappa_scale=0.1,
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
        zero_nan_grads=True,
    )

    save_dir = Path(save_dir)
    model.save_to_disk(save_dir)
    calibrate_model(model)
    np.save(save_dir / 'losses.npy', np.array(losses))

    return model


# -------------------------------------------------------------------------
# Synthetic data generation
# -------------------------------------------------------------------------

def generate_synthetic_data(model, item_keys, response_cardinality,
                            abilities=None, missingness_rate=0.0, seed=0):
    """Generate synthetic responses from a fitted NeuralGRModel or GRModel.

    Args:
        model: A fitted IRT model with calibrated_expectations set.
        item_keys: List of item column names.
        response_cardinality: Number of response categories.
        abilities: Optional ability array to use. If None, uses model's calibrated.
        missingness_rate: Fraction of responses to set missing (MCAR). Default 0.
        seed: Random seed for missingness.

    Returns:
        data_dict with keys: person, item_key1, ..., item_keyN
    """
    responses = np.array(model.simulate_data(abilities=abilities))
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
        rng = np.random.default_rng(seed)
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
