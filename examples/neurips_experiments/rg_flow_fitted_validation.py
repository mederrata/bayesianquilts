#!/usr/bin/env python3
"""
RG Flow Validation Using Fitted Model Parameters

Uses parameters from a fitted German Credit model as ground truth to validate
the RG flow theory. The fitted model has pairwise interactions larger than
main effects, making K=2 the optimal truncation order.

Reference:
    Chang (2025), "A renormalization-group inspired hierarchical Bayesian
    framework for piecewise linear regression models"
"""

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


@dataclass
class ValidationConfig:
    """Configuration for validation experiments."""
    n_train: int = 2000
    n_test: int = 500
    noise_std: float = 0.5
    n_replications: int = 30
    max_order: int = 3
    output_dir: str = "results/rg_flow_fitted"


def load_fitted_params(pkl_path: str) -> Tuple[Dict, List[Tuple], Dict]:
    """Load fitted parameters from a pickle file.

    Args:
        pkl_path: Path to the pickle file

    Returns:
        Tuple of (params, dimensions, decomp_info)
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    params = data['fold_models'][0]['params']
    dimensions = data['dimensions']
    decomp_info = data['decomp_info']

    return params, dimensions, decomp_info


def create_decomposition(dimensions: List[Tuple], n_features: int) -> Decomposed:
    """Create a Decomposed object from dimension specification.

    Args:
        dimensions: List of (name, n_levels) tuples
        n_features: Number of regression features

    Returns:
        Decomposed parameter object
    """
    dims = [Dimension(name, n_levels) for name, n_levels in dimensions]
    interactions = Interactions(dimensions=dims)

    decomp = Decomposed(
        interactions=interactions,
        param_shape=[n_features],
        name="beta",
    )

    return decomp


def generate_synthetic_data(
    true_params: Dict,
    decomp: Decomposed,
    n_obs: int,
    noise_std: float,
    rng_key: jax.random.PRNGKey,
) -> Dict:
    """Generate synthetic data from fitted parameters.

    Args:
        true_params: Dictionary of true parameter values
        decomp: Parameter decomposition
        n_obs: Number of observations
        noise_std: Standard deviation of noise
        rng_key: Random key

    Returns:
        Data dictionary with X, y, and factor indices
    """
    keys = jax.random.split(rng_key, 10)

    n_features = decomp._param_shape[0]
    n_factors = len(decomp._interactions._dimensions)

    factor_indices = {}
    for i, dim in enumerate(decomp._interactions._dimensions):
        factor_indices[dim.name] = jax.random.randint(
            keys[i], shape=(n_obs,), minval=0, maxval=dim.cardinality
        )

    X = jax.random.normal(keys[5], shape=(n_obs, n_features))

    interaction_indices = jnp.stack(
        [factor_indices[dim.name] for dim in decomp._interactions._dimensions],
        axis=-1
    )

    full_params = {}
    for name, shape in decomp._tensor_part_shapes.items():
        if name in true_params:
            full_params[name] = jnp.array(true_params[name])
        else:
            full_params[name] = jnp.zeros(shape)

    beta = decomp.lookup(interaction_indices, tensors=full_params)
    eta = jnp.sum(X * beta, axis=-1)

    if '_intercept' in true_params:
        eta = eta + true_params['_intercept'][0]

    noise = noise_std * jax.random.normal(keys[6], shape=(n_obs,))
    y = eta + noise

    data = {
        "X": X,
        "y": y,
        **factor_indices,
    }

    return data


def fit_truncated_model(
    data_train: Dict,
    data_test: Dict,
    decomp: Decomposed,
    truncation_order: int,
    noise_std: float,
    n_steps: int = 500,
) -> Tuple[float, float, Dict]:
    """Fit a model truncated at a given order and evaluate.

    Args:
        data_train: Training data
        data_test: Test data
        decomp: Parameter decomposition
        truncation_order: Maximum interaction order to include
        noise_std: Noise standard deviation for prior scaling
        n_steps: Number of optimization steps

    Returns:
        Tuple of (train_loss, test_loss, fitted_params)
    """
    n_obs = len(data_train["y"])

    prior_scales = decomp.generalization_preserving_scales(
        noise_scale=noise_std,
        total_n=n_obs,
        c=0.5,
        per_component=True,
    )

    active_components = [
        name for name in decomp._tensor_parts.keys()
        if decomp.component_order(name) <= truncation_order
    ]

    params = {
        name: jnp.zeros(decomp._tensor_part_shapes[name])
        for name in active_components
    }
    params['_intercept'] = jnp.zeros((1,))

    def get_interaction_indices(data):
        return jnp.stack(
            [data[dim.name] for dim in decomp._interactions._dimensions],
            axis=-1
        )

    train_indices = get_interaction_indices(data_train)
    test_indices = get_interaction_indices(data_test)

    def loss_fn(params):
        full_params = {**params}
        for name in decomp._tensor_parts.keys():
            if name not in params:
                full_params[name] = jnp.zeros(decomp._tensor_part_shapes[name])

        intercept = params.get('_intercept', jnp.zeros((1,)))

        beta = decomp.lookup(train_indices, tensors=full_params)
        eta = jnp.sum(data_train["X"] * beta, axis=-1) + intercept[0]
        residuals = data_train["y"] - eta
        nll = 0.5 * jnp.sum(residuals ** 2) / (noise_std ** 2)

        log_prior = 0.0
        for name, param in params.items():
            if name == '_intercept':
                continue
            scale = prior_scales.get(name, 1.0)
            log_prior += 0.5 * jnp.sum(param ** 2) / (scale ** 2)

        return nll + log_prior

    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for _ in range(n_steps):
        params, opt_state, train_loss = step(params, opt_state)

    full_params = {**params}
    for name in decomp._tensor_parts.keys():
        if name not in params:
            full_params[name] = jnp.zeros(decomp._tensor_part_shapes[name])

    intercept = params.get('_intercept', jnp.zeros((1,)))

    beta_test = decomp.lookup(test_indices, tensors=full_params)
    eta_test = jnp.sum(data_test["X"] * beta_test, axis=-1) + intercept[0]
    test_residuals = data_test["y"] - eta_test
    test_mse = jnp.mean(test_residuals ** 2)

    beta_train = decomp.lookup(train_indices, tensors=full_params)
    eta_train = jnp.sum(data_train["X"] * beta_train, axis=-1) + intercept[0]
    train_residuals = data_train["y"] - eta_train
    train_mse = jnp.mean(train_residuals ** 2)

    return float(train_mse), float(test_mse), params


def compute_snr_from_params(
    true_params: Dict,
    decomp: Decomposed,
    order: int,
    noise_std: float,
    n_obs: int,
) -> float:
    """Compute SNR at a given order from true parameters.

    Args:
        true_params: True parameter values
        decomp: Parameter decomposition
        order: Interaction order
        noise_std: Noise standard deviation
        n_obs: Number of observations

    Returns:
        SNR at this order
    """
    components_at_order = [
        name for name in decomp._tensor_parts.keys()
        if decomp.component_order(name) == order
    ]

    if not components_at_order:
        return 0.0

    total_var = 0.0
    total_n_local = 0.0

    for name in components_at_order:
        if name in true_params:
            param = np.asarray(true_params[name])
            signal_var = np.var(param)
            total_var += signal_var

        shape = decomp._tensor_part_shapes[name]
        n_cells = int(np.prod(shape[:-1]))
        n_local = n_obs / max(n_cells, 1)
        total_n_local += n_local

    if len(components_at_order) > 0:
        avg_n_local = total_n_local / len(components_at_order)
    else:
        avg_n_local = n_obs

    sampling_var = noise_std**2 / max(avg_n_local, 1)
    snr = total_var / max(sampling_var, 1e-10)

    return float(snr)


def run_single_replication(
    true_params: Dict,
    decomp: Decomposed,
    config: ValidationConfig,
    rng_key: jax.random.PRNGKey,
) -> Dict:
    """Run a single replication of the validation.

    Args:
        true_params: True parameter values
        decomp: Parameter decomposition
        config: Validation configuration
        rng_key: Random key

    Returns:
        Dictionary of results
    """
    keys = jax.random.split(rng_key, 2)

    train_data = generate_synthetic_data(
        true_params, decomp, config.n_train, config.noise_std, keys[0]
    )
    test_data = generate_synthetic_data(
        true_params, decomp, config.n_test, config.noise_std, keys[1]
    )

    train_mses = []
    test_mses = []

    for order in range(config.max_order + 1):
        train_mse, test_mse, _ = fit_truncated_model(
            train_data, test_data, decomp, order, config.noise_std
        )
        train_mses.append(train_mse)
        test_mses.append(test_mse)

    snr_values = []
    for order in range(config.max_order + 1):
        snr = compute_snr_from_params(
            true_params, decomp, order, config.noise_std, config.n_train
        )
        snr_values.append(snr)

    delta_s = [0.0]
    for i in range(1, len(test_mses)):
        delta_s.append(test_mses[i] - test_mses[i-1])

    optimal_k = int(np.argmin(test_mses))

    return {
        "train_mses": train_mses,
        "test_mses": test_mses,
        "snr_values": snr_values,
        "delta_s": delta_s,
        "optimal_k": optimal_k,
    }


def run_validation(config: ValidationConfig, pkl_path: str):
    """Run the full validation experiment.

    Args:
        config: Validation configuration
        pkl_path: Path to fitted model pickle file
    """
    os.makedirs(config.output_dir, exist_ok=True)

    true_params, dimensions, decomp_info = load_fitted_params(pkl_path)
    n_features = decomp_info['tensor_part_shapes']['beta__'][-1]
    decomp = create_decomposition(dimensions, n_features)

    print("Loaded fitted parameters from:", pkl_path)
    print("Dimensions:", dimensions)
    print("N features:", n_features)
    print()
    print("Parameter magnitudes by order:")
    for name, shape in decomp._tensor_part_shapes.items():
        order = decomp.component_order(name)
        if name in true_params:
            mag = abs(np.asarray(true_params[name])).mean()
            print(f"  Order {order}: {name} |mean|={mag:.4f}")
    print()

    all_results = []

    for rep in tqdm(range(config.n_replications), desc="Replications"):
        rng_key = jax.random.PRNGKey(rep + 42)
        results = run_single_replication(true_params, decomp, config, rng_key)
        results["replication"] = rep
        all_results.append(results)

    results_path = Path(config.output_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

    print(f"Results saved to {results_path}")

    plot_results(all_results, config)
    print_summary(all_results, config)


def plot_results(results: List[Dict], config: ValidationConfig):
    """Generate plots for validation results.

    Args:
        results: List of result dictionaries
        config: Validation configuration
    """
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    test_mses = np.array([r["test_mses"] for r in results])
    train_mses = np.array([r["train_mses"] for r in results])
    snr_values = np.array([r["snr_values"] for r in results])
    delta_s = np.array([r["delta_s"] for r in results])

    orders = np.arange(test_mses.shape[1])

    ax = axes[0]
    ax.errorbar(orders, test_mses.mean(axis=0), yerr=test_mses.std(axis=0),
                fmt='o', capsize=6, capthick=2, elinewidth=2,
                markersize=10, color='steelblue', ecolor='steelblue', label='Test MSE')
    ax.errorbar(orders + 0.1, train_mses.mean(axis=0), yerr=train_mses.std(axis=0),
                fmt='s', capsize=6, capthick=2, elinewidth=2,
                markersize=8, color='gray', ecolor='gray', alpha=0.6, label='Train MSE')
    ax.set_xlabel(r"Truncation Order $K$")
    ax.set_ylabel("MSE")
    ax.set_title("Generalization vs Truncation Order")
    ax.legend(frameon=False)
    ax.set_xticks(orders)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1]
    mean_snr = snr_values.mean(axis=0)
    std_snr = snr_values.std(axis=0)
    ax.errorbar(orders, mean_snr, yerr=std_snr, fmt='o', capsize=6, capthick=2, elinewidth=2,
                markersize=10, color='steelblue', ecolor='steelblue')
    ax.axhline(y=1.0, color='firebrick', linestyle='--', alpha=0.7, linewidth=1.5, label='SNR=1')
    ax.set_xlabel("Interaction Order")
    ax.set_ylabel("SNR")
    ax.set_title("Signal-to-Noise Ratio by Order")
    ax.legend(frameon=False)
    ax.set_xticks(orders)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[2]
    mean_delta = delta_s.mean(axis=0)
    std_delta = delta_s.std(axis=0)
    colors = ['forestgreen' if d < 0 else 'firebrick' for d in mean_delta]
    for i, (x, y, err, c) in enumerate(zip(orders, mean_delta, std_delta, colors)):
        ax.errorbar(x, y, yerr=err, fmt='o', capsize=6, capthick=2, elinewidth=2,
                    markersize=10, color=c, ecolor=c)
    ax.axhline(y=0.0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel(r"Truncation Order $K$")
    ax.set_ylabel(r"$\Delta S_K$ (negative = better)")
    ax.set_title("Generalization Gap")
    ax.set_xticks(orders)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(Path(config.output_dir) / "validation_results.png", dpi=150, bbox_inches='tight')
    plt.savefig(Path(config.output_dir) / "validation_results.pdf", bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {config.output_dir}")


def print_summary(results: List[Dict], config: ValidationConfig):
    """Print summary statistics.

    Args:
        results: List of result dictionaries
        config: Validation configuration
    """
    test_mses = np.array([r["test_mses"] for r in results])
    optimal_ks = [r["optimal_k"] for r in results]

    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    print("\nTest MSE by truncation order:")
    for k in range(test_mses.shape[1]):
        mse_mean = test_mses[:, k].mean()
        mse_std = test_mses[:, k].std()
        print(f"  K={k}: {mse_mean:.4f} ± {mse_std:.4f}")

    print(f"\nOptimal K distribution:")
    for k in range(test_mses.shape[1]):
        count = optimal_ks.count(k)
        pct = 100 * count / len(optimal_ks)
        print(f"  K={k}: {count}/{len(optimal_ks)} ({pct:.1f}%)")

    k2_better = sum(1 for r in results if r["test_mses"][2] < r["test_mses"][0])
    print(f"\nK=2 beats K=0: {k2_better}/{len(results)} ({100*k2_better/len(results):.1f}%)")

    mean_improvement = np.mean([r["test_mses"][0] - r["test_mses"][2] for r in results])
    print(f"Mean MSE improvement (K=0 → K=2): {mean_improvement:.4f}")


def main():
    parser = argparse.ArgumentParser(description="RG flow validation with fitted parameters")
    parser.add_argument("--pkl_path", type=str,
                        default="results/german_credit_full.pkl",
                        help="Path to fitted model pickle file")
    parser.add_argument("--n_replications", type=int, default=30)
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--noise_std", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="results/rg_flow_fitted")
    parser.add_argument("--quick", action="store_true", help="Run quick version")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only regenerate plots from existing results.json")
    args = parser.parse_args()

    config = ValidationConfig(
        n_replications=args.n_replications if not args.quick else 5,
        n_train=args.n_train if not args.quick else 500,
        n_test=args.n_test if not args.quick else 100,
        noise_std=args.noise_std,
        output_dir=args.output_dir,
    )

    if args.plot_only:
        results_path = Path(args.output_dir) / "results.json"
        if not results_path.exists():
            print(f"No results.json found at {results_path}")
            return
        with open(results_path) as f:
            all_results = json.load(f)
        plot_results(all_results, config)
        print_summary(all_results, config)
    else:
        run_validation(config, args.pkl_path)


if __name__ == "__main__":
    main()
