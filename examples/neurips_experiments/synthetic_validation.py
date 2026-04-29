#!/usr/bin/env python3
"""
Experiment 1: Synthetic Validation of Generalization-Preserving Regularization

Validates that the generalization-preserving scaling τ = σ/√(D·N^(α)) prevents
overfitting when adding model complexity, even when true effects are zero.

Reference:
    Chang (2025), "A renormalization-group inspired hierarchical Bayesian
    framework for piecewise linear regression models"
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    n_obs: int = 10000
    d_factors: int = 3
    L_levels: int = 4
    rho_decay: float = 0.3
    noise_std: float = 1.0
    n_features: int = 5
    true_max_order: int = 2
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    n_replications: int = 50
    max_order: int = 3
    n_obs_values: List[int] = None
    rho_values: List[float] = None
    output_dir: str = "results/synthetic_validation"

    def __post_init__(self):
        if self.n_obs_values is None:
            self.n_obs_values = [1000, 5000, 10000, 50000]
        if self.rho_values is None:
            self.rho_values = [0.1, 0.3, 0.5, 0.7]


def generate_synthetic_data(
    config: SyntheticDataConfig,
    rng_key: jax.random.PRNGKey,
) -> Tuple[Dict, Dict, Decomposed]:
    """Generate synthetic hierarchical data with known structure.

    Args:
        config: Data generation configuration
        rng_key: JAX random key

    Returns:
        Tuple of (data_dict, true_params, decomposition)
    """
    keys = jax.random.split(rng_key, 10)

    # Create interaction structure
    dimensions = [
        Dimension(f"factor_{i}", config.L_levels)
        for i in range(config.d_factors)
    ]
    interactions = Interactions(dimensions=dimensions)

    # Create decomposition for coefficients
    decomp = Decomposed(
        interactions=interactions,
        param_shape=[config.n_features],
        name="beta",
    )

    # Generate true parameters with decaying effect sizes
    true_params = {}
    key_idx = 0
    for name, shape in decomp._tensor_part_shapes.items():
        order = decomp.component_order(name)
        if order <= config.true_max_order:
            scale = config.rho_decay ** order
            true_params[name] = scale * jax.random.normal(
                keys[key_idx % 10], shape=shape
            )
        else:
            true_params[name] = jnp.zeros(shape)
        key_idx += 1

    # Generate factor assignments for each observation
    factor_indices = {}
    for i, dim in enumerate(dimensions):
        factor_indices[dim.name] = jax.random.randint(
            keys[i], shape=(config.n_obs,), minval=0, maxval=config.L_levels
        )

    # Generate features
    X = jax.random.normal(keys[5], shape=(config.n_obs, config.n_features))

    # Compute true linear predictor
    interaction_indices = jnp.stack(
        [factor_indices[f"factor_{i}"] for i in range(config.d_factors)],
        axis=-1
    )
    beta = decomp.lookup(interaction_indices, tensors=true_params)

    # y = X @ beta + noise (element-wise for each observation)
    eta = jnp.sum(X * beta, axis=-1)
    noise = config.noise_std * jax.random.normal(keys[6], shape=(config.n_obs,))
    y = eta + noise

    data = {
        "X": X,
        "y": y,
        **factor_indices,
    }

    return data, true_params, decomp


def fit_model_at_order(
    data: Dict,
    decomp: Decomposed,
    max_order: int,
    prior_scales: Dict[str, float],
    noise_std: float,
    n_steps: int = 1000,
    learning_rate: float = 0.01,
) -> Tuple[Dict, float]:
    """Fit model truncated at given order using MAP estimation.

    Args:
        data: Data dictionary with X, y, and factor indices
        decomp: Parameter decomposition
        max_order: Maximum interaction order to include
        prior_scales: Prior standard deviations for each component
        noise_std: Known noise standard deviation
        n_steps: Number of optimization steps
        learning_rate: Learning rate for Adam

    Returns:
        Tuple of (fitted_params, final_loss)
    """
    import optax

    # Get components at or below max_order
    active_components = [
        name for name in decomp._tensor_parts.keys()
        if decomp.component_order(name) <= max_order
    ]

    # Initialize parameters
    params = {
        name: jnp.zeros(decomp._tensor_part_shapes[name])
        for name in active_components
    }

    # Build interaction indices
    interaction_indices = jnp.stack(
        [data[f"factor_{i}"] for i in range(len(decomp._interactions._dimensions))],
        axis=-1
    )

    def loss_fn(params):
        # Lookup coefficients
        full_params = {**params}
        for name in decomp._tensor_parts.keys():
            if name not in params:
                full_params[name] = jnp.zeros(decomp._tensor_part_shapes[name])

        beta = decomp.lookup(interaction_indices, tensors=full_params)
        eta = jnp.sum(data["X"] * beta, axis=-1)

        # Gaussian log-likelihood
        residuals = data["y"] - eta
        nll = 0.5 * jnp.sum(residuals ** 2) / (noise_std ** 2)

        # Log-prior (Gaussian)
        log_prior = 0.0
        for name, param in params.items():
            scale = prior_scales.get(name, 1.0)
            log_prior += 0.5 * jnp.sum(param ** 2) / (scale ** 2)

        return nll + log_prior

    # Optimize
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for _ in range(n_steps):
        params, opt_state, loss = step(params, opt_state)

    return params, float(loss)


def compute_test_metrics(
    data_train: Dict,
    data_test: Dict,
    decomp: Decomposed,
    params: Dict,
    noise_std: float,
) -> Dict[str, float]:
    """Compute test set metrics.

    Args:
        data_train: Training data
        data_test: Test data
        decomp: Parameter decomposition
        params: Fitted parameters
        noise_std: Noise standard deviation

    Returns:
        Dictionary of metrics
    """
    # Fill in zeros for missing components
    full_params = {}
    for name in decomp._tensor_parts.keys():
        if name in params:
            full_params[name] = params[name]
        else:
            full_params[name] = jnp.zeros(decomp._tensor_part_shapes[name])

    def compute_ll(data):
        interaction_indices = jnp.stack(
            [data[f"factor_{i}"] for i in range(len(decomp._interactions._dimensions))],
            axis=-1
        )
        beta = decomp.lookup(interaction_indices, tensors=full_params)
        eta = jnp.sum(data["X"] * beta, axis=-1)
        residuals = data["y"] - eta
        ll = -0.5 * jnp.sum(residuals ** 2) / (noise_std ** 2)
        ll -= 0.5 * len(data["y"]) * jnp.log(2 * jnp.pi * noise_std ** 2)
        return float(ll)

    train_ll = compute_ll(data_train)
    test_ll = compute_ll(data_test)

    # Compute effective degrees of freedom
    df_eff = 0.0
    n_train = len(data_train["y"])
    for name, param in params.items():
        n_cells = int(np.prod(decomp._tensor_part_shapes[name][:-1]))
        n_local = n_train / max(n_cells, 1)
        p = int(np.prod(decomp._param_shape))
        # Per-parameter df_eff
        tau = 1.0  # Would need actual prior scale
        df_param = n_local * tau**2 / (n_local * tau**2 + noise_std**2)
        df_eff += p * n_cells * df_param

    return {
        "train_ll": train_ll,
        "test_ll": test_ll,
        "test_ll_per_obs": test_ll / len(data_test["y"]),
        "df_eff": df_eff,
    }


def compute_prior_scales(
    decomp: Decomposed,
    method: str,
    noise_std: float,
    n_obs: int,
    per_component: bool = True,
) -> Dict[str, float]:
    """Compute prior scales using different methods.

    Args:
        decomp: Parameter decomposition
        method: One of "none", "fixed", "empirical_bayes", "gen_preserving"
        noise_std: Noise standard deviation
        n_obs: Number of observations
        per_component: Whether to use per-component bound

    Returns:
        Dictionary of prior scales
    """
    if method == "none":
        return {name: 1e6 for name in decomp._tensor_parts.keys()}

    elif method == "fixed":
        return {name: 1.0 for name in decomp._tensor_parts.keys()}

    elif method == "gen_preserving":
        return decomp.generalization_preserving_scales(
            noise_scale=noise_std,
            total_n=n_obs,
            c=0.5,
            per_component=per_component,
        )

    elif method == "decay":
        # Ad-hoc decay factor (baseline from original code)
        scales = {}
        decay = 0.9
        for name in decomp._tensor_parts.keys():
            order = decomp.component_order(name)
            scales[name] = 5.0 * (decay ** order)
        return scales

    else:
        raise ValueError(f"Unknown method: {method}")


def run_single_experiment(
    data_config: SyntheticDataConfig,
    exp_config: ExperimentConfig,
    rng_key: jax.random.PRNGKey,
) -> Dict:
    """Run a single replication of the experiment.

    Args:
        data_config: Data configuration
        exp_config: Experiment configuration
        rng_key: Random key

    Returns:
        Dictionary of results
    """
    keys = jax.random.split(rng_key, 3)

    # Generate train and test data
    train_config = data_config
    test_config = SyntheticDataConfig(
        **{**vars(data_config), "n_obs": data_config.n_obs // 5}
    )

    train_data, true_params, decomp = generate_synthetic_data(train_config, keys[0])
    test_data, _, _ = generate_synthetic_data(test_config, keys[1])

    results = {"orders": [], "methods": []}

    methods = ["none", "fixed", "decay", "gen_preserving"]

    for method in methods:
        prior_scales = compute_prior_scales(
            decomp, method, data_config.noise_std, data_config.n_obs
        )

        for order in range(exp_config.max_order + 1):
            params, train_loss = fit_model_at_order(
                train_data, decomp, order, prior_scales, data_config.noise_std
            )

            metrics = compute_test_metrics(
                train_data, test_data, decomp, params, data_config.noise_std
            )

            results["orders"].append(order)
            results["methods"].append(method)
            for key, value in metrics.items():
                if key not in results:
                    results[key] = []
                results[key].append(value)

    return results


def run_full_experiment(exp_config: ExperimentConfig):
    """Run the full experiment across all configurations.

    Args:
        exp_config: Experiment configuration
    """
    os.makedirs(exp_config.output_dir, exist_ok=True)

    all_results = []

    for n_obs in tqdm(exp_config.n_obs_values, desc="N values"):
        for rho in tqdm(exp_config.rho_values, desc="rho values", leave=False):
            for rep in tqdm(range(exp_config.n_replications), desc="Replications", leave=False):
                data_config = SyntheticDataConfig(
                    n_obs=n_obs,
                    rho_decay=rho,
                    seed=rep,
                )

                rng_key = jax.random.PRNGKey(rep + 1000 * int(rho * 10) + 10000 * n_obs)

                results = run_single_experiment(data_config, exp_config, rng_key)
                results["n_obs"] = n_obs
                results["rho"] = rho
                results["replication"] = rep

                all_results.append(results)

    # Save results
    results_path = Path(exp_config.output_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

    print(f"Results saved to {results_path}")

    # Generate plots
    plot_results(all_results, exp_config.output_dir)


def plot_results(results: List[Dict], output_dir: str):
    """Generate plots from experiment results.

    Args:
        results: List of result dictionaries
        output_dir: Output directory for plots
    """
    import pandas as pd

    # Convert to DataFrame
    rows = []
    for r in results:
        n_entries = len(r["orders"])
        for i in range(n_entries):
            row = {
                "n_obs": r["n_obs"],
                "rho": r["rho"],
                "replication": r["replication"],
                "order": r["orders"][i],
                "method": r["methods"][i],
                "test_ll_per_obs": r["test_ll_per_obs"][i],
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Plot 1: Test log-likelihood vs order for each method
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, n_obs in enumerate(sorted(df["n_obs"].unique())):
        ax = axes[idx // 2, idx % 2]
        subset = df[df["n_obs"] == n_obs]

        for method in subset["method"].unique():
            method_data = subset[subset["method"] == method]
            means = method_data.groupby("order")["test_ll_per_obs"].mean()
            stds = method_data.groupby("order")["test_ll_per_obs"].std()

            ax.errorbar(means.index, means.values, yerr=stds.values,
                       label=method, marker='o', capsize=3)

        ax.set_xlabel("Truncation Order K")
        ax.set_ylabel("Test Log-Likelihood per Observation")
        ax.set_title(f"N = {n_obs}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "test_ll_vs_order.png", dpi=150)
    plt.close()

    # Plot 2: Generalization gap (train - test) vs order
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, rho in enumerate(sorted(df["rho"].unique())):
        ax = axes[idx // 2, idx % 2]
        subset = df[df["rho"] == rho]

        for method in ["none", "gen_preserving"]:
            method_data = subset[subset["method"] == method]
            means = method_data.groupby("order")["test_ll_per_obs"].mean()

            ax.plot(means.index, means.values, label=method, marker='o')

        ax.set_xlabel("Truncation Order K")
        ax.set_ylabel("Test Log-Likelihood per Observation")
        ax.set_title(f"ρ = {rho} (effect decay rate)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "test_ll_by_rho.png", dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Synthetic validation experiment")
    parser.add_argument("--n_replications", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="results/synthetic_validation")
    parser.add_argument("--quick", action="store_true", help="Run quick version")
    args = parser.parse_args()

    if args.quick:
        config = ExperimentConfig(
            n_replications=5,
            n_obs_values=[1000, 5000],
            rho_values=[0.3, 0.5],
            output_dir=args.output_dir,
        )
    else:
        config = ExperimentConfig(
            n_replications=args.n_replications,
            output_dir=args.output_dir,
        )

    run_full_experiment(config)


if __name__ == "__main__":
    main()
