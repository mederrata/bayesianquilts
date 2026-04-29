#!/usr/bin/env python3
"""
Experiment 5: Ablation Studies

Systematic analysis of generalization-preserving regularization components:
1. Per-parameter vs per-component bounding
2. Effect of bound constant c
3. Sensitivity to noise scale estimation
4. Impact of α (order-specific) vs uniform scaling

Reference:
    Chang (2025), "A renormalization-group inspired hierarchical Bayesian
    framework for piecewise linear regression models"
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


@dataclass
class AblationConfig:
    """Configuration for ablation experiments."""
    n_obs: int = 5000
    d_factors: int = 3
    L_levels: int = 4
    n_features: int = 3
    noise_std: float = 1.0
    true_max_order: int = 2
    rho_decay: float = 0.5
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    n_replications: int = 30
    max_order: int = 3
    c_values: List[float] = None
    noise_misspec_factors: List[float] = None
    output_dir: str = "results/ablation_studies"

    def __post_init__(self):
        if self.c_values is None:
            self.c_values = [0.1, 0.25, 0.5, 1.0, 2.0]
        if self.noise_misspec_factors is None:
            self.noise_misspec_factors = [0.5, 0.75, 1.0, 1.5, 2.0]


def generate_synthetic_data(
    config: AblationConfig,
    rng_key: jax.random.PRNGKey,
) -> Tuple[Dict, Dict, Decomposed]:
    """Generate synthetic hierarchical data."""
    keys = jax.random.split(rng_key, 10)

    dimensions = [
        Dimension(f"factor_{i}", config.L_levels)
        for i in range(config.d_factors)
    ]
    interactions = Interactions(dimensions=dimensions)

    decomp = Decomposed(
        interactions=interactions,
        param_shape=[config.n_features],
        name="beta",
    )

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

    factor_indices = {}
    for i, dim in enumerate(dimensions):
        factor_indices[dim.name] = jax.random.randint(
            keys[i], shape=(config.n_obs,), minval=0, maxval=config.L_levels
        )

    X = jax.random.normal(keys[5], shape=(config.n_obs, config.n_features))

    interaction_indices = jnp.stack(
        [factor_indices[f"factor_{i}"] for i in range(config.d_factors)],
        axis=-1
    )
    beta = decomp.lookup(interaction_indices, tensors=true_params)
    eta = jnp.sum(X * beta, axis=-1)
    noise = config.noise_std * jax.random.normal(keys[6], shape=(config.n_obs,))
    y = eta + noise

    data = {
        "X": X,
        "y": y,
        **factor_indices,
    }

    return data, true_params, decomp


def fit_model(
    data: Dict,
    decomp: Decomposed,
    max_order: int,
    prior_scales: Dict[str, float],
    noise_std: float,
    n_steps: int = 500,
) -> Dict:
    """Fit model with given prior scales."""
    import optax

    active_components = [
        name for name in decomp._tensor_parts.keys()
        if decomp.component_order(name) <= max_order
    ]

    params = {
        name: jnp.zeros(decomp._tensor_part_shapes[name])
        for name in active_components
    }

    interaction_indices = jnp.stack(
        [data[f"factor_{i}"] for i in range(len(decomp._interactions._dimensions))],
        axis=-1
    )

    def loss_fn(params):
        full_params = {**params}
        for name in decomp._tensor_parts.keys():
            if name not in params:
                full_params[name] = jnp.zeros(decomp._tensor_part_shapes[name])

        beta = decomp.lookup(interaction_indices, tensors=full_params)
        eta = jnp.sum(data["X"] * beta, axis=-1)
        residuals = data["y"] - eta
        nll = 0.5 * jnp.sum(residuals ** 2) / (noise_std ** 2)

        log_prior = 0.0
        for name, param in params.items():
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
        params, opt_state, loss = step(params, opt_state)

    return params


def evaluate_model(
    data_train: Dict,
    data_test: Dict,
    decomp: Decomposed,
    params: Dict,
    noise_std: float,
) -> Dict[str, float]:
    """Evaluate model on train and test data."""
    full_params = {}
    for name in decomp._tensor_parts.keys():
        if name in params:
            full_params[name] = params[name]
        else:
            full_params[name] = jnp.zeros(decomp._tensor_part_shapes[name])

    def compute_metrics(data):
        interaction_indices = jnp.stack(
            [data[f"factor_{i}"] for i in range(len(decomp._interactions._dimensions))],
            axis=-1
        )
        beta = decomp.lookup(interaction_indices, tensors=full_params)
        eta = jnp.sum(data["X"] * beta, axis=-1)
        residuals = data["y"] - eta

        mse = float(jnp.mean(residuals ** 2))
        ll = -0.5 * float(jnp.sum(residuals ** 2)) / (noise_std ** 2)
        ll -= 0.5 * len(data["y"]) * float(jnp.log(2 * jnp.pi * noise_std ** 2))

        return mse, ll / len(data["y"])

    train_mse, train_ll = compute_metrics(data_train)
    test_mse, test_ll = compute_metrics(data_test)

    return {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_ll": train_ll,
        "test_ll": test_ll,
    }


def ablation_per_component(
    config: AblationConfig,
    exp_config: ExperimentConfig,
) -> List[Dict]:
    """Ablation: per-parameter vs per-component bounding."""
    results = []

    for rep in tqdm(range(exp_config.n_replications), desc="Per-component ablation"):
        rng_key = jax.random.PRNGKey(rep)
        keys = jax.random.split(rng_key, 3)

        train_data, true_params, decomp = generate_synthetic_data(config, keys[0])
        test_config = AblationConfig(**{**vars(config), "n_obs": config.n_obs // 5})
        test_data, _, _ = generate_synthetic_data(test_config, keys[1])

        for per_component in [False, True]:
            prior_scales = decomp.generalization_preserving_scales(
                noise_scale=config.noise_std,
                total_n=config.n_obs,
                c=0.5,
                per_component=per_component,
            )

            for order in range(exp_config.max_order + 1):
                params = fit_model(
                    train_data, decomp, order, prior_scales, config.noise_std
                )
                metrics = evaluate_model(
                    train_data, test_data, decomp, params, config.noise_std
                )

                results.append({
                    "ablation": "per_component",
                    "per_component": per_component,
                    "order": order,
                    "replication": rep,
                    **metrics,
                })

    return results


def ablation_c_value(
    config: AblationConfig,
    exp_config: ExperimentConfig,
) -> List[Dict]:
    """Ablation: effect of bound constant c."""
    results = []

    for rep in tqdm(range(exp_config.n_replications), desc="c-value ablation"):
        rng_key = jax.random.PRNGKey(rep)
        keys = jax.random.split(rng_key, 3)

        train_data, true_params, decomp = generate_synthetic_data(config, keys[0])
        test_config = AblationConfig(**{**vars(config), "n_obs": config.n_obs // 5})
        test_data, _, _ = generate_synthetic_data(test_config, keys[1])

        for c in exp_config.c_values:
            prior_scales = decomp.generalization_preserving_scales(
                noise_scale=config.noise_std,
                total_n=config.n_obs,
                c=c,
                per_component=True,
            )

            for order in range(exp_config.max_order + 1):
                params = fit_model(
                    train_data, decomp, order, prior_scales, config.noise_std
                )
                metrics = evaluate_model(
                    train_data, test_data, decomp, params, config.noise_std
                )

                results.append({
                    "ablation": "c_value",
                    "c": c,
                    "order": order,
                    "replication": rep,
                    **metrics,
                })

    return results


def ablation_noise_misspecification(
    config: AblationConfig,
    exp_config: ExperimentConfig,
) -> List[Dict]:
    """Ablation: sensitivity to noise scale misspecification."""
    results = []

    for rep in tqdm(range(exp_config.n_replications), desc="Noise misspec ablation"):
        rng_key = jax.random.PRNGKey(rep)
        keys = jax.random.split(rng_key, 3)

        train_data, true_params, decomp = generate_synthetic_data(config, keys[0])
        test_config = AblationConfig(**{**vars(config), "n_obs": config.n_obs // 5})
        test_data, _, _ = generate_synthetic_data(test_config, keys[1])

        for noise_factor in exp_config.noise_misspec_factors:
            assumed_noise = config.noise_std * noise_factor

            prior_scales = decomp.generalization_preserving_scales(
                noise_scale=assumed_noise,
                total_n=config.n_obs,
                c=0.5,
                per_component=True,
            )

            for order in range(exp_config.max_order + 1):
                params = fit_model(
                    train_data, decomp, order, prior_scales, config.noise_std
                )
                metrics = evaluate_model(
                    train_data, test_data, decomp, params, config.noise_std
                )

                results.append({
                    "ablation": "noise_misspec",
                    "noise_factor": noise_factor,
                    "order": order,
                    "replication": rep,
                    **metrics,
                })

    return results


def ablation_uniform_vs_order(
    config: AblationConfig,
    exp_config: ExperimentConfig,
) -> List[Dict]:
    """Ablation: order-specific (α) vs uniform scaling."""
    results = []

    for rep in tqdm(range(exp_config.n_replications), desc="Uniform vs order ablation"):
        rng_key = jax.random.PRNGKey(rep)
        keys = jax.random.split(rng_key, 3)

        train_data, true_params, decomp = generate_synthetic_data(config, keys[0])
        test_config = AblationConfig(**{**vars(config), "n_obs": config.n_obs // 5})
        test_data, _, _ = generate_synthetic_data(test_config, keys[1])

        for scaling_type in ["order_specific", "uniform"]:
            if scaling_type == "order_specific":
                prior_scales = decomp.generalization_preserving_scales(
                    noise_scale=config.noise_std,
                    total_n=config.n_obs,
                    c=0.5,
                    per_component=True,
                )
            else:
                global_scale = config.noise_std / np.sqrt(config.n_obs)
                prior_scales = {
                    name: global_scale
                    for name in decomp._tensor_parts.keys()
                }

            for order in range(exp_config.max_order + 1):
                params = fit_model(
                    train_data, decomp, order, prior_scales, config.noise_std
                )
                metrics = evaluate_model(
                    train_data, test_data, decomp, params, config.noise_std
                )

                results.append({
                    "ablation": "uniform_vs_order",
                    "scaling_type": scaling_type,
                    "order": order,
                    "replication": rep,
                    **metrics,
                })

    return results


def run_full_experiment(exp_config: ExperimentConfig):
    """Run all ablation studies."""
    os.makedirs(exp_config.output_dir, exist_ok=True)

    config = AblationConfig()
    all_results = []

    print("Running per-component ablation...")
    all_results.extend(ablation_per_component(config, exp_config))

    print("Running c-value ablation...")
    all_results.extend(ablation_c_value(config, exp_config))

    print("Running noise misspecification ablation...")
    all_results.extend(ablation_noise_misspecification(config, exp_config))

    print("Running uniform vs order ablation...")
    all_results.extend(ablation_uniform_vs_order(config, exp_config))

    results_path = Path(exp_config.output_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {results_path}")

    plot_results(all_results, exp_config)


def plot_results(results: List[Dict], exp_config: ExperimentConfig):
    """Generate plots for ablation results."""
    import pandas as pd

    df = pd.DataFrame(results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    subset = df[df["ablation"] == "per_component"]
    for per_comp in [False, True]:
        data = subset[subset["per_component"] == per_comp]
        label = "per-component" if per_comp else "per-parameter"
        means = data.groupby("order")["test_ll"].mean()
        stds = data.groupby("order")["test_ll"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   label=label, marker='o', capsize=3)

    ax.set_xlabel("Truncation Order K")
    ax.set_ylabel("Test Log-Likelihood per Obs")
    ax.set_title("Per-Parameter vs Per-Component Bounding")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    subset = df[df["ablation"] == "c_value"]
    for c in exp_config.c_values:
        data = subset[subset["c"] == c]
        means = data.groupby("order")["test_ll"].mean()
        stds = data.groupby("order")["test_ll"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   label=f"c={c}", marker='o', capsize=3)

    ax.set_xlabel("Truncation Order K")
    ax.set_ylabel("Test Log-Likelihood per Obs")
    ax.set_title("Effect of Bound Constant c")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    subset = df[df["ablation"] == "noise_misspec"]
    for factor in exp_config.noise_misspec_factors:
        data = subset[subset["noise_factor"] == factor]
        means = data.groupby("order")["test_ll"].mean()
        stds = data.groupby("order")["test_ll"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   label=f"σ×{factor}", marker='o', capsize=3)

    ax.set_xlabel("Truncation Order K")
    ax.set_ylabel("Test Log-Likelihood per Obs")
    ax.set_title("Noise Scale Misspecification Sensitivity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    subset = df[df["ablation"] == "uniform_vs_order"]
    for scaling in ["order_specific", "uniform"]:
        data = subset[subset["scaling_type"] == scaling]
        means = data.groupby("order")["test_ll"].mean()
        stds = data.groupby("order")["test_ll"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   label=scaling, marker='o', capsize=3)

    ax.set_xlabel("Truncation Order K")
    ax.set_ylabel("Test Log-Likelihood per Obs")
    ax.set_title("Order-Specific vs Uniform Scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(exp_config.output_dir) / "ablation_results.png", dpi=150)
    plt.close()

    print(f"Plots saved to {exp_config.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Ablation studies")
    parser.add_argument("--n_replications", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="results/ablation_studies")
    parser.add_argument("--quick", action="store_true", help="Run quick version")
    args = parser.parse_args()

    if args.quick:
        config = ExperimentConfig(
            n_replications=5,
            max_order=2,
            c_values=[0.25, 0.5, 1.0],
            noise_misspec_factors=[0.75, 1.0, 1.5],
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
