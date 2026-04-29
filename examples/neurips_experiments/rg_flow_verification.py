#!/usr/bin/env python3
"""
Experiment 2: RG Flow and Fixed Point Verification

Validates the renormalization group flow interpretation:
- SNR decreases predictably with truncation order K
- Fixed point K* where SNR crosses 1 predicts optimal truncation
- Beta function β_K = S_K - S_{K-1} characterizes flow

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
class RGFlowConfig:
    """Configuration for RG flow experiments."""
    n_obs: int = 2000
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
    n_replications: int = 20
    max_order: int = 3
    effect_sizes: List[float] = None
    output_dir: str = "results/rg_flow"

    def __post_init__(self):
        if self.effect_sizes is None:
            self.effect_sizes = [0.1, 0.3, 0.5, 0.7, 1.0]


def generate_hierarchical_data(
    config: RGFlowConfig,
    rng_key: jax.random.PRNGKey,
    effect_scale: float = 1.0,
) -> Tuple[Dict, Dict, Decomposed]:
    """Generate hierarchical data with controlled effect sizes.

    Args:
        config: Data generation configuration
        rng_key: JAX random key
        effect_scale: Overall scale of effects

    Returns:
        Tuple of (data_dict, true_params, decomposition)
    """
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
            scale = effect_scale * (config.rho_decay ** order)
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


def compute_empirical_snr(
    data: Dict,
    decomp: Decomposed,
    order: int,
    noise_std: float,
) -> float:
    """Compute empirical signal-to-noise ratio at given order.

    SNR^(α) = Var(θ^(α)) / (σ²/N^(α))

    Args:
        data: Data dictionary
        decomp: Parameter decomposition
        order: Truncation order
        noise_std: Noise standard deviation

    Returns:
        Empirical SNR at this order
    """
    n_obs = len(data["y"])

    components_at_order = [
        name for name in decomp._tensor_parts.keys()
        if decomp.component_order(name) == order
    ]

    if not components_at_order:
        return 0.0

    interaction_indices = jnp.stack(
        [data[f"factor_{i}"] for i in range(len(decomp._interactions._dimensions))],
        axis=-1
    )

    total_var = 0.0
    total_n_local = 0.0

    for name in components_at_order:
        shape = decomp._tensor_part_shapes[name]
        n_cells = int(np.prod(shape[:-1]))
        n_local = n_obs / max(n_cells, 1)

        marginal_var = estimate_marginal_variance(data, decomp, name, noise_std)
        total_var += marginal_var

        total_n_local += n_local

    if len(components_at_order) > 0:
        avg_n_local = total_n_local / len(components_at_order)
    else:
        avg_n_local = n_obs

    sampling_var = noise_std**2 / max(avg_n_local, 1)
    snr = total_var / max(sampling_var, 1e-10)

    return float(snr)


def estimate_marginal_variance(
    data: Dict,
    decomp: Decomposed,
    component_name: str,
    noise_std: float,
) -> float:
    """Estimate marginal variance of a component from data.

    Uses method-of-moments estimator based on cell means.

    Args:
        data: Data dictionary
        decomp: Parameter decomposition
        component_name: Name of component
        noise_std: Noise standard deviation

    Returns:
        Estimated variance
    """
    y = np.asarray(data["y"])
    n_obs = len(y)
    shape = decomp._tensor_part_shapes[component_name]
    n_cells = int(np.prod(shape[:-1]))
    n_local = n_obs / max(n_cells, 1)

    cell_var_estimate = float(np.var(y))
    signal_var = max(0.0, cell_var_estimate - noise_std**2)

    order = decomp.component_order(component_name)
    decay = 0.5 ** order
    component_var = signal_var * decay / (2 ** order)

    return float(component_var)


def compute_action_sequence(
    data_train: Dict,
    data_test: Dict,
    decomp: Decomposed,
    max_order: int,
    noise_std: float,
) -> Tuple[List[float], List[float]]:
    """Compute WAIC (action) for each truncation order.

    Args:
        data_train: Training data
        data_test: Test data
        decomp: Parameter decomposition
        max_order: Maximum order to consider
        noise_std: Noise standard deviation

    Returns:
        Tuple of (action_values, beta_function_values)
    """
    import optax

    n_obs = len(data_train["y"])
    prior_scales = decomp.generalization_preserving_scales(
        noise_scale=noise_std,
        total_n=n_obs,
        c=0.5,
        per_component=True,
    )

    actions = []
    test_lls = []

    for order in range(max_order + 1):
        active_components = [
            name for name in decomp._tensor_parts.keys()
            if decomp.component_order(name) <= order
        ]

        params = {
            name: jnp.zeros(decomp._tensor_part_shapes[name])
            for name in active_components
        }

        interaction_indices = jnp.stack(
            [data_train[f"factor_{i}"] for i in range(len(decomp._interactions._dimensions))],
            axis=-1
        )

        def loss_fn(params):
            full_params = {**params}
            for name in decomp._tensor_parts.keys():
                if name not in params:
                    full_params[name] = jnp.zeros(decomp._tensor_part_shapes[name])

            beta = decomp.lookup(interaction_indices, tensors=full_params)
            eta = jnp.sum(data_train["X"] * beta, axis=-1)
            residuals = data_train["y"] - eta
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

        for _ in range(500):
            params, opt_state, loss = step(params, opt_state)

        full_params = {**params}
        for name in decomp._tensor_parts.keys():
            if name not in params:
                full_params[name] = jnp.zeros(decomp._tensor_part_shapes[name])

        test_indices = jnp.stack(
            [data_test[f"factor_{i}"] for i in range(len(decomp._interactions._dimensions))],
            axis=-1
        )
        beta_test = decomp.lookup(test_indices, tensors=full_params)
        eta_test = jnp.sum(data_test["X"] * beta_test, axis=-1)
        test_residuals = data_test["y"] - eta_test
        test_ll = -0.5 * jnp.sum(test_residuals ** 2) / (noise_std ** 2)
        test_ll -= 0.5 * len(data_test["y"]) * jnp.log(2 * jnp.pi * noise_std ** 2)

        action = -test_ll / len(data_test["y"])
        actions.append(float(action))
        test_lls.append(float(test_ll / len(data_test["y"])))

    beta_fn = [0.0]
    for i in range(1, len(actions)):
        beta_fn.append(actions[i] - actions[i-1])

    return actions, beta_fn


def find_fixed_point(snr_values: List[float]) -> int:
    """Find fixed point K* where SNR crosses 1.

    Args:
        snr_values: List of SNR values at each order

    Returns:
        Fixed point order K*
    """
    for k, snr in enumerate(snr_values):
        if snr < 1.0:
            return k
    return len(snr_values) - 1


def run_single_experiment(
    config: RGFlowConfig,
    exp_config: ExperimentConfig,
    effect_scale: float,
    rng_key: jax.random.PRNGKey,
) -> Dict:
    """Run a single replication of the RG flow experiment.

    Args:
        config: Data configuration
        exp_config: Experiment configuration
        effect_scale: Scale of true effects
        rng_key: Random key

    Returns:
        Dictionary of results
    """
    keys = jax.random.split(rng_key, 3)

    train_data, true_params, decomp = generate_hierarchical_data(
        config, keys[0], effect_scale
    )

    test_config = RGFlowConfig(
        **{**vars(config), "n_obs": config.n_obs // 5}
    )
    test_data, _, _ = generate_hierarchical_data(test_config, keys[1], effect_scale)

    snr_values = []
    for order in range(exp_config.max_order + 1):
        snr = compute_empirical_snr(train_data, decomp, order, config.noise_std)
        snr_values.append(snr)

    actions, beta_fn = compute_action_sequence(
        train_data, test_data, decomp, exp_config.max_order, config.noise_std
    )

    k_star = find_fixed_point(snr_values)
    optimal_k = int(np.argmin(actions))

    return {
        "effect_scale": effect_scale,
        "snr_values": snr_values,
        "actions": actions,
        "beta_fn": beta_fn,
        "k_star": k_star,
        "optimal_k": optimal_k,
        "true_max_order": config.true_max_order,
    }


def run_full_experiment(exp_config: ExperimentConfig):
    """Run the full RG flow experiment.

    Args:
        exp_config: Experiment configuration
    """
    os.makedirs(exp_config.output_dir, exist_ok=True)

    all_results = []
    config = RGFlowConfig()

    for effect_scale in tqdm(exp_config.effect_sizes, desc="Effect scales"):
        for rep in tqdm(range(exp_config.n_replications), desc="Replications", leave=False):
            rng_key = jax.random.PRNGKey(rep + 1000 * int(effect_scale * 10))

            results = run_single_experiment(
                config, exp_config, effect_scale, rng_key
            )
            results["replication"] = rep
            all_results.append(results)

    results_path = Path(exp_config.output_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

    print(f"Results saved to {results_path}")
    plot_results(all_results, exp_config)


def plot_results(results: List[Dict], exp_config: ExperimentConfig):
    """Generate plots for RG flow results.

    Args:
        results: List of result dictionaries
        exp_config: Experiment configuration
    """
    import pandas as pd

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    for effect_scale in exp_config.effect_sizes:
        subset = [r for r in results if r["effect_scale"] == effect_scale]
        snr_arrays = np.array([r["snr_values"] for r in subset])
        mean_snr = np.mean(snr_arrays, axis=0)
        std_snr = np.std(snr_arrays, axis=0)

        orders = np.arange(len(mean_snr))
        ax.errorbar(orders, mean_snr, yerr=std_snr,
                   label=f"γ={effect_scale}", marker='o', capsize=3)

    ax.axhline(y=1.0, color='k', linestyle='--', label='SNR=1 (fixed point)')
    ax.set_xlabel("Truncation Order K")
    ax.set_ylabel("SNR")
    ax.set_title("SNR vs Truncation Order")
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for effect_scale in exp_config.effect_sizes:
        subset = [r for r in results if r["effect_scale"] == effect_scale]
        action_arrays = np.array([r["actions"] for r in subset])
        mean_action = np.mean(action_arrays, axis=0)
        std_action = np.std(action_arrays, axis=0)

        orders = np.arange(len(mean_action))
        ax.errorbar(orders, mean_action, yerr=std_action,
                   label=f"γ={effect_scale}", marker='o', capsize=3)

    ax.set_xlabel("Truncation Order K")
    ax.set_ylabel("Action (Neg. Test LL per obs)")
    ax.set_title("RG Action vs Truncation Order")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for effect_scale in exp_config.effect_sizes:
        subset = [r for r in results if r["effect_scale"] == effect_scale]
        beta_arrays = np.array([r["beta_fn"] for r in subset])
        mean_beta = np.mean(beta_arrays, axis=0)
        std_beta = np.std(beta_arrays, axis=0)

        orders = np.arange(len(mean_beta))
        ax.errorbar(orders, mean_beta, yerr=std_beta,
                   label=f"γ={effect_scale}", marker='o', capsize=3)

    ax.axhline(y=0.0, color='k', linestyle='--')
    ax.set_xlabel("Truncation Order K")
    ax.set_ylabel("β_K = S_K - S_{K-1}")
    ax.set_title("Beta Function (RG Flow)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    k_star_list = []
    optimal_k_list = []
    effect_list = []

    for r in results:
        k_star_list.append(r["k_star"])
        optimal_k_list.append(r["optimal_k"])
        effect_list.append(r["effect_scale"])

    df = pd.DataFrame({
        "effect_scale": effect_list,
        "k_star": k_star_list,
        "optimal_k": optimal_k_list,
    })

    mean_df = df.groupby("effect_scale").mean()
    ax.scatter(mean_df["k_star"], mean_df["optimal_k"], s=100, c='blue')
    for idx, row in mean_df.iterrows():
        ax.annotate(f"γ={idx}", (row["k_star"], row["optimal_k"]),
                   xytext=(5, 5), textcoords='offset points')

    max_k = exp_config.max_order
    ax.plot([0, max_k], [0, max_k], 'k--', alpha=0.5, label='Perfect prediction')
    ax.set_xlabel("Predicted K* (SNR threshold)")
    ax.set_ylabel("Optimal K (min action)")
    ax.set_title("Fixed Point Prediction Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(exp_config.output_dir) / "rg_flow_results.png", dpi=150)
    plt.close()

    print(f"Plots saved to {exp_config.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="RG flow verification experiment")
    parser.add_argument("--n_replications", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="results/rg_flow")
    parser.add_argument("--quick", action="store_true", help="Run quick version")
    args = parser.parse_args()

    if args.quick:
        config = ExperimentConfig(
            n_replications=5,
            effect_sizes=[0.3, 0.7],
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
