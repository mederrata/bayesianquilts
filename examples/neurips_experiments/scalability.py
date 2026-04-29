#!/usr/bin/env python3
"""
Experiment 7: Scalability Analysis

Analyzes computational scalability of hierarchical decomposition:
1. Runtime vs number of observations N
2. Runtime vs number of factors d
3. Runtime vs number of levels L
4. Memory usage analysis
5. Comparison with naive implementations

Reference:
    Chang (2025), "A renormalization-group inspired hierarchical Bayesian
    framework for piecewise linear regression models"
"""

import argparse
import gc
import json
import os
import time
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
class ScalabilityConfig:
    """Configuration for scalability experiments."""
    n_features: int = 3
    noise_std: float = 1.0
    n_warmup: int = 2
    n_timing: int = 5
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    n_obs_values: List[int] = None
    d_factors_values: List[int] = None
    L_levels_values: List[int] = None
    output_dir: str = "results/scalability"

    def __post_init__(self):
        if self.n_obs_values is None:
            self.n_obs_values = [1000, 5000, 10000, 50000, 100000]
        if self.d_factors_values is None:
            self.d_factors_values = [2, 3, 4, 5]
        if self.L_levels_values is None:
            self.L_levels_values = [3, 5, 10, 20]


def generate_data(
    n_obs: int,
    d_factors: int,
    L_levels: int,
    n_features: int,
    rng_key: jax.random.PRNGKey,
) -> Tuple[Dict, Decomposed]:
    """Generate synthetic data for scalability testing."""
    keys = jax.random.split(rng_key, d_factors + 3)

    dimensions = [
        Dimension(f"factor_{i}", L_levels)
        for i in range(d_factors)
    ]
    interactions = Interactions(dimensions=dimensions)

    decomp = Decomposed(
        interactions=interactions,
        param_shape=[n_features],
        name="beta",
    )

    factor_indices = {}
    for i, dim in enumerate(dimensions):
        factor_indices[dim.name] = jax.random.randint(
            keys[i], shape=(n_obs,), minval=0, maxval=L_levels
        )

    X = jax.random.normal(keys[d_factors], shape=(n_obs, n_features))
    y = jax.random.normal(keys[d_factors + 1], shape=(n_obs,))

    data = {
        "X": X,
        "y": y,
        **factor_indices,
    }

    return data, decomp


def time_lookup(
    data: Dict,
    decomp: Decomposed,
    config: ScalabilityConfig,
) -> Dict[str, float]:
    """Time the lookup operation."""
    dim_names = [d.name for d in decomp._interactions._dimensions]
    interaction_indices = jnp.stack(
        [jnp.array(data[name]) for name in dim_names],
        axis=-1
    )

    tensors = {
        name: jnp.zeros(shape)
        for name, shape in decomp._tensor_part_shapes.items()
    }

    @jax.jit
    def lookup_fn(indices, tensors):
        return decomp.lookup(indices, tensors=tensors)

    for _ in range(config.n_warmup):
        _ = lookup_fn(interaction_indices, tensors)
        jax.block_until_ready(_)

    times = []
    for _ in range(config.n_timing):
        start = time.perf_counter()
        result = lookup_fn(interaction_indices, tensors)
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
    }


def time_gradient(
    data: Dict,
    decomp: Decomposed,
    config: ScalabilityConfig,
    noise_std: float = 1.0,
) -> Dict[str, float]:
    """Time the gradient computation."""
    dim_names = [d.name for d in decomp._interactions._dimensions]
    interaction_indices = jnp.stack(
        [jnp.array(data[name]) for name in dim_names],
        axis=-1
    )

    params = {
        name: jnp.zeros(shape)
        for name, shape in decomp._tensor_part_shapes.items()
    }

    X = data["X"]
    y = data["y"]

    def loss_fn(params):
        beta = decomp.lookup(interaction_indices, tensors=params)
        eta = jnp.sum(X * beta, axis=-1)
        return 0.5 * jnp.mean((y - eta) ** 2)

    grad_fn = jax.jit(jax.grad(loss_fn))

    for _ in range(config.n_warmup):
        grads = grad_fn(params)
        jax.tree_util.tree_map(jax.block_until_ready, grads)

    times = []
    for _ in range(config.n_timing):
        start = time.perf_counter()
        grads = grad_fn(params)
        jax.tree_util.tree_map(jax.block_until_ready, grads)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
    }


def count_parameters(decomp: Decomposed, max_order: int = None) -> Dict[str, int]:
    """Count parameters at each order."""
    counts = {}
    total = 0

    for name, shape in decomp._tensor_part_shapes.items():
        order = decomp.component_order(name)
        if max_order is not None and order > max_order:
            continue

        n_params = int(np.prod(shape))

        if order not in counts:
            counts[order] = 0
        counts[order] += n_params
        total += n_params

    counts["total"] = total
    return counts


def estimate_memory(decomp: Decomposed) -> Dict[str, float]:
    """Estimate memory usage in MB."""
    bytes_per_param = 4

    total_bytes = 0
    for name, shape in decomp._tensor_part_shapes.items():
        n_params = int(np.prod(shape))
        total_bytes += n_params * bytes_per_param

    return {
        "parameters_mb": total_bytes / (1024 ** 2),
        "n_components": len(decomp._tensor_part_shapes),
    }


def run_n_obs_scaling(
    config: ScalabilityConfig,
    exp_config: ExperimentConfig,
) -> List[Dict]:
    """Test scaling with number of observations."""
    results = []
    d_factors = 3
    L_levels = 5

    for n_obs in tqdm(exp_config.n_obs_values, desc="N scaling"):
        gc.collect()

        rng_key = jax.random.PRNGKey(config.seed)
        data, decomp = generate_data(
            n_obs, d_factors, L_levels, config.n_features, rng_key
        )

        lookup_timing = time_lookup(data, decomp, config)
        grad_timing = time_gradient(data, decomp, config, config.noise_std)
        param_counts = count_parameters(decomp)
        memory = estimate_memory(decomp)

        results.append({
            "experiment": "n_obs_scaling",
            "n_obs": n_obs,
            "d_factors": d_factors,
            "L_levels": L_levels,
            "lookup_time_mean": lookup_timing["mean_time"],
            "lookup_time_std": lookup_timing["std_time"],
            "gradient_time_mean": grad_timing["mean_time"],
            "gradient_time_std": grad_timing["std_time"],
            "n_parameters": param_counts["total"],
            "memory_mb": memory["parameters_mb"],
        })

    return results


def run_d_factors_scaling(
    config: ScalabilityConfig,
    exp_config: ExperimentConfig,
) -> List[Dict]:
    """Test scaling with number of factors."""
    results = []
    n_obs = 10000
    L_levels = 4

    for d_factors in tqdm(exp_config.d_factors_values, desc="d scaling"):
        gc.collect()

        rng_key = jax.random.PRNGKey(config.seed)
        data, decomp = generate_data(
            n_obs, d_factors, L_levels, config.n_features, rng_key
        )

        lookup_timing = time_lookup(data, decomp, config)
        grad_timing = time_gradient(data, decomp, config, config.noise_std)
        param_counts = count_parameters(decomp)
        memory = estimate_memory(decomp)

        n_components = 0
        for order in range(d_factors + 1):
            from math import comb
            n_components += comb(d_factors, order)

        results.append({
            "experiment": "d_factors_scaling",
            "n_obs": n_obs,
            "d_factors": d_factors,
            "L_levels": L_levels,
            "lookup_time_mean": lookup_timing["mean_time"],
            "lookup_time_std": lookup_timing["std_time"],
            "gradient_time_mean": grad_timing["mean_time"],
            "gradient_time_std": grad_timing["std_time"],
            "n_parameters": param_counts["total"],
            "n_components": n_components,
            "memory_mb": memory["parameters_mb"],
        })

    return results


def run_L_levels_scaling(
    config: ScalabilityConfig,
    exp_config: ExperimentConfig,
) -> List[Dict]:
    """Test scaling with number of levels per factor."""
    results = []
    n_obs = 10000
    d_factors = 3

    for L_levels in tqdm(exp_config.L_levels_values, desc="L scaling"):
        gc.collect()

        rng_key = jax.random.PRNGKey(config.seed)
        data, decomp = generate_data(
            n_obs, d_factors, L_levels, config.n_features, rng_key
        )

        lookup_timing = time_lookup(data, decomp, config)
        grad_timing = time_gradient(data, decomp, config, config.noise_std)
        param_counts = count_parameters(decomp)
        memory = estimate_memory(decomp)

        results.append({
            "experiment": "L_levels_scaling",
            "n_obs": n_obs,
            "d_factors": d_factors,
            "L_levels": L_levels,
            "lookup_time_mean": lookup_timing["mean_time"],
            "lookup_time_std": lookup_timing["std_time"],
            "gradient_time_mean": grad_timing["mean_time"],
            "gradient_time_std": grad_timing["std_time"],
            "n_parameters": param_counts["total"],
            "memory_mb": memory["parameters_mb"],
        })

    return results


def run_truncation_comparison(
    config: ScalabilityConfig,
) -> List[Dict]:
    """Compare full vs truncated models."""
    results = []
    n_obs = 10000
    d_factors = 4
    L_levels = 5

    rng_key = jax.random.PRNGKey(config.seed)
    data, decomp = generate_data(
        n_obs, d_factors, L_levels, config.n_features, rng_key
    )

    for max_order in range(d_factors + 1):
        param_counts = count_parameters(decomp, max_order)

        results.append({
            "experiment": "truncation_comparison",
            "max_order": max_order,
            "n_parameters": param_counts["total"],
            "order_breakdown": {str(k): v for k, v in param_counts.items() if k != "total"},
        })

    return results


def run_full_experiment(exp_config: ExperimentConfig):
    """Run all scalability experiments."""
    os.makedirs(exp_config.output_dir, exist_ok=True)

    config = ScalabilityConfig()
    all_results = []

    print("Running N scaling experiment...")
    all_results.extend(run_n_obs_scaling(config, exp_config))

    print("Running d factors scaling experiment...")
    all_results.extend(run_d_factors_scaling(config, exp_config))

    print("Running L levels scaling experiment...")
    all_results.extend(run_L_levels_scaling(config, exp_config))

    print("Running truncation comparison...")
    all_results.extend(run_truncation_comparison(config))

    results_path = Path(exp_config.output_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {results_path}")

    plot_results(all_results, exp_config)


def plot_results(results: List[Dict], exp_config: ExperimentConfig):
    """Generate scalability plots."""
    import pandas as pd

    df = pd.DataFrame(results)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    subset = df[df["experiment"] == "n_obs_scaling"]
    ax.errorbar(subset["n_obs"], subset["lookup_time_mean"],
               yerr=subset["lookup_time_std"], label="Lookup", marker='o', capsize=3)
    ax.errorbar(subset["n_obs"], subset["gradient_time_mean"],
               yerr=subset["gradient_time_std"], label="Gradient", marker='s', capsize=3)
    ax.set_xlabel("Number of Observations (N)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Runtime vs Sample Size")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    subset = df[df["experiment"] == "d_factors_scaling"]
    ax.errorbar(subset["d_factors"], subset["gradient_time_mean"],
               yerr=subset["gradient_time_std"], marker='o', capsize=3)
    ax2 = ax.twinx()
    ax2.plot(subset["d_factors"], subset["n_parameters"], 'r--', marker='s', label="Parameters")
    ax.set_xlabel("Number of Factors (d)")
    ax.set_ylabel("Gradient Time (seconds)", color='blue')
    ax2.set_ylabel("Number of Parameters", color='red')
    ax.set_title("Runtime vs Number of Factors")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    subset = df[df["experiment"] == "L_levels_scaling"]
    ax.errorbar(subset["L_levels"], subset["gradient_time_mean"],
               yerr=subset["gradient_time_std"], marker='o', capsize=3)
    ax2 = ax.twinx()
    ax2.plot(subset["L_levels"], subset["memory_mb"], 'r--', marker='s')
    ax.set_xlabel("Levels per Factor (L)")
    ax.set_ylabel("Gradient Time (seconds)", color='blue')
    ax2.set_ylabel("Memory (MB)", color='red')
    ax.set_title("Runtime & Memory vs Levels")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    subset = df[df["experiment"] == "truncation_comparison"]
    ax.bar(subset["max_order"], subset["n_parameters"], color='steelblue')
    ax.set_xlabel("Maximum Interaction Order (K)")
    ax.set_ylabel("Number of Parameters")
    ax.set_title("Parameter Count by Truncation Order")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    for i, (order, params) in enumerate(zip(subset["max_order"], subset["n_parameters"])):
        ax.annotate(f"{params:,}", (order, params), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(Path(exp_config.output_dir) / "scalability_results.png", dpi=150)
    plt.close()

    print(f"Plots saved to {exp_config.output_dir}")

    print("\nScalability Summary:")
    print("\nN scaling (gradient time):")
    subset = df[df["experiment"] == "n_obs_scaling"]
    for _, row in subset.iterrows():
        print(f"  N={row['n_obs']:>7}: {row['gradient_time_mean']*1000:.2f} ms")

    print("\nd factors scaling:")
    subset = df[df["experiment"] == "d_factors_scaling"]
    for _, row in subset.iterrows():
        print(f"  d={row['d_factors']}: {row['n_parameters']:>10,} params, {row['gradient_time_mean']*1000:.2f} ms")

    print("\nTruncation effect:")
    subset = df[df["experiment"] == "truncation_comparison"]
    full_params = subset["n_parameters"].max()
    for _, row in subset.iterrows():
        pct = 100 * row['n_parameters'] / full_params
        print(f"  K≤{row['max_order']}: {row['n_parameters']:>10,} params ({pct:.1f}% of full)")


def main():
    parser = argparse.ArgumentParser(description="Scalability analysis")
    parser.add_argument("--output_dir", type=str, default="results/scalability")
    parser.add_argument("--quick", action="store_true", help="Run quick version")
    args = parser.parse_args()

    if args.quick:
        config = ExperimentConfig(
            n_obs_values=[1000, 10000],
            d_factors_values=[2, 3, 4],
            L_levels_values=[3, 5, 10],
            output_dir=args.output_dir,
        )
    else:
        config = ExperimentConfig(
            output_dir=args.output_dir,
        )

    run_full_experiment(config)


if __name__ == "__main__":
    main()
