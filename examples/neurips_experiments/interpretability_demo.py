#!/usr/bin/env python3
"""
Experiment 6: Interpretability Demonstration

Demonstrates interpretability benefits of hierarchical decomposition:
1. Effect attribution to different scales
2. Variance decomposition across interaction orders
3. Visualization of learned hierarchical structure
4. Comparison with black-box alternatives

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
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


@dataclass
class InterpretabilityConfig:
    """Configuration for interpretability experiments."""
    n_obs: int = 10000
    d_factors: int = 3
    L_levels: int = 5
    n_features: int = 4
    noise_std: float = 1.0
    seed: int = 42


def generate_interpretable_data(
    config: InterpretabilityConfig,
    rng_key: jax.random.PRNGKey,
) -> Tuple[Dict, Dict, Decomposed, Dict]:
    """Generate data with known interpretable structure.

    Creates data where:
    - Global effect is strongest
    - First-order effects represent factor main effects
    - Second-order effects are sparse interactions

    Returns:
        Tuple of (data, true_params, decomp, effect_descriptions)
    """
    keys = jax.random.split(rng_key, 10)

    factor_names = ["region", "age_group", "income_level"]
    level_names = {
        "region": ["North", "South", "East", "West", "Central"],
        "age_group": ["18-25", "26-35", "36-45", "46-55", "56+"],
        "income_level": ["Low", "Medium-Low", "Medium", "Medium-High", "High"],
    }

    dimensions = [
        Dimension(name, config.L_levels)
        for name in factor_names
    ]
    interactions = Interactions(dimensions=dimensions)

    feature_names = ["base_rate", "seasonal", "trend", "volatility"]

    decomp = Decomposed(
        interactions=interactions,
        param_shape=[config.n_features],
        name="beta",
    )

    true_params = {}
    effect_descriptions = {}

    for name, shape in decomp._tensor_part_shapes.items():
        order = decomp.component_order(name)
        parts = name.replace("beta_", "").split("__")

        if order == 0:
            # Global effect: shape is (1, 1, 1, n_features)
            true_params[name] = jnp.array([[[[1.0, 0.2, 0.1, 0.05]]]])
            effect_descriptions[name] = "Global baseline effect"

        elif order == 1:
            factor = parts[0]
            if factor == "region":
                values = jnp.array([0.3, -0.2, 0.1, -0.1, 0.0])
                desc = "Regional variation"
            elif factor == "age_group":
                values = jnp.array([-0.4, -0.1, 0.1, 0.2, 0.3])
                desc = "Age-related trend"
            else:  # income_level
                values = jnp.array([-0.5, -0.2, 0.0, 0.3, 0.6])
                desc = "Income effect"

            feature_weights = jnp.array([1.0, 0.5, 0.3, 0.1])
            # Create effect tensor matching the expected shape
            effect_tensor = jnp.zeros(shape)
            # Find which axis has L_levels (not 1)
            for axis_idx, dim_size in enumerate(shape[:-1]):
                if dim_size == config.L_levels:
                    # Expand values along this axis
                    expand_shape = [1] * len(shape)
                    expand_shape[axis_idx] = config.L_levels
                    expand_shape[-1] = config.n_features
                    effect_tensor = (values.reshape(expand_shape[:axis_idx+1] + [1]*(len(shape)-axis_idx-2) + [1])
                                    * feature_weights.reshape([1]*(len(shape)-1) + [config.n_features]))
                    break
            true_params[name] = effect_tensor
            effect_descriptions[name] = desc

        elif order == 2:
            base = 0.1 * jax.random.normal(keys[order], shape=shape)
            if "region" in name and "income_level" in name:
                # Add specific interaction effects at corners
                # Find which axes correspond to region and income_level
                base = base.at[0, :, config.L_levels-1, :].set(0.3)
                base = base.at[config.L_levels-1, :, 0, :].set(0.3)
                effect_descriptions[name] = "Region × Income interaction"
            else:
                effect_descriptions[name] = f"Interaction: {' × '.join(parts)}"
            true_params[name] = base

        else:
            true_params[name] = jnp.zeros(shape)
            effect_descriptions[name] = "Higher-order (zeroed)"

    factor_indices = {}
    for i, dim in enumerate(dimensions):
        factor_indices[dim.name] = jax.random.randint(
            keys[i], shape=(config.n_obs,), minval=0, maxval=config.L_levels
        )

    X = jax.random.normal(keys[5], shape=(config.n_obs, config.n_features))
    X = X.at[:, 0].set(1.0)

    interaction_indices = jnp.stack(
        [factor_indices[name] for name in factor_names],
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

    metadata = {
        "factor_names": factor_names,
        "level_names": level_names,
        "feature_names": feature_names,
    }

    return data, true_params, decomp, effect_descriptions, metadata


def fit_hierarchical_model(
    data: Dict,
    decomp: Decomposed,
    max_order: int,
    noise_std: float,
    n_steps: int = 1000,
) -> Dict:
    """Fit hierarchical model and return parameters."""
    import optax

    prior_scales = decomp.generalization_preserving_scales(
        noise_scale=noise_std,
        total_n=len(data["y"]),
        c=0.5,
        per_component=True,
    )

    active_components = [
        name for name in decomp._tensor_parts.keys()
        if decomp.component_order(name) <= max_order
    ]

    params = {
        name: jnp.zeros(decomp._tensor_part_shapes[name])
        for name in active_components
    }

    dim_names = [d.name for d in decomp._interactions._dimensions]
    interaction_indices = jnp.stack(
        [jnp.array(data[name]) for name in dim_names],
        axis=-1
    )

    X = jnp.array(data["X"])
    y = jnp.array(data["y"])

    def loss_fn(params):
        full_params = {**params}
        for name in decomp._tensor_parts.keys():
            if name not in params:
                full_params[name] = jnp.zeros(decomp._tensor_part_shapes[name])

        beta = decomp.lookup(interaction_indices, tensors=full_params)
        eta = jnp.sum(X * beta, axis=-1)
        residuals = y - eta
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


def compute_variance_decomposition(
    decomp: Decomposed,
    params: Dict,
) -> Dict[str, float]:
    """Compute variance contribution of each component.

    Returns:
        Dictionary mapping component names to variance contributions
    """
    variances = {}

    for name, param in params.items():
        var = float(jnp.var(param))
        variances[name] = var

    total_var = sum(variances.values())
    if total_var > 0:
        variances = {k: v / total_var for k, v in variances.items()}

    return variances


def compute_effect_attribution(
    data: Dict,
    decomp: Decomposed,
    params: Dict,
) -> Dict[str, np.ndarray]:
    """Compute contribution of each component to predictions.

    Returns:
        Dictionary mapping component names to effect arrays
    """
    dim_names = [d.name for d in decomp._interactions._dimensions]
    interaction_indices = jnp.stack(
        [jnp.array(data[name]) for name in dim_names],
        axis=-1
    )

    effects = {}

    for name in params.keys():
        single_param = {name: params[name]}
        for other_name in decomp._tensor_parts.keys():
            if other_name != name:
                single_param[other_name] = jnp.zeros(decomp._tensor_part_shapes[other_name])

        beta = decomp.lookup(interaction_indices, tensors=single_param)
        eta = jnp.sum(data["X"] * beta, axis=-1)
        effects[name] = np.array(eta)

    return effects


def plot_effect_heatmap(
    params: Dict,
    decomp: Decomposed,
    metadata: Dict,
    output_path: str,
):
    """Plot heatmaps of first-order effects."""
    first_order = [
        name for name in params.keys()
        if decomp.component_order(name) == 1
    ]

    fig, axes = plt.subplots(1, len(first_order), figsize=(5 * len(first_order), 4))
    if len(first_order) == 1:
        axes = [axes]

    for idx, name in enumerate(first_order):
        ax = axes[idx]
        param = np.array(params[name])

        effect_on_base = param[:, 0]

        parts = name.replace("beta_", "").split("__")
        factor = parts[0]
        level_names = metadata["level_names"].get(factor, [f"L{i}" for i in range(len(effect_on_base))])

        colors = plt.cm.RdBu_r(np.linspace(0.1, 0.9, len(effect_on_base)))
        bars = ax.bar(range(len(effect_on_base)), effect_on_base, color=colors)

        ax.set_xticks(range(len(effect_on_base)))
        ax.set_xticklabels(level_names, rotation=45, ha='right')
        ax.set_ylabel("Effect on Outcome")
        ax.set_title(f"{factor.replace('_', ' ').title()} Effects")
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_interaction_heatmap(
    params: Dict,
    decomp: Decomposed,
    metadata: Dict,
    output_path: str,
):
    """Plot heatmaps of second-order interactions."""
    second_order = [
        name for name in params.keys()
        if decomp.component_order(name) == 2
    ]

    n_plots = len(second_order)
    if n_plots == 0:
        return

    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, name in enumerate(second_order):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        param = np.array(params[name])
        effect_matrix = param[:, :, 0]

        parts = name.replace("beta_", "").split("__")

        vmax = np.abs(effect_matrix).max()
        im = ax.imshow(effect_matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax)

        ax.set_xlabel(parts[1].replace('_', ' ').title())
        ax.set_ylabel(parts[0].replace('_', ' ').title())
        ax.set_title(f"{parts[0]} × {parts[1]}")

        plt.colorbar(im, ax=ax, shrink=0.8)

    for idx in range(n_plots, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_variance_decomposition(
    variances: Dict[str, float],
    decomp: Decomposed,
    output_path: str,
):
    """Plot variance decomposition by order."""
    order_variances = {}
    for name, var in variances.items():
        order = decomp.component_order(name)
        if order not in order_variances:
            order_variances[order] = 0
        order_variances[order] += var

    orders = sorted(order_variances.keys())
    values = [order_variances[o] for o in orders]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(orders)))
    ax.bar(orders, values, color=colors)
    ax.set_xlabel("Interaction Order")
    ax.set_ylabel("Proportion of Variance")
    ax.set_title("Variance Decomposition by Order")
    ax.set_xticks(orders)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    labels = [f"Order {o}" for o in orders]
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors,
           startangle=90)
    ax.set_title("Variance Attribution")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def compare_with_ground_truth(
    true_params: Dict,
    fitted_params: Dict,
    decomp: Decomposed,
) -> Dict[str, float]:
    """Compare fitted parameters with ground truth."""
    results = {}

    for name in fitted_params.keys():
        if name in true_params:
            true_val = np.array(true_params[name])
            fitted_val = np.array(fitted_params[name])

            mse = float(np.mean((true_val - fitted_val) ** 2))
            correlation = float(np.corrcoef(true_val.flatten(), fitted_val.flatten())[0, 1])

            results[name] = {
                "mse": mse,
                "correlation": correlation,
            }

    return results


def run_interpretability_demo(output_dir: str):
    """Run full interpretability demonstration."""
    os.makedirs(output_dir, exist_ok=True)

    config = InterpretabilityConfig()
    rng_key = jax.random.PRNGKey(config.seed)

    print("Generating interpretable synthetic data...")
    data, true_params, decomp, effect_descriptions, metadata = generate_interpretable_data(
        config, rng_key
    )

    print("Fitting hierarchical model...")
    fitted_params = fit_hierarchical_model(data, decomp, max_order=2, noise_std=config.noise_std)

    print("Computing variance decomposition...")
    variances = compute_variance_decomposition(decomp, fitted_params)

    print("Computing effect attribution...")
    effects = compute_effect_attribution(data, decomp, fitted_params)

    print("Comparing with ground truth...")
    comparison = compare_with_ground_truth(true_params, fitted_params, decomp)

    print("Generating visualizations...")
    plot_effect_heatmap(
        fitted_params, decomp, metadata,
        str(Path(output_dir) / "first_order_effects.png")
    )

    plot_interaction_heatmap(
        fitted_params, decomp, metadata,
        str(Path(output_dir) / "interaction_effects.png")
    )

    plot_variance_decomposition(
        variances, decomp,
        str(Path(output_dir) / "variance_decomposition.png")
    )

    results = {
        "effect_descriptions": effect_descriptions,
        "variance_decomposition": variances,
        "ground_truth_comparison": comparison,
        "effect_means": {name: float(np.mean(np.abs(eff))) for name, eff in effects.items()},
    }

    results_path = Path(output_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))

    print(f"\nResults saved to {output_dir}")
    print("\nVariance decomposition by order:")
    order_var = {}
    for name, var in variances.items():
        order = decomp.component_order(name)
        order_var[order] = order_var.get(order, 0) + var
    for order in sorted(order_var.keys()):
        print(f"  Order {order}: {order_var[order]:.1%}")

    print("\nGround truth recovery (correlation):")
    for name, comp in comparison.items():
        order = decomp.component_order(name)
        desc = effect_descriptions.get(name, "")
        print(f"  {name}: r={comp['correlation']:.3f} ({desc})")


def main():
    parser = argparse.ArgumentParser(description="Interpretability demonstration")
    parser.add_argument("--output_dir", type=str, default="results/interpretability")
    args = parser.parse_args()

    run_interpretability_demo(args.output_dir)


if __name__ == "__main__":
    main()
