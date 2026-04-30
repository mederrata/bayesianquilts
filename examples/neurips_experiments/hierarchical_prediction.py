#!/usr/bin/env python3
"""
Experiment 4: Hierarchical Prediction on Real Datasets

Demonstrates benefits of hierarchical decomposition on datasets with
natural hierarchical structure: MovieLens (users × items × time).

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
from urllib.request import urlretrieve
import zipfile

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


@dataclass
class ExperimentConfig:
    """Configuration for hierarchical prediction experiments."""
    n_replications: int = 10
    max_order: int = 2
    train_frac: float = 0.8
    output_dir: str = "results/hierarchical_prediction"
    data_dir: str = "data/movielens"
    n_users_subsample: int = 1000
    n_items_subsample: int = 500


def download_movielens(data_dir: str) -> pd.DataFrame:
    """Download MovieLens 100K dataset.

    Args:
        data_dir: Directory to store data

    Returns:
        DataFrame with ratings
    """
    os.makedirs(data_dir, exist_ok=True)
    filepath = Path(data_dir) / "ratings.csv"

    if filepath.exists():
        return pd.read_csv(filepath)

    print("Downloading MovieLens 100K dataset...")
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = Path(data_dir) / "ml-100k.zip"

    if not zip_path.exists():
        urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open("ml-100k/u.data") as f:
            df = pd.read_csv(f, sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open("ml-100k/u.user") as f:
            users = pd.read_csv(f, sep="|", names=["user_id", "age", "gender", "occupation", "zip"],
                              encoding='latin-1')
        with z.open("ml-100k/u.item") as f:
            items = pd.read_csv(f, sep="|", names=["item_id", "title", "release_date", "video_date",
                                                    "imdb_url", "unknown", "action", "adventure",
                                                    "animation", "children", "comedy", "crime",
                                                    "documentary", "drama", "fantasy", "film_noir",
                                                    "horror", "musical", "mystery", "romance",
                                                    "sci_fi", "thriller", "war", "western"],
                              encoding='latin-1')

    df = df.merge(users[["user_id", "age", "gender", "occupation"]], on="user_id")
    df["age_bin"] = pd.cut(df["age"], bins=[0, 20, 30, 40, 50, 100],
                          labels=["<20", "20-30", "30-40", "40-50", "50+"])

    genre_cols = ["action", "adventure", "animation", "children", "comedy", "crime",
                 "documentary", "drama", "fantasy", "film_noir", "horror", "musical",
                 "mystery", "romance", "sci_fi", "thriller", "war", "western"]
    items["primary_genre"] = items[genre_cols].idxmax(axis=1)
    df = df.merge(items[["item_id", "primary_genre"]], on="item_id")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["time_of_day"] = pd.cut(df["hour_of_day"], bins=[0, 6, 12, 18, 24],
                               labels=["night", "morning", "afternoon", "evening"])

    df.to_csv(filepath, index=False)
    return df


def prepare_movielens_data(
    df: pd.DataFrame,
    exp_config: ExperimentConfig,
    rng_key: jax.random.PRNGKey,
) -> Tuple[Dict, List[Dimension]]:
    """Prepare MovieLens data for hierarchical modeling.

    Args:
        df: Raw DataFrame
        exp_config: Experiment configuration
        rng_key: Random key for subsampling

    Returns:
        Tuple of (data_dict, dimensions)
    """
    unique_users = df["user_id"].unique()
    unique_items = df["item_id"].unique()

    if len(unique_users) > exp_config.n_users_subsample:
        np.random.seed(int(rng_key[0]))
        selected_users = np.random.choice(unique_users, exp_config.n_users_subsample, replace=False)
        df = df[df["user_id"].isin(selected_users)]

    if len(unique_items) > exp_config.n_items_subsample:
        np.random.seed(int(rng_key[1]))
        selected_items = np.random.choice(df["item_id"].unique(), exp_config.n_items_subsample, replace=False)
        df = df[df["item_id"].isin(selected_items)]

    user_map = {u: i for i, u in enumerate(df["user_id"].unique())}
    item_map = {it: i for i, it in enumerate(df["item_id"].unique())}
    age_map = {a: i for i, a in enumerate(df["age_bin"].unique())}
    gender_map = {g: i for i, g in enumerate(df["gender"].unique())}
    genre_map = {g: i for i, g in enumerate(df["primary_genre"].unique())}
    time_map = {t: i for i, t in enumerate(df["time_of_day"].unique())}

    dimensions = [
        Dimension("user", len(user_map)),
        Dimension("item", len(item_map)),
        Dimension("age_bin", len(age_map)),
        Dimension("gender", len(gender_map)),
        Dimension("genre", len(genre_map)),
        Dimension("time_of_day", len(time_map)),
    ]

    data = {
        "y": (df["rating"].values - 3.0) / 2.0,
        "user": df["user_id"].map(user_map).values,
        "item": df["item_id"].map(item_map).values,
        "age_bin": df["age_bin"].map(age_map).values,
        "gender": df["gender"].map(gender_map).values,
        "genre": df["primary_genre"].map(genre_map).values,
        "time_of_day": df["time_of_day"].map(time_map).values,
        "X": np.ones((len(df), 1), dtype=np.float32),
    }

    return data, dimensions


def split_data(
    data: Dict,
    train_frac: float,
    seed: int,
) -> Tuple[Dict, Dict]:
    """Split data into train and test sets.

    Args:
        data: Full data dictionary
        train_frac: Fraction of data for training
        seed: Random seed

    Returns:
        Tuple of (train_data, test_data)
    """
    n = len(data["y"])
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(indices, train_size=train_frac,
                                           random_state=seed)

    train_data = {key: val[train_idx] for key, val in data.items()}
    test_data = {key: val[test_idx] for key, val in data.items()}

    return train_data, test_data


def fit_rating_model(
    data: Dict,
    decomp: Decomposed,
    max_order: int,
    prior_scales: Dict[str, float],
    noise_std: float = 0.5,
    n_steps: int = 1000,
    learning_rate: float = 0.01,
) -> Tuple[Dict, float]:
    """Fit rating prediction model.

    Args:
        data: Training data
        decomp: Parameter decomposition
        max_order: Maximum interaction order
        prior_scales: Prior scales
        noise_std: Noise standard deviation
        n_steps: Number of optimization steps
        learning_rate: Learning rate

    Returns:
        Tuple of (params, final_loss)
    """
    import optax

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

        mse = 0.5 * jnp.mean((y - eta) ** 2) / (noise_std ** 2)

        log_prior = 0.0
        for name, param in params.items():
            scale = prior_scales.get(name, 1.0)
            log_prior += 0.5 * jnp.sum(param ** 2) / (scale ** 2)

        return mse + log_prior / len(y)

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


def evaluate_rating_model(
    data: Dict,
    decomp: Decomposed,
    params: Dict,
) -> Dict[str, float]:
    """Evaluate rating prediction model.

    Args:
        data: Data dictionary
        decomp: Parameter decomposition
        params: Model parameters

    Returns:
        Dictionary of metrics
    """
    full_params = {}
    for name in decomp._tensor_parts.keys():
        if name in params:
            full_params[name] = params[name]
        else:
            full_params[name] = jnp.zeros(decomp._tensor_part_shapes[name])

    dim_names = [d.name for d in decomp._interactions._dimensions]
    interaction_indices = jnp.stack(
        [jnp.array(data[name]) for name in dim_names],
        axis=-1
    )

    X = jnp.array(data["X"])
    y = jnp.array(data["y"])

    beta = decomp.lookup(interaction_indices, tensors=full_params)
    eta = jnp.sum(X * beta, axis=-1)

    mse = float(jnp.mean((y - eta) ** 2))
    rmse = float(jnp.sqrt(mse))
    mae = float(jnp.mean(jnp.abs(y - eta)))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }


def run_movielens_experiment(exp_config: ExperimentConfig) -> List[Dict]:
    """Run MovieLens experiment.

    Args:
        exp_config: Experiment configuration

    Returns:
        List of result dictionaries
    """
    df = download_movielens(exp_config.data_dir)

    results = []
    methods = ["none", "fixed", "decay", "gen_preserving"]

    for rep in tqdm(range(exp_config.n_replications), desc="Replications"):
        rng_key = jax.random.PRNGKey(rep)

        data, dimensions = prepare_movielens_data(df, exp_config, rng_key)
        train_data, test_data = split_data(data, exp_config.train_frac, rep)

        interactions = Interactions(dimensions=dimensions)
        decomp = Decomposed(
            interactions=interactions,
            param_shape=[1],
            name="bias",
        )

        for method in methods:
            if method == "none":
                prior_scales = {name: 1e6 for name in decomp._tensor_parts.keys()}
            elif method == "fixed":
                prior_scales = {name: 1.0 for name in decomp._tensor_parts.keys()}
            elif method == "decay":
                prior_scales = {}
                for name in decomp._tensor_parts.keys():
                    order = decomp.component_order(name)
                    prior_scales[name] = 2.0 * (0.8 ** order)
            else:
                prior_scales = decomp.generalization_preserving_scales(
                    noise_scale=0.5,
                    total_n=len(train_data["y"]),
                    c=0.5,
                    per_component=True,
                )

            for order in range(exp_config.max_order + 1):
                params, train_loss = fit_rating_model(
                    train_data, decomp, order, prior_scales
                )

                train_metrics = evaluate_rating_model(train_data, decomp, params)
                test_metrics = evaluate_rating_model(test_data, decomp, params)

                n_params = sum(
                    int(np.prod(params[name].shape))
                    for name in params.keys()
                )

                result = {
                    "dataset": "movielens",
                    "method": method,
                    "order": order,
                    "replication": rep,
                    "n_params": n_params,
                    "train_rmse": train_metrics["rmse"],
                    "train_mae": train_metrics["mae"],
                    "test_rmse": test_metrics["rmse"],
                    "test_mae": test_metrics["mae"],
                    "generalization_gap": train_metrics["rmse"] - test_metrics["rmse"],
                }
                results.append(result)

    return results


def run_full_experiment(exp_config: ExperimentConfig):
    """Run full hierarchical prediction experiment.

    Args:
        exp_config: Experiment configuration
    """
    os.makedirs(exp_config.output_dir, exist_ok=True)

    results = run_movielens_experiment(exp_config)

    results_path = Path(exp_config.output_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")

    plot_results(results, exp_config)


def plot_results(results: List[Dict], exp_config: ExperimentConfig):
    """Generate plots for hierarchical prediction results.

    Args:
        results: List of result dictionaries
        exp_config: Experiment configuration
    """
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    for method in df["method"].unique():
        method_data = df[df["method"] == method]
        means = method_data.groupby("order")["test_rmse"].mean()
        stds = method_data.groupby("order")["test_rmse"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   label=method, marker='o', capsize=3)

    ax.set_xlabel("Truncation Order K")
    ax.set_ylabel("Test RMSE")
    ax.set_title("MovieLens: Test RMSE vs Order")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for method in df["method"].unique():
        method_data = df[df["method"] == method]
        means = method_data.groupby("order")["test_mae"].mean()
        stds = method_data.groupby("order")["test_mae"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   label=method, marker='o', capsize=3)

    ax.set_xlabel("Truncation Order K")
    ax.set_ylabel("Test MAE")
    ax.set_title("MovieLens: Test MAE vs Order")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for method in df["method"].unique():
        method_data = df[df["method"] == method]
        means = method_data.groupby("order")["generalization_gap"].mean()
        stds = method_data.groupby("order")["generalization_gap"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   label=method, marker='o', capsize=3)

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel("Truncation Order K")
    ax.set_ylabel("Train RMSE - Test RMSE")
    ax.set_title("Generalization Gap")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for method in df["method"].unique():
        method_data = df[df["method"] == method]
        grouped = method_data.groupby("order").agg({
            "n_params": "mean",
            "test_rmse": "mean"
        })
        ax.plot(grouped["n_params"], grouped["test_rmse"],
               label=method, marker='o')

    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Test RMSE vs Model Complexity")
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(exp_config.output_dir) / "hierarchical_results.png", dpi=150)
    plt.close()

    summary = df.groupby(["method", "order"]).agg({
        "test_rmse": ["mean", "std"],
        "test_mae": ["mean", "std"],
        "n_params": "mean",
    }).round(4)

    print("\nSummary Results:")
    print(summary)

    summary.to_csv(Path(exp_config.output_dir) / "summary.csv")
    print(f"Summary saved to {exp_config.output_dir}/summary.csv")


def main():
    parser = argparse.ArgumentParser(description="Hierarchical prediction experiments")
    parser.add_argument("--n_replications", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="results/hierarchical_prediction")
    parser.add_argument("--quick", action="store_true", help="Run quick version")
    args = parser.parse_args()

    if args.quick:
        config = ExperimentConfig(
            n_replications=3,
            max_order=1,
            n_users_subsample=200,
            n_items_subsample=100,
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
