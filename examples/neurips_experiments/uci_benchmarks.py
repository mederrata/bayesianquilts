#!/usr/bin/env python3
"""
Experiment 3: UCI Dataset Benchmarks

Compares generalization-preserving regularization against baselines on
standard UCI classification datasets: Adult, German Credit, Bank Marketing.

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
from urllib.request import urlretrieve

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


UCI_DATASETS = {
    "adult": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "columns": [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
        ],
        "target": "income",
        "categorical": ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "sex", "native_country"],
        "numeric": ["age", "fnlwgt", "education_num", "capital_gain",
                   "capital_loss", "hours_per_week"],
    },
    "german": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
        "columns": [f"A{i}" for i in range(1, 21)] + ["target"],
        "target": "target",
        "categorical": ["A1", "A3", "A4", "A6", "A7", "A9", "A10", "A12",
                       "A14", "A15", "A17", "A19", "A20"],
        "numeric": ["A2", "A5", "A8", "A11", "A13", "A16", "A18"],
    },
    "bank": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip",
        "columns": ["age", "job", "marital", "education", "default", "balance",
                   "housing", "loan", "contact", "day", "month", "duration",
                   "campaign", "pdays", "previous", "poutcome", "y"],
        "target": "y",
        "categorical": ["job", "marital", "education", "default", "housing",
                       "loan", "contact", "month", "poutcome"],
        "numeric": ["age", "balance", "day", "duration", "campaign",
                   "pdays", "previous"],
    },
}


@dataclass
class ExperimentConfig:
    """Configuration for UCI benchmark experiments."""
    n_folds: int = 5
    n_replications: int = 10
    max_order: int = 2
    datasets: List[str] = None
    output_dir: str = "results/uci_benchmarks"
    data_dir: str = "data/uci"

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["adult", "german"]


def download_dataset(dataset_name: str, data_dir: str) -> pd.DataFrame:
    """Download UCI dataset if not present.

    Args:
        dataset_name: Name of dataset
        data_dir: Directory to store data

    Returns:
        DataFrame with dataset
    """
    os.makedirs(data_dir, exist_ok=True)
    config = UCI_DATASETS[dataset_name]
    filepath = Path(data_dir) / f"{dataset_name}.csv"

    if filepath.exists():
        df = pd.read_csv(filepath)
        return df

    print(f"Downloading {dataset_name} dataset...")

    if dataset_name == "adult":
        local_path = Path(data_dir) / "adult.data"
        if not local_path.exists():
            urlretrieve(config["url"], local_path)
        df = pd.read_csv(local_path, names=config["columns"],
                        skipinitialspace=True, na_values="?")
        df = df.dropna()

    elif dataset_name == "german":
        local_path = Path(data_dir) / "german.data"
        if not local_path.exists():
            urlretrieve(config["url"], local_path)
        df = pd.read_csv(local_path, names=config["columns"],
                        sep=" ", header=None)

    elif dataset_name == "bank":
        import zipfile
        local_path = Path(data_dir) / "bank.zip"
        if not local_path.exists():
            urlretrieve(config["url"], local_path)
        with zipfile.ZipFile(local_path, 'r') as z:
            with z.open("bank.csv") as f:
                df = pd.read_csv(f, sep=";")

    df.to_csv(filepath, index=False)
    return df


def prepare_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    hierarchical_factors: List[str],
) -> Tuple[Dict, np.ndarray]:
    """Prepare dataset for hierarchical modeling.

    Args:
        df: Raw DataFrame
        dataset_name: Name of dataset
        hierarchical_factors: List of columns to use as hierarchical factors

    Returns:
        Tuple of (data_dict, target)
    """
    config = UCI_DATASETS[dataset_name]

    for col in hierarchical_factors:
        if col not in df.columns:
            raise ValueError(f"Column {col} not in dataset")

    numeric_cols = [c for c in config["numeric"] if c in df.columns]
    X_numeric = df[numeric_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(X_numeric)

    factor_encoders = {}
    factor_indices = {}
    dimensions = []

    for col in hierarchical_factors:
        le = LabelEncoder()
        indices = le.fit_transform(df[col].astype(str))
        factor_encoders[col] = le
        factor_indices[col] = indices
        dimensions.append(Dimension(col, len(le.classes_)))

    target_col = config["target"]
    le_target = LabelEncoder()
    y = le_target.fit_transform(df[target_col].astype(str))

    data = {
        "X": X_numeric,
        "y": y,
        **factor_indices,
    }

    return data, dimensions


def create_train_test_split(
    data: Dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[Dict, Dict]:
    """Split data into train and test sets.

    Args:
        data: Full data dictionary
        train_idx: Training indices
        test_idx: Test indices

    Returns:
        Tuple of (train_data, test_data)
    """
    train_data = {
        key: val[train_idx] if isinstance(val, np.ndarray) else val
        for key, val in data.items()
    }
    test_data = {
        key: val[test_idx] if isinstance(val, np.ndarray) else val
        for key, val in data.items()
    }
    return train_data, test_data


def fit_logistic_model(
    data: Dict,
    decomp: Decomposed,
    max_order: int,
    prior_scales: Dict[str, float],
    n_steps: int = 1000,
    learning_rate: float = 0.01,
) -> Dict:
    """Fit logistic regression with hierarchical coefficients.

    Args:
        data: Training data
        decomp: Parameter decomposition
        max_order: Maximum interaction order
        prior_scales: Prior scales for each component
        n_steps: Number of optimization steps
        learning_rate: Learning rate

    Returns:
        Fitted parameters
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
        logits = jnp.sum(X * beta, axis=-1)

        bce = jnp.mean(
            jnp.logaddexp(0, logits) - y * logits
        )

        log_prior = 0.0
        for name, param in params.items():
            scale = prior_scales.get(name, 1.0)
            log_prior += 0.5 * jnp.sum(param ** 2) / (scale ** 2)

        return bce + log_prior / len(y)

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

    return params


def evaluate_model(
    data: Dict,
    decomp: Decomposed,
    params: Dict,
) -> Dict[str, float]:
    """Evaluate model on data.

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
    logits = jnp.sum(X * beta, axis=-1)
    probs = jax.nn.sigmoid(logits)

    predictions = (probs > 0.5).astype(jnp.float32)
    accuracy = float(jnp.mean(predictions == y))

    bce = float(jnp.mean(jnp.logaddexp(0, logits) - y * logits))

    auc = compute_auc(np.array(y), np.array(probs))

    return {
        "accuracy": accuracy,
        "bce": bce,
        "auc": auc,
    }


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC score.

    Args:
        y_true: True labels
        y_score: Predicted probabilities

    Returns:
        AUC score
    """
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.5


def run_single_dataset(
    dataset_name: str,
    exp_config: ExperimentConfig,
) -> List[Dict]:
    """Run experiment on a single dataset.

    Args:
        dataset_name: Name of dataset
        exp_config: Experiment configuration

    Returns:
        List of result dictionaries
    """
    df = download_dataset(dataset_name, exp_config.data_dir)

    config = UCI_DATASETS[dataset_name]
    hierarchical_factors = config["categorical"][:3]

    data, dimensions = prepare_dataset(df, dataset_name, hierarchical_factors)

    interactions = Interactions(dimensions=dimensions)
    n_features = data["X"].shape[1]
    decomp = Decomposed(
        interactions=interactions,
        param_shape=[n_features],
        name="beta",
    )

    results = []

    methods = ["none", "fixed", "decay", "gen_preserving"]

    for rep in tqdm(range(exp_config.n_replications), desc=f"{dataset_name} replications"):
        skf = StratifiedKFold(n_splits=exp_config.n_folds, shuffle=True,
                             random_state=rep)

        for fold, (train_idx, test_idx) in enumerate(skf.split(data["X"], data["y"])):
            train_data, test_data = create_train_test_split(data, train_idx, test_idx)

            for method in methods:
                if method == "none":
                    prior_scales = {name: 1e6 for name in decomp._tensor_parts.keys()}
                elif method == "fixed":
                    prior_scales = {name: 1.0 for name in decomp._tensor_parts.keys()}
                elif method == "decay":
                    prior_scales = {}
                    for name in decomp._tensor_parts.keys():
                        order = decomp.component_order(name)
                        prior_scales[name] = 5.0 * (0.9 ** order)
                else:
                    prior_scales = decomp.generalization_preserving_scales(
                        noise_scale=1.0,
                        total_n=len(train_idx),
                        c=0.5,
                        per_component=True,
                    )

                for order in range(exp_config.max_order + 1):
                    params = fit_logistic_model(
                        train_data, decomp, order, prior_scales
                    )

                    train_metrics = evaluate_model(train_data, decomp, params)
                    test_metrics = evaluate_model(test_data, decomp, params)

                    result = {
                        "dataset": dataset_name,
                        "method": method,
                        "order": order,
                        "fold": fold,
                        "replication": rep,
                        "train_accuracy": train_metrics["accuracy"],
                        "train_bce": train_metrics["bce"],
                        "train_auc": train_metrics["auc"],
                        "test_accuracy": test_metrics["accuracy"],
                        "test_bce": test_metrics["bce"],
                        "test_auc": test_metrics["auc"],
                    }
                    results.append(result)

    return results


def run_full_experiment(exp_config: ExperimentConfig):
    """Run UCI benchmark experiment.

    Args:
        exp_config: Experiment configuration
    """
    os.makedirs(exp_config.output_dir, exist_ok=True)

    all_results = []

    for dataset_name in tqdm(exp_config.datasets, desc="Datasets"):
        results = run_single_dataset(dataset_name, exp_config)
        all_results.extend(results)

    results_path = Path(exp_config.output_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {results_path}")

    plot_results(all_results, exp_config)


def plot_results(results: List[Dict], exp_config: ExperimentConfig):
    """Generate plots for UCI benchmark results.

    Args:
        results: List of result dictionaries
        exp_config: Experiment configuration
    """
    df = pd.DataFrame(results)

    n_datasets = len(exp_config.datasets)
    fig, axes = plt.subplots(n_datasets, 2, figsize=(12, 5 * n_datasets))

    if n_datasets == 1:
        axes = axes.reshape(1, -1)

    for idx, dataset in enumerate(exp_config.datasets):
        subset = df[df["dataset"] == dataset]

        ax = axes[idx, 0]
        for method in subset["method"].unique():
            method_data = subset[subset["method"] == method]
            means = method_data.groupby("order")["test_accuracy"].mean()
            stds = method_data.groupby("order")["test_accuracy"].std()
            ax.errorbar(means.index, means.values, yerr=stds.values,
                       label=method, marker='o', capsize=3)

        ax.set_xlabel("Truncation Order K")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"{dataset.upper()}: Accuracy vs Order")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[idx, 1]
        for method in subset["method"].unique():
            method_data = subset[subset["method"] == method]
            means = method_data.groupby("order")["test_auc"].mean()
            stds = method_data.groupby("order")["test_auc"].std()
            ax.errorbar(means.index, means.values, yerr=stds.values,
                       label=method, marker='o', capsize=3)

        ax.set_xlabel("Truncation Order K")
        ax.set_ylabel("Test AUC")
        ax.set_title(f"{dataset.upper()}: AUC vs Order")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(exp_config.output_dir) / "uci_results.png", dpi=150)
    plt.close()

    summary = df.groupby(["dataset", "method"]).agg({
        "test_accuracy": ["mean", "std"],
        "test_auc": ["mean", "std"],
    }).round(4)

    print("\nSummary Results:")
    print(summary)

    summary.to_csv(Path(exp_config.output_dir) / "summary.csv")
    print(f"Summary saved to {exp_config.output_dir}/summary.csv")


def main():
    parser = argparse.ArgumentParser(description="UCI benchmark experiments")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_replications", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="results/uci_benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick version")
    args = parser.parse_args()

    if args.quick:
        config = ExperimentConfig(
            n_folds=3,
            n_replications=2,
            max_order=1,
            datasets=["german"],
            output_dir=args.output_dir,
        )
    else:
        config = ExperimentConfig(
            n_folds=args.n_folds,
            n_replications=args.n_replications,
            output_dir=args.output_dir,
        )

    run_full_experiment(config)


if __name__ == "__main__":
    main()
