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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    HAS_EBM = True
except ImportError:
    HAS_EBM = False

try:
    from sklearn.datasets import fetch_openml
    HAS_OPENML = True
except ImportError:
    HAS_OPENML = False

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


UCI_DATASETS = {
    "german": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
        "columns": [f"A{i}" for i in range(1, 21)] + ["target"],
        "target": "target",
        "categorical": ["A1", "A3", "A4", "A6", "A7"],
        "numeric": ["A2", "A5", "A8", "A11", "A13", "A16", "A18"],
        "N": 1000,
        "p": 7,
    },
    "adult": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "columns": [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
        ],
        "target": "income",
        "categorical": ["workclass", "education", "marital_status", "occupation", "sex"],
        "numeric": ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"],
        "N": 48842,
        "p": 6,
    },
    "bank": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip",
        "columns": ["age", "job", "marital", "education", "default", "balance",
                   "housing", "loan", "contact", "day", "month", "duration",
                   "campaign", "pdays", "previous", "poutcome", "y"],
        "target": "y",
        "categorical": ["job", "marital", "education", "housing", "loan", "contact", "poutcome"],
        "numeric": ["age", "balance", "duration", "campaign", "pdays", "previous"],
        "N": 45211,
        "p": 7,
    },
    "taiwan": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
        "loader": "openml",
        "openml_id": 42477,
        "target": "default_payment_next_month",
        "categorical": ["SEX", "EDUCATION", "MARRIAGE"],
        "numeric": ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                   "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                   "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"],
        "N": 30000,
        "p": 23,
    },
    "heart": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "loader": "openml",
        "openml_id": 43,
        "target": "target",
        "categorical": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
        "numeric": ["age", "trestbps", "chol", "thalach", "oldpeak"],
        "N": 303,
        "p": 13,
    },
    "bioresponse": {
        "loader": "openml",
        "openml_id": 4134,
        "target": "target",
        "categorical": [],
        "numeric": "all",
        "use_pca": True,
        "pca_components": 50,
        "N": 3751,
        "p": 1776,
    },
    "spambase": {
        "loader": "openml",
        "openml_id": 44,
        "target": "class",
        "categorical": [],
        "numeric": "all",
        "N": 4601,
        "p": 57,
    },
    "mushroom": {
        "loader": "openml",
        "openml_id": 24,
        "target": "class",
        "categorical": "all",
        "numeric": [],
        "N": 8124,
        "p": 22,
    },
    "phoneme": {
        "loader": "openml",
        "openml_id": 1489,
        "target": "Class",
        "categorical": [],
        "numeric": "all",
        "N": 5404,
        "p": 5,
    },
    "electricity": {
        "loader": "openml",
        "openml_id": 151,
        "target": "class",
        "categorical": ["day"],
        "numeric": ["date", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"],
        "N": 45312,
        "p": 8,
    },
}


@dataclass
class ExperimentConfig:
    """Configuration for UCI benchmark experiments."""
    n_folds: int = 5
    n_replications: int = 5
    max_order: int = 2
    datasets: List[str] = None
    output_dir: str = "results/uci_benchmarks"
    data_dir: str = "data/uci"

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = [
                "german", "adult", "bank", "taiwan", "heart",
                "bioresponse", "spambase", "mushroom", "phoneme", "electricity"
            ]


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

    loader = config.get("loader", "url")

    if loader == "openml":
        if not HAS_OPENML:
            raise ImportError("sklearn.datasets.fetch_openml required for OpenML datasets")
        openml_id = config["openml_id"]
        data = fetch_openml(data_id=openml_id, as_frame=True, parser='auto')
        df = data.frame
        if config["target"] not in df.columns and hasattr(data, 'target_names'):
            df[config["target"]] = data.target

    elif dataset_name == "adult":
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

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    df = df.dropna()
    df.to_csv(filepath, index=False)
    return df


def prepare_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    hierarchical_factors: List[str] = None,
    n_pca_bins: int = 4,
) -> Tuple[Dict, List]:
    """Prepare dataset for hierarchical modeling.

    Args:
        df: Raw DataFrame
        dataset_name: Name of dataset
        hierarchical_factors: List of columns to use as hierarchical factors
        n_pca_bins: Number of bins for PCA-discretized latent dimensions

    Returns:
        Tuple of (data_dict, dimensions)
    """
    config = UCI_DATASETS[dataset_name]

    # Handle numeric columns
    numeric_cols = config.get("numeric", [])
    if numeric_cols == "all":
        numeric_cols = [c for c in df.columns if c != config["target"]
                       and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    else:
        numeric_cols = [c for c in numeric_cols if c in df.columns]

    # Handle categorical columns
    categorical_cols = config.get("categorical", [])
    if categorical_cols == "all":
        categorical_cols = [c for c in df.columns if c != config["target"]
                          and c not in numeric_cols]
    else:
        categorical_cols = [c for c in categorical_cols if c in df.columns]

    if hierarchical_factors is None:
        hierarchical_factors = categorical_cols

    # Process numeric features
    if numeric_cols:
        X_numeric = df[numeric_cols].values.astype(np.float32)
        X_numeric = np.nan_to_num(X_numeric, nan=0.0)
        scaler = StandardScaler()
        X_numeric = scaler.fit_transform(X_numeric)

        # Apply PCA for high-dimensional datasets
        if config.get("use_pca", False):
            n_components = min(config.get("pca_components", 50), X_numeric.shape[1], X_numeric.shape[0] - 1)
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_numeric)
            # Use first few PCA components as discretized lattice dimensions
            n_lattice_dims = min(3, n_components)
            pca_factors = {}
            pca_dimensions = []
            for i in range(n_lattice_dims):
                col_name = f"pca_{i}"
                bins = np.percentile(X_pca[:, i], np.linspace(0, 100, n_pca_bins + 1))
                bins[0] = -np.inf
                bins[-1] = np.inf
                pca_factors[col_name] = np.digitize(X_pca[:, i], bins[1:-1])
                pca_dimensions.append(Dimension(col_name, n_pca_bins))
            X_numeric = X_pca
        else:
            pca_factors = {}
            pca_dimensions = []
    else:
        X_numeric = np.zeros((len(df), 1), dtype=np.float32)
        pca_factors = {}
        pca_dimensions = []

    # Process categorical factors
    factor_indices = {}
    dimensions = []

    for col in hierarchical_factors:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        indices = le.fit_transform(df[col].astype(str))
        n_levels = len(le.classes_)
        # Limit number of levels for memory efficiency (max 8 to avoid explosion)
        if n_levels > 8:
            from collections import Counter
            counts = Counter(indices)
            top_cats = [k for k, v in counts.most_common(7)]
            indices = np.array([i if i in top_cats else 7 for i in indices])
            n_levels = 8
        factor_indices[col] = indices
        dimensions.append(Dimension(col, n_levels))

    # Add PCA-derived dimensions
    factor_indices.update(pca_factors)
    dimensions.extend(pca_dimensions)

    # Process target
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
    n_steps: int = 2000,
    learning_rate: float = 0.01,
    sparse: bool = True,
    l1_weight: float = 0.01,
) -> Dict:
    """Fit logistic regression with hierarchical coefficients.

    Args:
        data: Training data
        decomp: Parameter decomposition
        max_order: Maximum interaction order
        prior_scales: Prior scales for each component
        n_steps: Number of optimization steps
        learning_rate: Learning rate
        sparse: Whether to use L1 sparsity penalty
        l1_weight: Weight for L1 penalty

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

        # L2 regularization (ridge)
        l2_reg = 0.0
        for name, param in params.items():
            scale = prior_scales.get(name, 1.0)
            l2_reg += 0.5 * jnp.sum(param ** 2) / (scale ** 2)

        # L1 regularization for sparsity (elastic net style)
        l1_reg = 0.0
        if sparse:
            for name, param in params.items():
                order = decomp.component_order(name)
                # Stronger sparsity for higher-order interactions
                l1_reg += l1_weight * (1 + order) * jnp.sum(jnp.abs(param))

        return bce + l2_reg / len(y) + l1_reg / len(y)

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


def fit_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> List[Dict]:
    """Fit baseline ML models and return metrics.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels

    Returns:
        List of result dictionaries for each baseline
    """
    results = []

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    results.append({
        "method": "logistic_regression",
        "order": -1,
        "test_auc": compute_auc(y_test, lr_probs),
        "test_accuracy": float(np.mean(lr.predict(X_test) == y_test)),
    })

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    results.append({
        "method": "random_forest",
        "order": -1,
        "test_auc": compute_auc(y_test, rf_probs),
        "test_accuracy": float(np.mean(rf.predict(X_test) == y_test)),
    })

    # Gradient Boosting (sklearn)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    gb_probs = gb.predict_proba(X_test)[:, 1]
    results.append({
        "method": "gradient_boosting",
        "order": -1,
        "test_auc": compute_auc(y_test, gb_probs),
        "test_accuracy": float(np.mean(gb.predict(X_test) == y_test)),
    })

    # XGBoost
    if HAS_XGBOOST:
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        results.append({
            "method": "xgboost",
            "order": -1,
            "test_auc": compute_auc(y_test, xgb_probs),
            "test_accuracy": float(np.mean(xgb_model.predict(X_test) == y_test)),
        })

    # LightGBM
    if HAS_LIGHTGBM:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
        results.append({
            "method": "lightgbm",
            "order": -1,
            "test_auc": compute_auc(y_test, lgb_probs),
            "test_accuracy": float(np.mean(lgb_model.predict(X_test) == y_test)),
        })

    # MLP (Neural Network)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
        early_stopping=True, validation_fraction=0.1
    )
    mlp.fit(X_train, y_train)
    mlp_probs = mlp.predict_proba(X_test)[:, 1]
    results.append({
        "method": "mlp",
        "order": -1,
        "test_auc": compute_auc(y_test, mlp_probs),
        "test_accuracy": float(np.mean(mlp.predict(X_test) == y_test)),
    })

    # EBM (Explainable Boosting Machine)
    if HAS_EBM:
        ebm = ExplainableBoostingClassifier(random_state=42, n_jobs=1)
        ebm.fit(X_train, y_train)
        ebm_probs = ebm.predict_proba(X_test)[:, 1]
        results.append({
            "method": "ebm",
            "order": -1,
            "test_auc": compute_auc(y_test, ebm_probs),
            "test_accuracy": float(np.mean(ebm.predict(X_test) == y_test)),
        })

    return results


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
    try:
        df = download_dataset(dataset_name, exp_config.data_dir)
    except Exception as e:
        print(f"Failed to download {dataset_name}: {e}")
        return []

    config = UCI_DATASETS[dataset_name]

    # Use categorical features, limited for memory
    # Limit to 3 factors to avoid combinatorial explosion
    categorical = config.get("categorical", [])
    if categorical == "all":
        categorical = [c for c in df.columns if c != config["target"]][:3]
    hierarchical_factors = categorical[:3]

    try:
        data, dimensions = prepare_dataset(df, dataset_name, hierarchical_factors)
    except Exception as e:
        print(f"Failed to prepare {dataset_name}: {e}")
        return []

    if not dimensions:
        print(f"No dimensions for {dataset_name}, adding dummy")
        data["dummy"] = np.zeros(len(data["y"]), dtype=np.int32)
        dimensions = [Dimension("dummy", 1)]

    interactions = Interactions(dimensions=dimensions)
    n_features = data["X"].shape[1]
    decomp = Decomposed(
        interactions=interactions,
        param_shape=[n_features],
        name="beta",
    )

    # Compute lattice size and limit max_order dynamically
    total_cells = 1
    for dim in dimensions:
        total_cells *= dim.cardinality
    # If lattice is large, limit to order 1 to prevent OOM
    effective_max_order = exp_config.max_order
    if total_cells * n_features > 500000:
        effective_max_order = min(effective_max_order, 1)
        print(f"  Limiting to order 1 due to lattice size ({total_cells} cells)")

    results = []

    # Include sparse method for better performance
    methods = ["gen_preserving", "sparse"]

    for rep in tqdm(range(exp_config.n_replications), desc=f"{dataset_name} replications"):
        skf = StratifiedKFold(n_splits=exp_config.n_folds, shuffle=True,
                             random_state=rep)

        for fold, (train_idx, test_idx) in enumerate(skf.split(data["X"], data["y"])):
            train_data, test_data = create_train_test_split(data, train_idx, test_idx)

            # Fit baseline models (XGBoost, LightGBM, RF, etc.)
            baseline_results = fit_baseline_models(
                train_data["X"], train_data["y"],
                test_data["X"], test_data["y"]
            )
            for br in baseline_results:
                br["dataset"] = dataset_name
                br["fold"] = fold
                br["replication"] = rep
                results.append(br)

            for method in methods:
                use_sparse = (method == "sparse")

                if method == "sparse":
                    # Sparse: gen-preserving scales + L1 penalty
                    prior_scales = decomp.generalization_preserving_scales(
                        noise_scale=1.0,
                        total_n=len(train_idx),
                        c=0.3,  # Tighter bound
                        per_component=True,
                    )
                else:
                    prior_scales = decomp.generalization_preserving_scales(
                        noise_scale=1.0,
                        total_n=len(train_idx),
                        c=0.5,
                        per_component=True,
                    )

                for order in range(effective_max_order + 1):
                    params = fit_logistic_model(
                        train_data, decomp, order, prior_scales,
                        sparse=use_sparse, l1_weight=0.01 if use_sparse else 0.0,
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
    parser.add_argument("--n_replications", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="results/uci_benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick version")
    parser.add_argument("--dataset", type=str, default=None, help="Run single dataset")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated list of datasets")
    args = parser.parse_args()

    if args.quick:
        config = ExperimentConfig(
            n_folds=3,
            n_replications=2,
            max_order=1,
            datasets=["german"],
            output_dir=args.output_dir,
        )
    elif args.dataset:
        config = ExperimentConfig(
            n_folds=args.n_folds,
            n_replications=args.n_replications,
            datasets=[args.dataset],
            output_dir=args.output_dir,
        )
    elif args.datasets:
        config = ExperimentConfig(
            n_folds=args.n_folds,
            n_replications=args.n_replications,
            datasets=args.datasets.split(","),
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
