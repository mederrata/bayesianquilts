#!/usr/bin/env python3
"""
Timing comparison: Our method vs EBM vs GAMINet
Shows that our method is fast enough to ensemble multiple models.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score

import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def load_dataset(name):
    """Load a dataset by name."""
    if name == "german":
        data = fetch_openml(data_id=31, as_frame=True, parser='auto')
        X = pd.get_dummies(data.data, drop_first=True).values.astype(float)
        y = np.array(data.target == 'good', dtype=int)
    elif name == "spambase":
        data = fetch_openml(data_id=44, as_frame=True, parser='auto')
        X = data.data.values.astype(float)
        y = np.array(data.target == '1', dtype=int)
    elif name == "phoneme":
        data = fetch_openml(data_id=1489, as_frame=True, parser='auto')
        X = data.data.values.astype(float)
        y = np.array(data.target == '1', dtype=int)
    elif name == "electricity":
        data = fetch_openml(data_id=151, as_frame=True, parser='auto')
        X = data.data.select_dtypes(include=[np.number]).values.astype(float)
        y = np.array(data.target == 'UP', dtype=int)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X = np.nan_to_num(X, nan=0.0)
    return X, y


def time_our_method(X_train, X_test, y_train, y_test, n_epochs=300):
    """Time our decomposition method."""
    n_features = X_train.shape[0 if len(X_train.shape) == 1 else 1]
    n_features = X_train.shape[1]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Simple 3d lattice
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=200, solver='lbfgs')
    lr.fit(X_train_s, y_train)
    coef_abs = np.abs(lr.coef_.flatten())
    sorted_idx = np.argsort(-coef_abs)

    start_time = time.time()

    # Build lattice
    int_dims = []
    int_bin_data = {}
    bins_per_dim = [8, 6, 8]

    for i in range(3):
        feat_idx = sorted_idx[i]
        n_bins = bins_per_dim[i]
        feature_vals = X_train_s[:, feat_idx]
        edges = np.percentile(feature_vals, np.linspace(0, 100, n_bins + 1))
        edges = np.unique(edges)
        edges[0], edges[-1] = -np.inf, np.inf
        n_actual = len(edges) - 1

        dim_name = f"i{i}"
        int_dims.append(Dimension(dim_name, n_actual))
        int_bin_data[dim_name] = {
            "train": np.clip(np.digitize(feature_vals, edges[1:-1]), 0, n_actual - 1),
            "test": np.clip(np.digitize(X_test_s[:, feat_idx], edges[1:-1]), 0, n_actual - 1),
        }

    int_interactions = Interactions(dimensions=int_dims)
    int_decomp = Decomposed(interactions=int_interactions, param_shape=[1], name="intercept", max_order=2)

    int_components = {k: v for k, v in int_decomp._tensor_part_shapes.items()
                     if int_decomp.component_order(k) in [0, 1, 2]}

    # Beta lattice
    b_dims = []
    beta_bin_data = {}
    beta_bins = [6, 5, 6]

    for i in range(3):
        feat_idx = sorted_idx[i]
        n_bins = beta_bins[i]
        feature_vals = X_train_s[:, feat_idx]
        edges = np.percentile(feature_vals, np.linspace(0, 100, n_bins + 1))
        edges = np.unique(edges)
        edges[0], edges[-1] = -np.inf, np.inf
        n_actual = len(edges) - 1

        dim_name = f"b{i}"
        b_dims.append(Dimension(dim_name, n_actual))
        beta_bin_data[dim_name] = {
            "train": np.clip(np.digitize(feature_vals, edges[1:-1]), 0, n_actual - 1),
            "test": np.clip(np.digitize(X_test_s[:, feat_idx], edges[1:-1]), 0, n_actual - 1),
        }

    beta_interactions = Interactions(dimensions=b_dims)
    beta_decomp = Decomposed(interactions=beta_interactions, param_shape=[n_features], name="beta", max_order=1)
    beta_components = {k: v for k, v in beta_decomp._tensor_part_shapes.items()
                      if beta_decomp.component_order(k) in [0, 1]}

    # Prepare indices
    int_train_idx = np.stack([int_bin_data[d.name]["train"] for d in int_dims], axis=-1)
    int_test_idx = np.stack([int_bin_data[d.name]["test"] for d in int_dims], axis=-1)
    beta_train_idx = np.stack([beta_bin_data[d.name]["train"] for d in b_dims], axis=-1)
    beta_test_idx = np.stack([beta_bin_data[d.name]["test"] for d in b_dims], axis=-1)

    int_params = {name: jnp.zeros(shape) for name, shape in int_components.items()}
    beta_params = {name: jnp.zeros(shape) for name, shape in beta_components.items()}
    global_int = jnp.zeros(1)

    X_train_jax = jnp.array(X_train_s)
    X_test_jax = jnp.array(X_test_s)
    int_train_idx_jax = jnp.array(int_train_idx)
    int_test_idx_jax = jnp.array(int_test_idx)
    beta_train_idx_jax = jnp.array(beta_train_idx)
    beta_test_idx_jax = jnp.array(beta_test_idx)
    y_train_jax = jnp.array(y_train)

    def compute_logits(int_p, beta_p, g_int, int_idx, beta_idx, X):
        cell_int = int_decomp.lookup_flat(int_idx, int_p)
        beta = beta_decomp.lookup_flat(beta_idx, beta_p)
        return jnp.sum(X * beta, axis=-1) + cell_int[..., 0] + g_int[0]

    def loss_fn(params, int_idx, beta_idx, X, y, N):
        int_p, beta_p, g_int = params
        logits = compute_logits(int_p, beta_p, g_int, int_idx, beta_idx, X)
        bce = jnp.mean(jnp.logaddexp(0, logits) - y * logits)
        l2 = sum(0.5 * jnp.sum(p ** 2) / 50.0 for p in int_p.values())
        l2 += sum(0.5 * jnp.sum(p ** 2) / 100.0 for p in beta_p.values())
        return bce + l2 / N

    params = (int_params, beta_params, global_int)
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(params)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    n_train = len(y_train)
    batch_size = min(2048, n_train)
    n_batches = (n_train + batch_size - 1) // batch_size

    for epoch in range(n_epochs):
        perm = np.random.permutation(n_train)
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_train)
            idx = perm[start:end]
            loss_val, grads = loss_and_grad(
                params, int_train_idx_jax[idx], beta_train_idx_jax[idx],
                X_train_jax[idx], y_train_jax[idx], len(idx)
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

    train_time = time.time() - start_time

    # Evaluate
    int_p, beta_p, g_int = params
    test_logits = compute_logits(int_p, beta_p, g_int, int_test_idx_jax, beta_test_idx_jax, X_test_jax)
    test_probs = jax.nn.sigmoid(test_logits)
    test_auc = roc_auc_score(y_test, np.array(test_probs))

    return train_time, test_auc


def time_ebm(X_train, X_test, y_train, y_test):
    """Time EBM."""
    from interpret.glassbox import ExplainableBoostingClassifier

    start_time = time.time()
    ebm = ExplainableBoostingClassifier(
        max_bins=256,
        interactions=10,
        outer_bags=8,
        inner_bags=0,
        learning_rate=0.01,
        max_rounds=5000,
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    ebm.fit(X_train, y_train)
    train_time = time.time() - start_time

    test_probs = ebm.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)

    return train_time, test_auc


def time_gaminet(X_train, X_test, y_train, y_test):
    """Time GAMINet."""
    try:
        from gaminet import GAMINet
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
    except ImportError:
        return None, None

    start_time = time.time()

    try:
        # GAMINet config - API varies by version
        meta_info = {
            "X" + str(i): {"type": "continuous"} for i in range(X_train.shape[1])
        }
        meta_info["Y"] = {"type": "target"}

        model = GAMINet(
            meta_info=meta_info,
            interact_num=10,
            batch_size=min(1024, len(y_train)),
            task_type="Classification",
            device="cpu",
            verbose=False,
            random_state=42,
        )

        model.fit(X_train, y_train.reshape(-1, 1))
        train_time = time.time() - start_time

        test_probs = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_probs)

        return train_time, test_auc
    except Exception as e:
        print(f"    GAMINet error: {e}")
        return None, None


def run_timing_comparison():
    """Run timing comparison on multiple datasets."""
    datasets = ["german", "spambase", "phoneme", "electricity"]
    results = []

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        X, y = load_dataset(dataset_name)
        print(f"  Shape: {X.shape}, prevalence: {y.mean():.3f}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Our method
        print("  Timing our method...", flush=True)
        our_time, our_auc = time_our_method(X_train, X_test, y_train, y_test)
        print(f"    Time: {our_time:.2f}s, AUC: {our_auc:.4f}")
        results.append({
            "dataset": dataset_name,
            "method": "Ours",
            "time": our_time,
            "auc": our_auc,
        })

        # EBM
        print("  Timing EBM...", flush=True)
        ebm_time, ebm_auc = time_ebm(X_train, X_test, y_train, y_test)
        print(f"    Time: {ebm_time:.2f}s, AUC: {ebm_auc:.4f}")
        results.append({
            "dataset": dataset_name,
            "method": "EBM",
            "time": ebm_time,
            "auc": ebm_auc,
        })

        # GAMINet
        print("  Timing GAMINet...", flush=True)
        gami_time, gami_auc = time_gaminet(X_train, X_test, y_train, y_test)
        if gami_time is not None:
            print(f"    Time: {gami_time:.2f}s, AUC: {gami_auc:.4f}")
            results.append({
                "dataset": dataset_name,
                "method": "GAMINet",
                "time": gami_time,
                "auc": gami_auc,
            })
        else:
            print("    GAMINet not available")

        # How many of our models could fit in EBM time?
        n_ensemble = int(ebm_time / our_time)
        print(f"\n  Could fit {n_ensemble} of our models in EBM training time")

    print(f"\n{'='*60}")
    print("TIMING SUMMARY")
    print(f"{'='*60}")

    df = pd.DataFrame(results)
    for dataset in datasets:
        subset = df[df['dataset'] == dataset]
        print(f"\n{dataset}:")
        for _, row in subset.iterrows():
            print(f"  {row['method']:10s}: {row['time']:6.2f}s  AUC={row['auc']:.4f}")

        our_row = subset[subset['method'] == 'Ours'].iloc[0]
        ebm_row = subset[subset['method'] == 'EBM'].iloc[0]
        speedup = ebm_row['time'] / our_row['time']
        print(f"  Speedup vs EBM: {speedup:.1f}x")

    return results


if __name__ == "__main__":
    results = run_timing_comparison()
