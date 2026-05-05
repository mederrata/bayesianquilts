#!/usr/bin/env python3
"""
HIGGS: 10-fold CV comparison of our method vs EBM.

Question: Which method benefits more from additional training data?
- 5-fold: 80% train, 20% test
- 10-fold: 90% train, 10% test

If one method improves more with 10-fold, it suggests better sample efficiency.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension

print("Loading HIGGS...", flush=True)
higgs = fetch_openml(data_id=23512, as_frame=False, parser='auto')
X_full, y_full = higgs.data, higgs.target
X_full = np.nan_to_num(X_full, nan=0.0)
y_full = (y_full.astype(float) > 0.5).astype(int)

N, n_features = X_full.shape
print(f"Dataset: {N} samples, {n_features} features")

# Get feature ranking
print("\nFitting LR for feature selection...", flush=True)
scaler_full = StandardScaler()
X_scaled = scaler_full.fit_transform(X_full)
lr = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
lr.fit(X_scaled, y_full)
coef_abs = np.abs(lr.coef_.flatten())
sorted_idx = np.argsort(-coef_abs)
print(f"Top 8 features: {sorted_idx[:8].tolist()}")


def percentile_edges(vals, n_bins):
    edges = np.percentile(vals, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def run_our_method(X_train, X_test, y_train, y_test, config):
    """Run our decomposition method."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Intercept lattice
    int_features = sorted_idx[:config['int_dims']].tolist()
    int_dims = []
    int_bin_data = {}

    for i, feat_idx in enumerate(int_features):
        n_bins = config['int_bins'][i]
        feature_vals = X_train_s[:, feat_idx]
        bins = percentile_edges(feature_vals, n_bins)
        n_actual = len(bins) - 1
        dim_name = f"i{i}"
        int_dims.append(Dimension(dim_name, n_actual))
        int_bin_data[dim_name] = {
            "train": np.clip(np.digitize(feature_vals, bins[1:-1]), 0, n_actual - 1),
            "test": np.clip(np.digitize(X_test_s[:, feat_idx], bins[1:-1]), 0, n_actual - 1),
        }

    int_interactions = Interactions(dimensions=int_dims)
    int_decomp = Decomposed(
        interactions=int_interactions,
        param_shape=[1],
        name="intercept",
        max_order=config['int_order']
    )
    int_components = {k: v for k, v in int_decomp._tensor_part_shapes.items()
                     if int_decomp.component_order(k) in config.get('int_active_orders', [0, 1, 2])}

    # Beta lattice
    beta_features = sorted_idx[:config['beta_dims']].tolist()
    b_dims = []
    beta_bin_data = {}

    for i, feat_idx in enumerate(beta_features):
        n_bins = config['beta_bins'][i]
        feature_vals = X_train_s[:, feat_idx]
        bins = percentile_edges(feature_vals, n_bins)
        n_actual = len(bins) - 1
        dim_name = f"b{i}"
        b_dims.append(Dimension(dim_name, n_actual))
        beta_bin_data[dim_name] = {
            "train": np.clip(np.digitize(feature_vals, bins[1:-1]), 0, n_actual - 1),
            "test": np.clip(np.digitize(X_test_s[:, feat_idx], bins[1:-1]), 0, n_actual - 1),
        }

    beta_interactions = Interactions(dimensions=b_dims)
    beta_decomp = Decomposed(
        interactions=beta_interactions,
        param_shape=[n_features],
        name="beta",
        max_order=1
    )
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

    l2_int = config.get('l2_int', 30.0)
    l2_beta = config.get('l2_beta', 80.0)

    def loss_fn(params, int_idx, beta_idx, X, y, N):
        int_p, beta_p, g_int = params
        logits = compute_logits(int_p, beta_p, g_int, int_idx, beta_idx, X)
        bce = jnp.mean(jnp.logaddexp(0, logits) - y * logits)
        l2_reg = sum(0.5 * jnp.sum(p ** 2) / l2_int for p in int_p.values())
        l2_reg += sum(0.5 * jnp.sum(p ** 2) / l2_beta for p in beta_p.values())
        return bce + l2_reg / N

    params = (int_params, beta_params, global_int)

    batch_size = config.get('batch_size', 32768)
    n_epochs = config.get('epochs', 800)
    lr_peak = config.get('lr', 0.012)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0001,
        peak_value=lr_peak,
        warmup_steps=50,
        decay_steps=n_epochs - 50,
        end_value=0.0001
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    n_train = len(y_train)
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

    int_p, beta_p, g_int = params
    test_logits = compute_logits(int_p, beta_p, g_int, int_test_idx_jax, beta_test_idx_jax, X_test_jax)
    test_probs = jax.nn.sigmoid(test_logits)
    test_auc = roc_auc_score(y_test, np.array(test_probs))

    return test_auc


def run_ebm(X_train, X_test, y_train, y_test):
    """Run EBM."""
    from interpret.glassbox import ExplainableBoostingClassifier

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
    test_probs = ebm.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)

    return test_auc


# Our best single-model config
OUR_CONFIG = {
    'int_dims': 4,
    'int_bins': [10, 8, 10, 6],
    'int_order': 2,
    'int_active_orders': [0, 1, 2],
    'beta_dims': 6,
    'beta_bins': [8, 6, 8, 4, 8, 4],
    'lr': 0.012,
    'epochs': 800,
    'l2_int': 30.0,
    'l2_beta': 80.0,
}


def run_comparison(n_folds, seed=42):
    """Run n-fold CV comparison."""
    print(f"\n{'='*70}")
    print(f"{n_folds}-FOLD CROSS-VALIDATION")
    print(f"  Training size: {100*(n_folds-1)/n_folds:.0f}%")
    print(f"  Test size: {100/n_folds:.0f}%")
    print(f"{'='*70}")

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    our_aucs = []
    ebm_aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_full, y_full)):
        print(f"\n  Fold {fold_idx + 1}/{n_folds}:", flush=True)

        X_train, X_test = X_full[train_idx], X_full[test_idx]
        y_train, y_test = y_full[train_idx], y_full[test_idx]

        print(f"    Train: {len(y_train)}, Test: {len(y_test)}")

        # Our method
        print(f"    Running our method...", flush=True)
        t0 = time.time()
        our_auc = run_our_method(X_train, X_test, y_train, y_test, OUR_CONFIG)
        our_time = time.time() - t0
        our_aucs.append(our_auc)
        print(f"      Ours: {our_auc:.4f} ({our_time:.1f}s)")

        # EBM
        print(f"    Running EBM...", flush=True)
        t0 = time.time()
        ebm_auc = run_ebm(X_train, X_test, y_train, y_test)
        ebm_time = time.time() - t0
        ebm_aucs.append(ebm_auc)
        print(f"      EBM:  {ebm_auc:.4f} ({ebm_time:.1f}s)")

    our_mean, our_std = np.mean(our_aucs), np.std(our_aucs)
    ebm_mean, ebm_std = np.mean(ebm_aucs), np.std(ebm_aucs)

    print(f"\n  {n_folds}-fold Results:")
    print(f"    Ours: {our_mean:.4f} +/- {our_std:.4f}")
    print(f"    EBM:  {ebm_mean:.4f} +/- {ebm_std:.4f}")

    return {
        'n_folds': n_folds,
        'our_mean': our_mean, 'our_std': our_std, 'our_aucs': our_aucs,
        'ebm_mean': ebm_mean, 'ebm_std': ebm_std, 'ebm_aucs': ebm_aucs,
    }


# Run both 5-fold and 10-fold
results_5 = run_comparison(5)
results_10 = run_comparison(10)

print(f"\n{'='*70}")
print("SUMMARY: How does more training data help?")
print(f"{'='*70}")
print(f"\n5-fold (80% train):")
print(f"  Ours: {results_5['our_mean']:.4f} +/- {results_5['our_std']:.4f}")
print(f"  EBM:  {results_5['ebm_mean']:.4f} +/- {results_5['ebm_std']:.4f}")
print(f"\n10-fold (90% train):")
print(f"  Ours: {results_10['our_mean']:.4f} +/- {results_10['our_std']:.4f}")
print(f"  EBM:  {results_10['ebm_mean']:.4f} +/- {results_10['ebm_std']:.4f}")

our_improvement = results_10['our_mean'] - results_5['our_mean']
ebm_improvement = results_10['ebm_mean'] - results_5['ebm_mean']

print(f"\nImprovement from 5-fold to 10-fold:")
print(f"  Ours: {our_improvement:+.4f}")
print(f"  EBM:  {ebm_improvement:+.4f}")

if our_improvement > ebm_improvement:
    print(f"\n=> Our method benefits MORE from additional data (+{our_improvement - ebm_improvement:.4f})")
else:
    print(f"\n=> EBM benefits MORE from additional data (+{ebm_improvement - our_improvement:.4f})")

print(f"{'='*70}")
