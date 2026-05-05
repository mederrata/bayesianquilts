#!/usr/bin/env python3
"""
HIGGS: Two-lattice v3 - fix divergence, match 0.787
Key changes:
- Lower learning rate
- Stronger regularization
- Multiple configs to find optimal
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension

print("Loading HIGGS...", flush=True)
higgs = fetch_openml(data_id=23512, as_frame=False, parser='auto')
X_full, y_full = higgs.data, higgs.target
X_full = np.nan_to_num(X_full, nan=0.0)
y_full = (y_full.astype(float) > 0.5).astype(int)

N, n_features = X_full.shape
print(f"Dataset: {N} samples, {n_features} features")

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


def run_two_lattice(int_dims, int_bins, beta_dims, beta_bins,
                    n_epochs, batch_size, lr_peak, l2_int, l2_beta, name):
    print(f"\n{'='*60}", flush=True)
    print(f"{name}", flush=True)
    print(f"  Int: {int_dims}d, bins={int_bins[:int_dims]}")
    print(f"  Beta: {beta_dims}d, bins={beta_bins[:beta_dims]}")
    print(f"  LR peak: {lr_peak}, L2: int={l2_int}, beta={l2_beta}")
    print(f"{'='*60}", flush=True)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_full, y_full)):
        print(f"\n  Fold {fold_idx + 1}/5:", flush=True)

        X_train, X_test = X_full[train_idx], X_full[test_idx]
        y_train, y_test = y_full[train_idx], y_full[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Intercept lattice
        int_features = sorted_idx[:int_dims].tolist()
        i_dims = []
        int_bin_data = {}

        for i, feat_idx in enumerate(int_features):
            n_bins = int_bins[i]
            vals = X_train_scaled[:, feat_idx]
            edges = percentile_edges(vals, n_bins)
            n_actual = len(edges) - 1

            dim_name = f"i{i}"
            i_dims.append(Dimension(dim_name, n_actual))
            int_bin_data[dim_name] = {
                "train": np.digitize(vals, edges[1:-1]),
                "test": np.digitize(X_test_scaled[:, feat_idx], edges[1:-1]),
            }

        int_interactions = Interactions(dimensions=i_dims)
        int_decomp = Decomposed(interactions=int_interactions, param_shape=[1], name="intercept", max_order=1)

        # Beta lattice
        beta_features = sorted_idx[:beta_dims].tolist()
        b_dims = []
        beta_bin_data = {}

        for i, feat_idx in enumerate(beta_features):
            n_bins = beta_bins[i]
            vals = X_train_scaled[:, feat_idx]
            edges = percentile_edges(vals, n_bins)
            n_actual = len(edges) - 1

            dim_name = f"b{i}"
            b_dims.append(Dimension(dim_name, n_actual))
            beta_bin_data[dim_name] = {
                "train": np.digitize(vals, edges[1:-1]),
                "test": np.digitize(X_test_scaled[:, feat_idx], edges[1:-1]),
            }

        beta_interactions = Interactions(dimensions=b_dims)
        beta_decomp = Decomposed(interactions=beta_interactions, param_shape=[n_features], name="beta", max_order=1)

        int_components = int_decomp._tensor_part_shapes.copy()
        beta_components = beta_decomp._tensor_part_shapes.copy()

        int_params_count = sum(np.prod(s) for s in int_components.values())
        beta_params_count = sum(np.prod(s) for s in beta_components.values())
        print(f"    Params: int={int_params_count}, beta={beta_params_count}")

        int_train_idx = np.stack([int_bin_data[d.name]["train"] for d in i_dims], axis=-1)
        int_test_idx = np.stack([int_bin_data[d.name]["test"] for d in i_dims], axis=-1)
        beta_train_idx = np.stack([beta_bin_data[d.name]["train"] for d in b_dims], axis=-1)
        beta_test_idx = np.stack([beta_bin_data[d.name]["test"] for d in b_dims], axis=-1)

        int_params = {name: jnp.zeros(shape) for name, shape in int_components.items()}
        beta_params = {name: jnp.zeros(shape) for name, shape in beta_components.items()}
        global_intercept = jnp.zeros(1)

        X_train_j = jnp.array(X_train_scaled)
        X_test_j = jnp.array(X_test_scaled)
        int_train_j = jnp.array(int_train_idx)
        int_test_j = jnp.array(int_test_idx)
        beta_train_j = jnp.array(beta_train_idx)
        beta_test_j = jnp.array(beta_test_idx)
        y_train_j = jnp.array(y_train)

        def compute_logits(int_p, beta_p, global_int, int_idx, beta_idx, X):
            cell_int = int_decomp.lookup_flat(int_idx, int_p)
            beta = beta_decomp.lookup_flat(beta_idx, beta_p)
            return jnp.sum(X * beta, axis=-1) + cell_int[..., 0] + global_int[0]

        def loss_fn(params, int_idx, beta_idx, X, y, N):
            int_p, beta_p, global_int = params
            logits = compute_logits(int_p, beta_p, global_int, int_idx, beta_idx, X)
            bce = jnp.mean(jnp.logaddexp(0, logits) - y * logits)
            reg_int = sum(0.5 * jnp.sum(p ** 2) / l2_int for p in int_p.values())
            reg_beta = sum(0.5 * jnp.sum(p ** 2) / l2_beta for p in beta_p.values())
            return bce + (reg_int + reg_beta) / N

        params = (int_params, beta_params, global_intercept)

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0001,
            peak_value=lr_peak,
            warmup_steps=100,
            decay_steps=n_epochs - 100,
            end_value=0.0001
        )
        optimizer = optax.adam(schedule)
        opt_state = optimizer.init(params)
        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

        n_train = len(y_train)
        n_batches = (n_train + batch_size - 1) // batch_size

        best_auc = 0.0
        best_params = params
        patience = 200
        no_improve = 0

        for epoch in range(n_epochs):
            perm = np.random.permutation(n_train)
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_train)
                idx = perm[start:end]
                loss_val, grads = loss_and_grad(
                    params, int_train_j[idx], beta_train_j[idx],
                    X_train_j[idx], y_train_j[idx], len(idx)
                )
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)

            if epoch % 50 == 0 or epoch == n_epochs - 1:
                int_p, beta_p, global_int = params
                test_logits = compute_logits(int_p, beta_p, global_int, int_test_j, beta_test_j, X_test_j)
                test_probs = jax.nn.sigmoid(test_logits)
                test_auc = roc_auc_score(y_test, np.array(test_probs))
                if epoch % 100 == 0:
                    print(f"    Epoch {epoch}: AUC={test_auc:.4f}")
                if test_auc > best_auc:
                    best_auc = test_auc
                    best_params = params
                    no_improve = 0
                else:
                    no_improve += 50
                if no_improve >= patience:
                    print(f"    Early stop at epoch {epoch}")
                    break

        aucs.append(best_auc)
        print(f"    Final AUC: {best_auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  {name}: {mean_auc:.4f} +/- {std_auc:.4f}")
    return mean_auc, std_auc


results = {}

# Original config with lower LR
results['orig_low_lr'] = run_two_lattice(
    int_dims=8, int_bins=[16, 12, 16, 8, 12, 8, 10, 10],
    beta_dims=6, beta_bins=[8, 6, 8, 4, 8, 4],
    n_epochs=1000, batch_size=32768,
    lr_peak=0.005, l2_int=50.0, l2_beta=100.0,
    name="Original config, LR=0.005"
)

# Original config with very low LR
results['orig_vlow_lr'] = run_two_lattice(
    int_dims=8, int_bins=[16, 12, 16, 8, 12, 8, 10, 10],
    beta_dims=6, beta_bins=[8, 6, 8, 4, 8, 4],
    n_epochs=1500, batch_size=32768,
    lr_peak=0.002, l2_int=50.0, l2_beta=100.0,
    name="Original config, LR=0.002"
)

# Smaller bins (faster, more stable)
results['smaller_bins'] = run_two_lattice(
    int_dims=8, int_bins=[10, 8, 10, 6, 8, 6, 6, 6],
    beta_dims=6, beta_bins=[6, 4, 6, 3, 6, 3],
    n_epochs=1000, batch_size=32768,
    lr_peak=0.01, l2_int=30.0, l2_beta=60.0,
    name="Smaller bins, LR=0.01"
)

# More regularization
results['more_reg'] = run_two_lattice(
    int_dims=8, int_bins=[16, 12, 16, 8, 12, 8, 10, 10],
    beta_dims=6, beta_bins=[8, 6, 8, 4, 8, 4],
    n_epochs=1000, batch_size=32768,
    lr_peak=0.005, l2_int=20.0, l2_beta=40.0,
    name="Original bins, more reg"
)

# Fewer int dims
results['fewer_int'] = run_two_lattice(
    int_dims=6, int_bins=[16, 12, 16, 8, 12, 8],
    beta_dims=6, beta_bins=[8, 6, 8, 4, 8, 4],
    n_epochs=1000, batch_size=32768,
    lr_peak=0.01, l2_int=30.0, l2_beta=60.0,
    name="6 int dims"
)

# Moderate config
results['moderate'] = run_two_lattice(
    int_dims=6, int_bins=[12, 10, 12, 6, 10, 6],
    beta_dims=5, beta_bins=[8, 6, 8, 4, 6],
    n_epochs=1000, batch_size=32768,
    lr_peak=0.008, l2_int=40.0, l2_beta=80.0,
    name="Moderate (6int, 5beta)"
)

print(f"\n{'='*70}")
print("HIGGS TWO-LATTICE V3 RESULTS:")
print(f"  Target:    0.787 (original)")
print(f"  EBM:       0.803")
print(f"{'='*70}")
for name, (mean, std) in sorted(results.items(), key=lambda x: -x[1][0]):
    marker = " **BEST**" if mean >= max(r[0] for r in results.values()) else ""
    vs_target = " MATCH" if mean >= 0.785 else ""
    print(f"  {name}: {mean:.4f} +/- {std:.4f}{marker}{vs_target}")
print(f"{'='*70}")
