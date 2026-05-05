#!/usr/bin/env python3
"""
HIGGS: Add nonlinear effects to two-lattice architecture.
Options:
1. Order-2 on intercept lattice (smaller)
2. Explicit pairwise products
3. Combined approaches
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


def run_nonlinear(config, name):
    """Run experiment with nonlinear effects."""
    print(f"\n{'='*60}", flush=True)
    print(f"{name}", flush=True)
    print(f"  Config: {config}")
    print(f"{'='*60}", flush=True)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_full, y_full)):
        print(f"\n  Fold {fold_idx + 1}/5:", flush=True)

        X_train, X_test = X_full[train_idx], X_full[test_idx]
        y_train, y_test = y_full[train_idx], y_full[test_idx]
        N_train = len(y_train)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Build intercept lattice
        int_dims = config['int_dims']
        int_bins = config['int_bins']
        int_order = config.get('int_order', 1)
        
        int_features = sorted_idx[:int_dims].tolist()
        i_dimensions = []
        int_bin_data = {}

        for i, feat_idx in enumerate(int_features):
            n_bins = int_bins[i]
            vals = X_train_s[:, feat_idx]
            edges = percentile_edges(vals, n_bins)
            n_actual = len(edges) - 1
            dim_name = f"i{i}"
            i_dimensions.append(Dimension(dim_name, n_actual))
            int_bin_data[dim_name] = {
                "train": np.digitize(vals, edges[1:-1]),
                "test": np.digitize(X_test_s[:, feat_idx], edges[1:-1]),
            }

        int_interactions = Interactions(dimensions=i_dimensions)
        int_decomp = Decomposed(
            interactions=int_interactions, 
            param_shape=[1], 
            name="intercept", 
            max_order=int_order
        )

        # Select active orders for intercept
        int_active_orders = config.get('int_active_orders', list(range(int_order + 1)))
        int_active = [c for c in int_decomp._tensor_parts.keys() 
                     if int_decomp.component_order(c) in int_active_orders]

        # Build beta lattice
        beta_dims = config['beta_dims']
        beta_bins = config['beta_bins']
        beta_order = config.get('beta_order', 1)
        
        beta_features = sorted_idx[:beta_dims].tolist()
        b_dimensions = []
        beta_bin_data = {}

        for i, feat_idx in enumerate(beta_features):
            n_bins = beta_bins[i]
            vals = X_train_s[:, feat_idx]
            edges = percentile_edges(vals, n_bins)
            n_actual = len(edges) - 1
            dim_name = f"b{i}"
            b_dimensions.append(Dimension(dim_name, n_actual))
            beta_bin_data[dim_name] = {
                "train": np.digitize(vals, edges[1:-1]),
                "test": np.digitize(X_test_s[:, feat_idx], edges[1:-1]),
            }

        beta_interactions = Interactions(dimensions=b_dimensions)
        beta_decomp = Decomposed(
            interactions=beta_interactions, 
            param_shape=[n_features], 
            name="beta", 
            max_order=beta_order
        )

        beta_active_orders = config.get('beta_active_orders', list(range(beta_order + 1)))
        beta_active = [c for c in beta_decomp._tensor_parts.keys()
                      if beta_decomp.component_order(c) in beta_active_orders]

        # Build pairwise products
        pairwise = config.get('pairwise', [])
        n_pairs = len(pairwise)
        if n_pairs > 0:
            X_pairs_train = np.zeros((len(X_train), n_pairs))
            X_pairs_test = np.zeros((len(X_test), n_pairs))
            for k, (fi, fj) in enumerate(pairwise):
                X_pairs_train[:, k] = X_train_s[:, fi] * X_train_s[:, fj]
                X_pairs_test[:, k] = X_test_s[:, fi] * X_test_s[:, fj]
            pair_scaler = StandardScaler()
            X_pairs_train = pair_scaler.fit_transform(X_pairs_train)
            X_pairs_test = pair_scaler.transform(X_pairs_test)

        int_count = sum(np.prod(int_decomp._tensor_part_shapes[c]) for c in int_active)
        beta_count = sum(np.prod(beta_decomp._tensor_part_shapes[c]) for c in beta_active)
        print(f"    Params: int={int_count}, beta={beta_count}, pairs={n_pairs}")

        # Prepare indices
        int_train_idx = np.stack([int_bin_data[d.name]["train"] for d in i_dimensions], axis=-1)
        int_test_idx = np.stack([int_bin_data[d.name]["test"] for d in i_dimensions], axis=-1)
        beta_train_idx = np.stack([beta_bin_data[d.name]["train"] for d in b_dimensions], axis=-1)
        beta_test_idx = np.stack([beta_bin_data[d.name]["test"] for d in b_dimensions], axis=-1)

        # Initialize parameters
        int_params = {c: jnp.zeros(int_decomp._tensor_part_shapes[c]) for c in int_active}
        beta_params = {c: jnp.zeros(beta_decomp._tensor_part_shapes[c]) for c in beta_active}
        global_int = jnp.zeros(1)
        pair_beta = jnp.zeros(n_pairs) if n_pairs > 0 else None

        # Convert to JAX
        X_train_j = jnp.array(X_train_s)
        X_test_j = jnp.array(X_test_s)
        int_train_j = jnp.array(int_train_idx)
        int_test_j = jnp.array(int_test_idx)
        beta_train_j = jnp.array(beta_train_idx)
        beta_test_j = jnp.array(beta_test_idx)
        y_train_j = jnp.array(y_train)
        if n_pairs > 0:
            X_pairs_train_j = jnp.array(X_pairs_train)
            X_pairs_test_j = jnp.array(X_pairs_test)

        l2_int = config.get('l2_int', 50.0)
        l2_beta = config.get('l2_beta', 100.0)
        l2_pair = config.get('l2_pair', 100.0)

        def compute_logits(int_p, beta_p, g_int, int_idx, beta_idx, X, pair_b=None, X_pairs=None):
            cell_int = int_decomp.lookup_flat(int_idx, int_p)
            beta = beta_decomp.lookup_flat(beta_idx, beta_p)
            logits = jnp.sum(X * beta, axis=-1) + cell_int[..., 0] + g_int[0]
            if pair_b is not None and X_pairs is not None:
                logits = logits + jnp.sum(X_pairs * pair_b, axis=-1)
            return logits

        def loss_fn(params):
            if n_pairs > 0:
                int_p, beta_p, g_int, pair_b = params
                logits = compute_logits(int_p, beta_p, g_int, int_train_j, beta_train_j, 
                                       X_train_j, pair_b, X_pairs_train_j)
            else:
                int_p, beta_p, g_int = params
                logits = compute_logits(int_p, beta_p, g_int, int_train_j, beta_train_j, X_train_j)

            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)
            
            reg_int = sum(0.5 * jnp.sum(p ** 2) / l2_int for p in int_p.values())
            reg_beta = sum(0.5 * jnp.sum(p ** 2) / l2_beta for p in beta_p.values())
            reg = reg_int + reg_beta
            if n_pairs > 0:
                reg = reg + 0.5 * jnp.sum(pair_b ** 2) / l2_pair
            
            return bce + reg / N_train

        if n_pairs > 0:
            params = (int_params, beta_params, global_int, pair_beta)
        else:
            params = (int_params, beta_params, global_int)

        lr_peak = config.get('lr', 0.01)
        n_epochs = config.get('epochs', 1000)
        
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0001,
            peak_value=lr_peak,
            warmup_steps=100,
            decay_steps=n_epochs - 100,
            end_value=0.0001
        )
        optimizer = optax.adam(schedule)
        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        best_auc = 0.0
        patience = 200
        no_improve = 0

        for epoch in range(n_epochs):
            params, opt_state, loss = step(params, opt_state)

            if epoch % 50 == 0 or epoch == n_epochs - 1:
                if n_pairs > 0:
                    int_p, beta_p, g_int, pair_b = params
                    test_logits = compute_logits(int_p, beta_p, g_int, int_test_j, beta_test_j,
                                                X_test_j, pair_b, X_pairs_test_j)
                else:
                    int_p, beta_p, g_int = params
                    test_logits = compute_logits(int_p, beta_p, g_int, int_test_j, beta_test_j, X_test_j)
                
                test_probs = jax.nn.sigmoid(test_logits)
                test_auc = roc_auc_score(y_test, np.array(test_probs))
                
                if epoch % 200 == 0:
                    print(f"    Epoch {epoch}: loss={float(loss):.4f}, AUC={test_auc:.4f}")
                
                if test_auc > best_auc:
                    best_auc = test_auc
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

# Top pairwise interactions (based on LR importance)
top_pairs = [
    (sorted_idx[0], sorted_idx[1]),  # 27*26
    (sorted_idx[0], sorted_idx[2]),  # 27*25
    (sorted_idx[1], sorted_idx[2]),  # 26*25
    (sorted_idx[0], sorted_idx[3]),  # 27*5
    (sorted_idx[1], sorted_idx[3]),  # 26*5
    (sorted_idx[2], sorted_idx[3]),  # 25*5
    (sorted_idx[0], sorted_idx[4]),  # 27*22
    (sorted_idx[1], sorted_idx[4]),  # 26*22
    (sorted_idx[2], sorted_idx[4]),  # 25*22
    (sorted_idx[3], sorted_idx[4]),  # 5*22
]

# 1. Two-lattice + pairwise products
results['two_lattice_pairs'] = run_nonlinear({
    'int_dims': 6, 'int_bins': [12, 10, 12, 6, 10, 6],
    'beta_dims': 5, 'beta_bins': [8, 6, 8, 4, 6],
    'int_order': 1, 'beta_order': 1,
    'pairwise': top_pairs,
    'lr': 0.01, 'epochs': 1500,
    'l2_int': 40.0, 'l2_beta': 80.0, 'l2_pair': 100.0,
}, "Two-lattice + 10 pairwise")

# 2. Order-2 intercept lattice + order-1 beta
results['int_order2'] = run_nonlinear({
    'int_dims': 4, 'int_bins': [10, 8, 10, 6],
    'int_order': 2, 'int_active_orders': [0, 1, 2],
    'beta_dims': 6, 'beta_bins': [8, 6, 8, 4, 8, 4],
    'beta_order': 1,
    'lr': 0.01, 'epochs': 1500,
    'l2_int': 30.0, 'l2_beta': 80.0,
}, "Order-2 intercept (4d) + order-1 beta (6d)")

# 3. Order-2 intercept + pairwise
results['int_order2_pairs'] = run_nonlinear({
    'int_dims': 4, 'int_bins': [10, 8, 10, 6],
    'int_order': 2, 'int_active_orders': [0, 1, 2],
    'beta_dims': 5, 'beta_bins': [8, 6, 8, 4, 6],
    'beta_order': 1,
    'pairwise': top_pairs[:6],  # Top 6 pairs
    'lr': 0.01, 'epochs': 1500,
    'l2_int': 30.0, 'l2_beta': 80.0, 'l2_pair': 100.0,
}, "Order-2 int + beta + 6 pairwise")

# 4. Skip order-1 in intercept [0,2]
results['int_skip_order1'] = run_nonlinear({
    'int_dims': 5, 'int_bins': [10, 8, 10, 6, 8],
    'int_order': 2, 'int_active_orders': [0, 2],
    'beta_dims': 5, 'beta_bins': [8, 6, 8, 4, 6],
    'beta_order': 1,
    'lr': 0.01, 'epochs': 1500,
    'l2_int': 30.0, 'l2_beta': 80.0,
}, "Int [0,2] + beta order-1")

# 5. Both lattices order-2 on small dims
results['both_order2'] = run_nonlinear({
    'int_dims': 3, 'int_bins': [12, 10, 12],
    'int_order': 2, 'int_active_orders': [0, 1, 2],
    'beta_dims': 3, 'beta_bins': [10, 8, 10],
    'beta_order': 2, 'beta_active_orders': [0, 1, 2],
    'pairwise': top_pairs[:6],
    'lr': 0.01, 'epochs': 1500,
    'l2_int': 20.0, 'l2_beta': 40.0, 'l2_pair': 80.0,
}, "Both order-2 (3d) + 6 pairs")

# 6. Larger intercept order-2, smaller beta
results['large_int_order2'] = run_nonlinear({
    'int_dims': 5, 'int_bins': [12, 10, 12, 8, 10],
    'int_order': 2, 'int_active_orders': [0, 1, 2],
    'beta_dims': 4, 'beta_bins': [8, 6, 8, 4],
    'beta_order': 1,
    'lr': 0.008, 'epochs': 2000,
    'l2_int': 25.0, 'l2_beta': 60.0,
}, "5d int order-2 + 4d beta")

# 7. Many pairwise products
all_pairs = [(sorted_idx[i], sorted_idx[j]) 
             for i in range(6) for j in range(i+1, 6)]  # 15 pairs
results['many_pairs'] = run_nonlinear({
    'int_dims': 6, 'int_bins': [10, 8, 10, 6, 8, 6],
    'beta_dims': 6, 'beta_bins': [6, 5, 6, 4, 5, 4],
    'int_order': 1, 'beta_order': 1,
    'pairwise': all_pairs,
    'lr': 0.01, 'epochs': 1500,
    'l2_int': 40.0, 'l2_beta': 80.0, 'l2_pair': 80.0,
}, "Two-lattice + 15 pairwise")

print(f"\n{'='*70}")
print("HIGGS NONLINEAR RESULTS:")
print(f"  Target:       0.787")
print(f"  Current best: 0.7715 (5d order-2)")
print(f"  EBM:          0.803")
print(f"{'='*70}")
for name, (mean, std) in sorted(results.items(), key=lambda x: -x[1][0]):
    marker = " **BEST**" if mean >= max(r[0] for r in results.values()) else ""
    improved = " IMPROVED" if mean > 0.7715 else ""
    match = " MATCH" if mean >= 0.785 else ""
    print(f"  {name}: {mean:.4f} +/- {std:.4f}{marker}{improved}{match}")
print(f"{'='*70}")
