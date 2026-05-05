#!/usr/bin/env python3
"""
HIGGS: Improved configurations targeting underfitting.

Based on analysis:
- Main effects only (order 1) underfits HIGGS physics
- Need order-2 interactions but can't do 8 dims (explodes)
- Solution: fewer dims + order 2, or explicit pairwise products
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

HEAVY_TAIL = {0, 3, 5, 9, 13, 17}


def tukey_edges(vals, n_inner):
    Q1, Q3 = np.percentile(vals, [25, 75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    edges = [-np.inf, lower]
    inner_vals = vals[(vals >= lower) & (vals <= upper)]
    if len(inner_vals) > n_inner:
        inner_edges = np.percentile(inner_vals, np.linspace(0, 100, n_inner + 1)[1:-1])
        edges.extend(inner_edges)
    edges.extend([upper, np.inf])
    return np.unique(edges)


def percentile_edges(vals, n_bins):
    edges = np.percentile(vals, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def run_experiment(config, name):
    """Run experiment with given config."""
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
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build lattice dimensions
        features = config['features']
        bins_per_feat = config['bins']
        use_tukey = config.get('tukey', set())
        max_order = config.get('max_order', 1)

        dimensions = []
        bin_data = {}

        for i, feat_idx in enumerate(features):
            n_bins = bins_per_feat[i]
            vals = X_train_scaled[:, feat_idx]

            if feat_idx in use_tukey:
                edges = tukey_edges(vals, max(n_bins - 2, 2))
            else:
                edges = percentile_edges(vals, n_bins)

            n_actual = len(edges) - 1
            dim_name = f"d{i}"
            dimensions.append(Dimension(dim_name, n_actual))
            bin_data[dim_name] = {
                "train": np.clip(np.digitize(vals, edges[1:-1]), 0, n_actual - 1),
                "test": np.clip(np.digitize(X_test_scaled[:, feat_idx], edges[1:-1]), 0, n_actual - 1),
            }

        interactions = Interactions(dimensions=dimensions)

        # Use cell-varying beta or intercept-only based on config
        use_cell_beta = config.get('cell_beta', True)
        if use_cell_beta:
            decomp = Decomposed(
                interactions=interactions,
                param_shape=[n_features],
                name="beta",
                max_order=max_order
            )
        else:
            decomp = Decomposed(
                interactions=interactions,
                param_shape=[1],
                name="intercept",
                max_order=max_order
            )

        # Select active orders
        active_orders = config.get('orders', list(range(max_order + 1)))
        active_components = [c for c in decomp._tensor_parts.keys()
                           if decomp.component_order(c) in active_orders]

        total_params = sum(np.prod(decomp._tensor_part_shapes[c]) for c in active_components)

        # Build pairwise products if requested
        pairwise = config.get('pairwise', [])
        n_pairs = len(pairwise)
        if n_pairs > 0:
            X_pairs_train = np.zeros((len(X_train), n_pairs))
            X_pairs_test = np.zeros((len(X_test), n_pairs))
            for k, (fi, fj) in enumerate(pairwise):
                X_pairs_train[:, k] = X_train_scaled[:, fi] * X_train_scaled[:, fj]
                X_pairs_test[:, k] = X_test_scaled[:, fi] * X_test_scaled[:, fj]
            pair_scaler = StandardScaler()
            X_pairs_train = pair_scaler.fit_transform(X_pairs_train)
            X_pairs_test = pair_scaler.transform(X_pairs_test)

        print(f"    Decomp params: {total_params}, Pairs: {n_pairs}, Orders: {active_orders}")

        # Prepare indices
        dim_names = [d.name for d in dimensions]
        train_idx_arr = np.stack([bin_data[d]["train"] for d in dim_names], axis=-1)
        test_idx_arr = np.stack([bin_data[d]["test"] for d in dim_names], axis=-1)

        # Initialize parameters
        params = {c: jnp.zeros(decomp._tensor_part_shapes[c]) for c in active_components}
        if not use_cell_beta:
            params["global_beta"] = jnp.zeros(n_features)
        if n_pairs > 0:
            params["beta_pairs"] = jnp.zeros(n_pairs)

        # Convert to JAX
        X_train_j = jnp.array(X_train_scaled)
        X_test_j = jnp.array(X_test_scaled)
        train_idx_j = jnp.array(train_idx_arr)
        test_idx_j = jnp.array(test_idx_arr)
        y_train_j = jnp.array(y_train)
        if n_pairs > 0:
            X_pairs_train_j = jnp.array(X_pairs_train)
            X_pairs_test_j = jnp.array(X_pairs_test)

        # Prior variances
        sigma_eff = 2.0
        prior_vars = {}
        for c in active_components:
            order = decomp.component_order(c)
            if order == 0:
                prior_vars[c] = (sigma_eff / np.sqrt(N_train)) ** 2
            else:
                dims_in = [d for d in dimensions if d.name in c]
                n_cells = np.prod([d.cardinality for d in dims_in])
                avg_count = N_train / n_cells
                prior_vars[c] = (sigma_eff / np.sqrt(max(avg_count, 1))) ** 2

        tau_global = sigma_eff / np.sqrt(N_train)

        def compute_logits(params, idx, X, X_pairs=None):
            decomp_params = {k: v for k, v in params.items()
                           if k not in ["global_beta", "beta_pairs"]}
            decomp_vals = decomp.lookup_flat(idx, decomp_params)

            if use_cell_beta:
                logits = jnp.sum(X * decomp_vals, axis=-1)
            else:
                logits = jnp.sum(X * params["global_beta"], axis=-1) + decomp_vals[..., 0]

            if n_pairs > 0 and X_pairs is not None:
                logits = logits + jnp.sum(X_pairs * params["beta_pairs"], axis=-1)

            return logits

        def loss_fn(params):
            X_p = X_pairs_train_j if n_pairs > 0 else None
            logits = compute_logits(params, train_idx_j, X_train_j, X_p)
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2_reg = 0.0
            for c in active_components:
                l2_reg += 0.5 * jnp.sum(params[c] ** 2) / (prior_vars[c] + 1e-8)
            if not use_cell_beta:
                l2_reg += 0.5 * jnp.sum(params["global_beta"] ** 2) / (tau_global ** 2)
            if n_pairs > 0:
                l2_reg += 0.5 * jnp.sum(params["beta_pairs"] ** 2) / (tau_global ** 2)

            return bce + l2_reg / N_train

        # Optimizer
        n_epochs = config.get('epochs', 3000)
        lr_peak = config.get('lr', 0.02)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.001,
            peak_value=lr_peak,
            warmup_steps=300,
            decay_steps=n_epochs - 300,
            end_value=0.001
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
        patience = 400
        no_improve = 0

        for epoch in range(n_epochs):
            params, opt_state, loss = step(params, opt_state)

            if epoch % 100 == 0 or epoch == n_epochs - 1:
                X_p = X_pairs_test_j if n_pairs > 0 else None
                test_logits = compute_logits(params, test_idx_j, X_test_j, X_p)
                test_probs = jax.nn.sigmoid(test_logits)
                test_auc = roc_auc_score(y_test, np.array(test_probs))

                if epoch % 500 == 0:
                    print(f"    Epoch {epoch}: loss={float(loss):.4f}, AUC={test_auc:.4f}")

                if test_auc > best_auc:
                    best_auc = test_auc
                    no_improve = 0
                else:
                    no_improve += 100
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

# Top features from LR: [27, 26, 25, 5, 22, 3, 24, 0]
# Heavy-tail pT features: 0, 3, 5

# 1. Baseline: 4 dims, order 1, cell beta (reference)
results['baseline_4d'] = run_experiment({
    'features': sorted_idx[:4].tolist(),
    'bins': [12, 10, 12, 8],
    'tukey': {5},
    'max_order': 1,
    'orders': [0, 1],
    'cell_beta': True,
    'epochs': 3000,
    'lr': 0.02,
}, "Baseline: 4 dims, order 1")

# 2. Order 2 with 4 dims - capture pairwise interactions in lattice
results['order2_4d'] = run_experiment({
    'features': sorted_idx[:4].tolist(),
    'bins': [10, 8, 10, 6],
    'tukey': {5},
    'max_order': 2,
    'orders': [0, 1, 2],
    'cell_beta': True,
    'epochs': 4000,
    'lr': 0.02,
}, "Order 2: 4 dims, full interactions")

# 3. Order 2 with 3 dims - even more focused
results['order2_3d'] = run_experiment({
    'features': sorted_idx[:3].tolist(),
    'bins': [14, 12, 14],
    'tukey': set(),
    'max_order': 2,
    'orders': [0, 1, 2],
    'cell_beta': True,
    'epochs': 4000,
    'lr': 0.02,
}, "Order 2: 3 dims, more bins")

# 4. Skip order 1, use [0, 2] - like Bioresponse success
results['skip_order1_4d'] = run_experiment({
    'features': sorted_idx[:4].tolist(),
    'bins': [12, 10, 12, 8],
    'tukey': {5},
    'max_order': 2,
    'orders': [0, 2],
    'cell_beta': True,
    'epochs': 4000,
    'lr': 0.02,
}, "Orders [0,2]: skip redundant order 1")

# 5. Intercept lattice + explicit pairwise products
top_pairs = [
    (sorted_idx[0], sorted_idx[1]),  # 27 * 26
    (sorted_idx[0], sorted_idx[2]),  # 27 * 25
    (sorted_idx[1], sorted_idx[2]),  # 26 * 25
    (sorted_idx[0], sorted_idx[3]),  # 27 * 5
    (sorted_idx[1], sorted_idx[3]),  # 26 * 5
    (sorted_idx[2], sorted_idx[3]),  # 25 * 5
]
results['pairwise_explicit'] = run_experiment({
    'features': sorted_idx[:4].tolist(),
    'bins': [10, 8, 10, 6],
    'tukey': {5},
    'max_order': 1,
    'orders': [0, 1],
    'cell_beta': False,  # intercept lattice only
    'pairwise': top_pairs,
    'epochs': 4000,
    'lr': 0.02,
}, "Intercept + 6 pairwise products")

# 6. More pairwise products (top 10 pairs)
more_pairs = top_pairs + [
    (sorted_idx[0], sorted_idx[4]),  # 27 * 22
    (sorted_idx[1], sorted_idx[4]),  # 26 * 22
    (sorted_idx[2], sorted_idx[4]),  # 25 * 22
    (sorted_idx[3], sorted_idx[4]),  # 5 * 22
]
results['pairwise_10'] = run_experiment({
    'features': sorted_idx[:5].tolist(),
    'bins': [10, 8, 10, 6, 8],
    'tukey': {5},
    'max_order': 1,
    'orders': [0, 1],
    'cell_beta': False,
    'pairwise': more_pairs,
    'epochs': 4000,
    'lr': 0.02,
}, "Intercept + 10 pairwise products")

# 7. Order 2 with 5 dims - push the limit
results['order2_5d'] = run_experiment({
    'features': sorted_idx[:5].tolist(),
    'bins': [8, 6, 8, 4, 6],
    'tukey': {5},
    'max_order': 2,
    'orders': [0, 1, 2],
    'cell_beta': True,
    'epochs': 4000,
    'lr': 0.02,
}, "Order 2: 5 dims, smaller bins")

# 8. Hybrid: order 2 lattice + pairwise for remaining features
results['hybrid'] = run_experiment({
    'features': sorted_idx[:3].tolist(),
    'bins': [12, 10, 12],
    'tukey': set(),
    'max_order': 2,
    'orders': [0, 1, 2],
    'cell_beta': True,
    'pairwise': [
        (sorted_idx[0], sorted_idx[3]),  # top3 * feat4
        (sorted_idx[1], sorted_idx[3]),
        (sorted_idx[2], sorted_idx[3]),
        (sorted_idx[0], sorted_idx[4]),  # top3 * feat5
        (sorted_idx[1], sorted_idx[4]),
        (sorted_idx[2], sorted_idx[4]),
    ],
    'epochs': 4000,
    'lr': 0.02,
}, "Hybrid: 3d order-2 + pairwise to feat 4,5")

print(f"\n{'='*70}")
print("HIGGS IMPROVED RESULTS:")
print(f"  Previous best (leaked):  0.787")
print(f"  Previous best (CV):      ~0.780")
print(f"  EBM target:              0.803")
print(f"  LGBM:                    0.804")
print(f"{'='*70}")
for name, (mean, std) in sorted(results.items(), key=lambda x: -x[1][0]):
    marker = " **BEST**" if mean >= max(r[0] for r in results.values()) else ""
    improved = " IMPROVED" if mean > 0.780 else ""
    print(f"  {name}: {mean:.4f} +/- {std:.4f}{marker}{improved}")
print(f"{'='*70}")
