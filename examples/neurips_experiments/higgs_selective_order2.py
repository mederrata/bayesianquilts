#!/usr/bin/env python3
"""
HIGGS: Selective order-2 - keep 6+ dimensions but only add order-2 for top pairs.
Based on finding that 8 dims order-1 (~0.78) > 4 dims order-2 (0.76).
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


def run_selective_order2(n_dims, bins, n_order2_dims, pairwise, epochs, name):
    """
    Run experiment with selective order-2 interactions.

    n_dims: total dimensions in lattice
    n_order2_dims: how many top dims to include in order-2 components
    pairwise: explicit pairwise product features (on top of lattice)
    """
    print(f"\n{'='*60}", flush=True)
    print(f"{name}", flush=True)
    print(f"  Total dims: {n_dims}, Order-2 dims: {n_order2_dims}")
    print(f"  Bins: {bins}")
    print(f"  Explicit pairs: {len(pairwise)}")
    print(f"{'='*60}", flush=True)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    features = sorted_idx[:n_dims].tolist()

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_full, y_full)):
        print(f"\n  Fold {fold_idx + 1}/5:", flush=True)

        X_train, X_test = X_full[train_idx], X_full[test_idx]
        y_train, y_test = y_full[train_idx], y_full[test_idx]
        N_train = len(y_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build lattice
        dimensions = []
        bin_data = {}

        for i, feat_idx in enumerate(features):
            n_bins = bins[i]
            vals = X_train_scaled[:, feat_idx]

            if feat_idx in HEAVY_TAIL:
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
        decomp = Decomposed(
            interactions=interactions,
            param_shape=[n_features],
            name="beta",
            max_order=2
        )

        # Select components: all order 0-1, plus order-2 only for top n_order2_dims
        top_dim_names = {f"d{i}" for i in range(n_order2_dims)}
        active_components = []
        for comp_name in decomp._tensor_parts.keys():
            order = decomp.component_order(comp_name)
            if order <= 1:
                active_components.append(comp_name)
            elif order == 2:
                # Check if both dims in this component are in top_dim_names
                dims_in = [d.name for d in dimensions if d.name in comp_name]
                if all(d in top_dim_names for d in dims_in):
                    active_components.append(comp_name)

        total_params = sum(np.prod(decomp._tensor_part_shapes[c]) for c in active_components)

        # Build pairwise products
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

        print(f"    Decomp params: {total_params}, Pairs: {n_pairs}")

        # Prepare indices
        dim_names = [d.name for d in dimensions]
        train_idx_arr = np.stack([bin_data[d]["train"] for d in dim_names], axis=-1)
        test_idx_arr = np.stack([bin_data[d]["test"] for d in dim_names], axis=-1)

        # Initialize parameters
        params = {c: jnp.zeros(decomp._tensor_part_shapes[c]) for c in active_components}
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
            decomp_params = {k: v for k, v in params.items() if k != "beta_pairs"}
            decomp_vals = decomp.lookup_flat(idx, decomp_params)
            logits = jnp.sum(X * decomp_vals, axis=-1)
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
            if n_pairs > 0:
                l2_reg += 0.5 * jnp.sum(params["beta_pairs"] ** 2) / (tau_global ** 2)

            return bce + l2_reg / N_train

        # Optimizer
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.001,
            peak_value=0.02,
            warmup_steps=300,
            decay_steps=epochs - 300,
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

        for epoch in range(epochs):
            params, opt_state, loss = step(params, opt_state)

            if epoch % 100 == 0 or epoch == epochs - 1:
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

# 1. 6 dims, order-2 only on top 3
results['6d_top3_order2'] = run_selective_order2(
    n_dims=6,
    bins=[10, 8, 10, 6, 8, 6],
    n_order2_dims=3,
    pairwise=[],
    epochs=4000,
    name="6 dims, order-2 on top 3"
)

# 2. 6 dims, order-2 on top 4
results['6d_top4_order2'] = run_selective_order2(
    n_dims=6,
    bins=[10, 8, 10, 6, 8, 6],
    n_order2_dims=4,
    pairwise=[],
    epochs=4000,
    name="6 dims, order-2 on top 4"
)

# 3. 8 dims, order-2 only on top 3
results['8d_top3_order2'] = run_selective_order2(
    n_dims=8,
    bins=[8, 6, 8, 4, 6, 4, 4, 4],
    n_order2_dims=3,
    pairwise=[],
    epochs=4000,
    name="8 dims, order-2 on top 3"
)

# 4. 6 dims order-2 top 3 + explicit pairs for features 4-5
remaining_pairs = [
    (sorted_idx[0], sorted_idx[4]),  # 27 * 22
    (sorted_idx[0], sorted_idx[5]),  # 27 * 3
    (sorted_idx[1], sorted_idx[4]),  # 26 * 22
    (sorted_idx[1], sorted_idx[5]),  # 26 * 3
    (sorted_idx[2], sorted_idx[4]),  # 25 * 22
    (sorted_idx[2], sorted_idx[5]),  # 25 * 3
]
results['6d_explicit_pairs'] = run_selective_order2(
    n_dims=6,
    bins=[10, 8, 10, 6, 8, 6],
    n_order2_dims=3,
    pairwise=remaining_pairs,
    epochs=4000,
    name="6 dims + 6 explicit pairs"
)

# 5. 8 dims, order-2 on top 2 only
results['8d_top2_order2'] = run_selective_order2(
    n_dims=8,
    bins=[8, 6, 8, 4, 6, 4, 4, 4],
    n_order2_dims=2,
    pairwise=[],
    epochs=4000,
    name="8 dims, order-2 on top 2"
)

# 6. 5 dims full order-2 (sweet spot?)
results['5d_full_order2'] = run_selective_order2(
    n_dims=5,
    bins=[10, 8, 10, 6, 8],
    n_order2_dims=5,
    pairwise=[],
    epochs=4000,
    name="5 dims, full order-2"
)

# 7. 6 dims, bigger bins, order-2 on top 3
results['6d_big_bins'] = run_selective_order2(
    n_dims=6,
    bins=[14, 12, 14, 8, 10, 8],
    n_order2_dims=3,
    pairwise=[],
    epochs=4000,
    name="6 dims, bigger bins, order-2 top 3"
)

print(f"\n{'='*70}")
print("HIGGS SELECTIVE ORDER-2 RESULTS:")
print(f"  Previous 4d order-2:     0.7611")
print(f"  Previous 8d order-1:     ~0.780")
print(f"  EBM target:              0.803")
print(f"{'='*70}")
for name, (mean, std) in sorted(results.items(), key=lambda x: -x[1][0]):
    marker = " **BEST**" if mean >= max(r[0] for r in results.values()) else ""
    improved = " IMPROVED" if mean > 0.780 else ""
    print(f"  {name}: {mean:.4f} +/- {std:.4f}{marker}{improved}")
print(f"{'='*70}")
