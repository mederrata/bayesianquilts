#!/usr/bin/env python3
"""
HIGGS: Ensemble with LOO-approximated stacking weights (no leakage).
Uses leverage-based LOO reweighting when learning ensemble weights.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent / "python"))
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


def percentile_edges(vals, n_bins):
    edges = np.percentile(vals, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


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


def build_base_model(config, X_train_s, X_test_s, y_train, sorted_idx, n_features):
    """Build a single base model and return test logits only."""
    offset = config.get('feature_offset', 0)

    # Intercept lattice
    int_features = sorted_idx[offset:offset + config['int_dims']].tolist()
    int_dims = []
    int_bin_data = {}

    for i, feat_idx in enumerate(int_features):
        n_bins = config['int_bins'][i]
        feature_vals = X_train_s[:, feat_idx]

        if config.get('use_tukey', False) and feat_idx in HEAVY_TAIL:
            bins = tukey_edges(feature_vals, max(n_bins - 2, 2))
        else:
            bins = percentile_edges(feature_vals, n_bins)

        n_actual = len(bins) - 1
        dim_name = f"i{i}"
        int_dims.append(Dimension(dim_name, n_actual))
        int_bin_data[dim_name] = {
            "bins": bins,
            "train": np.clip(np.digitize(feature_vals, bins[1:-1]), 0, n_actual - 1),
            "test": np.clip(np.digitize(X_test_s[:, feat_idx], bins[1:-1]), 0, n_actual - 1),
        }

    int_interactions = Interactions(dimensions=int_dims)
    int_order = config.get('int_order', 2)
    int_decomp = Decomposed(interactions=int_interactions, param_shape=[1], name="intercept", max_order=int_order)

    int_active = config.get('int_active_orders', list(range(int_order + 1)))
    int_components = {k: v for k, v in int_decomp._tensor_part_shapes.items()
                     if int_decomp.component_order(k) in int_active}

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
            "bins": bins,
            "train": np.clip(np.digitize(feature_vals, bins[1:-1]), 0, n_actual - 1),
            "test": np.clip(np.digitize(X_test_s[:, feat_idx], bins[1:-1]), 0, n_actual - 1),
        }

    beta_interactions = Interactions(dimensions=b_dims)
    beta_order = config.get('beta_order', 1)
    beta_decomp = Decomposed(interactions=beta_interactions, param_shape=[n_features], name="beta", max_order=beta_order)
    beta_active_orders = config.get('beta_active_orders', list(range(beta_order + 1)))
    beta_components = {k: v for k, v in beta_decomp._tensor_part_shapes.items()
                      if beta_decomp.component_order(k) in beta_active_orders}

    # Prepare indices
    int_train_idx = np.stack([int_bin_data[d.name]["train"] for d in int_dims], axis=-1)
    int_test_idx = np.stack([int_bin_data[d.name]["test"] for d in int_dims], axis=-1)
    beta_train_idx = np.stack([beta_bin_data[d.name]["train"] for d in b_dims], axis=-1)
    beta_test_idx = np.stack([beta_bin_data[d.name]["test"] for d in b_dims], axis=-1)

    int_params_dict = {name: jnp.zeros(shape) for name, shape in int_components.items()}
    beta_params_dict = {name: jnp.zeros(shape) for name, shape in beta_components.items()}
    global_intercept = jnp.zeros(1)

    X_train_jax = jnp.array(X_train_s)
    X_test_jax = jnp.array(X_test_s)
    int_train_idx_jax = jnp.array(int_train_idx)
    int_test_idx_jax = jnp.array(int_test_idx)
    beta_train_idx_jax = jnp.array(beta_train_idx)
    beta_test_idx_jax = jnp.array(beta_test_idx)
    y_train_jax = jnp.array(y_train)

    def compute_logits(int_params, beta_params, global_int, int_idx, beta_idx, X):
        cell_intercept = int_decomp.lookup_flat(int_idx, int_params)
        beta = beta_decomp.lookup_flat(beta_idx, beta_params)
        return jnp.sum(X * beta, axis=-1) + cell_intercept[..., 0] + global_int[0]

    l2_int = config.get('l2_int', 30.0)
    l2_beta = config.get('l2_beta', 80.0)

    def loss_fn(params, int_idx, beta_idx, X, y, N):
        int_p, beta_p, global_int = params
        logits = compute_logits(int_p, beta_p, global_int, int_idx, beta_idx, X)
        bce = jnp.mean(jnp.logaddexp(0, logits) - y * logits)
        l2_int_reg = sum(0.5 * jnp.sum(p ** 2) / l2_int for p in int_p.values())
        l2_beta_reg = sum(0.5 * jnp.sum(p ** 2) / l2_beta for p in beta_p.values())
        return bce + (l2_int_reg + l2_beta_reg) / N

    params = (int_params_dict, beta_params_dict, global_intercept)

    batch_size = config.get('batch_size', 32768)
    n_epochs = config.get('epochs', 800)
    lr_peak = config.get('lr', 0.012)

    lr_scale = np.sqrt(batch_size / 2048)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0001 * lr_scale,
        peak_value=lr_peak * lr_scale,
        warmup_steps=50,
        decay_steps=n_epochs - 50,
        end_value=0.0001 * lr_scale
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

    int_p, beta_p, global_int = params
    test_logits = compute_logits(int_p, beta_p, global_int, int_test_idx_jax, beta_test_idx_jax, X_test_jax)

    return np.array(test_logits), int_bin_data, beta_bin_data


def learn_weights_with_loo(train_logits_list, y_train, X_train_s, sorted_idx,
                           weight_config, n_models):
    """Learn decomposed local stacking weights with LOO approximation.

    Uses leverage-based LOO reweighting: weight_i = 1/(1 - h_ii)
    where h_ii ≈ n_models / n_cell for the cell containing sample i.
    """
    N = len(y_train)

    # Build weight lattice
    weight_features = sorted_idx[:weight_config['weight_dims']].tolist()
    w_dims = []
    weight_bin_data = {}
    edges_dict = {}

    for i, feat_idx in enumerate(weight_features):
        n_bins = weight_config['weight_bins'][i]
        feature_vals = X_train_s[:, feat_idx]
        bins = percentile_edges(feature_vals, n_bins)
        n_actual = len(bins) - 1
        dim_name = f"w{i}"
        w_dims.append(Dimension(dim_name, n_actual))
        train_idx = np.clip(np.digitize(feature_vals, bins[1:-1]), 0, n_actual - 1)
        weight_bin_data[dim_name] = train_idx
        edges_dict[feat_idx] = bins

    weight_interactions = Interactions(dimensions=w_dims)
    weight_order = weight_config.get('weight_order', 1)
    weight_decomp = Decomposed(
        interactions=weight_interactions,
        param_shape=[n_models],
        name="weights",
        max_order=weight_order
    )

    weight_active = list(range(weight_order + 1))
    weight_components = {k: v for k, v in weight_decomp._tensor_part_shapes.items()
                       if weight_decomp.component_order(k) in weight_active}

    weight_params = {name: jnp.zeros(shape) for name, shape in weight_components.items()}

    # Prior variance for regularization
    prior_vars = {}
    n_bins = weight_config['weight_bins'][0]
    for name in weight_components:
        order = weight_decomp.component_order(name)
        avg_count = N / (n_bins ** order) if order > 0 else N
        tau = 1.0 / np.sqrt(max(avg_count, 1))
        prior_vars[name] = tau ** 2

    dim_names = [d.name for d in w_dims]
    train_idx = jnp.stack([jnp.array(weight_bin_data[name]) for name in dim_names], axis=-1)

    # Compute cell counts for leverage
    cell_key = tuple(weight_bin_data[name] for name in dim_names)
    cell_ids = np.ravel_multi_index(cell_key, [d.cardinality for d in w_dims])
    unique_cells, cell_counts = np.unique(cell_ids, return_counts=True)
    count_map = dict(zip(unique_cells, cell_counts))
    n_per_cell = np.array([count_map[c] for c in cell_ids])

    # Leverage approximation: h_ii ≈ n_models / n_cell
    leverage = n_models / n_per_cell
    leverage = np.clip(leverage, 0, 0.9)  # Cap to avoid division issues
    loo_weight = 1.0 / (1.0 - leverage)
    loo_weight = loo_weight / loo_weight.mean()  # Normalize
    loo_weight_j = jnp.array(loo_weight)

    logits_stack = jnp.stack(train_logits_list, axis=-1)
    y_j = jnp.array(y_train)

    def loss_fn(params):
        w_raw = weight_decomp.lookup_flat(train_idx, params)
        w = jax.nn.softmax(w_raw, axis=-1)
        ensemble_logits = jnp.sum(logits_stack * w, axis=-1)

        # BCE with LOO reweighting
        bce_per_obs = jnp.logaddexp(0, ensemble_logits) - y_j * ensemble_logits
        bce = jnp.mean(bce_per_obs * loo_weight_j)

        # L2 regularization on non-global components
        l2 = sum(0.5 * jnp.sum(p ** 2) / (prior_vars.get(n, 1.0) + 1e-8)
                for n, p in params.items() if weight_decomp.component_order(n) > 0)

        return bce + l2 / N

    opt = optax.adam(0.01)
    opt_state = opt.init(weight_params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    n_steps = weight_config.get('weight_steps', 2000)
    for i in range(n_steps):
        weight_params, opt_state, loss = step(weight_params, opt_state)

    return weight_params, weight_decomp, edges_dict, w_dims


def apply_weights(test_logits_list, X_test_s, sorted_idx, weight_params,
                  weight_decomp, edges_dict, w_dims, weight_config):
    """Apply learned weights to test data."""
    weight_features = sorted_idx[:weight_config['weight_dims']].tolist()

    test_indices = {}
    for i, feat_idx in enumerate(weight_features):
        bins = edges_dict[feat_idx]
        n_actual = len(bins) - 1
        dim_name = f"w{i}"
        test_indices[dim_name] = np.clip(
            np.digitize(X_test_s[:, feat_idx], bins[1:-1]), 0, n_actual - 1
        )

    dim_names = [d.name for d in w_dims]
    test_idx = jnp.stack([jnp.array(test_indices[name]) for name in dim_names], axis=-1)

    w_raw = weight_decomp.lookup_flat(test_idx, weight_params)
    w = jax.nn.softmax(w_raw, axis=-1)

    logits_stack = jnp.stack(test_logits_list, axis=-1)
    ensemble_logits = jnp.sum(logits_stack * w, axis=-1)

    # Compute weight statistics
    w_np = np.array(w)
    w_std = np.std(w_np, axis=0).mean()

    return np.array(ensemble_logits), w_std


# Base model configurations
BASE_CONFIGS = [
    {
        'name': 'best_4d_o2',
        'int_dims': 4, 'int_bins': [10, 8, 10, 6],
        'int_order': 2, 'int_active_orders': [0, 1, 2],
        'beta_dims': 6, 'beta_bins': [8, 6, 8, 4, 8, 4],
        'lr': 0.012, 'epochs': 800,
        'l2_int': 30.0, 'l2_beta': 80.0,
    },
    {
        'name': 'deep_3d_o3',
        'int_dims': 3, 'int_bins': [8, 6, 8],
        'int_order': 3, 'int_active_orders': [0, 1, 2, 3],
        'beta_dims': 3, 'beta_bins': [4, 4, 4],
        'lr': 0.01, 'epochs': 900,
        'l2_int': 25.0, 'l2_beta': 100.0,
    },
    {
        'name': 'wide_8d_o1',
        'int_dims': 8, 'int_bins': [8, 6, 8, 5, 6, 5, 6, 5],
        'int_order': 1, 'int_active_orders': [0, 1],
        'beta_dims': 8, 'beta_bins': [6, 5, 6, 4, 5, 4, 5, 4],
        'lr': 0.012, 'epochs': 700,
        'l2_int': 40.0, 'l2_beta': 60.0,
    },
    {
        'name': 'hires_1d',
        'int_dims': 1, 'int_bins': [32],
        'int_order': 1, 'int_active_orders': [0, 1],
        'beta_dims': 4, 'beta_bins': [6, 5, 6, 4],
        'lr': 0.015, 'epochs': 600,
        'l2_int': 20.0, 'l2_beta': 80.0,
    },
]

WEIGHT_CONFIG = {
    'weight_dims': 3,
    'weight_bins': [8, 6, 8],
    'weight_order': 1,
    'weight_steps': 2000,
}


def run_experiment():
    print("\n" + "="*70)
    print("HIGGS - Ensemble with LOO-approximated stacking (no leakage)")
    print("="*70)

    K = len(BASE_CONFIGS)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    ensemble_aucs = []
    avg_aucs = []
    base_aucs_all = [[] for _ in range(K)]

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_full, y_full)):
        print(f"\n  Fold {fold_idx + 1}/5:", flush=True)

        X_train, X_test = X_full[train_idx], X_full[test_idx]
        y_train, y_test = y_full[train_idx], y_full[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train base models and collect test logits
        test_logits_all = []
        train_logits_all = []

        for k, config in enumerate(BASE_CONFIGS):
            print(f"    Base {k+1}/{K} ({config['name']})...", flush=True)

            # Train model once, get both train and test logits
            # Train logits will be used with LOO reweighting for weight learning
            test_logits, train_logits, _, _ = build_base_model_with_train(
                config, X_train_s, X_test_s, y_train, sorted_idx, n_features
            )
            test_logits_all.append(test_logits)
            train_logits_all.append(train_logits)

            test_probs = 1 / (1 + np.exp(-test_logits))
            base_auc = roc_auc_score(y_test, test_probs)
            base_aucs_all[k].append(base_auc)
            print(f"      AUC: {base_auc:.4f}")

        # Learn weights with LOO approximation
        print("    Learning weights with LOO approximation...", flush=True)
        weight_params, weight_decomp, edges_dict, w_dims = learn_weights_with_loo(
            train_logits_all, y_train, X_train_s, sorted_idx, WEIGHT_CONFIG, K
        )

        # Apply to test
        ensemble_logits, w_std = apply_weights(
            test_logits_all, X_test_s, sorted_idx, weight_params,
            weight_decomp, edges_dict, w_dims, WEIGHT_CONFIG
        )
        ensemble_probs = 1 / (1 + np.exp(-ensemble_logits))
        ensemble_auc = roc_auc_score(y_test, ensemble_probs)
        ensemble_aucs.append(ensemble_auc)
        print(f"    Ensemble AUC: {ensemble_auc:.4f} (w_std={w_std:.4f})")

        # Simple average baseline (logit averaging)
        avg_logits = np.mean(np.stack(test_logits_all, axis=-1), axis=-1)
        avg_probs = 1 / (1 + np.exp(-avg_logits))
        avg_auc = roc_auc_score(y_test, avg_probs)
        avg_aucs.append(avg_auc)
        print(f"    Simple avg AUC: {avg_auc:.4f}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Ensemble (LOO weights): {np.mean(ensemble_aucs):.4f} ± {np.std(ensemble_aucs):.4f}")
    print(f"  Simple logit avg:       {np.mean(avg_aucs):.4f} ± {np.std(avg_aucs):.4f}")
    print(f"\n  Base models:")
    for k, config in enumerate(BASE_CONFIGS):
        print(f"    {config['name']}: {np.mean(base_aucs_all[k]):.4f} ± {np.std(base_aucs_all[k]):.4f}")

    return np.mean(ensemble_aucs), np.mean(avg_aucs)


def build_base_model_with_train(config, X_train_s, X_test_s, y_train, sorted_idx, n_features):
    """Build base model and return both train and test logits."""
    offset = config.get('feature_offset', 0)

    int_features = sorted_idx[offset:offset + config['int_dims']].tolist()
    int_dims = []
    int_bin_data = {}

    for i, feat_idx in enumerate(int_features):
        n_bins = config['int_bins'][i]
        feature_vals = X_train_s[:, feat_idx]

        if config.get('use_tukey', False) and feat_idx in HEAVY_TAIL:
            bins = tukey_edges(feature_vals, max(n_bins - 2, 2))
        else:
            bins = percentile_edges(feature_vals, n_bins)

        n_actual = len(bins) - 1
        dim_name = f"i{i}"
        int_dims.append(Dimension(dim_name, n_actual))
        int_bin_data[dim_name] = {
            "bins": bins,
            "train": np.clip(np.digitize(feature_vals, bins[1:-1]), 0, n_actual - 1),
            "test": np.clip(np.digitize(X_test_s[:, feat_idx], bins[1:-1]), 0, n_actual - 1),
        }

    int_interactions = Interactions(dimensions=int_dims)
    int_order = config.get('int_order', 2)
    int_decomp = Decomposed(interactions=int_interactions, param_shape=[1], name="intercept", max_order=int_order)

    int_active = config.get('int_active_orders', list(range(int_order + 1)))
    int_components = {k: v for k, v in int_decomp._tensor_part_shapes.items()
                     if int_decomp.component_order(k) in int_active}

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
            "bins": bins,
            "train": np.clip(np.digitize(feature_vals, bins[1:-1]), 0, n_actual - 1),
            "test": np.clip(np.digitize(X_test_s[:, feat_idx], bins[1:-1]), 0, n_actual - 1),
        }

    beta_interactions = Interactions(dimensions=b_dims)
    beta_order = config.get('beta_order', 1)
    beta_decomp = Decomposed(interactions=beta_interactions, param_shape=[n_features], name="beta", max_order=beta_order)
    beta_active_orders = config.get('beta_active_orders', list(range(beta_order + 1)))
    beta_components = {k: v for k, v in beta_decomp._tensor_part_shapes.items()
                      if beta_decomp.component_order(k) in beta_active_orders}

    int_train_idx = np.stack([int_bin_data[d.name]["train"] for d in int_dims], axis=-1)
    int_test_idx = np.stack([int_bin_data[d.name]["test"] for d in int_dims], axis=-1)
    beta_train_idx = np.stack([beta_bin_data[d.name]["train"] for d in b_dims], axis=-1)
    beta_test_idx = np.stack([beta_bin_data[d.name]["test"] for d in b_dims], axis=-1)

    int_params_dict = {name: jnp.zeros(shape) for name, shape in int_components.items()}
    beta_params_dict = {name: jnp.zeros(shape) for name, shape in beta_components.items()}
    global_intercept = jnp.zeros(1)

    X_train_jax = jnp.array(X_train_s)
    X_test_jax = jnp.array(X_test_s)
    int_train_idx_jax = jnp.array(int_train_idx)
    int_test_idx_jax = jnp.array(int_test_idx)
    beta_train_idx_jax = jnp.array(beta_train_idx)
    beta_test_idx_jax = jnp.array(beta_test_idx)
    y_train_jax = jnp.array(y_train)

    def compute_logits(int_params, beta_params, global_int, int_idx, beta_idx, X):
        cell_intercept = int_decomp.lookup_flat(int_idx, int_params)
        beta = beta_decomp.lookup_flat(beta_idx, beta_params)
        return jnp.sum(X * beta, axis=-1) + cell_intercept[..., 0] + global_int[0]

    l2_int = config.get('l2_int', 30.0)
    l2_beta = config.get('l2_beta', 80.0)

    def loss_fn(params, int_idx, beta_idx, X, y, N):
        int_p, beta_p, global_int = params
        logits = compute_logits(int_p, beta_p, global_int, int_idx, beta_idx, X)
        bce = jnp.mean(jnp.logaddexp(0, logits) - y * logits)
        l2_int_reg = sum(0.5 * jnp.sum(p ** 2) / l2_int for p in int_p.values())
        l2_beta_reg = sum(0.5 * jnp.sum(p ** 2) / l2_beta for p in beta_p.values())
        return bce + (l2_int_reg + l2_beta_reg) / N

    params = (int_params_dict, beta_params_dict, global_intercept)

    batch_size = config.get('batch_size', 32768)
    n_epochs = config.get('epochs', 800)
    lr_peak = config.get('lr', 0.012)

    lr_scale = np.sqrt(batch_size / 2048)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0001 * lr_scale,
        peak_value=lr_peak * lr_scale,
        warmup_steps=50,
        decay_steps=n_epochs - 50,
        end_value=0.0001 * lr_scale
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

    int_p, beta_p, global_int = params
    train_logits = compute_logits(int_p, beta_p, global_int, int_train_idx_jax, beta_train_idx_jax, X_train_jax)
    test_logits = compute_logits(int_p, beta_p, global_int, int_test_idx_jax, beta_test_idx_jax, X_test_jax)

    return np.array(test_logits), np.array(train_logits), int_bin_data, beta_bin_data


if __name__ == "__main__":
    run_experiment()
