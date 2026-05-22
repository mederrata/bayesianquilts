"""Bioresponse: EBM-style interaction selection + theory-based regularization."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
from itertools import combinations
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def compute_interaction_strength(y, bin_i, bin_j, n_bins):
    """
    Compute interaction strength between two binned features.
    Uses ANOVA-like test: variance of cell means vs overall mean.
    """
    cell_means = np.zeros((n_bins, n_bins))
    cell_counts = np.zeros((n_bins, n_bins))

    for i in range(n_bins):
        for j in range(n_bins):
            mask = (bin_i == i) & (bin_j == j)
            if mask.sum() > 0:
                cell_means[i, j] = y[mask].mean()
                cell_counts[i, j] = mask.sum()

    # Overall mean
    overall_mean = y.mean()

    # Between-cell variance (interaction effect)
    valid_mask = cell_counts > 0
    if valid_mask.sum() < 4:
        return 0.0

    weighted_var = np.sum(cell_counts[valid_mask] * (cell_means[valid_mask] - overall_mean) ** 2)
    total_weight = np.sum(cell_counts[valid_mask])

    # Main effects (row and column means)
    row_means = np.zeros(n_bins)
    col_means = np.zeros(n_bins)
    for i in range(n_bins):
        mask_i = bin_i == i
        if mask_i.sum() > 0:
            row_means[i] = y[mask_i].mean()
    for j in range(n_bins):
        mask_j = bin_j == j
        if mask_j.sum() > 0:
            col_means[j] = y[mask_j].mean()

    # Interaction = cell effect - row effect - col effect + overall
    interaction_var = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if cell_counts[i, j] > 0:
                expected = row_means[i] + col_means[j] - overall_mean
                interaction = cell_means[i, j] - expected
                interaction_var += cell_counts[i, j] * interaction ** 2

    return interaction_var / (total_weight + 1e-8)


def run_bioresponse_hybrid():
    """Bioresponse with EBM-style selection + theory-based regularization."""
    print("\n" + "="*60)
    print("BIORESPONSE - Hybrid: EBM Selection + Theory Regularization")
    print("="*60)

    data = fetch_openml(data_id=4134, as_frame=True, parser="auto")
    df = data.frame

    X = df.drop(columns=["target"]).values.astype(np.float32)
    y = df["target"].astype(int).values

    N, p_orig = X.shape
    print(f"  N = {N}, p = {p_orig}")

    class_balance = y.mean()
    avg_fisher_weight = class_balance * (1 - class_balance)
    sigma_eff = 1 / np.sqrt(avg_fisher_weight)
    print(f"  Class balance: {class_balance:.3f}")
    print(f"  Effective σ: {sigma_eff:.3f}")

    # Fixed top features
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X)
    lr_full = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    lr_full.fit(X_full_s, y)
    importance = np.abs(lr_full.coef_[0])

    n_shape_features = 6  # More features for interaction candidates
    n_bins = 8
    top_features = np.argsort(importance)[::-1][:n_shape_features]
    print(f"  Top features: {top_features.tolist()}")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        N_train = len(y_train)

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Build bins for all top features
        train_bins = {}
        test_bins = {}

        for feat_idx in top_features:
            train_vals = X_train_s[:, feat_idx]
            test_vals = X_test_s[:, feat_idx]

            edges = np.percentile(train_vals, np.linspace(0, 100, n_bins + 1))
            train_bins[feat_idx] = np.clip(np.digitize(train_vals, edges[1:-1]), 0, n_bins - 1)
            test_bins[feat_idx] = np.clip(np.digitize(test_vals, edges[1:-1]), 0, n_bins - 1)

        # EBM-STYLE: Test all pairs for interaction strength
        pair_strengths = []
        for i, j in combinations(top_features, 2):
            strength = compute_interaction_strength(
                y_train, train_bins[i], train_bins[j], n_bins
            )
            pair_strengths.append((i, j, strength))

        pair_strengths.sort(key=lambda x: -x[2])

        # Select top interactions (those above median strength)
        if pair_strengths:
            median_strength = np.median([s for _, _, s in pair_strengths])
            selected_pairs = [(i, j) for i, j, s in pair_strengths if s > median_strength]
            print(f"    Selected {len(selected_pairs)} / {len(pair_strengths)} interactions")
        else:
            selected_pairs = []

        # Build lattice using Decomposed
        dimensions = []
        train_indices = {}
        test_indices = {}

        for feat_idx in top_features:
            dim_name = f"F{feat_idx}"
            train_indices[dim_name] = train_bins[feat_idx]
            test_indices[dim_name] = test_bins[feat_idx]
            dimensions.append(Dimension(dim_name, n_bins))

        interactions = Interactions(dimensions=dimensions)
        decomp = Decomposed(interactions=interactions, param_shape=[1], name="intercept")

        # Build interaction indices
        dim_names = [d.name for d in decomp._interactions._dimensions]
        train_int_idx = jnp.stack([jnp.array(train_indices[name]) for name in dim_names], axis=-1)
        test_int_idx = jnp.stack([jnp.array(test_indices[name]) for name in dim_names], axis=-1)

        X_train_j = jnp.array(X_train_s)
        X_test_j = jnp.array(X_test_s)
        y_train_j = jnp.array(y_train)

        # THEORY-BASED: Select components based on order and data support
        # Include: global, all main effects, and SELECTED pairwise only
        c = 0.5
        bound_factor_sq = c / (1 - c)

        active_components = []
        prior_vars = {}

        # Map selected pairs to component names
        selected_pair_names = set()
        for i, j in selected_pairs:
            # Component name format: intercept__F{i}_F{j}
            name1 = f"intercept__F{i}_F{j}"
            name2 = f"intercept__F{j}_F{i}"
            selected_pair_names.add(name1)
            selected_pair_names.add(name2)

        for name in decomp._tensor_parts.keys():
            order = decomp.component_order(name)

            # Include global and main effects
            if order <= 1:
                active_components.append(name)
                n_cells = n_bins ** order if order > 0 else 1
                avg_samples = N_train / n_cells
                tau_sq = bound_factor_sq * (sigma_eff ** 2) / avg_samples
                prior_vars[name] = tau_sq

            # Include only SELECTED pairwise interactions
            elif order == 2 and name in selected_pair_names:
                active_components.append(name)
                n_cells = n_bins ** 2
                avg_samples = N_train / n_cells
                tau_sq = bound_factor_sq * (sigma_eff ** 2) / avg_samples
                prior_vars[name] = tau_sq

        print(f"    Active: {len(active_components)} components (global + main + selected pairs)")

        intercept_params = {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in active_components}
        beta = jnp.zeros(p_orig)

        params = {"intercept": intercept_params, "beta": beta}

        tau_sq_beta = bound_factor_sq * (sigma_eff ** 2) / N_train
        print(f"    τ_beta: {np.sqrt(tau_sq_beta):.4f}")

        def loss_fn(params):
            int_vals = decomp.lookup_flat(train_int_idx, params["intercept"])
            intercept = int_vals[:, 0]
            logits = jnp.sum(X_train_j * params["beta"], axis=-1) + intercept
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2_int = 0.0
            for name, param in params["intercept"].items():
                tau_sq = prior_vars[name]
                l2_int += 0.5 * jnp.sum(param ** 2) / (tau_sq + 1e-8)

            l2_beta = 0.5 * jnp.sum(params["beta"] ** 2) / (tau_sq_beta + 1e-8)

            return bce + (l2_int + l2_beta) / N_train

        opt = optax.adam(0.01)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        for i in range(3000):
            params, opt_state, loss = step(params, opt_state)
            if i % 1000 == 0:
                print(f"    Step {i}: loss = {loss:.4f}")

        # Evaluate
        int_vals = decomp.lookup_flat(test_int_idx, params["intercept"])
        intercept = int_vals[:, 0]
        logits = jnp.sum(X_test_j * params["beta"], axis=-1) + intercept
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (hybrid): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Previous: 0.844, EBM: 0.866, LGBM: 0.872")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bioresponse_hybrid()
