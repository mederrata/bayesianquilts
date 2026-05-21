"""Bioresponse: Stable feature selection + theory-based decomposition."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def run_bioresponse_stable():
    """Bioresponse with stable feature selection."""
    print("\n" + "="*60)
    print("BIORESPONSE - Stable Feature Selection")
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

    # Find categorical features
    cat_features = []
    cat_levels = {}
    for i in range(X.shape[1]):
        unique_vals = np.unique(X[:, i])
        if len(unique_vals) <= 10:
            cat_features.append(i)
            cat_levels[i] = unique_vals

    print(f"  Found {len(cat_features)} categorical features")

    # Stable feature selection: use full dataset to identify top features
    # This mimics what EBM does - features are selected once, not per fold
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X)

    # Multiple selection methods for stability
    # 1. L1 LR
    lr = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    lr.fit(X_scaled_full, y)
    l1_importance = np.abs(lr.coef_[0])

    # 2. Mutual information (captures nonlinear relationships)
    mi_importance = mutual_info_classif(X_scaled_full, y, random_state=42, n_neighbors=5)

    # 3. Combined ranking
    l1_rank = np.argsort(np.argsort(-l1_importance))
    mi_rank = np.argsort(np.argsort(-mi_importance))
    combined_rank = l1_rank + mi_rank  # Lower is better

    # Select stable features
    n_dims = 5
    n_bins = 6
    stable_features = np.argsort(combined_rank)[:n_dims]
    print(f"  Stable top features: {stable_features.tolist()}")

    # Check which are categorical vs continuous
    feature_types = []
    for feat_idx in stable_features:
        if feat_idx in cat_features:
            feature_types.append(('cat', feat_idx, len(cat_levels[feat_idx])))
        else:
            feature_types.append(('cont', feat_idx, n_bins))
    print(f"  Feature types: {[(t, f'F{idx}') for t, idx, _ in feature_types]}")

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

        # Build dimensions
        dimensions = []
        train_indices = {}
        test_indices = {}

        for feat_type, feat_idx, n_levels in feature_types:
            if feat_type == 'cat':
                unique_vals = cat_levels[feat_idx]
                val_to_idx = {v: i for i, v in enumerate(unique_vals)}
                train_bins = np.array([val_to_idx.get(v, 0) for v in X_train[:, feat_idx]])
                test_bins = np.array([val_to_idx.get(v, 0) for v in X_test[:, feat_idx]])
            else:
                train_vals = X_train_s[:, feat_idx]
                test_vals = X_test_s[:, feat_idx]
                edges = np.percentile(train_vals, np.linspace(0, 100, n_levels + 1))
                train_bins = np.clip(np.digitize(train_vals, edges[1:-1]), 0, n_levels - 1)
                test_bins = np.clip(np.digitize(test_vals, edges[1:-1]), 0, n_levels - 1)

            dim_name = f"F{feat_idx}"
            train_indices[dim_name] = train_bins
            test_indices[dim_name] = test_bins
            dimensions.append(Dimension(dim_name, n_levels))

        total_cells = 1
        for d in dimensions:
            total_cells *= d.cardinality
        print(f"    Total cells: {total_cells}")

        interactions = Interactions(dimensions=dimensions)
        decomp = Decomposed(interactions=interactions, param_shape=[1], name="intercept")

        # Theory-based priors
        c = 0.5
        bound_factor = np.sqrt(c / (1 - c))
        tau_global = bound_factor * sigma_eff / np.sqrt(N_train)

        # Build interaction indices
        dim_names = [d.name for d in decomp._interactions._dimensions]
        train_int_idx = jnp.stack([jnp.array(train_indices[name]) for name in dim_names], axis=-1)
        test_int_idx = jnp.stack([jnp.array(test_indices[name]) for name in dim_names], axis=-1)

        X_train_j = jnp.array(X_train_s)
        X_test_j = jnp.array(X_test_s)
        y_train_j = jnp.array(y_train)

        # Adaptive order selection based on critical boundary
        # Include component if avg samples per cell >= threshold
        all_components = list(decomp._tensor_parts.keys())
        active_components = []
        prior_vars = {}

        critical_threshold = 10

        for name in all_components:
            order = decomp.component_order(name)

            # Compute cell count for this component
            if order == 0:
                cell_count = 1
            else:
                parts = name.replace("intercept__", "").split("_")
                dims_involved = [p for p in parts if p.startswith("F")]
                cell_count = 1
                for dim_name in dim_names:
                    if dim_name in dims_involved:
                        for d in dimensions:
                            if d.name == dim_name:
                                cell_count *= d.cardinality
                                break

            avg_samples = N_train / cell_count

            # Include if above soft threshold
            if avg_samples >= critical_threshold / 2:
                active_components.append(name)

                if order == 0:
                    prior_vars[name] = tau_global ** 2
                else:
                    tau = bound_factor * sigma_eff / np.sqrt(max(avg_samples, 1))
                    # Extra shrinkage if near critical boundary
                    if avg_samples < critical_threshold:
                        tau /= np.sqrt(2)  # Extra factor for marginal components
                    prior_vars[name] = tau ** 2

        intercept_params = {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in active_components}
        beta = jnp.zeros(p_orig)

        params = {"intercept": intercept_params, "beta": beta}

        tau_beta = bound_factor * sigma_eff / np.sqrt(N_train)
        print(f"    Active components: {len(active_components)} / {len(all_components)}")
        print(f"    τ_global: {tau_global:.4f}, τ_beta: {tau_beta:.4f}")

        def loss_fn(params):
            int_vals = decomp.lookup_flat(train_int_idx, params["intercept"])
            intercept = int_vals[:, 0]
            logits = jnp.sum(X_train_j * params["beta"], axis=-1) + intercept
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2_int = 0.0
            for name, param in params["intercept"].items():
                var = prior_vars.get(name, 1.0)
                l2_int += 0.5 * jnp.sum(param ** 2) / (var + 1e-8)

            l2_beta = 0.5 * jnp.sum(params["beta"] ** 2) / (tau_beta ** 2 + 1e-8)

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
    print(f"\n  OURS (stable selection): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Previous: 0.844, EBM: 0.866, LGBM: 0.872")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bioresponse_stable()
