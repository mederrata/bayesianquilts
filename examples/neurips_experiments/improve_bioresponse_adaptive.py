"""Bioresponse: Adaptive interaction order with horseshoe-like priors."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def run_bioresponse_adaptive():
    """Bioresponse with adaptive interaction order via horseshoe priors."""
    print("\n" + "="*60)
    print("BIORESPONSE - Adaptive Interaction Order")
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

    # Find categorical features (≤10 unique values)
    cat_features = []
    cat_levels = {}
    for i in range(X.shape[1]):
        unique_vals = np.unique(X[:, i])
        if len(unique_vals) <= 10:
            cat_features.append(i)
            cat_levels[i] = unique_vals

    print(f"  Found {len(cat_features)} categorical features")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        N_train = len(y_train)

        # Scale for linear part
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Find most predictive categorical features via L1-LR
        lr = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        lr.fit(X_train_s, y_train)
        importance = np.abs(lr.coef_[0])

        # Rank categorical features by importance
        cat_importance = [(i, importance[i], len(cat_levels[i])) for i in cat_features]
        cat_importance.sort(key=lambda x: -x[1])

        # Select top categorical features - use fewer dimensions for higher-order interactions
        selected_cat = []
        for idx, imp, n_levels in cat_importance:
            if n_levels >= 2:
                selected_cat.append((idx, n_levels))
                if len(selected_cat) == 4:  # 4 dims allows meaningful higher-order
                    break

        print(f"    Top cat features: {[(f'D{idx+1}', n) for idx, n in selected_cat]}")

        # Build dimensions
        dimensions = []
        train_indices = {}
        test_indices = {}

        for feat_idx, n_levels in selected_cat:
            unique_vals = cat_levels[feat_idx]
            val_to_idx = {v: i for i, v in enumerate(unique_vals)}

            train_labels = np.array([val_to_idx.get(v, 0) for v in X_train[:, feat_idx]])
            test_labels = np.array([val_to_idx.get(v, 0) for v in X_test[:, feat_idx]])

            dim_name = f"D{feat_idx+1}"
            train_indices[dim_name] = train_labels
            test_indices[dim_name] = test_labels
            dimensions.append(Dimension(dim_name, n_levels))

        total_cells = 1
        for d in dimensions:
            total_cells *= d.cardinality
        print(f"    Total dimensions: {len(dimensions)}, Total cells: {total_cells}")

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

        # Use ALL components but with adaptive horseshoe-like priors
        all_components = list(decomp._tensor_parts.keys())
        intercept_params = {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in all_components}
        beta = jnp.zeros(p_orig)

        # Learn local shrinkage per component
        local_log_scales = {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in all_components}

        params = {"intercept": intercept_params, "beta": beta, "local_scales": local_log_scales}

        # Global scale per order (horseshoe global shrinkage)
        tau_by_order = {}
        for name in all_components:
            order = decomp.component_order(name)
            if order == 0:
                tau_by_order[name] = tau_global
            else:
                # Stronger shrinkage for higher order
                avg_count = N_train / (total_cells ** (order / len(dimensions)))
                tau_by_order[name] = bound_factor * sigma_eff / np.sqrt(max(avg_count, 1))

        tau_beta = bound_factor * sigma_eff / np.sqrt(N_train)
        print(f"    τ_global: {tau_global:.4f}, τ_beta: {tau_beta:.4f}")
        print(f"    Total components: {len(all_components)}")

        def loss_fn(params):
            int_vals = decomp.lookup_flat(train_int_idx, params["intercept"])
            intercept = int_vals[:, 0]
            logits = jnp.sum(X_train_j * params["beta"], axis=-1) + intercept
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # Horseshoe-like regularization
            reg = 0.0
            for name, param in params["intercept"].items():
                local_scale = jax.nn.softplus(params["local_scales"][name]) + 1e-6
                tau = tau_by_order[name]
                # Horseshoe: p(theta) propto 1/sqrt(1 + (theta/(tau*lambda))^2)
                # Approximated as: |theta| / (tau*lambda) + log(tau*lambda)
                scale = tau * local_scale
                reg += jnp.sum(jnp.abs(param) / scale + jnp.log(scale))

            l2_beta = 0.5 * jnp.sum(params["beta"] ** 2) / (tau_beta ** 2 + 1e-8)

            return bce + (reg + l2_beta) / N_train

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
                # Count active components (|param| > 0.01)
                active = sum(
                    (jnp.abs(params["intercept"][name]) > 0.01).sum()
                    for name in all_components
                )
                print(f"    Step {i}: loss = {loss:.4f}, active params = {active}")

        # Evaluate
        int_vals = decomp.lookup_flat(test_int_idx, params["intercept"])
        intercept = int_vals[:, 0]
        logits = jnp.sum(X_test_j * params["beta"], axis=-1) + intercept
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

        # Show which orders are active
        for name in all_components:
            order = decomp.component_order(name)
            active = (jnp.abs(params["intercept"][name]) > 0.01).sum()
            total = params["intercept"][name].size
            if active > 0:
                print(f"      {name} (order {order}): {active}/{total} active")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (adaptive order): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Previous: 0.844, EBM: 0.866, LGBM: 0.872")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bioresponse_adaptive()
