"""Bioresponse: improved theory-based model with top L1-selected features."""
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


def run_bioresponse():
    """Bioresponse with top L1-selected features for lattice."""
    print("\n" + "="*60)
    print("BIORESPONSE - L1 Feature Selection")
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

    # Strategy: Use top L1-selected features with more bins
    # d=2 dims with L=10 bins = 100 cells
    # Constraint: L < sqrt(N) for p=1 → L < 61, so L=10 is safe
    d_lat = 2
    n_bins = 10
    print(f"\n  Lattice: {d_lat} L1 features × {n_bins} bins = {n_bins**d_lat} cells")
    print(f"  Bin constraint check: L={n_bins} < N^(1/{d_lat}) = {N**(1/d_lat):.1f} ✓")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        N_train = len(y_train)
        n_features = X_train_s.shape[1]

        # Use L1-LR to find top features for lattice
        lr = LogisticRegression(max_iter=500, C=0.1, solver='saga', penalty='l1')
        lr.fit(X_train_s, y_train)
        importance = np.abs(lr.coef_[0])
        top_idx = np.argsort(importance)[::-1][:d_lat]
        print(f"    Top features for lattice: {top_idx}")

        # Discretize top features into bins
        dimensions = []
        train_indices = {}
        test_indices = {}

        for i, feat_idx in enumerate(top_idx):
            percentiles = np.linspace(0, 100, n_bins + 1)
            edges = np.percentile(X_train_s[:, feat_idx], percentiles)
            train_indices[f"f{i}"] = np.clip(
                np.digitize(X_train_s[:, feat_idx], edges[1:-1]), 0, n_bins - 1
            )
            test_indices[f"f{i}"] = np.clip(
                np.digitize(X_test_s[:, feat_idx], edges[1:-1]), 0, n_bins - 1
            )
            dimensions.append(Dimension(f"f{i}", n_bins))

        interactions = Interactions(dimensions=dimensions)
        decomp = Decomposed(interactions=interactions, param_shape=[1], name="intercept")

        # Theory-based priors
        c = 0.5
        bound_factor = np.sqrt(c / (1 - c))

        # Count observations per cell
        cell_counts = np.zeros([n_bins] * d_lat)
        for i in range(N_train):
            idx = tuple(train_indices[f"f{j}"][i] for j in range(d_lat))
            cell_counts[idx] += 1

        tau_global = bound_factor * sigma_eff / np.sqrt(N_train)

        # Build interaction indices
        dim_names = [d.name for d in decomp._interactions._dimensions]
        train_int_idx = jnp.stack([jnp.array(train_indices[name]) for name in dim_names], axis=-1)
        test_int_idx = jnp.stack([jnp.array(test_indices[name]) for name in dim_names], axis=-1)

        X_train_j = jnp.array(X_train_s)
        X_test_j = jnp.array(X_test_s)
        y_train_j = jnp.array(y_train)

        # Initialize parameters - order 2 (include pairwise interactions)
        active_components = [name for name in decomp._tensor_parts.keys() if decomp.component_order(name) <= 2]
        intercept_params = {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in active_components}
        beta = jnp.zeros(n_features)

        params = {"intercept": intercept_params, "beta": beta}

        # Prior variances
        prior_vars = {}
        for name in active_components:
            order = decomp.component_order(name)
            if order == 0:
                prior_vars[name] = tau_global ** 2
            elif order == 1:
                # Main effects: use average count per level
                avg_count = N_train / n_bins
                tau = bound_factor * sigma_eff / np.sqrt(max(avg_count, 1))
                prior_vars[name] = tau ** 2
            else:
                # Order 2: use minimum cell count
                min_cell = max(cell_counts.min(), 1)
                tau_int = bound_factor * sigma_eff / np.sqrt(max(min_cell, 1))
                prior_vars[name] = tau_int ** 2

        tau_beta = bound_factor * sigma_eff / np.sqrt(N_train)
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
    print(f"\n  OURS (L1 + order 2): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Previous: 0.844, LGBM: 0.872")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bioresponse()
