"""Bioresponse: Strict theory-based generalization-preserving regularization."""
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


def run_bioresponse_theory():
    """Bioresponse with strict generalization-preserving theory."""
    print("\n" + "="*60)
    print("BIORESPONSE - Strict Theory-Based Regularization")
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

    # Use fewer dimensions with more bins for better coverage
    n_shape_features = 3
    n_bins = 10
    top_features = np.argsort(importance)[::-1][:n_shape_features]
    print(f"  Top features: {top_features.tolist()}")
    print(f"  Lattice: {n_shape_features} × {n_bins} = {n_bins**n_shape_features} cells")

    # Critical condition: need avg n_cell >> 1 for each interaction order
    N_per_fold = int(N * 0.8)
    for order in range(n_shape_features + 1):
        n_cells = n_bins ** order if order > 0 else 1
        avg_per_cell = N_per_fold / n_cells
        status = "OK" if avg_per_cell >= 10 else ("marginal" if avg_per_cell >= 5 else "SPARSE")
        print(f"    Order {order}: {n_cells} cells, ~{avg_per_cell:.0f}/cell [{status}]")

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

        # Build lattice dimensions
        dimensions = []
        train_indices = {}
        test_indices = {}

        for feat_idx in top_features:
            train_vals = X_train_s[:, feat_idx]
            test_vals = X_test_s[:, feat_idx]

            edges = np.percentile(train_vals, np.linspace(0, 100, n_bins + 1))
            train_bins = np.clip(np.digitize(train_vals, edges[1:-1]), 0, n_bins - 1)
            test_bins = np.clip(np.digitize(test_vals, edges[1:-1]), 0, n_bins - 1)

            dim_name = f"F{feat_idx}"
            train_indices[dim_name] = train_bins
            test_indices[dim_name] = test_bins
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

        # THEORY-BASED REGULARIZATION
        # τ² = c/(1-c) × σ²_eff / n_cell
        # where c controls the "bound factor" (c=0.5 gives balanced prior)
        c = 0.5
        bound_factor_sq = c / (1 - c)  # = 1 when c = 0.5

        # Select components based on critical condition
        # Only include if avg samples per cell >= threshold
        critical_threshold = 5
        active_components = []
        prior_vars = {}

        for name in decomp._tensor_parts.keys():
            order = decomp.component_order(name)

            # Compute cell count for this component
            if order == 0:
                n_cells = 1
            else:
                n_cells = n_bins ** order

            avg_samples = N_train / n_cells

            # Only include if above critical threshold
            if avg_samples >= critical_threshold:
                active_components.append(name)

                # Theory: τ² = bound_factor_sq × σ²_eff / n_cell
                tau_sq = bound_factor_sq * (sigma_eff ** 2) / avg_samples
                prior_vars[name] = tau_sq

        print(f"    Active components: {len(active_components)} / {len(decomp._tensor_parts)}")

        intercept_params = {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in active_components}
        beta = jnp.zeros(p_orig)

        params = {"intercept": intercept_params, "beta": beta}

        # Linear term prior: treat p features as p "cells"
        # τ²_beta = bound_factor_sq × σ²_eff / (N / p) = bound_factor_sq × σ²_eff × p / N
        # But this is too weak. Use N directly as "effective sample size per coefficient"
        tau_sq_beta = bound_factor_sq * (sigma_eff ** 2) / N_train
        tau_beta = np.sqrt(tau_sq_beta)

        print(f"    τ_beta: {tau_beta:.4f}")

        def loss_fn(params):
            int_vals = decomp.lookup_flat(train_int_idx, params["intercept"])
            intercept = int_vals[:, 0]
            logits = jnp.sum(X_train_j * params["beta"], axis=-1) + intercept
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # Theory-based regularization
            l2_int = 0.0
            for name, param in params["intercept"].items():
                tau_sq = prior_vars[name]
                l2_int += 0.5 * jnp.sum(param ** 2) / (tau_sq + 1e-8)

            l2_beta = 0.5 * jnp.sum(params["beta"] ** 2) / (tau_sq_beta + 1e-8)

            # Scale by 1/N to get proper posterior
            return bce + (l2_int + l2_beta) / N_train

        # Use cosine schedule
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.001,
            peak_value=0.02,
            warmup_steps=500,
            decay_steps=4500,
            end_value=0.001,
        )
        opt = optax.adam(learning_rate=schedule)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        for i in range(5000):
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
    print(f"\n  OURS (theory-based): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Previous: 0.844, EBM: 0.866, LGBM: 0.872")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bioresponse_theory()
