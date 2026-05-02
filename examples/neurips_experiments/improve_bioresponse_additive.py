"""Bioresponse: Pure additive model - top features with shape functions, no interactions."""
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


def run_bioresponse_additive():
    """Bioresponse with pure additive shape functions."""
    print("\n" + "="*60)
    print("BIORESPONSE - Pure Additive (No Interactions)")
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

    # Parameters
    n_shape_features = 8  # Top features to get shape functions
    n_bins = 10  # Bins per feature (finer resolution)

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

        # Find most predictive features via L1-LR
        lr = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        lr.fit(X_train_s, y_train)
        importance = np.abs(lr.coef_[0])
        top_features = np.argsort(importance)[::-1][:n_shape_features]
        print(f"    Top features: {top_features.tolist()}")

        # Build bin indices for each feature (no Decomposed - direct implementation)
        train_bins = np.zeros((N_train, n_shape_features), dtype=int)
        test_bins = np.zeros((len(y_test), n_shape_features), dtype=int)

        for i, feat_idx in enumerate(top_features):
            train_vals = X_train_s[:, feat_idx]
            test_vals = X_test_s[:, feat_idx]
            edges = np.percentile(train_vals, np.linspace(0, 100, n_bins + 1))
            train_bins[:, i] = np.clip(np.digitize(train_vals, edges[1:-1]), 0, n_bins - 1)
            test_bins[:, i] = np.clip(np.digitize(test_vals, edges[1:-1]), 0, n_bins - 1)

        train_bins_j = jnp.array(train_bins)
        test_bins_j = jnp.array(test_bins)
        X_train_j = jnp.array(X_train_s)
        X_test_j = jnp.array(X_test_s)
        y_train_j = jnp.array(y_train)

        # Theory-based priors
        c = 0.5
        bound_factor = np.sqrt(c / (1 - c))
        tau_global = bound_factor * sigma_eff / np.sqrt(N_train)

        # Shape function prior: each bin has ~N/n_bins samples
        avg_bin_count = N_train / n_bins
        tau_shape = bound_factor * sigma_eff / np.sqrt(avg_bin_count)

        tau_beta = bound_factor * sigma_eff / np.sqrt(N_train)

        print(f"    τ_global: {tau_global:.4f}, τ_shape: {tau_shape:.4f}, τ_beta: {tau_beta:.4f}")
        print(f"    Shape params: {n_shape_features} × {n_bins} = {n_shape_features * n_bins}")

        # Initialize parameters
        # Shape functions: (n_features, n_bins)
        # Global intercept: scalar
        # Linear coefficients: (p_orig,)
        params = {
            "intercept": jnp.zeros(1),
            "shapes": jnp.zeros((n_shape_features, n_bins)),
            "beta": jnp.zeros(p_orig),
        }

        def loss_fn(params):
            # Additive shape functions
            shape_contrib = jnp.zeros(N_train)
            for i in range(n_shape_features):
                shape_contrib += params["shapes"][i, train_bins_j[:, i]]

            logits = params["intercept"][0] + shape_contrib + jnp.sum(X_train_j * params["beta"], axis=-1)
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # Regularization
            l2_intercept = 0.5 * params["intercept"][0] ** 2 / (tau_global ** 2 + 1e-8)
            l2_shapes = 0.5 * jnp.sum(params["shapes"] ** 2) / (tau_shape ** 2 + 1e-8)
            l2_beta = 0.5 * jnp.sum(params["beta"] ** 2) / (tau_beta ** 2 + 1e-8)

            return bce + (l2_intercept + l2_shapes + l2_beta) / N_train

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
        shape_contrib = jnp.zeros(len(y_test))
        for i in range(n_shape_features):
            shape_contrib += params["shapes"][i, test_bins_j[:, i]]

        logits = params["intercept"][0] + shape_contrib + jnp.sum(X_test_j * params["beta"], axis=-1)
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (pure additive): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Previous: 0.844, EBM: 0.866, LGBM: 0.872")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bioresponse_additive()
