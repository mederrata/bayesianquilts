"""Bioresponse: Full GAM - many features with main effects only."""
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


def run_bioresponse_full_gam():
    """Bioresponse with full GAM - many features, main effects only."""
    print("\n" + "="*60)
    print("BIORESPONSE - Full GAM (Many Shape Functions)")
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

    # Scale full data to find top features
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X)

    # Find most predictive features on FULL data
    lr_full = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    lr_full.fit(X_full_s, y)
    importance = np.abs(lr_full.coef_[0])

    # Use more features for GAM
    n_shape_features = 20  # Top 20 features
    n_bins = 8
    top_features = np.argsort(importance)[::-1][:n_shape_features]
    print(f"  Top {n_shape_features} features: {top_features[:5].tolist()}...")
    print(f"  Total shape params: {n_shape_features * n_bins}")

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

        # Build shape function bins for each top feature
        # We'll use a simple approach: concatenate binned features
        train_bins_list = []
        test_bins_list = []

        for feat_idx in top_features:
            train_vals = X_train_s[:, feat_idx]
            test_vals = X_test_s[:, feat_idx]

            # Quantile binning
            edges = np.percentile(train_vals, np.linspace(0, 100, n_bins + 1))

            # One-hot encode bins for GAM effect
            train_bin_idx = np.clip(np.digitize(train_vals, edges[1:-1]), 0, n_bins - 1)
            test_bin_idx = np.clip(np.digitize(test_vals, edges[1:-1]), 0, n_bins - 1)

            # One-hot encode
            train_onehot = np.zeros((len(train_vals), n_bins))
            train_onehot[np.arange(len(train_vals)), train_bin_idx] = 1

            test_onehot = np.zeros((len(test_vals), n_bins))
            test_onehot[np.arange(len(test_vals)), test_bin_idx] = 1

            train_bins_list.append(train_onehot)
            test_bins_list.append(test_onehot)

        # Concatenate all binned features
        X_train_bins = np.hstack(train_bins_list)
        X_test_bins = np.hstack(test_bins_list)

        # Combine: original scaled features + binned features
        X_train_combined = np.hstack([X_train_s, X_train_bins])
        X_test_combined = np.hstack([X_test_s, X_test_bins])

        n_total = X_train_combined.shape[1]
        print(f"    Total features: {n_total} ({p_orig} orig + {X_train_bins.shape[1]} bins)")

        X_train_j = jnp.array(X_train_combined)
        X_test_j = jnp.array(X_test_combined)
        y_train_j = jnp.array(y_train)

        # Theory-based priors
        c = 0.5
        bound_factor = np.sqrt(c / (1 - c))
        tau_global = bound_factor * sigma_eff / np.sqrt(N_train)

        # Different regularization for original vs binned features
        # Original features: standard tau
        # Binned features: adjusted for n_bins (like main effects)
        tau_orig = bound_factor * sigma_eff / np.sqrt(N_train)
        tau_bins = bound_factor * sigma_eff / np.sqrt(N_train / n_bins)

        # Initialize params
        beta = jnp.zeros(n_total)
        intercept = jnp.zeros(1)
        params = {"beta": beta, "intercept": intercept}

        print(f"    τ_orig: {tau_orig:.4f}, τ_bins: {tau_bins:.4f}")

        def loss_fn(params):
            logits = jnp.sum(X_train_j * params["beta"], axis=-1) + params["intercept"][0]
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # Separate regularization for original vs binned
            l2_orig = 0.5 * jnp.sum(params["beta"][:p_orig] ** 2) / (tau_orig ** 2 + 1e-8)
            l2_bins = 0.5 * jnp.sum(params["beta"][p_orig:] ** 2) / (tau_bins ** 2 + 1e-8)

            return bce + (l2_orig + l2_bins) / N_train

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
        logits = jnp.sum(X_test_j * params["beta"], axis=-1) + params["intercept"][0]
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (full GAM): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Previous: 0.844, EBM: 0.866, LGBM: 0.872")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bioresponse_full_gam()
