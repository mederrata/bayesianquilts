"""Bioresponse: Shape functions + selected pairwise interactions."""
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
from itertools import combinations


def run_bioresponse_pairs():
    """Bioresponse with shape functions + pairwise interactions."""
    print("\n" + "="*60)
    print("BIORESPONSE - Shape Functions + Pairwise Interactions")
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

    # Find most predictive features
    lr_full = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    lr_full.fit(X_full_s, y)
    importance = np.abs(lr_full.coef_[0])

    # Top features for shape functions
    n_shape_features = 10
    n_bins = 6
    n_pair_features = 5  # Top 5 for pairwise
    top_features = np.argsort(importance)[::-1][:n_shape_features]
    pair_features = top_features[:n_pair_features]

    print(f"  Shape features: {n_shape_features} × {n_bins} bins")
    print(f"  Pairwise features: {n_pair_features} → {len(list(combinations(range(n_pair_features), 2)))} pairs")

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

        # Build features:
        # 1. Original scaled features
        # 2. One-hot bins for shape functions
        # 3. Interaction bins for top pairs

        train_bins_list = []
        test_bins_list = []
        train_bin_idx = {}
        test_bin_idx = {}

        # Shape functions for all top features
        for feat_idx in top_features:
            train_vals = X_train_s[:, feat_idx]
            test_vals = X_test_s[:, feat_idx]

            edges = np.percentile(train_vals, np.linspace(0, 100, n_bins + 1))
            train_idx_f = np.clip(np.digitize(train_vals, edges[1:-1]), 0, n_bins - 1)
            test_idx_f = np.clip(np.digitize(test_vals, edges[1:-1]), 0, n_bins - 1)

            # Store bin indices for pairwise interactions
            if feat_idx in pair_features:
                train_bin_idx[feat_idx] = train_idx_f
                test_bin_idx[feat_idx] = test_idx_f

            # One-hot encode
            train_onehot = np.zeros((len(train_vals), n_bins))
            train_onehot[np.arange(len(train_vals)), train_idx_f] = 1

            test_onehot = np.zeros((len(test_vals), n_bins))
            test_onehot[np.arange(len(test_vals)), test_idx_f] = 1

            train_bins_list.append(train_onehot)
            test_bins_list.append(test_onehot)

        # Pairwise interactions for top 5 features
        for i, j in combinations(pair_features, 2):
            # Create interaction bins (cartesian product of bin indices)
            # Total bins for pair: n_bins * n_bins
            n_pair_bins = n_bins * n_bins

            train_pair_idx = train_bin_idx[i] * n_bins + train_bin_idx[j]
            test_pair_idx = test_bin_idx[i] * n_bins + test_bin_idx[j]

            train_pair_onehot = np.zeros((len(y_train), n_pair_bins))
            train_pair_onehot[np.arange(len(y_train)), train_pair_idx] = 1

            test_pair_onehot = np.zeros((len(y_test), n_pair_bins))
            test_pair_onehot[np.arange(len(y_test)), test_pair_idx] = 1

            train_bins_list.append(train_pair_onehot)
            test_bins_list.append(test_pair_onehot)

        # Concatenate all binned features
        X_train_bins = np.hstack(train_bins_list)
        X_test_bins = np.hstack(test_bins_list)

        # Combine: original scaled features + binned features
        X_train_combined = np.hstack([X_train_s, X_train_bins])
        X_test_combined = np.hstack([X_test_s, X_test_bins])

        n_main_bins = n_shape_features * n_bins
        n_pair_bins_total = len(list(combinations(pair_features, 2))) * n_bins * n_bins
        n_total = X_train_combined.shape[1]
        print(f"    Features: {p_orig} orig + {n_main_bins} main + {n_pair_bins_total} pair = {n_total}")

        X_train_j = jnp.array(X_train_combined)
        X_test_j = jnp.array(X_test_combined)
        y_train_j = jnp.array(y_train)

        # Theory-based priors
        c = 0.5
        bound_factor = np.sqrt(c / (1 - c))

        tau_orig = bound_factor * sigma_eff / np.sqrt(N_train)
        tau_main = bound_factor * sigma_eff / np.sqrt(N_train / n_bins)
        tau_pair = bound_factor * sigma_eff / np.sqrt(N_train / (n_bins * n_bins))

        # Initialize params
        beta = jnp.zeros(n_total)
        intercept = jnp.zeros(1)
        params = {"beta": beta, "intercept": intercept}

        print(f"    τ_orig: {tau_orig:.4f}, τ_main: {tau_main:.4f}, τ_pair: {tau_pair:.4f}")

        def loss_fn(params):
            logits = jnp.sum(X_train_j * params["beta"], axis=-1) + params["intercept"][0]
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # Separate regularization
            l2_orig = 0.5 * jnp.sum(params["beta"][:p_orig] ** 2) / (tau_orig ** 2 + 1e-8)
            l2_main = 0.5 * jnp.sum(params["beta"][p_orig:p_orig + n_main_bins] ** 2) / (tau_main ** 2 + 1e-8)
            l2_pair = 0.5 * jnp.sum(params["beta"][p_orig + n_main_bins:] ** 2) / (tau_pair ** 2 + 1e-8)

            return bce + (l2_orig + l2_main + l2_pair) / N_train

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
    print(f"\n  OURS (shape + pairs): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Previous: 0.844, EBM: 0.866, LGBM: 0.872")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bioresponse_pairs()
