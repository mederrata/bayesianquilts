"""Bioresponse: Sequential/residual fitting inspired by EBM."""
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


def run_bioresponse_sequential():
    """Bioresponse with sequential/residual fitting (EBM-inspired)."""
    print("\n" + "="*60)
    print("BIORESPONSE - Sequential Fitting (EBM-inspired)")
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

    n_shape_features = 10
    n_bins = 8
    top_features = np.argsort(importance)[::-1][:n_shape_features]
    print(f"  Shape features: {n_shape_features} × {n_bins} bins")

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

        # Build bin indices for all features
        train_bin_idx = {}
        test_bin_idx = {}
        train_onehot = {}
        test_onehot = {}

        for feat_idx in top_features:
            train_vals = X_train_s[:, feat_idx]
            test_vals = X_test_s[:, feat_idx]

            edges = np.percentile(train_vals, np.linspace(0, 100, n_bins + 1))
            t_idx = np.clip(np.digitize(train_vals, edges[1:-1]), 0, n_bins - 1)
            v_idx = np.clip(np.digitize(test_vals, edges[1:-1]), 0, n_bins - 1)

            train_bin_idx[feat_idx] = t_idx
            test_bin_idx[feat_idx] = v_idx

            # One-hot
            t_oh = np.zeros((N_train, n_bins))
            t_oh[np.arange(N_train), t_idx] = 1
            v_oh = np.zeros((len(y_test), n_bins))
            v_oh[np.arange(len(y_test)), v_idx] = 1

            train_onehot[feat_idx] = t_oh
            test_onehot[feat_idx] = v_oh

        # Theory priors
        c = 0.5
        bound_factor = np.sqrt(c / (1 - c))
        tau_global = bound_factor * sigma_eff / np.sqrt(N_train)
        tau_main = bound_factor * sigma_eff / np.sqrt(N_train / n_bins)
        tau_pair = bound_factor * sigma_eff / np.sqrt(N_train / (n_bins * n_bins))

        y_train_j = jnp.array(y_train)

        # SEQUENTIAL FITTING (EBM-style)
        # 1. Fit global intercept
        # 2. Fit main effects (shape functions) one at a time
        # 3. Select and fit important interactions

        # Current predictions (logits)
        current_logits_train = jnp.zeros(N_train)
        current_logits_test = jnp.zeros(len(y_test))

        # Step 1: Global intercept
        log_odds = np.log(y_train.mean() / (1 - y_train.mean() + 1e-8))
        intercept = jnp.array([log_odds])
        current_logits_train += intercept[0]
        current_logits_test += intercept[0]
        print(f"    Intercept: {float(intercept[0]):.4f}")

        # Step 2: Cyclic main effects (shape functions)
        n_cycles = 5
        shape_params = {feat_idx: jnp.zeros(n_bins) for feat_idx in top_features}

        for cycle in range(n_cycles):
            for feat_idx in top_features:
                X_feat_train = jnp.array(train_onehot[feat_idx])
                X_feat_test = jnp.array(test_onehot[feat_idx])

                # Remove current feature's contribution
                old_contrib = X_feat_train @ shape_params[feat_idx]
                residual_logits = current_logits_train - old_contrib

                # Fit on residuals
                params = {"theta": shape_params[feat_idx].copy()}

                def loss_fn(params):
                    contrib = X_feat_train @ params["theta"]
                    logits = residual_logits + contrib
                    bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)
                    l2 = 0.5 * jnp.sum(params["theta"] ** 2) / (tau_main ** 2 + 1e-8)
                    return bce + l2 / N_train

                opt = optax.adam(0.05)
                opt_state = opt.init(params)

                @jax.jit
                def step(params, opt_state):
                    loss, grads = jax.value_and_grad(loss_fn)(params)
                    updates, opt_state = opt.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    return params, opt_state, loss

                for _ in range(200):  # Fewer steps per feature
                    params, opt_state, loss = step(params, opt_state)

                shape_params[feat_idx] = params["theta"]

                # Update current logits
                new_contrib = X_feat_train @ params["theta"]
                current_logits_train = residual_logits + new_contrib

            if cycle % 2 == 0:
                probs = 1 / (1 + jnp.exp(-current_logits_train))
                train_auc = roc_auc_score(y_train, np.array(probs))
                print(f"    Cycle {cycle}: train AUC = {train_auc:.4f}")

        # Update test logits with shape functions
        for feat_idx in top_features:
            X_feat_test = jnp.array(test_onehot[feat_idx])
            current_logits_test += X_feat_test @ shape_params[feat_idx]

        # Step 3: Add linear term for remaining features
        X_train_j = jnp.array(X_train_s)
        X_test_j = jnp.array(X_test_s)

        beta = jnp.zeros(p_orig)
        tau_beta = bound_factor * sigma_eff / np.sqrt(N_train)

        params = {"beta": beta}
        residual_logits = current_logits_train

        def loss_fn_linear(params):
            linear_contrib = X_train_j @ params["beta"]
            logits = residual_logits + linear_contrib
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)
            l2 = 0.5 * jnp.sum(params["beta"] ** 2) / (tau_beta ** 2 + 1e-8)
            return bce + l2 / N_train

        opt = optax.adam(0.01)
        opt_state = opt.init(params)

        @jax.jit
        def step_linear(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn_linear)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        for i in range(2000):
            params, opt_state, loss = step_linear(params, opt_state)
            if i % 500 == 0:
                print(f"    Linear step {i}: loss = {loss:.4f}")

        # Final logits
        final_logits_train = current_logits_train + X_train_j @ params["beta"]
        final_logits_test = current_logits_test + X_test_j @ params["beta"]

        probs = 1 / (1 + jnp.exp(-final_logits_test))
        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (sequential): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Previous: 0.844, EBM: 0.866, LGBM: 0.872")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bioresponse_sequential()
