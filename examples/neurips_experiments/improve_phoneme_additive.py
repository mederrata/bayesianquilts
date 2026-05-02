"""Phoneme: pure additive model with all 5 features."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
import jax
import jax.numpy as jnp
import optax


def run_phoneme_additive():
    """Phoneme with additive shape functions for all 5 features."""
    print("\n" + "="*60)
    print("PHONEME - Pure Additive (5 shape functions)")
    print("="*60)

    data = fetch_openml(data_id=1489, as_frame=True, parser="auto")
    df = data.frame

    X = df.drop(columns=["Class"]).values.astype(np.float32)
    y = (df["Class"].astype(str) == "1").astype(int).values

    N, p = X.shape
    print(f"  N = {N}, p = {p}")

    class_balance = y.mean()
    avg_fisher_weight = class_balance * (1 - class_balance)
    sigma_eff = 1 / np.sqrt(avg_fisher_weight)
    print(f"  Class balance: {class_balance:.3f}")
    print(f"  Effective σ: {sigma_eff:.3f}")

    # Dyadic-style binning: use 16 bins (2^4) for hierarchical structure
    n_bins = 16
    print(f"\n  Shape functions: {p} features × {n_bins} bins each")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        N_train = len(y_train)

        # Build one-hot encoded bins for each feature
        train_bins = {}
        test_bins = {}
        train_onehot = {}
        test_onehot = {}

        for feat_idx in range(p):
            edges = np.percentile(X_train_s[:, feat_idx], np.linspace(0, 100, n_bins + 1))
            t_idx = np.clip(np.digitize(X_train_s[:, feat_idx], edges[1:-1]), 0, n_bins - 1)
            v_idx = np.clip(np.digitize(X_test_s[:, feat_idx], edges[1:-1]), 0, n_bins - 1)

            train_bins[feat_idx] = t_idx
            test_bins[feat_idx] = v_idx

            t_oh = np.zeros((N_train, n_bins))
            t_oh[np.arange(N_train), t_idx] = 1
            v_oh = np.zeros((len(y_test), n_bins))
            v_oh[np.arange(len(y_test)), v_idx] = 1

            train_onehot[feat_idx] = jnp.array(t_oh)
            test_onehot[feat_idx] = jnp.array(v_oh)

        # Theory-based priors
        c = 0.5
        bound_factor = np.sqrt(c / (1 - c))

        tau_global = bound_factor * sigma_eff / np.sqrt(N_train)
        tau_main = bound_factor * sigma_eff / np.sqrt(N_train / n_bins)

        y_train_j = jnp.array(y_train)

        # Initialize parameters
        intercept = jnp.zeros(1)
        shape_params = {i: jnp.zeros(n_bins) for i in range(p)}

        params = {"intercept": intercept, "shapes": shape_params}

        print(f"    τ_global: {tau_global:.4f}, τ_main: {tau_main:.4f}")

        def loss_fn(params):
            logits = params["intercept"][0] * jnp.ones(N_train)

            for feat_idx in range(p):
                logits = logits + train_onehot[feat_idx] @ params["shapes"][feat_idx]

            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # Regularization
            l2_int = 0.5 * jnp.sum(params["intercept"] ** 2) / (tau_global ** 2)
            l2_shapes = 0.0
            for feat_idx in range(p):
                l2_shapes += 0.5 * jnp.sum(params["shapes"][feat_idx] ** 2) / (tau_main ** 2)

            return bce + (l2_int + l2_shapes) / N_train

        opt = optax.adam(0.02)
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
        logits_test = params["intercept"][0] * jnp.ones(len(y_test))
        for feat_idx in range(p):
            logits_test = logits_test + test_onehot[feat_idx] @ params["shapes"][feat_idx]

        probs = 1 / (1 + jnp.exp(-logits_test))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (additive): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Table: 0.937, RF: 0.961")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_phoneme_additive()
