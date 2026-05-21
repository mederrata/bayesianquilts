"""Bioresponse: Sparse linear latent factors (z = X @ A) with horseshoe prior."""
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

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def run_bioresponse_sparse_latent():
    """Bioresponse with learned sparse linear latents for lattice."""
    print("\n" + "="*60)
    print("BIORESPONSE - Sparse Linear Latents (z = X @ A)")
    print("="*60)

    data = fetch_openml(data_id=4134, as_frame=True, parser="auto")
    df = data.frame

    X = df.drop(columns=["target"]).values.astype(np.float32)
    y = df["target"].astype(int).values

    N, p_orig = X.shape
    print(f"  N = {N}, p = {p_orig}")
    print(f"  Sparsity: {(X == 0).sum() / X.size:.1%}")

    class_balance = y.mean()
    avg_fisher_weight = class_balance * (1 - class_balance)
    sigma_eff = 1 / np.sqrt(avg_fisher_weight)
    print(f"  Class balance: {class_balance:.3f}")
    print(f"  Effective σ: {sigma_eff:.3f}")

    # Latent dimensions for lattice
    n_latent = 3
    n_bins = 6
    print(f"\n  Latents: {n_latent} dimensions × {n_bins} bins = {n_bins**n_latent} cells")
    print(f"  Encoding matrix: {p_orig} → {n_latent}")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        N_train = len(y_train)
        n_features = X_train.shape[1]

        # Normalize (but keep non-negative - just scale by max)
        X_max = X_train.max(axis=0, keepdims=True) + 1e-8
        X_train_n = X_train / X_max
        X_test_n = X_test / X_max

        X_train_j = jnp.array(X_train_n)
        X_test_j = jnp.array(X_test_n)
        y_train_j = jnp.array(y_train)

        # Initialize encoding matrix A (p x n_latent) - sparse via horseshoe-like prior
        # Use small random init
        key = jax.random.PRNGKey(fold_idx)
        A_init = jax.random.normal(key, (n_features, n_latent)) * 0.01

        # Global scale for horseshoe (small = more sparse)
        tau_A = 0.1 / np.sqrt(n_features)

        # Parameters: encoding matrix A + classification weights
        params = {
            "A": A_init,
            "intercept_global": jnp.zeros(1),
            "beta_latent": jnp.zeros(n_latent),
        }

        def loss_fn(params):
            # Compute latents: z = softplus(X @ A) to ensure non-negative
            A = params["A"]
            z_train = jax.nn.softplus(X_train_j @ A)

            # Classification: logit = intercept + z @ beta
            logits = params["intercept_global"][0] + z_train @ params["beta_latent"]

            # BCE loss
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # Horseshoe-like prior on A (L1 + L2 approximation)
            # Encourages sparsity in encoding matrix
            l1_A = jnp.sum(jnp.abs(A))
            l2_A = jnp.sum(A ** 2)
            reg_A = tau_A * l1_A + 0.5 * tau_A**2 * l2_A

            # L2 on beta
            l2_beta = 0.5 * jnp.sum(params["beta_latent"] ** 2) / (sigma_eff**2 + 1e-8)

            return bce + (reg_A + l2_beta) / N_train

        opt = optax.adam(0.01)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        # Train encoding + classifier jointly
        for i in range(3000):
            params, opt_state, loss = step(params, opt_state)
            if i % 1000 == 0:
                # Count non-zero elements in A
                A_sparsity = (jnp.abs(params["A"]) < 0.001).mean()
                print(f"    Step {i}: loss = {loss:.4f}, A sparsity = {A_sparsity:.1%}")

        # Evaluate
        z_test = jax.nn.softplus(X_test_j @ params["A"])
        logits = params["intercept_global"][0] + z_test @ params["beta_latent"]
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

        # Show top features per latent
        A_np = np.array(params["A"])
        for k in range(n_latent):
            top_idx = np.argsort(np.abs(A_np[:, k]))[-5:][::-1]
            top_vals = A_np[top_idx, k]
            print(f"    Latent {k}: top features = {top_idx}, weights = {top_vals}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (sparse linear latents): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Previous: 0.844, LGBM: 0.872")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bioresponse_sparse_latent()
