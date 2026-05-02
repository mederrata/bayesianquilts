"""Bioresponse: Multiple clusterings as lattice dimensions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def run_bioresponse_cluster():
    """Bioresponse with multiple clusterings as lattice."""
    print("\n" + "="*60)
    print("BIORESPONSE - Multiple Clusterings as Bins")
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

    # Multiple clusterings: different random seeds, different feature subsets
    n_clusters_per_dim = 6
    n_cluster_dims = 3
    total_cells = n_clusters_per_dim ** n_cluster_dims
    print(f"\n  Lattice: {n_cluster_dims} clusterings × {n_clusters_per_dim} clusters = {total_cells} cells")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        N_train = len(y_train)
        n_features = X_train.shape[1]

        # Create multiple clusterings
        dimensions = []
        train_indices = {}
        test_indices = {}

        for d in range(n_cluster_dims):
            # Each clustering uses different random init + feature subset
            np.random.seed(42 + d * 100 + fold_idx)

            # Random feature subset (50% of features)
            feat_mask = np.random.choice(n_features, size=n_features//2, replace=False)
            X_train_sub = X_train[:, feat_mask]
            X_test_sub = X_test[:, feat_mask]

            # K-means clustering
            km = KMeans(n_clusters=n_clusters_per_dim, random_state=42 + d, n_init=3)
            train_labels = km.fit_predict(X_train_sub)
            test_labels = km.predict(X_test_sub)

            dim_name = f"cluster{d}"
            train_indices[dim_name] = train_labels
            test_indices[dim_name] = test_labels
            dimensions.append(Dimension(dim_name, n_clusters_per_dim))

            # Check cluster balance
            counts = np.bincount(train_labels, minlength=n_clusters_per_dim)
            print(f"    Cluster dim {d}: sizes = {counts}")

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

        X_train_j = jnp.array(X_train)
        X_test_j = jnp.array(X_test)
        y_train_j = jnp.array(y_train)

        # Use up to order 2 interactions
        active_components = [name for name in decomp._tensor_parts.keys() if decomp.component_order(name) <= 2]
        intercept_params = {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in active_components}

        params = {"intercept": intercept_params}

        # Prior variances
        prior_vars = {}
        for name in active_components:
            order = decomp.component_order(name)
            if order == 0:
                prior_vars[name] = tau_global ** 2
            elif order == 1:
                avg_count = N_train / n_clusters_per_dim
                tau = bound_factor * sigma_eff / np.sqrt(max(avg_count, 1))
                prior_vars[name] = tau ** 2
            else:
                avg_count = N_train / (n_clusters_per_dim ** order)
                tau = bound_factor * sigma_eff / np.sqrt(max(avg_count, 1))
                prior_vars[name] = tau ** 2

        print(f"    τ_global: {tau_global:.4f}")
        print(f"    Active components: {len(active_components)}")

        def loss_fn(params):
            int_vals = decomp.lookup_flat(train_int_idx, params["intercept"])
            intercept = int_vals[:, 0]
            logits = intercept  # Pure cluster-based model
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2_int = 0.0
            for name, param in params["intercept"].items():
                var = prior_vars.get(name, 1.0)
                l2_int += 0.5 * jnp.sum(param ** 2) / (var + 1e-8)

            return bce + l2_int / N_train

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
        logits = intercept
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (cluster lattice): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Previous: 0.844, LGBM: 0.872")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bioresponse_cluster()
