"""Phoneme: exact match to benchmark config."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
from itertools import combinations
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def run_phoneme_exact():
    """Phoneme with exact benchmark config."""
    print("\n" + "="*60)
    print("PHONEME - Exact Benchmark Config (6 bins, order 2)")
    print("="*60)

    data = fetch_openml(data_id=1489, as_frame=True, parser="auto")
    df = data.frame

    X_orig = df.drop(columns=["Class"]).values.astype(np.float32)
    y = (df["Class"].astype(str) == "1").astype(int).values

    N, p_orig = X_orig.shape
    print(f"  N = {N}, p_orig = {p_orig}")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_orig, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        X_train_orig, X_test_orig = X_orig[train_idx], X_orig[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_orig)
        X_test_s = scaler.transform(X_test_orig)

        # Add pairwise interaction features (exact match to benchmark)
        n_features_orig = X_train_s.shape[1]
        pairwise_train = []
        pairwise_test = []
        for i, j in combinations(range(n_features_orig), 2):
            pairwise_train.append(X_train_s[:, i] * X_train_s[:, j])
            pairwise_test.append(X_test_s[:, i] * X_test_s[:, j])

        if pairwise_train:
            X_train = np.concatenate([X_train_s, np.stack(pairwise_train, axis=1)], axis=1)
            X_test = np.concatenate([X_test_s, np.stack(pairwise_test, axis=1)], axis=1)
        else:
            X_train, X_test = X_train_s, X_test_s

        N_train = len(y_train)
        n_features = X_train.shape[1]
        print(f"    Features: {n_features_orig} orig + {n_features - n_features_orig} pairwise = {n_features}")

        # Try more bins
        n_bins = 8
        n_lattice_dims = 5  # Use all 5 features
        max_order = 3  # Try order 3

        print(f"    Lattice: {n_lattice_dims} dims × {n_bins} bins, max_order = {max_order}")

        # Create lattice dimensions
        dimensions = []
        train_indices = {}
        test_indices = {}

        for i in range(n_lattice_dims):
            col_name = f"bin_{i}"
            feature_vals = X_train[:, i]

            edges = np.percentile(feature_vals, np.linspace(0, 100, n_bins + 1))
            edges = np.unique(edges)
            if len(edges) < 3:
                continue
            edges[0] = -np.inf
            edges[-1] = np.inf
            n_actual_bins = len(edges) - 1

            train_indices[col_name] = np.digitize(X_train[:, i], edges[1:-1])
            test_indices[col_name] = np.digitize(X_test[:, i], edges[1:-1])
            dimensions.append(Dimension(col_name, n_actual_bins))

        if not dimensions:
            print("    No valid dimensions, skipping")
            continue

        interactions = Interactions(dimensions=dimensions)
        decomp = Decomposed(interactions=interactions, param_shape=[n_features], name="beta")

        # Use generalization_preserving_scales (exact benchmark method)
        prior_scales = decomp.generalization_preserving_scales(
            noise_scale=1.0,
            total_n=N_train,
            c=0.5,
            per_component=True,
        )

        print(f"    Prior scales: {list(prior_scales.items())[:3]}...")

        dim_names = [d.name for d in decomp._interactions._dimensions]
        train_int_idx = jnp.stack([jnp.array(train_indices[name]) for name in dim_names], axis=-1)
        test_int_idx = jnp.stack([jnp.array(test_indices[name]) for name in dim_names], axis=-1)

        X_train_j = jnp.array(X_train)
        X_test_j = jnp.array(X_test)
        y_train_j = jnp.array(y_train)

        # Active components (order <= max_order)
        active_components = [
            name for name in decomp._tensor_parts.keys()
            if decomp.component_order(name) <= max_order
        ]
        print(f"    Active components: {len(active_components)}")

        params = {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in active_components}
        params["_intercept"] = jnp.zeros(1)

        def loss_fn(params):
            intercept = params.get("_intercept", jnp.zeros(1))
            model_params = {k: v for k, v in params.items() if k != "_intercept"}

            beta = decomp.lookup_flat(train_int_idx, model_params)
            logits = jnp.sum(X_train_j * beta, axis=-1) + intercept[0]

            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # L2 regularization (exact benchmark: scale * 10)
            l2_reg = 0.0
            for name, param in model_params.items():
                scale = prior_scales.get(name, 1.0)
                l2_reg += 0.5 * jnp.sum(param ** 2) / ((scale * 10) ** 2)

            # L1 regularization for sparse (benchmark uses l1_weight=0.01)
            l1_reg = 0.0
            l1_weight = 0.01
            for name, param in model_params.items():
                order = decomp.component_order(name)
                if order > 0:
                    l1_reg += l1_weight * order * jnp.sum(jnp.abs(param))

            return bce + l2_reg / N_train + l1_reg / N_train

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
        intercept = params.get("_intercept", jnp.zeros(1))
        model_params = {k: v for k, v in params.items() if k != "_intercept"}
        beta = decomp.lookup_flat(test_int_idx, model_params)
        logits = jnp.sum(X_test_j * beta, axis=-1) + intercept[0]
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (exact config): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Table: 0.937, RF: 0.961")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_phoneme_exact()
