"""Phoneme: adaptive truncation order based on local sample size."""
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


def run_phoneme_adaptive():
    """Phoneme with adaptive truncation order based on cell counts."""
    print("\n" + "="*60)
    print("PHONEME - Adaptive Order (up to order 3)")
    print("="*60)

    data = fetch_openml(data_id=1489, as_frame=True, parser="auto")
    df = data.frame

    X_orig = df.drop(columns=["Class"]).values.astype(np.float32)
    y = (df["Class"].astype(str) == "1").astype(int).values

    N, p_orig = X_orig.shape
    print(f"  N = {N}, p_orig = {p_orig}")

    class_balance = y.mean()
    avg_fisher_weight = class_balance * (1 - class_balance)
    sigma_eff = 1 / np.sqrt(avg_fisher_weight)
    print(f"  Class balance: {class_balance:.3f}")
    print(f"  Effective σ: {sigma_eff:.3f}")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_orig, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        X_train_orig, X_test_orig = X_orig[train_idx], X_orig[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_orig)
        X_test_s = scaler.transform(X_test_orig)

        # Add pairwise interaction features
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

        # Use optimal config: 5 dims × 8 bins
        # But with non-uniform order: cells with sufficient N^(α) get order 3+
        n_bins = 8
        n_lattice_dims = min(5, n_features_orig)
        max_order = 4  # Allow up to order 4 but with adaptive inclusion

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

        # Compute cell counts for adaptive regularization
        dim_names = [d.name for d in decomp._interactions._dimensions]
        train_int_idx = jnp.stack([jnp.array(train_indices[name]) for name in dim_names], axis=-1)
        test_int_idx = jnp.stack([jnp.array(test_indices[name]) for name in dim_names], axis=-1)

        # Count samples per cell (for full lattice)
        cell_counts = {}
        for comp_name, shape in decomp._tensor_part_shapes.items():
            order = decomp.component_order(comp_name)
            if order == 0:
                cell_counts[comp_name] = N_train
            else:
                # Get the dimensions involved in this component
                # For higher orders, estimate based on uniform distribution
                n_cells = np.prod(shape[:-1])  # Exclude param_shape dimension
                avg_per_cell = N_train / n_cells
                cell_counts[comp_name] = avg_per_cell

        # Determine which components to include based on critical condition
        # γ_local = p / N^(α) < 1 means we need N^(α) > p
        c = 0.5
        active_components = []
        prior_scales = {}

        for name, shape in decomp._tensor_part_shapes.items():
            order = decomp.component_order(name)
            n_local = cell_counts[name]

            # Critical condition: p / N_local < 1
            gamma_local = n_features / n_local

            if order <= max_order and gamma_local < 1.0:
                active_components.append(name)
                # Generalization-preserving scale: τ = σ / √(p · N^(α))
                # With factor for c
                scale = sigma_eff * np.sqrt(c / (1 - c)) / np.sqrt(n_features * n_local)
                prior_scales[name] = scale
                print(f"      {name}: order={order}, N_local≈{n_local:.0f}, γ={gamma_local:.3f}, τ={scale:.4f}")
            elif order <= max_order:
                print(f"      {name}: order={order}, EXCLUDED (γ={gamma_local:.3f} >= 1)")

        print(f"    Active components: {len(active_components)}")

        X_train_j = jnp.array(X_train)
        X_test_j = jnp.array(X_test)
        y_train_j = jnp.array(y_train)

        # Initialize parameters
        params = {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in active_components}
        params["_intercept"] = jnp.zeros(1)

        def loss_fn(params):
            intercept = params.get("_intercept", jnp.zeros(1))
            model_params = {k: v for k, v in params.items() if k != "_intercept"}

            beta = decomp.lookup_flat(train_int_idx, model_params)
            logits = jnp.sum(X_train_j * beta, axis=-1) + intercept[0]

            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # L2 regularization with adaptive scales
            l2_reg = 0.0
            for name, param in model_params.items():
                scale = prior_scales.get(name, 1.0)
                # Scale up by factor to allow fitting
                l2_reg += 0.5 * jnp.sum(param ** 2) / ((scale * 10) ** 2)

            # L1 for sparsity on higher-order terms
            l1_reg = 0.0
            l1_weight = 0.002
            for name, param in model_params.items():
                order = decomp.component_order(name)
                if order > 1:
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

        for i in range(4000):
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
    print(f"\n  OURS (adaptive order 3): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Table: 0.937, RF: 0.961")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_phoneme_adaptive()
