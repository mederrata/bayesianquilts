"""Adult: Full features + lattice with relaxed regularization."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def run_adult():
    """Adult with full LR features + intercept lattice, relaxed regularization."""
    print("\n" + "="*60, flush=True)
    print("ADULT - Full Features + Relaxed Regularization", flush=True)
    print("="*60, flush=True)

    data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    df = data.frame
    y = (df["class"].astype(str) == ">50K").astype(int).values
    N = len(y)
    print(f"  N = {N}, class balance = {y.mean():.3f}", flush=True)

    cats = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    nums = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

    X_numeric = df[nums].values.astype(np.float32)

    # Lattice: marital-status × occupation (key predictors)
    lattice_cats = ["marital-status", "occupation"]

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_numeric, y)):
        print(f"\n  Fold {fold_idx + 1}/5:", flush=True)

        X_train_num, X_test_num = X_numeric[train_idx], X_numeric[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_num_s = scaler.fit_transform(X_train_num)
        X_test_num_s = scaler.transform(X_test_num)

        # One-hot all categoricals
        enc = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        X_train_cat = enc.fit_transform(df.iloc[train_idx][cats].astype(str))
        X_test_cat = enc.transform(df.iloc[test_idx][cats].astype(str))

        X_train = np.concatenate([X_train_num_s, X_train_cat], axis=1)
        X_test = np.concatenate([X_test_num_s, X_test_cat], axis=1)

        N_train = len(y_train)
        n_features = X_train.shape[1]
        print(f"    Features: {len(nums)} + {X_train_cat.shape[1]} = {n_features}", flush=True)

        # Encode lattice
        train_indices = {}
        test_indices = {}
        for col in lattice_cats:
            le = LabelEncoder()
            col_vals = df[col].astype(str).values
            le.fit(col_vals)
            train_indices[col] = le.transform(col_vals[train_idx])
            test_indices[col] = le.transform(col_vals[test_idx])

        dims = [Dimension(col, len(np.unique(train_indices[col]))) for col in lattice_cats]
        total_cells = np.prod([d.cardinality for d in dims])

        interactions = Interactions(dimensions=dims)
        decomp = Decomposed(interactions=interactions, param_shape=[1], name="intercept")

        prior_scales = decomp.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=True,
        )

        dim_names = [d.name for d in decomp._interactions._dimensions]
        train_idx_arr = jnp.stack([jnp.array(train_indices[name]) for name in dim_names], axis=-1)
        test_idx_arr = jnp.stack([jnp.array(test_indices[name]) for name in dim_names], axis=-1)

        X_train_j = jnp.array(X_train)
        X_test_j = jnp.array(X_test)
        y_train_j = jnp.array(y_train)

        max_order = 2
        active = [name for name in decomp._tensor_parts.keys() if decomp.component_order(name) <= max_order]
        print(f"    Lattice: {total_cells} cells, order {max_order}, {len(active)} components", flush=True)

        # Initialize parameters
        params = {
            "intercept": {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in active},
            "beta": jnp.zeros(n_features),
        }

        # RELAXED regularization - use multiplier on prior scales
        scale_multiplier = 50.0  # More relaxed (was effectively 1.0)

        def loss_fn(params):
            int_vals = decomp.lookup_flat(train_idx_arr, params["intercept"])
            intercept = int_vals[:, 0]
            logits = jnp.sum(X_train_j * params["beta"], axis=-1) + intercept
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # L2 on intercept with theory-based scales
            l2_int = 0.0
            for name, param in params["intercept"].items():
                scale = prior_scales.get(name, 1.0) * scale_multiplier
                l2_int += 0.5 * jnp.sum(param ** 2) / (scale ** 2 + 1e-8)

            # L2 on beta - use LR-like regularization
            l2_beta = 0.5 * jnp.sum(params["beta"] ** 2) * 0.01  # C=100 equivalent

            return bce + (l2_int + l2_beta) / N_train

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
                print(f"    Step {i}: loss = {loss:.4f}", flush=True)

        # Evaluate
        int_vals = decomp.lookup_flat(test_idx_arr, params["intercept"])
        intercept = int_vals[:, 0]
        logits = jnp.sum(X_test_j * params["beta"], axis=-1) + intercept
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}", flush=True)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (relaxed): {mean_auc:.4f} +/- {std_auc:.4f}", flush=True)
    print(f"  LR: 0.907, EBM: 0.930", flush=True)
    return mean_auc, std_auc


if __name__ == "__main__":
    run_adult()
