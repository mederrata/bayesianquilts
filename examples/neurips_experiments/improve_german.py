"""German Credit: two-lattice approach using top categorical features."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score
from itertools import combinations
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def load_german_data():
    """Load German Credit dataset."""
    import urllib.request
    from pathlib import Path

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    cache_path = Path("data/german/german.data")

    cols = [f"A{i}" for i in range(1, 21)] + ["target"]

    if cache_path.exists():
        df = pd.read_csv(cache_path, sep=" ", header=None, names=cols)
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, cache_path)
        df = pd.read_csv(cache_path, sep=" ", header=None, names=cols)

    return df


def run_german():
    """German Credit with two-lattice approach."""
    print("\n" + "="*60)
    print("GERMAN CREDIT - Two-Lattice Approach")
    print("="*60)

    df = load_german_data()
    y = (df["target"] == 1).astype(int).values  # 1=good, 2=bad
    N = len(y)
    print(f"  N = {N}, class balance = {y.mean():.3f}")

    # LR analysis shows:
    # - A1 (checking status, 4 cats): 1.65 sum abs coef
    # - A3 (credit history, 5 cats): 1.12
    # - A4 (purpose, 10 cats): 1.91 (but many categories)
    # - A6 (savings, 5 cats): 1.25
    # - A2 (duration, num): -0.30
    # - A5 (credit amount, num): -0.27

    # Numeric features
    numeric_cols = ["A2", "A5", "A8", "A11", "A13", "A16", "A18"]
    X_numeric = df[numeric_cols].values.astype(np.float32)

    # Categoricals for lattice (LR top 4)
    # Intercept: A1 × A3 × A6 = 4 × 5 × 5 = 100 cells
    # With N=800 train per fold, ~8 per cell avg - use order 2
    # Beta: A1 only (4 cells) for cell-varying slopes
    lattice_cats_int = ["A1", "A3", "A6"]
    lattice_cats_beta = ["A1"]

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_numeric, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        X_train_num, X_test_num = X_numeric[train_idx], X_numeric[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_num_s = scaler.fit_transform(X_train_num)
        X_test_num_s = scaler.transform(X_test_num)

        # One-hot encode A4 (purpose) for regression features since it's important
        enc = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        A4_train = enc.fit_transform(df.iloc[train_idx][["A4"]].astype(str))
        A4_test = enc.transform(df.iloc[test_idx][["A4"]].astype(str))

        # Combine numeric + one-hot A4
        X_train = np.concatenate([X_train_num_s, A4_train], axis=1)
        X_test = np.concatenate([X_test_num_s, A4_test], axis=1)

        N_train = len(y_train)
        n_features = X_train.shape[1]
        print(f"    Features: {len(numeric_cols)} numeric + {A4_train.shape[1]} A4_onehot = {n_features}")

        # Encode categoricals for lattice
        train_indices = {}
        test_indices = {}
        for col in set(lattice_cats_int + lattice_cats_beta):
            le = LabelEncoder()
            col_vals = df[col].astype(str).values
            le.fit(col_vals)
            train_indices[col] = le.transform(col_vals[train_idx])
            test_indices[col] = le.transform(col_vals[test_idx])

        # Build TWO lattices
        dims_int = []
        for col in lattice_cats_int:
            cardinality = len(np.unique(train_indices[col]))
            dims_int.append(Dimension(col, cardinality))

        dims_beta = []
        for col in lattice_cats_beta:
            cardinality = len(np.unique(train_indices[col]))
            dims_beta.append(Dimension(col, cardinality))

        interactions_int = Interactions(dimensions=dims_int)
        interactions_beta = Interactions(dimensions=dims_beta)

        decomp_intercept = Decomposed(interactions=interactions_int, param_shape=[1], name="intercept")
        decomp_beta = Decomposed(interactions=interactions_beta, param_shape=[n_features], name="beta")

        prior_scales_int = decomp_intercept.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=True,
        )
        prior_scales_beta = decomp_beta.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=True,
        )

        dim_names_int = [d.name for d in decomp_intercept._interactions._dimensions]
        dim_names_beta = [d.name for d in decomp_beta._interactions._dimensions]
        train_int_idx = jnp.stack([jnp.array(train_indices[name]) for name in dim_names_int], axis=-1)
        test_int_idx = jnp.stack([jnp.array(test_indices[name]) for name in dim_names_int], axis=-1)
        train_beta_idx = jnp.stack([jnp.array(train_indices[name]) for name in dim_names_beta], axis=-1)
        test_beta_idx = jnp.stack([jnp.array(test_indices[name]) for name in dim_names_beta], axis=-1)

        X_train_j = jnp.array(X_train)
        X_test_j = jnp.array(X_test)
        y_train_j = jnp.array(y_train)

        # Small dataset - be conservative with order
        # Intercept: order 2 (main + pairwise on 3 dims)
        # Beta: order 1 (main effects only on A1)
        max_order_int = 2
        max_order_beta = 1

        active_int = [
            name for name in decomp_intercept._tensor_parts.keys()
            if decomp_intercept.component_order(name) <= max_order_int
        ]
        active_beta = [
            name for name in decomp_beta._tensor_parts.keys()
            if decomp_beta.component_order(name) <= max_order_beta
        ]
        print(f"    Intercept: {len(dims_int)} dims, order {max_order_int}, {len(active_int)} components")
        print(f"    Beta: {len(dims_beta)} dims, order {max_order_beta}, {len(active_beta)} components")

        params = {}
        for name in active_int:
            params[f"int_{name}"] = jnp.zeros(decomp_intercept._tensor_part_shapes[name])
        for name in active_beta:
            params[f"beta_{name}"] = jnp.zeros(decomp_beta._tensor_part_shapes[name])

        def loss_fn(params):
            int_params = {k[4:]: v for k, v in params.items() if k.startswith("int_")}
            beta_params = {k[5:]: v for k, v in params.items() if k.startswith("beta_")}

            cell_intercept = decomp_intercept.lookup_flat(train_int_idx, int_params)[:, 0]
            cell_beta = decomp_beta.lookup_flat(train_beta_idx, beta_params)

            logits = jnp.sum(X_train_j * cell_beta, axis=-1) + cell_intercept
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # Stronger regularization for small dataset
            l2_int = 0.0
            for name, param in int_params.items():
                scale = prior_scales_int.get(name, 1.0)
                l2_int += 0.5 * jnp.sum(param ** 2) / ((scale * 10) ** 2)

            l2_beta = 0.0
            for name, param in beta_params.items():
                scale = prior_scales_beta.get(name, 1.0)
                l2_beta += 0.5 * jnp.sum(param ** 2) / ((scale * 10) ** 2)

            return bce + (l2_int + l2_beta) / N_train

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
        int_params = {k[4:]: v for k, v in params.items() if k.startswith("int_")}
        beta_params = {k[5:]: v for k, v in params.items() if k.startswith("beta_")}

        cell_intercept = decomp_intercept.lookup_flat(test_int_idx, int_params)[:, 0]
        cell_beta = decomp_beta.lookup_flat(test_beta_idx, beta_params)
        logits = jnp.sum(X_test_j * cell_beta, axis=-1) + cell_intercept
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (two-lattice): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Table: 0.788, LGBM: 0.794")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_german()
