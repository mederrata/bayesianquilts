"""German Credit: simpler lattice on A1 × A3 with cell-varying beta."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score
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
    """German Credit with A1 × A3 lattice (checking status × credit history)."""
    print("\n" + "="*60)
    print("GERMAN CREDIT - A1 × A3 Lattice")
    print("="*60)

    df = load_german_data()
    y = (df["target"] == 1).astype(int).values
    N = len(y)
    print(f"  N = {N}, class balance = {y.mean():.3f}")

    # A1 (checking status): 4 categories - most predictive (1.65)
    # A3 (credit history): 5 categories - second most predictive (1.12)
    # Total: 4 × 5 = 20 cells, ~50 samples per cell

    numeric_cols = ["A2", "A5", "A8", "A11", "A13", "A16", "A18"]
    X_numeric = df[numeric_cols].values.astype(np.float32)

    # Lattice: A1 × A3 for both intercept and beta
    # Small lattice so can use same for both
    lattice_cats = ["A1", "A3"]

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_numeric, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        X_train_num, X_test_num = X_numeric[train_idx], X_numeric[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_num)
        X_test_s = scaler.transform(X_test_num)

        # One-hot encode A4 (purpose) and A6 (savings) for regression
        enc = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        cats_onehot = enc.fit_transform(df.iloc[train_idx][["A4", "A6"]].astype(str))
        cats_onehot_test = enc.transform(df.iloc[test_idx][["A4", "A6"]].astype(str))

        X_train = np.concatenate([X_train_s, cats_onehot], axis=1)
        X_test = np.concatenate([X_test_s, cats_onehot_test], axis=1)

        N_train = len(y_train)
        n_features = X_train.shape[1]
        print(f"    Features: {len(numeric_cols)} numeric + {cats_onehot.shape[1]} one-hot = {n_features}")

        # Encode lattice categoricals
        train_indices = {}
        test_indices = {}
        for col in lattice_cats:
            le = LabelEncoder()
            col_vals = df[col].astype(str).values
            le.fit(col_vals)
            train_indices[col] = le.transform(col_vals[train_idx])
            test_indices[col] = le.transform(col_vals[test_idx])

        dims = []
        for col in lattice_cats:
            cardinality = len(np.unique(train_indices[col]))
            dims.append(Dimension(col, cardinality))

        interactions = Interactions(dimensions=dims)
        decomp_intercept = Decomposed(interactions=interactions, param_shape=[1], name="intercept")
        decomp_beta = Decomposed(interactions=interactions, param_shape=[n_features], name="beta")

        prior_scales_int = decomp_intercept.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=True,
        )
        prior_scales_beta = decomp_beta.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=True,
        )

        dim_names = [d.name for d in decomp_intercept._interactions._dimensions]
        train_idx_arr = jnp.stack([jnp.array(train_indices[name]) for name in dim_names], axis=-1)
        test_idx_arr = jnp.stack([jnp.array(test_indices[name]) for name in dim_names], axis=-1)

        X_train_j = jnp.array(X_train)
        X_test_j = jnp.array(X_test)
        y_train_j = jnp.array(y_train)

        # Order 2 for both (full pairwise on 2 dims = 4 components total)
        max_order = 2

        active_int = [
            name for name in decomp_intercept._tensor_parts.keys()
            if decomp_intercept.component_order(name) <= max_order
        ]
        active_beta = [
            name for name in decomp_beta._tensor_parts.keys()
            if decomp_beta.component_order(name) <= max_order
        ]
        print(f"    Lattice: {len(dims)} dims, order {max_order}")
        print(f"    Intercept components: {len(active_int)}, Beta components: {len(active_beta)}")

        params = {}
        for name in active_int:
            params[f"int_{name}"] = jnp.zeros(decomp_intercept._tensor_part_shapes[name])
        for name in active_beta:
            params[f"beta_{name}"] = jnp.zeros(decomp_beta._tensor_part_shapes[name])

        def loss_fn(params):
            int_params = {k[4:]: v for k, v in params.items() if k.startswith("int_")}
            beta_params = {k[5:]: v for k, v in params.items() if k.startswith("beta_")}

            cell_intercept = decomp_intercept.lookup_flat(train_idx_arr, int_params)[:, 0]
            cell_beta = decomp_beta.lookup_flat(train_idx_arr, beta_params)

            logits = jnp.sum(X_train_j * cell_beta, axis=-1) + cell_intercept
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # Use theory-based prior scales
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

        cell_intercept = decomp_intercept.lookup_flat(test_idx_arr, int_params)[:, 0]
        cell_beta = decomp_beta.lookup_flat(test_idx_arr, beta_params)
        logits = jnp.sum(X_test_j * cell_beta, axis=-1) + cell_intercept
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (A1×A3 lattice): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Table: 0.788, LGBM: 0.794")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_german()
