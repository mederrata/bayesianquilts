"""German Credit: Simple intercept lattice on A1 (checking status) only."""
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
    """German Credit: match LR features + A1 intercept lattice."""
    print("\n" + "="*60, flush=True)
    print("GERMAN CREDIT - Full Features + A1 Lattice", flush=True)
    print("="*60, flush=True)

    df = load_german_data()
    y = (df["target"] == 1).astype(int).values
    N = len(y)
    print(f"  N = {N}, class balance = {y.mean():.3f}", flush=True)

    # All categoricals
    cats = ["A1", "A3", "A4", "A6", "A7", "A9", "A10", "A12", "A14", "A15", "A17", "A19", "A20"]
    nums = ["A2", "A5", "A8", "A11", "A13", "A16", "A18"]

    X_numeric = df[nums].values.astype(np.float32)

    # Intercept lattice: just A1 (checking status, 4 categories)
    # This is the most predictive single feature
    lattice_cats = ["A1"]

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_numeric, y)):
        print(f"\n  Fold {fold_idx + 1}/5:", flush=True)

        X_train_num, X_test_num = X_numeric[train_idx], X_numeric[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_num_s = scaler.fit_transform(X_train_num)
        X_test_num_s = scaler.transform(X_test_num)

        # One-hot ALL categoricals for regression (like LR baseline)
        enc = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        X_train_cat = enc.fit_transform(df.iloc[train_idx][cats].astype(str))
        X_test_cat = enc.transform(df.iloc[test_idx][cats].astype(str))

        X_train = np.concatenate([X_train_num_s, X_train_cat], axis=1)
        X_test = np.concatenate([X_test_num_s, X_test_cat], axis=1)

        N_train = len(y_train)
        n_features = X_train.shape[1]
        print(f"    Features: {len(nums)} numeric + {X_train_cat.shape[1]} one-hot = {n_features}", flush=True)

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

        # Full order (just main effects since 1 dim)
        max_order = 1
        active = [name for name in decomp._tensor_parts.keys() if decomp.component_order(name) <= max_order]
        print(f"    Intercept: A1 lattice ({dims[0].cardinality} cells), {len(active)} components", flush=True)

        # Global beta + cell-varying intercept
        params = {
            "intercept": {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in active},
            "beta": jnp.zeros(n_features),
        }

        c = 0.5
        bound_factor = np.sqrt(c / (1 - c))
        sigma_eff = 1 / np.sqrt(y.mean() * (1 - y.mean()))
        tau_beta = bound_factor * sigma_eff / np.sqrt(N_train)

        prior_vars = {}
        for name in active:
            order = decomp.component_order(name)
            if order == 0:
                prior_vars[name] = (bound_factor * sigma_eff / np.sqrt(N_train)) ** 2
            else:
                avg_count = N_train / dims[0].cardinality
                tau = bound_factor * sigma_eff / np.sqrt(avg_count)
                prior_vars[name] = tau ** 2

        def loss_fn(params):
            int_vals = decomp.lookup_flat(train_idx_arr, params["intercept"])
            intercept = int_vals[:, 0]
            logits = jnp.sum(X_train_j * params["beta"], axis=-1) + intercept
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            l2_int = 0.0
            for name, param in params["intercept"].items():
                var = prior_vars.get(name, 1.0)
                l2_int += 0.5 * jnp.sum(param ** 2) / (var + 1e-8)

            l2_beta = 0.5 * jnp.sum(params["beta"] ** 2) / (tau_beta ** 2 + 1e-8)

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
    print(f"\n  OURS (full features + A1 lattice): {mean_auc:.4f} +/- {std_auc:.4f}", flush=True)
    print(f"  Table: 0.788, LR: 0.786, LGBM: 0.794", flush=True)
    return mean_auc, std_auc


if __name__ == "__main__":
    run_german()
