"""Taiwan Credit: improve by using direct bins and higher order."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
from itertools import combinations
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def run_taiwan():
    """Taiwan with improved config."""
    print("\n" + "="*60)
    print("TAIWAN CREDIT - Improved Config")
    print("="*60)

    data = fetch_openml(data_id=42477, as_frame=True, parser="auto")
    df = data.frame.copy()

    # Target
    y = (df["y"].astype(int) == 1).astype(int).values
    N = len(y)
    print(f"  N = {N}")
    print(f"  Class balance: {y.mean():.3f}")

    # Numeric features
    numeric_cols = [f"x{i}" for i in [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
    X_numeric = df[numeric_cols].values.astype(np.float32)
    X_numeric = np.nan_to_num(X_numeric, nan=0.0)

    # Categorical features (for lattice AND one-hot for regression)
    cat_cols = ["x2", "x3", "x4"]  # SEX, EDUCATION, MARRIAGE

    # PAY variables are ordinal (-2 to 8) - use for lattice
    pay_cols = ["x6", "x7", "x8"]  # PAY_0, PAY_2, PAY_3 (most recent)

    # One-hot encode categorical features for regression
    from sklearn.preprocessing import OneHotEncoder
    cat_data = df[cat_cols].fillna(-999).astype(str)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat_onehot = ohe.fit_transform(cat_data).astype(np.float32)
    print(f"  Categorical one-hot shape: {X_cat_onehot.shape}")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_numeric, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        X_train_num, X_test_num = X_numeric[train_idx], X_numeric[test_idx]
        X_train_cat, X_test_cat = X_cat_onehot[train_idx], X_cat_onehot[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize numeric features
        scaler = StandardScaler()
        X_train_num_s = scaler.fit_transform(X_train_num)
        X_test_num_s = scaler.transform(X_test_num)

        # Combine numeric + categorical one-hot
        X_train_s = np.concatenate([X_train_num_s, X_train_cat], axis=1)
        X_test_s = np.concatenate([X_test_num_s, X_test_cat], axis=1)

        # Add pairwise interactions for top numeric features only
        n_top = 5  # Top 5 numeric features for interactions
        pairwise_train = []
        pairwise_test = []
        for i, j in combinations(range(n_top), 2):
            pairwise_train.append(X_train_num_s[:, i] * X_train_num_s[:, j])
            pairwise_test.append(X_test_num_s[:, i] * X_test_num_s[:, j])

        if pairwise_train:
            X_train = np.concatenate([X_train_s, np.stack(pairwise_train, axis=1)], axis=1)
            X_test = np.concatenate([X_test_s, np.stack(pairwise_test, axis=1)], axis=1)
        else:
            X_train, X_test = X_train_s, X_test_s

        N_train = len(y_train)
        n_features = X_train.shape[1]
        print(f"    Features: {X_train_s.shape[1]} + {n_features - X_train_s.shape[1]} pairwise = {n_features}")

        # Build lattice from categorical + top numeric bins
        dimensions = []
        train_indices = {}
        test_indices = {}

        # Add categorical dimensions
        for col in cat_cols:
            le = LabelEncoder()
            col_vals = df[col].fillna(-999).astype(str).values
            le.fit(col_vals)

            train_vals = le.transform(df.iloc[train_idx][col].fillna(-999).astype(str).values)
            test_vals = le.transform(df.iloc[test_idx][col].fillna(-999).astype(str).values)

            cardinality = len(le.classes_)
            train_indices[col] = train_vals
            test_indices[col] = test_vals
            dimensions.append(Dimension(col, cardinality))

        # Add PAY variables (ordinal) to lattice
        for col in pay_cols:
            vals = df[col].fillna(-999).astype(int).values
            le = LabelEncoder()
            le.fit(vals)
            train_vals = le.transform(df.iloc[train_idx][col].fillna(-999).astype(int).values)
            test_vals = le.transform(df.iloc[test_idx][col].fillna(-999).astype(int).values)
            cardinality = len(le.classes_)
            train_indices[col] = train_vals
            test_indices[col] = test_vals
            dimensions.append(Dimension(col, cardinality))

        if not dimensions:
            print("    No valid dimensions, skipping")
            continue

        interactions = Interactions(dimensions=dimensions)
        # Use param_shape=[1] for cell-specific intercepts + global linear coefficients
        # This is simpler than cell-specific coefficients
        decomp = Decomposed(interactions=interactions, param_shape=[1], name="intercept")

        # Get prior scales
        prior_scales = decomp.generalization_preserving_scales(
            noise_scale=1.0,
            total_n=N_train,
            c=0.5,
            per_component=True,
        )

        dim_names = [d.name for d in decomp._interactions._dimensions]
        train_int_idx = jnp.stack([jnp.array(train_indices[name]) for name in dim_names], axis=-1)
        test_int_idx = jnp.stack([jnp.array(test_indices[name]) for name in dim_names], axis=-1)

        X_train_j = jnp.array(X_train)
        X_test_j = jnp.array(X_test)
        y_train_j = jnp.array(y_train)

        # Order 2 is optimal (order 3 overfits)
        max_order = 2
        active_components = [
            name for name in decomp._tensor_parts.keys()
            if decomp.component_order(name) <= max_order
        ]
        print(f"    Lattice: {len(dimensions)} dims, order {max_order}, {len(active_components)} components")

        params = {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in active_components}
        params["_beta"] = jnp.zeros(n_features)  # Global linear coefficients

        def loss_fn(params):
            beta = params.get("_beta", jnp.zeros(n_features))
            decomp_params = {k: v for k, v in params.items() if k != "_beta"}

            # Cell-specific intercepts + global linear
            cell_intercept = decomp.lookup_flat(train_int_idx, decomp_params)[:, 0]
            logits = jnp.sum(X_train_j * beta, axis=-1) + cell_intercept

            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # L2 regularization on decomposed intercepts
            l2_decomp = 0.0
            for name, param in decomp_params.items():
                scale = prior_scales.get(name, 1.0)
                l2_decomp += 0.5 * jnp.sum(param ** 2) / ((scale * 20) ** 2)

            # L2 regularization on global beta (weak)
            l2_beta = 0.5 * jnp.sum(beta ** 2) / (1.0 ** 2)

            return bce + (l2_decomp + l2_beta) / N_train

        opt = optax.adam(0.01)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        for i in range(5000):
            params, opt_state, loss = step(params, opt_state)
            if i % 1000 == 0:
                print(f"    Step {i}: loss = {loss:.4f}")

        # Evaluate
        beta = params.get("_beta", jnp.zeros(n_features))
        decomp_params = {k: v for k, v in params.items() if k != "_beta"}
        cell_intercept = decomp.lookup_flat(test_int_idx, decomp_params)[:, 0]
        logits = jnp.sum(X_test_j * beta, axis=-1) + cell_intercept
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (improved): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Table: 0.761, EBM: 0.784")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_taiwan()
