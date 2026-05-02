"""Taiwan Credit: Apply learnings from bioresponse - ordinal smoothness, theory-based priors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
from itertools import combinations
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def ordinal_smoothness_penalty(params: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Penalize adjacent differences along ordinal axis."""
    diffs = jnp.diff(params, axis=axis)
    return jnp.sum(diffs ** 2)


def compute_max_order(idx: np.ndarray, min_samples: int = 20):
    """Compute max order based on cell occupancy."""
    N, k = idx.shape
    for order in range(1, k + 1):
        min_count = N
        for dims in combinations(range(k), order):
            cell_counts = {}
            for i in range(N):
                cell = tuple(idx[i, d] for d in dims)
                cell_counts[cell] = cell_counts.get(cell, 0) + 1
            if cell_counts:
                min_count = min(min_count, min(cell_counts.values()))
        if min_count < min_samples:
            return order - 1
    return k


def run_taiwan():
    """Taiwan with improved config using bioresponse learnings."""
    print("\n" + "="*60)
    print("TAIWAN CREDIT - v2 with Ordinal Smoothness")
    print("="*60)

    data = fetch_openml(data_id=42477, as_frame=True, parser="auto")
    df = data.frame.copy()

    y = (df["y"].astype(int) == 1).astype(int).values
    N = len(y)
    print(f"N = {N}, pos_rate = {y.mean():.3f}")

    # Numeric features
    numeric_cols = [f"x{i}" for i in [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
    X_numeric = df[numeric_cols].values.astype(np.float32)
    X_numeric = np.nan_to_num(X_numeric, nan=0.0)

    # Categorical features (unordered)
    cat_cols = ["x2", "x3", "x4"]  # SEX, EDUCATION, MARRIAGE

    # PAY variables are ORDINAL (-2 to 8) - apply smoothness
    pay_cols = ["x6", "x7", "x8"]  # PAY_0, PAY_2, PAY_3

    # One-hot encode categorical features
    cat_data = df[cat_cols].fillna(-999).astype(str)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat_onehot = ohe.fit_transform(cat_data).astype(np.float32)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_numeric, y)):
        print(f"\nFold {fold_idx + 1}/5:")

        X_train_num, X_test_num = X_numeric[train_idx], X_numeric[test_idx]
        X_train_cat, X_test_cat = X_cat_onehot[train_idx], X_cat_onehot[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        N_train = len(y_train)

        # Standardize
        scaler = StandardScaler()
        X_train_num_s = scaler.fit_transform(X_train_num)
        X_test_num_s = scaler.transform(X_test_num)

        # Combine
        X_train_base = np.concatenate([X_train_num_s, X_train_cat], axis=1)
        X_test_base = np.concatenate([X_test_num_s, X_test_cat], axis=1)

        # Add pairwise interactions for top numeric features
        n_top = 5
        pairwise_train = []
        pairwise_test = []
        for i, j in combinations(range(n_top), 2):
            pairwise_train.append(X_train_num_s[:, i] * X_train_num_s[:, j])
            pairwise_test.append(X_test_num_s[:, i] * X_test_num_s[:, j])

        if pairwise_train:
            X_train = np.concatenate([X_train_base, np.stack(pairwise_train, axis=1)], axis=1)
            X_test = np.concatenate([X_test_base, np.stack(pairwise_test, axis=1)], axis=1)
        else:
            X_train, X_test = X_train_base, X_test_base
        n_features = X_train.shape[1]

        # Build lattice dimensions
        dimensions = []
        train_indices = {}
        test_indices = {}
        ordinal_dims = set()  # Track which dimensions are ordinal

        # Add categorical dimensions (unordered)
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

        # Add PAY variables (ordinal) - mark for smoothness
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
            ordinal_dims.add(col)

        # Add LIMIT_BAL (credit limit) as binned dimension
        limit_bal = df["x1"].fillna(0).astype(float).values
        limit_train = limit_bal[train_idx]
        limit_test = limit_bal[test_idx]
        limit_edges = np.percentile(limit_train, [16.7, 33.3, 50, 66.7, 83.3])  # 6 bins
        limit_train_bins = np.digitize(limit_train, limit_edges)
        limit_test_bins = np.digitize(limit_test, limit_edges)
        train_indices["LIMIT_BAL"] = limit_train_bins
        test_indices["LIMIT_BAL"] = limit_test_bins
        dimensions.append(Dimension("LIMIT_BAL", 6))
        ordinal_dims.add("LIMIT_BAL")


        interactions = Interactions(dimensions=dimensions)
        decomp = Decomposed(interactions=interactions, param_shape=[1], name="intercept")

        # Build index arrays
        dim_names = [d.name for d in decomp._interactions._dimensions]
        train_int_idx = np.stack([train_indices[name] for name in dim_names], axis=-1)
        test_int_idx = np.stack([test_indices[name] for name in dim_names], axis=-1)

        # For Taiwan, order 2 works empirically - the theory-based regularization
        # handles the sparse cells appropriately
        max_order = 2
        print(f"  Using order {max_order} (regularization handles sparse cells)")

        # Get prior scales
        prior_scales = decomp.generalization_preserving_scales(
            noise_scale=1.0, total_n=N_train, c=0.5, per_component=True
        )

        active_components = [
            name for name in decomp._tensor_parts.keys()
            if decomp.component_order(name) <= max_order
        ]
        print(f"  Lattice: {len(dimensions)} dims, {len(active_components)} components")

        # Theory-based tau_beta
        class_balance = y_train.mean()
        sigma_eff = 1 / np.sqrt(class_balance * (1 - class_balance))
        tau_beta = sigma_eff / np.sqrt(N_train)
        print(f"  tau_beta = {tau_beta:.4f}")

        # Initialize parameters
        params = {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in active_components}
        params["_beta"] = jnp.zeros(n_features)

        train_int_idx_j = jnp.array(train_int_idx)
        X_train_j = jnp.array(X_train)
        y_train_j = jnp.array(y_train)

        # Map dimension names to their position and ordinal status
        ordinal_dim_indices = [i for i, name in enumerate(dim_names) if name in ordinal_dims]
        print(f"  Ordinal dimensions (for smoothness): {[dim_names[i] for i in ordinal_dim_indices]}")

        scale_mult = 20.0  # Relaxed for sparse cells
        smooth_wt = 0.5    # Moderate smoothness on ordinal PAY vars

        def loss_fn(params):
            beta = params["_beta"]
            decomp_params = {k: v for k, v in params.items() if k != "_beta"}

            cell_intercept = decomp.lookup_flat(train_int_idx_j, decomp_params)[:, 0]
            logits = jnp.sum(X_train_j * beta, axis=-1) + cell_intercept
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_train_j * logits)

            # L2 regularization with theory-based scales
            l2_decomp = 0.0
            for name, param in decomp_params.items():
                scale = prior_scales.get(name, 1.0)
                l2_decomp += 0.5 * jnp.sum(param ** 2) / ((scale * scale_mult) ** 2 + 1e-8)

            # L2 on beta - weak regularization like original config
            l2_beta = 0.5 * jnp.sum(beta ** 2) / (1.0 ** 2)

            # Ordinal smoothness on PAY dimensions
            smooth_penalty = 0.0
            for name, param in decomp_params.items():
                order = decomp.component_order(name)
                if order > 0:
                    # Apply smoothness along ordinal axes only
                    for axis, dim_name in enumerate(name.split("_")):
                        if dim_name in ordinal_dims and param.shape[axis] > 1:
                            smooth_penalty += ordinal_smoothness_penalty(param, axis=axis)

            reg = (l2_decomp + l2_beta) / N_train + smooth_wt * smooth_penalty / N_train
            return bce + reg

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.001, peak_value=0.02, warmup_steps=500,
            decay_steps=4500, end_value=0.001
        )
        opt = optax.adam(learning_rate=schedule)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        for i in range(5001):
            params, opt_state, loss = step(params, opt_state)
            if i % 1000 == 0:
                print(f"  Step {i}: loss = {loss:.4f}")

        # Evaluate
        test_int_idx_j = jnp.array(test_int_idx)
        X_test_j = jnp.array(X_test)
        beta = params["_beta"]
        decomp_params = {k: v for k, v in params.items() if k != "_beta"}
        cell_intercept = decomp.lookup_flat(test_int_idx_j, decomp_params)[:, 0]
        logits = jnp.sum(X_test_j * beta, axis=-1) + cell_intercept
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"  AUC = {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\nOURS: {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"Previous: 0.766, EBM: 0.784")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_taiwan()
