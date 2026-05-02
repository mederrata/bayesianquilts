"""Bioresponse: Binary/low-cardinality categoricals for lattice + proper encoding."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
import jax
import jax.numpy as jnp
import optax

from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def run_bioresponse_binary_cat():
    """Bioresponse with binary/low-cardinality categoricals for lattice."""
    print("\n" + "="*60)
    print("BIORESPONSE - Binary/Low-Card Categoricals + Continuous")
    print("="*60)

    data = fetch_openml(data_id=4134, as_frame=True, parser="auto")
    df = data.frame

    X = df.drop(columns=["target"]).values.astype(np.float32)
    y = df["target"].astype(int).values

    N, p_orig = X.shape
    print(f"  N = {N}, p = {p_orig}")

    class_balance = y.mean()
    avg_fisher_weight = class_balance * (1 - class_balance)
    sigma_eff = 1 / np.sqrt(avg_fisher_weight)
    print(f"  Class balance: {class_balance:.3f}")
    print(f"  Effective σ: {sigma_eff:.3f}")

    # Identify binary and low-cardinality categorical features (2-4 levels)
    binary_cat = []
    low_cat = []
    cont_features = []
    cat_levels = {}

    for i in range(X.shape[1]):
        unique_vals = np.unique(X[:, i])
        n_unique = len(unique_vals)
        if n_unique == 2:
            binary_cat.append(i)
            cat_levels[i] = unique_vals
        elif 3 <= n_unique <= 4:
            low_cat.append(i)
            cat_levels[i] = unique_vals
        elif n_unique > 10:
            cont_features.append(i)

    print(f"  Binary categorical: {len(binary_cat)}")
    print(f"  Low-card categorical (3-4 levels): {len(low_cat)}")
    print(f"  Continuous features: {len(cont_features)}")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        N_train = len(y_train)

        # Scale continuous features only
        X_train_cont = X_train[:, cont_features]
        X_test_cont = X_test[:, cont_features]
        scaler = StandardScaler()
        X_train_cont_s = scaler.fit_transform(X_train_cont)
        X_test_cont_s = scaler.transform(X_test_cont)

        # Find most predictive features via L1-LR on continuous only
        lr = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        lr.fit(X_train_cont_s, y_train)
        cont_importance = np.abs(lr.coef_[0])

        # Also fit LR with binary categoricals to find predictive ones
        X_train_binary = X_train[:, binary_cat].astype(np.float32)
        X_test_binary = X_test[:, binary_cat].astype(np.float32)

        # Standardize binary (0/1 already, but center)
        binary_means = X_train_binary.mean(axis=0)
        X_train_binary_c = X_train_binary - binary_means
        X_test_binary_c = X_test_binary - binary_means

        lr_binary = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        lr_binary.fit(X_train_binary_c, y_train)
        binary_importance = np.abs(lr_binary.coef_[0])

        # Select top binary features for lattice
        binary_ranking = sorted(
            [(binary_cat[i], binary_importance[i]) for i in range(len(binary_cat))],
            key=lambda x: -x[1]
        )

        # Select top low-card features for lattice
        X_train_low = X_train[:, low_cat].astype(np.float32)
        X_test_low = X_test[:, low_cat].astype(np.float32)
        if len(low_cat) > 0:
            lr_low = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
            lr_low.fit(X_train_low, y_train)
            low_importance = np.abs(lr_low.coef_[0])
            low_ranking = sorted(
                [(low_cat[i], low_importance[i], len(cat_levels[low_cat[i]])) for i in range(len(low_cat))],
                key=lambda x: -x[1]
            )
        else:
            low_ranking = []

        # Select: 3 binary + 2 low-card for lattice = 5 dimensions
        # Max cells: 2^3 * 4^2 = 8 * 16 = 128 cells (manageable)
        selected_binary = [(idx, 2) for idx, imp in binary_ranking[:3]]
        selected_low = [(idx, n_lev) for idx, imp, n_lev in low_ranking[:2]]
        selected_cat = selected_binary + selected_low

        print(f"    Lattice dims: {[(f'D{idx+1}', n) for idx, n in selected_cat]}")

        # Build lattice dimensions
        dimensions = []
        train_indices = {}
        test_indices = {}

        for feat_idx, n_levels in selected_cat:
            unique_vals = cat_levels[feat_idx]
            val_to_idx = {v: i for i, v in enumerate(unique_vals)}

            train_labels = np.array([val_to_idx.get(v, 0) for v in X_train[:, feat_idx]])
            test_labels = np.array([val_to_idx.get(v, 0) for v in X_test[:, feat_idx]])

            dim_name = f"D{feat_idx+1}"
            train_indices[dim_name] = train_labels
            test_indices[dim_name] = test_labels
            dimensions.append(Dimension(dim_name, n_levels))

        total_cells = 1
        for d in dimensions:
            total_cells *= d.cardinality
        print(f"    Total cells: {total_cells}")

        interactions = Interactions(dimensions=dimensions)
        decomp = Decomposed(interactions=interactions, param_shape=[1], name="intercept")

        # Linear term: continuous features only (scaled)
        # Don't duplicate lattice features in linear term
        X_train_j = jnp.array(X_train_cont_s)
        X_test_j = jnp.array(X_test_cont_s)
        y_train_j = jnp.array(y_train)
        n_linear = X_train_cont_s.shape[1]

        # Theory-based priors
        c = 0.5
        bound_factor = np.sqrt(c / (1 - c))
        tau_global = bound_factor * sigma_eff / np.sqrt(N_train)

        # Build interaction indices
        dim_names = [d.name for d in decomp._interactions._dimensions]
        train_int_idx = jnp.stack([jnp.array(train_indices[name]) for name in dim_names], axis=-1)
        test_int_idx = jnp.stack([jnp.array(test_indices[name]) for name in dim_names], axis=-1)

        # Include interactions based on critical boundary
        active_components = []
        prior_vars = {}
        critical_threshold = 10

        for name in decomp._tensor_parts.keys():
            order = decomp.component_order(name)
            if order == 0:
                avg_samples = N_train
                cell_count = 1
            else:
                # Compute actual cell count for this component
                parts = name.replace("intercept__", "").split("_")
                dims_involved = [p for p in parts if p.startswith("D")]
                cell_count = 1
                for dim in dims_involved:
                    for d, (feat_idx, n_levels) in zip(dimensions, selected_cat):
                        if d.name == dim:
                            cell_count *= n_levels
                            break
                avg_samples = N_train / cell_count

            if avg_samples >= critical_threshold:
                active_components.append(name)
                if order == 0:
                    prior_vars[name] = tau_global ** 2
                else:
                    tau = bound_factor * sigma_eff / np.sqrt(avg_samples)
                    # Extra regularization if close to boundary
                    if avg_samples < critical_threshold * 2:
                        tau *= 0.5  # Shrink more
                    prior_vars[name] = tau ** 2

        intercept_params = {name: jnp.zeros(decomp._tensor_part_shapes[name]) for name in active_components}
        beta = jnp.zeros(n_linear)

        params = {"intercept": intercept_params, "beta": beta}

        tau_beta = bound_factor * sigma_eff / np.sqrt(N_train)
        print(f"    τ_global: {tau_global:.4f}, τ_beta: {tau_beta:.4f}")
        print(f"    Active components: {len(active_components)}")

        def loss_fn(params):
            int_vals = decomp.lookup_flat(train_int_idx, params["intercept"])
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
                print(f"    Step {i}: loss = {loss:.4f}")

        # Evaluate
        int_vals = decomp.lookup_flat(test_int_idx, params["intercept"])
        intercept = int_vals[:, 0]
        logits = jnp.sum(X_test_j * params["beta"], axis=-1) + intercept
        probs = 1 / (1 + jnp.exp(-logits))

        auc = roc_auc_score(y_test, np.array(probs))
        aucs.append(auc)
        print(f"    AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\n  OURS (binary/low-card): {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Previous: 0.844, EBM: 0.866, LGBM: 0.872")
    return mean_auc, std_auc


if __name__ == "__main__":
    run_bioresponse_binary_cat()
